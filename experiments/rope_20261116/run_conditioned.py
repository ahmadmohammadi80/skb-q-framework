import json,math,hashlib,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
torch.set_num_threads(4)
out=Path(__file__).parent
name='HuggingFaceTB/SmolLM2-135M';revision='93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
model=AutoModelForCausalLM.from_pretrained(name,revision=revision).eval()
tokenizer=AutoTokenizer.from_pretrained(name,revision=revision)
original={k:v.clone() for k,v in model.state_dict().items() if 'q_proj.weight' in k or 'k_proj.weight' in k}
d=64;eye=torch.eye(d);freq=1/(100000**(torch.arange(0,d,2).float()/d))
def rotation(phase):
 c,s=torch.cos(phase),torch.sin(phase)
 return torch.cat([torch.cat([torch.diag(c),-torch.diag(s)],1),torch.cat([torch.diag(s),torch.diag(c)],1)],0)
bases=[rotation(torch.full((32,),a)) for a in [0,math.pi/16,math.pi/8,3*math.pi/16]]
rs=[rotation(freq*r) for r in [0,1,16,64]]
def quant(w,b):
 limit=2**(b-1)-1;scale=w.abs().amax(1)/limit
 return (w/scale.clamp_min(1e-30)[:,None]).round().clamp(-limit,limit)*scale[:,None]
def error(q,k,qh,kh):
 # ||qh.T R kh - q.T R k||_F^2 via 64-dimensional Gram matrices.
 qq=qh@qh.T;kk=kh@kh.T;qo=q@q.T;ko=k@k.T;crossq=q@qh.T;crossk=kh@k.T
 vals=[]
 for r in rs:
  vals.append(torch.trace(qq@r@kk@r.T)+torch.trace(qo@r@ko@r.T)-2*torch.trace(crossq@r@crossk@r.T))
 return max(0,float(torch.stack(vals).mean()))
options=json.load(open(out/'scores.json'))
def allocate(mode):
 states={0:(0,[])}
 for table in options:
  fixed=min(range(4),key=lambda i:table[i][0])
  best=[(min(range(4),key=lambda i:table[i][j]) if mode=='conditioned' else fixed if mode=='fixed' else 0) for j in range(4)]
  nxt={}
  for cost,(val,path) in states.items():
   for j,delta in enumerate([0,1,3,4]):
    nc=cost+delta; nv=val+table[best[j]][j]
    if nc<=180 and (nc not in nxt or nv<nxt[nc][0]):nxt[nc]=(nv,path+[(j,best[j])])
  states=nxt
 return states[180]
def apply(path):
 model.load_state_dict(original,strict=False)
 with torch.no_grad():
  for idx,(choice,bi) in enumerate(path):
   li,g=divmod(idx,3);att=model.model.layers[li].self_attn;qb,kb=[(4,4),(4,8),(8,4),(8,8)][choice];basis=bases[bi]
   for j in range(3):
    sl=slice((g*3+j)*d,(g*3+j+1)*d);att.q_proj.weight[sl].copy_(quant(basis@att.q_proj.weight[sl],qb))
   sl=slice(g*d,(g+1)*d);att.k_proj.weight[sl].copy_(quant(basis@att.k_proj.weight[sl],kb))
data=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='test')
text='\n\n'.join(data['text']);tokens=tokenizer(text,return_tensors='pt').input_ids
# Evaluate every complete non-overlapping 512-token test block.
# This includes the previously inspected 16-block pilot; no selector settings changed.
blocks=[tokens[:,i*512:(i+1)*512] for i in range(tokens.shape[1]//512)]
results={'model':name,'revision':revision,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'blocks':len(blocks),'predicted_tokens':len(blocks)*511,'scope':'Q/K weights only, simulated quantization, all other weights float32','results':{}}
for mode in ['conditioned']:
 if mode!='fp32':
  score,path=allocate(mode);apply(path)
 else:score,path=None,None
 losses=[]
 with torch.inference_mode():
  for block in blocks:losses.append(float(model(block,labels=block).loss));print(mode,len(losses),flush=True)
 results['results'][mode]={'ppl':math.exp(float(np.mean(losses))),'block_nll':losses,'surrogate':score,'allocation':path}
 json.dump(results,open(out/'conditioned_full.json','w'),indent=2)
print(json.dumps({k:v['ppl'] for k,v in results['results'].items()}))
