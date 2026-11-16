import json,math,torch,time
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
p=Path(__file__).parent;meta=json.load(open(p/'independent_model.json'));torch.set_num_threads(4)
m=AutoModelForCausalLM.from_pretrained(meta['model'],revision=meta['revision']).eval();tok=AutoTokenizer.from_pretrained(meta['model'],revision=meta['revision']);orig={k:v.clone() for k,v in m.state_dict().items() if 'query_key_value' in k}
def rot(a):
 c,s=torch.cos(a),torch.sin(a);u=torch.eye(64);u[:16,:16]=torch.cat([torch.cat([torch.diag(c),-torch.diag(s)],1),torch.cat([torch.diag(s),torch.diag(c)],1)],0);return u
bases=[rot(torch.full((8,),a)) for a in [0,math.pi/16,math.pi/8,3*math.pi/16]];freq=1/(10000**(torch.arange(0,16,2).float()/16));rs=[rot(freq*r).double() for r in [0,1,16,64]]
def quant(w,b):
 lim=2**(b-1)-1;s=w.abs().amax(1)/lim;return (w/s.clamp_min(1e-30)[:,None]).round().clamp(-lim,lim)*s[:,None]
def err(q,k,a,b):
 q,k,a,b=[x.double() for x in [q,k,a,b]]
 return max(0,float(torch.stack([torch.trace((a@a.T)@r@(b@b.T)@r.T)+torch.trace((q@q.T)@r@(k@k.T)@r.T)-2*torch.trace((q@a.T)@r@(b@k.T)@r.T) for r in rs]).mean()))
opts=[]
for layer in m.gpt_neox.layers:
 w=layer.attention.query_key_value.weight.detach().reshape(8,3,64,512)
 for h in range(8):
  q,k=w[h,0],w[h,1];opts.append([[err(q,k,quant(u@q,qb),quant(u@k,kb)) for qb,kb in [(4,4),(4,8),(8,4),(8,8)]] for u in bases])
 print('scored',len(opts),flush=True)
json.dump(opts,open(p/'independent_scores.json','w'))
def solve(mode,budget,indices=None):
 states={0:(0,[])}
 for i,t in enumerate(opts):
  fixed=min(range(4),key=lambda a:t[a][0]);best=[min(range(4),key=lambda a:t[a][j]) if mode=='conditioned' else (indices[i] if indices is not None else fixed if mode=='fixed' else 0) for j in range(4)];n={}
  for cost,(v,path) in states.items():
   for j,d in enumerate([0,1,1,2]):
    nc=cost+d;nv=v+t[best[j]][j]
    if nc<=budget and (nc not in n or nv<n[nc][0]):n[nc]=(nv,path+[(j,best[j])])
  states=n
 return states[budget]
def apply(path,quantize=True):
 m.load_state_dict(orig,strict=False)
 with torch.no_grad():
  for i,(j,a) in enumerate(path):
   li,h=divmod(i,8);proj=m.gpt_neox.layers[li].attention.query_key_value;w=proj.weight.reshape(8,3,64,512);bias=proj.bias.reshape(8,3,64);u=bases[a]
   for z,b in enumerate([(4,4),(4,8),(8,4),(8,8)][j]):
    wr=u@w[h,z];w[h,z].copy_(quant(wr,b) if quantize else wr);bias[h,z].copy_(u@bias[h,z])
data=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='test');text='\n\n'.join(data['text']);ids=tok(text,return_tensors='pt').input_ids;blocks=[ids[:,i:i+512] for i in range(0,ids.shape[1]-511,512)]
with torch.inference_mode():
 ref=m(blocks[0]).logits;apply([(0,3)]*48,False);delta=(m(blocks[0]).logits-ref).abs().max().item()
m.load_state_dict(orig,strict=False)
result={'model':meta,'blocks':len(blocks),'predicted_tokens':len(blocks)*511,'rotation_max_error':delta,'scope':'QK weights only; FP32 rotated biases; weight-only surrogate; simulated quantization','results':{},'joint_comparator':{}}
for budget in [24,48,72]:
 indices=[min(range(4),key=lambda a:t[a][0]) for t in opts]
 for iteration in range(100):
  val,path=solve('alternating',budget,indices);new=[min(range(4),key=lambda a:opts[i][a][j]) for i,(j,a) in enumerate(path)]
  if new==indices:break
  indices=new
 joint,jpath=solve('conditioned',budget);result['joint_comparator'][str(budget)]={'joint_objective':joint,'alternating_objective':val,'same_map':path==jpath,'iterations':iteration+1}
for mode in ['fp32','identity','fixed','conditioned']:
 if mode=='fp32':score,path=None,None;m.load_state_dict(orig,strict=False)
 else:score,path=solve(mode,48);apply(path)
 losses=[]
 with torch.inference_mode():
  for i,block in enumerate(blocks):
   losses.append(float(m(block,labels=block,use_cache=False).loss))
   if i%50==0:print(mode,i,len(blocks),flush=True)
 result['results'][mode]={'ppl':math.exp(sum(losses)/len(losses)),'block_nll':losses,'surrogate':score,'allocation':path,'theoretical_qk_payload_bytes':(48*(128*512*4//8+128*4)+48*64*512*4//8+48+48*128*4) if path else None}
 json.dump(result,open(p/'independent_results.json','w'),indent=2);print(mode,result['results'][mode]['ppl'],flush=True)
