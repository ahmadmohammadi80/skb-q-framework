import os,json,math,time,hashlib
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
from context_score import ContextScore,allocate,rotation
p=Path(__file__).parent;cfg=json.load(open(p/'protocol.json'));torch.set_num_threads(4)
m=AutoModelForCausalLM.from_pretrained(cfg['model'],revision=cfg['revision']).eval();tok=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision']);orig={k:v.clone() for k,v in m.state_dict().items() if 'q_proj.weight' in k or 'k_proj.weight' in k}
freq=1/(100000**(np.arange(0,64,2)/64));metric=ContextScore(freq,512)
def basis(theta):
 a=torch.full((32,),theta);c,s=torch.diag(a.cos()),torch.diag(a.sin());return torch.cat([torch.cat([c,-s],1),torch.cat([s,c],1)],0)
bases=[basis(a) for a in cfg['angles']]
def quant(w,b):
 L=2**(b-1)-1;scale=w.abs().amax(1)/L;return (w/scale.clamp_min(1e-30)[:,None]).round().clamp(-L,L)*scale[:,None]
def four(q,k,a,b):
 aa=a@a.T;bb=b@b.T;qq=q@q.T;kk=k@k.T;qa=q@a.T;bk=b@k.T
 return float(np.mean([np.trace(aa@r@bb@r.T)+np.trace(qq@r@kk@r.T)-2*np.trace(qa@r@bk@r.T) for r in [rotation(freq,-i) for i in [0,1,16,64]]]))
tables={mode:[] for mode in cfg['methods']};start=time.perf_counter()
for li,layer in enumerate(m.model.layers):
 att=layer.self_attn;assert att.q_proj.bias is None and att.k_proj.bias is None
 qw=att.q_proj.weight.detach();kw=att.k_proj.weight.detach()
 for g in range(3):
  qs=[qw[(g*3+j)*64:(g*3+j+1)*64] for j in range(3)];k=kw[g*64:(g+1)*64];qns=[q.numpy().astype(float) for q in qs];kn=k.numpy().astype(float);group={mode:[] for mode in cfg['methods']}
  for u in bases:
   vals={mode:[] for mode in cfg['methods']}
   for qb,kb in [(4,4),(4,8),(8,4),(8,8)]:
    a=[quant(u@q,qb).numpy().astype(float) for q in qs];b=quant(u@k,kb).numpy().astype(float);c=[metric.components(q,kn,x,b) for q,x in zip(qns,a)]
    vals['context_no_self'].append(sum(max(0,x['cross']) for x in c)/(4*64))
    vals['context_self'].append(sum(max(0,x['cross']+x['self_weight']*x['self_second']) for x in c)/(4*64))
    vals['causal_four'].append(sum(max(0,four(q,kn,x,b)) for q,x in zip(qns,a)))
   for mode in cfg['methods']:group[mode].append(vals[mode])
  for mode in cfg['methods']:tables[mode].append(group[mode])
 print('scored',li,flush=True)
json.dump({'seconds':time.perf_counter()-start,'tables':tables},open(p/'scores.json','w'),indent=2)
def apply(path):
 m.load_state_dict(orig,strict=False)
 with torch.no_grad():
  for i,(choice,bi) in enumerate(path):
   li,g=divmod(i,3);att=m.model.layers[li].self_attn;qb,kb=[(4,4),(4,8),(8,4),(8,8)][choice];u=bases[bi]
   for j in range(3):
    sl=slice((3*g+j)*64,(3*g+j+1)*64);att.q_proj.weight[sl].copy_(quant(u@att.q_proj.weight[sl],qb))
   sl=slice(g*64,(g+1)*64);att.k_proj.weight[sl].copy_(quant(u@att.k_proj.weight[sl],kb))
d=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='test');text='\n\n'.join(d['text']);ids=tok(text,return_tensors='pt').input_ids;blocks=[ids[:,i:i+512] for i in range(0,ids.shape[1]-511,512)];assert len(blocks)==595
out={'protocol':cfg,'test_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'blocks':len(blocks),'results':{}}
for mode in cfg['methods']:
 score,path=allocate(tables[mode],180);apply(path);losses=[]
 with torch.inference_mode():
  for i,b in enumerate(blocks):
   losses.append(float(m(b,labels=b,use_cache=False).loss))
   if i%100==0:print(mode,i,flush=True)
 out['results'][mode]={'ppl':math.exp(np.mean(losses)),'block_nll':losses,'allocation':path,'score':score,'qk_bytes':sum([74752,93184,130048,148480][j] for j,a in path)+90};assert out['results'][mode]['qk_bytes']==10045530
 json.dump(out,open(p/'results.json','w'),indent=2);print(mode,out['results'][mode]['ppl'],flush=True)
