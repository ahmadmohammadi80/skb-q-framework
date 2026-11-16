"""Search a deterministic small quantizer-generated separation witness.
Existence illustration only; not a randomly sampled performance benchmark.
"""
import json,numpy as np
from pathlib import Path
from context_score import ContextScore,rotation
p=Path(__file__).parent;d,n,L=4,5,16;freq=np.array([1.,.01]);s=ContextScore(freq,L);angles=np.arange(4)*np.pi/16

def quant(w,b):
 lim=2**(b-1)-1;scale=np.max(abs(w),axis=1)/lim;return np.round(w/scale[:,None]).clip(-lim,lim)*scale[:,None]
def old(q,k,a,b):return np.mean([np.linalg.norm(a.T@rotation(freq,-r)@b-q.T@rotation(freq,-r)@k)**2 for r in [0,1,16,64]])
for seed in range(1000):
 rng=np.random.default_rng(seed);q,k=rng.normal(size=(2,d,n));rows=[]
 for j,(qb,kb) in enumerate([(4,8),(8,4)]):
  for angle in angles:
   u=rotation(np.full(2,angle),1);a=quant(u@q,qb);b=quant(u@k,kb);rows.append({'bits':[qb,kb],'angle':float(angle),'old_score':float(old(q,k,a,b)),'new_score':s.score(q,k,a,b)})
 a=min(range(len(rows)),key=lambda i:rows[i]['old_score']);b=min(range(len(rows)),key=lambda i:rows[i]['new_score'])
 if rows[a]['bits']!=rows[b]['bits'] and rows[b]['new_score']<rows[a]['new_score']-1e-8:
  out={'seed':seed,'Q':q.tolist(),'K':k.tolist(),'frequencies':freq.tolist(),'context_length':L,'all_options':rows,'old_selected':a,'new_selected':b,'equal_weight_payload_bytes':d*n*12/8,'scale_bytes':2*d*8,'scope':'Constructed existence example, scores computed from actual row-RTN weights. No generalization or empirical attention-KL superiority claim.'};json.dump(out,open(p/'separation_example.json','w'),indent=2);print(seed,rows[a],rows[b]);break
else:raise RuntimeError('No reversal found')
