import json,math,numpy as np
from pathlib import Path
p=Path(__file__).parent;r=json.load(open(p/'budget_results.json'));base=json.load(open(p/'independent_results.json'));packed=json.load(open(p/'independent_packed_manifest.json'));assert set(r)=={'5bit_fixed','5bit_conditioned','7bit_fixed','7bit_conditioned'};out={}
for b in [5,6,7]:
 fixed,cond=[base['results'][m] if b==6 else r[f'{b}bit_{m}'] for m in ['fixed','conditioned']];a=np.array(cond['block_nll']);c=np.array(fixed['block_nll']);n=len(a);assert n==len(c)==563
 for mode,item in [('fixed',fixed),('conditioned',cond)]:
  assert abs(math.exp(np.mean(item['block_nll']))-item['ppl'])<1e-8
  assert sum([0,1,1,2][j] for j,z in item['allocation'])==(b-4)*24
  assert packed[mode if b==6 else f'{b}bit_{mode}']['bytes']==48*(128*512*4//8+128*4)+(b-4)*24*64*512*4//8+48+48*128*4
 vals=[];rng=np.random.default_rng(20261116)
 for _ in range(4000):
  ix=((rng.integers(n,size=math.ceil(n/8))[:,None]+np.arange(8))%n).ravel()[:n];vals.append(math.exp(a[ix].mean())-math.exp(c[ix].mean()))
 out[str(b)]={'fixed_ppl':fixed['ppl'],'conditioned_ppl':cond['ppl'],'delta_ppl':cond['ppl']-fixed['ppl'],'bootstrap_95pct_length8':np.quantile(vals,[.025,.975]).tolist(),'qk_bytes':packed['conditioned' if b==6 else f'{b}bit_conditioned']['bytes'],'bit_changes':sum(x[0]!=y[0] for x,y in zip(fixed['allocation'],cond['allocation'])),'basis_changes':sum(x[1]!=y[1] for x,y in zip(fixed['allocation'],cond['allocation']))}
json.dump(out,open(p/'budget_summary.json','w'),indent=2);print(json.dumps(out,indent=2))
