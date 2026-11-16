import json,math,hashlib
from pathlib import Path
import numpy as np
p=Path(__file__).parent;r=json.load(open(p/'independent_results.json'));assert set(r['results'])=={'fp32','identity','fixed','conditioned'};n=r['blocks'];assert n==563
packed=json.load(open(p/'independent_packed_manifest.json'));comparisons={}
for mode,x in r['results'].items():
 assert len(x['block_nll'])==n
 assert abs(math.exp(np.mean(x['block_nll']))-x['ppl'])<1e-8
 if mode!='fp32':
  assert len(x['allocation'])==48
  assert sum([0,1,1,2][j] for j,a in x['allocation'])==48
  assert packed[mode]['bytes']==2408496
for mode in ['identity','fixed']:
 a=np.array(r['results']['conditioned']['block_nll']);b=np.array(r['results'][mode]['block_nll']);ci={}
 for length in [1,8,16]:
  rng=np.random.default_rng(20261116);values=[]
  for _ in range(4000):
   ix=((rng.integers(n,size=math.ceil(n/length))[:,None]+np.arange(length))%n).ravel()[:n];values.append(math.exp(a[ix].mean())-math.exp(b[ix].mean()))
  ci[str(length)]=np.quantile(values,[.025,.975]).tolist()
 comparisons[mode]={'delta_ppl':math.exp(a.mean())-math.exp(b.mean()),'moving_block_bootstrap_95pct':ci}
x=r['results']['conditioned']['allocation'];y=r['results']['fixed']['allocation']
out={'PPL':{k:v['ppl'] for k,v in r['results'].items()},'blocks':n,'predicted_tokens':n*511,'actual_qk_bytes_including_bias_and_scale':2408496,'comparisons':comparisons,'bit_changes':sum(a[0]!=b[0] for a,b in zip(x,y)),'basis_changes':sum(a[1]!=b[1] for a,b in zip(x,y)),'joint_comparator':r['joint_comparator'],'uncertainty_scope':'Within one corpus/model; not independent model training repeats. Test split not used to tune grid or bit score.','metadata_correction':'Initial live script used an erroneous hardcoded theoretical byte count. This summary uses a recomputed count verified against serialized files; run_independent.py now computes it instead of hardcoding.'}
json.dump(out,open(p/'independent_summary.json','w'),indent=2);print(json.dumps(out,indent=2))
