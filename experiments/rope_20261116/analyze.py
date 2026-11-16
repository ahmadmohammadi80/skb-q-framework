"""Recompute matched-budget findings; refuse incomplete evaluation records."""
from pathlib import Path
import json,math,hashlib
import numpy as np
p=Path(__file__).parent
main=json.load(open(p/'results_full.json'))
extra=p/'conditioned_full.json'
if extra.exists():main['results'].update(json.load(open(extra))['results'])
pub=json.load(open(p/'published_rule_results.json'))
n=main['blocks'];assert n==595 and pub['blocks']==n
assert pub['test_text_sha256']==main['text_sha256']
assert set(main['results'])=={'fp32','identity','fixed','conditioned'}
main['results']['published_rule_transplant']=pub
for mode,x in main['results'].items():
 assert len(x['block_nll'])==n
 assert abs(math.exp(np.mean(x['block_nll']))-x['ppl'])<1e-8
 if mode!='fp32':
  assert len(x['allocation'])==90
  assert sum([0,1,3,4][j] for j,b in x['allocation'])==180
  x['storage_QK_bytes']=sum([74752,93184,130048,148480][j] for j,b in x['allocation'])+90
  assert x['storage_QK_bytes']==10045530
packed=json.load(open(p/'packed_qk_manifest.json'))
assert len(packed)==4 and all(x['actual_file_bytes']==10045530 and x['all_values_exactly_match_dequantized_model'] for x in packed.values())
comparisons={}
for other in ['identity','fixed','published_rule_transplant']:
 a=np.array(main['results']['conditioned']['block_nll']);b=np.array(main['results'][other]['block_nll']);dif=a-b
 cis={}
 for length in [1,8,16]:
  rng=np.random.default_rng(20261116);bs=[]
  for _ in range(4000):
   starts=rng.integers(n,size=math.ceil(n/length));ix=((starts[:,None]+np.arange(length))%n).ravel()[:n]
   bs.append(math.exp(a[ix].mean())-math.exp(b[ix].mean()))
  cis[str(length)]=np.quantile(bs,[.025,.975]).tolist()
 comparisons[other]={'delta_ppl_conditioned_minus_comparator':math.exp(a.mean())-math.exp(b.mean()),'mean_paired_nll_difference':float(dif.mean()),'blocks_lower_nll':int((dif<0).sum()),'circular_moving_block_bootstrap_95pct_ppl_difference':cis}
res={'model':main['model'],'revision':main['revision'],'blocks':n,'predicted_tokens':n*511,'tokenized_stream_length':304978,'unscored_tail_tokens':338,'budget_QK_bytes':10045530,'PPL':{m:v['ppl'] for m,v in main['results'].items()},'comparisons':comparisons,'bit_map_changes_conditioned_vs_fixed':sum(x[0]!=y[0] for x,y in zip(main['results']['conditioned']['allocation'],main['results']['fixed']['allocation'])),'basis_changes_conditioned_vs_fixed':sum(x[1]!=y[1] for x,y in zip(main['results']['conditioned']['allocation'],main['results']['fixed']['allocation'])),'inference_scope':'Q/K weight fake quantization; all other parameters and activations FP32; no packed-kernel speed claims','uncertainty_scope':'Exploratory paired block bootstrap of one fixed corpus and one model; not independent training or calibration repetitions.'}
json.dump(main,open(p/'verified_results.json','w'),indent=2);json.dump(res,open(p/'summary.json','w'),indent=2);print(json.dumps(res,indent=2))
