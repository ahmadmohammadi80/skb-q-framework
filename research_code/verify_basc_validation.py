#!/usr/bin/env python3
"""Independent offline audit of GPU-produced BASC-20 validation ZIP.
Usage: python verify_basc_validation.py SKBQ_BASC20_VALIDATION_FULL.zip --manifest BASC20_FROZEN_MANIFEST.json --out audit.json
This program does not perform any GPU evaluation and does not fetch network data.
"""
import argparse,csv,hashlib,json,math,zipfile
from pathlib import Path
import numpy as np

def canon(d): return json.dumps(d,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')
def sha(b): return hashlib.sha256(b).hexdigest()
def close(a,b,tol=1e-10): return abs(float(a)-float(b))<=tol

def audit(source,manifest,out):
 original=json.loads(Path(manifest).read_text(encoding='utf8'))
 mh=original['canonical_self_sha256']
 assert sha(canon({k:v for k,v in original.items() if k!='canonical_self_sha256'}))==mh,'Local frozen manifest checksum mismatch'
 expected_maps=original['chosen_maps']
 expected_order=['BASC20','Ridge20','Lasso20','Ridge30','Lasso30','BestDev20','BestDev30','WeightNormTopK','RTNErrorTopK']
 assert set(expected_order)==set(expected_maps)
 required=['FROZEN_MANIFEST_COPY.json','PLAN_BEFORE_OUTCOMES.json','MODEL_AND_BUDGET_LOCK.json',
           'VALIDATION_TOKENS_LOCK_BEFORE_OUTCOMES.json','POLICY_RESULTS.json','BASC20_PAIRED_COMPARISONS.json']
 with zipfile.ZipFile(source) as z:
  assert z.testzip() is None,'ZIP CRC error'
  names=z.namelist(); assert len(names)==len(set(names)), 'Duplicate filenames in ZIP'
  assert all(not n.startswith('/') and '..' not in n.split('/') for n in names),'Unsafe relative path'
  for f in required:assert f in names,f'Missing {f}'
  def get(f):return json.loads(z.read(f))
  copy=get(required[0]); assert copy==original,'Copied manifest diverges from frozen training manifest'
  plan=get(required[1]); ps=plan.pop('canonical_plan_sha256')
  assert sha(canon(plan))==ps,'Plan self-hash mismatch'
  assert plan['frozen_manifest_canonical_sha256']==mh and plan['target_split']=='validation'
  assert plan['maps']==expected_maps and plan['policy_order']==expected_order
  assert plan['n_blocks']==64 and plan['predictions_per_block']==1023
  plan['canonical_plan_sha256']=ps
  budget=get('MODEL_AND_BUDGET_LOCK.json')
  assert budget['model_revision']==original['model_revision']
  assert budget['embedding_sha256']=='fdb9e308aa4aab913119197abfbabf3bdee64178fc91348f6a759a2e53da07b8'
  assert (budget['linear_params'],budget['group_params'])==(1543569408,5505024)
  assert budget['frozen_manifest_sha256']==mh and budget['plan_sha256']==ps
  tokens=get('VALIDATION_TOKENS_LOCK_BEFORE_OUTCOMES.json')
  assert tokens['split']=='validation' and tokens['n_blocks']==64 and tokens['predictions']==65472
  assert tokens['plan_sha256']==ps and tokens['frozen_manifest_sha256']==mh
  th=tokens['token_sha256'];assert len(th)==64 and th!='ebf30d98a1f52b3ae8d96136565115ae8965e2170dc38dcdc7723e76d46cc163'
  summary=get('POLICY_RESULTS.json');assert len(summary)==9
  assert [x['policy'] for x in summary]==expected_order
  loss={};policy_rows=[]; avg_expected=4+4*14*5505024/1543569408
  for res in summary:
   name=res['policy'];ckname=f'checkpoint_{name}.json'
   assert ckname in names,f'Missing {ckname}'
   ck=get(ckname)
   assert ck['policy']==name and ck['selected_layers']==expected_maps[name]
   assert ck['blocks_complete']==64 and ck['target_token_sha256']==th
   assert ck['plan_sha256']==ps and ck['frozen_manifest_sha256']==mh
   assert ck['avgbit_linear']==res['avgbit_linear'] and close(ck['avgbit_linear'],avg_expected,1e-12)
   mapping={f'model.layers.{i}.self_attn.{p}':8 if i in expected_maps[name] else 4 for i in range(28) for p in ('q_proj','k_proj','v_proj','o_proj')}
   mapping.update({f'model.layers.{i}.mlp.{p}':4 for i in range(28) for p in ('gate_proj','up_proj','down_proj')})
   mapping['lm_head']=4
   map_sha=sha(canon(mapping))
   assert ck['allocation_sha256']==map_sha==res['allocation_sha256'],'Map hash differs'
   expected_fp=sha(canon({'plan_sha256':ps,'frozen_manifest_sha256':mh,
      'model_revision':original['model_revision'],'policy':name,'map_sha256':map_sha,
      'token_sha256':th,'avgbit':float(ck['avgbit_linear'])}))
   assert ck['fingerprint']==expected_fp,'Checkpoint fingerprint differs'
   v=np.asarray(ck['block_mean_nll'],dtype=float)
   assert v.shape==(64,) and np.all(np.isfinite(v))
   nll=float(v.mean());ppl=math.exp(nll)
   assert close(nll,res['mean_nll']) and close(ppl,res['ppl']) and res['predicted_tokens']==65472
   loss[name]=v
   policy_rows.append({'policy':name,'nll':nll,'ppl':ppl,'bits':res['avgbit_linear']})
  comp=get('BASC20_PAIRED_COMPARISONS.json')
  assert comp['target_split']=='validation' and comp['target_token_sha256']==th
  assert comp['frozen_manifest_sha256']==mh and comp['plan_sha256']==ps
  rng=np.random.default_rng(20261019); samples=rng.integers(0,64,size=(5000,64))
  pairs=[]
  for name in expected_order[1:]:
   diff=loss['BASC20']-loss[name];boot=diff[samples].mean(axis=1)
   c=comp['paired_comparisons'][name]
   delta=float(diff.mean()); ci=np.quantile(boot,[.025,.975]).tolist()
   assert close(c['BASC_minus_comparator_mean_NLL'],delta)
   assert np.allclose(c['bootstrap_iid_block_CI95_descriptive'],ci,atol=1e-10)
   assert c['BASC_lower_block_count']==int(np.count_nonzero(diff<0))
   assert c['BASC_higher_block_count']==int(np.count_nonzero(diff>0))
   assert close(c['relative_PPL_reduction_if_positive'],1-math.exp(delta))
   pairs.append({'comparison':'BASC20 - '+name,'delta_nll':delta,'bootstrap95_low':ci[0],
                 'bootstrap95_high':ci[1],'BASC_wins':int((diff<0).sum()),'BASC_loses':int((diff>0).sum()),
                 'relative_ppl_reduction':1-math.exp(delta)})
 result={'audit':'PASS','source_zip_sha256':sha(Path(source).read_bytes()),'frozen_manifest_sha256':mh,
         'plan_sha256':ps,'validation_token_sha256':th,
         'source_license_note':'Source evidence not bundled with this code-only release',
         'evidence_limit':'offline arithmetic and provenance only; not independently rerun on GPU',
         'results':policy_rows,'paired_comparisons':pairs}
 Path(out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 with open(Path(out).with_suffix('.csv'),'w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=policy_rows[0]);w.writeheader();w.writerows(policy_rows)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('gpu_zip');p.add_argument('--manifest',default=str(Path(__file__).with_name('BASC20_FROZEN_MANIFEST.json')));p.add_argument('--out',default='BASC20_AUDIT.json')
 a=p.parse_args();result=audit(a.gpu_zip,a.manifest,a.out)
 print('PASS',len(result['results']),'policies,',len(result['paired_comparisons']),'paired comparisons; audited result:',a.out)
