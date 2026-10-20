#!/usr/bin/env python3
"""Offline independent-arithmetic verifier of the final Qwen locked experiment ZIP.
Requires no GPU, no external model/corpus; does NOT independently reproduce model inference.
"""
import argparse, hashlib, json, math, sys, zipfile
from pathlib import Path

EXPECTED='188866e4c113a60cbee231473984aa52b2b8bcc1399269cb803a481f532a63a6'
def canon(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
def sha(x):return hashlib.sha256(x).hexdigest()
def readz(z,name):return json.loads(z.read(name).decode('utf-8'))
def mean(v):return sum(v)/len(v)
def cvar(v):return mean(sorted(v)[-16:])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('zipfile');args=ap.parse_args()
    with zipfile.ZipFile(args.zipfile) as z:
        names=set(z.namelist())
        assert not any(n.startswith('/') or '..' in Path(n).parts for n in names),'Invalid zip member path'
        f=readz(z,'FROZEN_MANIFEST_COPIED_BEFORE_EVALUATION.json')
        assert f['manifest_sha256']==EXPECTED
        assert sha(canon({k:v for k,v in f.items() if k!='manifest_sha256'}))==EXPECTED
        assert f['target_corpus']['block_start_index']==64 and f['target_corpus']['nonoverlapping_blocks']==64
        token=readz(z,'TARGET_TOKEN_LOCK_BEFORE_EVALUATION.json')
        assert token['manifest_sha256']==EXPECTED
        assert token['block_range']==[64,127] and token['predicted_tokens']==65472
        tokenhash=token['target_token_sha256']
        mlock=readz(z,'MODEL_AND_BUDGET_LOCK.json')
        assert mlock['manifest_sha256']==EXPECTED and mlock['model_sha']==f['model']['revision']
        budget=f['bit_allocation']['all_policy_avgbit_linear']
        assert math.isclose(mlock['avgbit_linear'],budget,rel_tol=0,abs_tol=1e-12)
        losses={}
        for name,selected in f['maps'].items():
            k='checkpoint_'+name+'.json'
            assert k in names,k+' missing'
            q=readz(z,k)
            assert q['policy']==name and q['selected_layers']==selected
            assert q['frozen_manifest_sha256']==EXPECTED and q['target_token_sha256']==tokenhash
            assert q['blocks_complete']==64 and math.isclose(q['avgbit_linear'],budget,rel_tol=0,abs_tol=1e-12)
            assert len(selected)==len(set(selected))==14 and set(selected).issubset(set(range(28)))
            assert sha(canon(selected))==f['map_sha256'][name]
            assert sha(canon({'model_revision':f['model']['revision'],'policy':name,'map':selected,'target_sha':tokenhash,
                              'manifest_sha':EXPECTED,'avgbit':budget,'blocks':64}))==q['fingerprint']
            v=q['block_mean_nll']
            assert len(v)==64 and all(isinstance(a,(int,float)) and math.isfinite(a) for a in v)
            losses[name]=v
        t=readz(z,'PREDECLARED_FINAL_ANALYSIS.json')
        assert t['frozen_manifest_sha256']==EXPECTED and t['input_tokens_sha256']==tokenhash
        assert t['target_prefix_blocks']==[64,127]
        late=losses['Late_30']
        for name,v in losses.items():
            z=t['outcomes'][name]
            assert abs(z['nll']-mean(v))<1e-9
            assert abs(z['ppl']-math.exp(mean(v)))<1e-9
            assert abs(z['cvar75_excess_vs_late']-cvar([a-b for a,b in zip(v,late)]))<1e-9
            print(f'{name:24s} NLL={mean(v):.8f} PPL={math.exp(mean(v)):.6f}  CVaR={cvar([a-b for a,b in zip(v,late)]):+.6f}')
        risk=mean(losses['RiskCVaR_30'])
        bases=f['primary_decision_rule']['comparator_names']
        pass0=all(risk<mean(losses[s]) for s in bases)
        assert pass0==t['mean_nll_primary_predeclared_pass_both_Lasso_and_OLS']
        tails=f['secondary_decision_rule']['comparator_names']
        pass1=all(cvar([a-b for a,b in zip(losses['RiskCVaR_30'],late)]) <
                  cvar([a-b for a,b in zip(losses[s],late)]) for s in tails)
        assert pass1==t['secondary_tail_better_than_all_three']
        for s,stat in t['risk_comparisons'].items():
            d=[a-b for a,b in zip(losses['RiskCVaR_30'],losses[s])]
            assert abs(mean(d)-stat['risk_minus_baseline_mean_nll'])<1e-9
            assert abs(100*(1-math.exp(mean(d)))-stat['relative_ppl_reduction_pct'])<1e-8
        print('\nVERIFIED offline numerical consistency / frozen manifest / all policy checkpoints.')
        print('Primary strictly better than Lasso and OLS:',pass0)
        print('Secondary tail criterion strictly better than Lasso, OLS, blockwise mean:',pass1)
        print('NOT VERIFIED: actual GPU inference provenance, independent documents, or superiority to complete CASQ.')
if __name__=='__main__':
    try:main()
    except Exception as e:print('VERIFICATION FAILED:',str(e),file=sys.stderr);sys.exit(1)
