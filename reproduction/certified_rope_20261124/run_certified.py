"""Frozen-model data-free RMSNorm-certified Q/K allocation; no test text used.

Usage:
 python run_certified.py --output-dir ./certified_allocation

The source of the isolated certificate is certified_score.py. This CLI does
NOT evaluate PPL; use the separate benchmark kit for equal-byte comparison.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from certified_score import make_certified_tables

MODEL='HuggingFaceTB/SmolLM2-135M'
REVISION='93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
LAYERS,GROUPS,Q_PER_K,DIM,WIDTH=30,3,3,64,576
LENGTH=512
ANGLES=(0.,np.pi/16,np.pi/8,3*np.pi/16)
BIT_PAIRS=((4,4),(4,8),(8,4),(8,8))
UPGRADES=(0,1,3,4)
BUDGET_UNITS=180
BUDGET_BYTES=10045530


def select_certified(table):
    t=np.asarray(table,dtype=float)
    if t.shape!=(90,4,4) or not np.isfinite(t).all() or (t<0).any():
        raise ValueError('expected 90x4x4 finite nonnegative options')
    # Exact multiple-choice cost DP: every group selects one precision and
    # one shared GQA rotation. No basis storage beyond one map byte per group.
    states={0:(0.,())}
    for group in t:
        best=group.argmin(axis=0)
        nxt={}
        for old,(value,path) in states.items():
            for j,delta in enumerate(UPGRADES):
                cost=old+delta
                if cost>BUDGET_UNITS:continue
                score=value+float(group[best[j],j])
                if cost not in nxt or score<nxt[cost][0]:
                    nxt[cost]=(score,path+((j,int(best[j])),))
        states=nxt
    value,path=states[BUDGET_UNITS]
    nbytes=90*74752 + BUDGET_UNITS*18432 + 90
    assert nbytes==BUDGET_BYTES
    assert len(path)==90
    return {'total_conditional_local_KL_bound':float(value),
            'allocation':[list(x) for x in path],
            'packed_qk_bytes':nbytes}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.float32).eval()
    cfg=model.config
    if getattr(cfg,'rope_scaling',None) not in (None,{}):
        raise ValueError('RoPE scaling unsupported')
    theta=float(getattr(cfg,'rope_theta',float('nan')))
    if not np.isclose(theta,100000.):
        raise ValueError('unexpected RoPE theta')
    if len(model.model.layers)!=LAYERS or int(cfg.hidden_size)!=WIDTH:
        raise ValueError('model architecture changed')
    options=make_certified_tables(model,ANGLES,BIT_PAIRS,GROUPS,Q_PER_K,DIM,LENGTH,theta)
    result=select_certified(options['tables'])
    result.update({'model':MODEL,'revision':REVISION,'horizon':LENGTH,
                   'angles':list(ANGLES),'precision_pairs':BIT_PAIRS,
                   'input_bounds_by_layer':options['input_bounds_by_layer'],
                   'scope':'Same-input local-head KL-bound sum; NOT PPL or whole-network KL',
                   'no_calibration_tokens_used':True})
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    (output/'certified_options.json').write_text(json.dumps(options,indent=2))
    (output/'certified_allocation.json').write_text(json.dumps(result,indent=2))
    for name in ('certified_options.json','certified_allocation.json'):
        f=output/name
        print(name,'sha256',hashlib.sha256(f.read_bytes()).hexdigest(),'bytes',f.stat().st_size)
    print('selected certificate sum',result['total_conditional_local_KL_bound'],'packed bytes',result['packed_qk_bytes'])


if __name__=='__main__':main()
