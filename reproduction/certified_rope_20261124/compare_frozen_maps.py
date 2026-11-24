"""Compare two frozen real-model allocation maps under SAME certified bound.

No PPL outcome or generalization claim; old Gaussian score is only used to
choose its own path. For the actual attention-KL score comparison, BOTH paths
are evaluated under the SAME deterministic certified objective.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from run_certified import select_certified


def compare(certified_path,gaussian_path):
    cert=json.loads(Path(certified_path).read_text())
    gauss=json.loads(Path(gaussian_path).read_text())
    a=np.asarray(cert['tables'],dtype=float)
    b=np.asarray(gauss['tables']['context_self'],dtype=float)
    if a.shape!=(90,4,4) or b.shape!=(90,4,4):
        raise ValueError('unmatched option dimensions')
    c=select_certified(a)
    g=select_certified(b)
    cp=c['allocation'];gp=g['allocation']
    b_on_a=float(sum(a[i,basis,choice] for i,(choice,basis) in enumerate(gp)))
    assert c['packed_qk_bytes']==g['packed_qk_bytes']==10045530
    assert c['total_conditional_local_KL_bound']<=b_on_a+1e-4
    return {
        'note':'weight-derived local attention bound, NOT a measurement of PPL/NLL/actual KL',
        'certified_bound_of_certified_selection':c['total_conditional_local_KL_bound'],
        'certified_bound_of_gaussian_selection':b_on_a,
        'percentage_bound_reduction':100*(b_on_a-c['total_conditional_local_KL_bound'])/b_on_a,
        'bit_pair_changes':sum(x[0]!=y[0] for x,y in zip(cp,gp)),
        'basis_changes':sum(x[1]!=y[1] for x,y in zip(cp,gp)),
        'both_packed_qk_budget':c['packed_qk_bytes'],
        'certified_allocation':cp,
        'gaussian_allocation':gp,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--certified-options',required=True)
    p.add_argument('--gaussian-options',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    result=compare(args.certified_options,args.gaussian_options)
    Path(args.output).write_text(json.dumps(result,indent=2))
    print({k:v for k,v in result.items() if not k.endswith('allocation')})

if __name__=='__main__':main()
