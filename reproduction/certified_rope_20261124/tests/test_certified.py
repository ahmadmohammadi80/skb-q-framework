"""Executable correctness controls, not pretrained-model benchmarks."""
import math
import pathlib
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from certified_score import NormBoundRoPE, rotation, rmsnorm_input_bound, make_certified_tables
from run_certified import select_certified


def test_spectral_equals_explicit_offset_traversal():
    rng=np.random.default_rng(1)
    m=NormBoundRoPE((.16,.03),7)
    for _ in range(8):
        q,k,a,z=(rng.normal(size=(4,6)) for _ in range(4))
        bound=math.sqrt(6)*2.1
        assert np.isclose(m.certificate(q,k,a,z,bound),
                          m.explicit(q,k,a,z,bound),rtol=2e-11,atol=2e-9)


def test_rmsnorm_norm_bound():
    rng=np.random.default_rng(5)
    gamma=rng.normal(size=7)
    B=rmsnorm_input_bound(gamma)
    for j in range(60):
        v=rng.normal(size=7)*10**(j/20-1)
        x=gamma*v/np.sqrt(np.mean(v*v)+1e-5)
        assert np.linalg.norm(x)<=B+1e-10


def test_real_softmax_kl_bound_on_correlated_non_gaussian_inputs():
    rng=np.random.default_rng(77)
    m=NormBoundRoPE((.17,.04),5)
    q=rng.normal(size=(4,7))*.45;k=rng.normal(size=(4,7))*.45
    a=q+rng.normal(size=(4,7))*.1;z=k+rng.normal(size=(4,7))*.1
    g=rng.uniform(.2,1.8,size=7)
    cert=m.certificate(q,k,a,z,rmsnorm_input_bound(g))
    for _ in range(24):
        data=[];last=np.zeros(7)
        for i in range(5):
            last=.6*last+rng.choice((-1.,1.),size=7)
            data.append(g*last/np.sqrt(np.mean(last*last)+1e-5))
        x=np.asarray(data)
        kls=[]
        for i in range(5):
            before=[];after=[]
            for j in range(i+1):
                R=rotation((.17,.04),j-i)
                before.append(x[i]@q.T@R@k@x[j]/2)
                after.append(x[i]@a.T@R@z@x[j]/2)
            p=torch.tensor(before,dtype=torch.float64)
            t=torch.tensor(after,dtype=torch.float64)
            kls.append(float((p.softmax(0)*(p.log_softmax(0)-t.log_softmax(0))).sum()))
        assert 0<=np.mean(kls)+1e-12<=cert+1e-10


def test_frozen_budget_dp_and_grid_objective():
    table=np.ones((90,4,4))
    table[:,:,0]=8;table[:,:,1]=6;table[:,:,2]=5;table[:,:,3]=1
    result=select_certified(table)
    assert result['packed_qk_bytes']==10045530
    assert len(result['allocation'])==90
    assert abs(result['total_conditional_local_KL_bound']-390)<1e-10


def test_generator_for_tiny_bias_free_rmsnorm_model():
    from types import SimpleNamespace
    torch.manual_seed(141)
    q=torch.nn.Linear(7,4,bias=False)
    k=torch.nn.Linear(7,4,bias=False)
    rms=SimpleNamespace(weight=torch.nn.Parameter(torch.tensor([1.,.8,1.4,.6,1.,.9,1.2])))
    model=SimpleNamespace(model=SimpleNamespace(layers=[
        SimpleNamespace(input_layernorm=rms,self_attn=SimpleNamespace(q_proj=q,k_proj=k))]))
    out=make_certified_tables(model,[0.,math.pi/16],[(4,4),(4,8)],1,1,4,8,10000.)
    values=np.asarray(out['tables'])
    assert values.shape==(1,2,2) and np.isfinite(values).all() and (values>=0).all()
