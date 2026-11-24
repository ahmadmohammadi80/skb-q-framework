"""Calibration-free per-head RoPE attention KL certificate and exact score.

In real arithmetic, for any causal sequence of length L, if each input vector to
both Q and K projections has Euclidean norm <= B, then mean attention-row KL
(teacher || student) <= B^4/(4*d) sum_{r=0}^{L-1} (L-r)/L ||E_r||_F^2,
where E_r = A.T @ R(-r) @ C - Q.T @ R(-r) @ K.

This does NOT certify output logits, perplexity or model-level composition.
Uses a data-free worst-case norm bound from the preceding RMSNorm scale.
"""
from __future__ import annotations
import math
import numpy as np


def rotation(freq,lag):
    f=np.asarray(freq,dtype=np.float64)
    c=np.diag(np.cos(f*lag));s=np.diag(np.sin(f*lag))
    return np.block([[c,-s],[s,c]])


class NormBoundRoPE:
    """Exact spectral contraction for a finite RoPE causal horizon."""
    def __init__(self, freqs, length):
        self.freqs=np.asarray(freqs,dtype=np.float64)
        self.length=int(length)
        if self.freqs.ndim!=1 or not np.all(np.isfinite(self.freqs)) or self.length<1:
            raise ValueError('invalid frequencies/context length')
        n=len(self.freqs)
        if n<1:raise ValueError('empty frequencies')
        self.d=2*n
        v=np.zeros((2*n,2*n),complex)
        for i in range(n):
            v[i,i]=v[i,i+n]=1/math.sqrt(2)
            v[i+n,i]=-1j/math.sqrt(2)
            v[i+n,i+n]=1j/math.sqrt(2)
        self.v=v
        eig=np.r_[self.freqs,-self.freqs]
        offsets=np.arange(self.length)
        self.weights=(self.length-offsets).astype(np.float64)/self.length
        ph=np.exp(-1j*eig[:,None]*offsets[None,:])
        self.kernel=(ph*self.weights[None,:])@ph.conj().T

    def _contract(self,m,n):
        vm=self.v.conj().T@m@self.v
        vn=self.v.conj().T@n@self.v
        return float(np.real(np.einsum('ij,ji->',vm,self.kernel*vn)))

    def frobenius_sum(self,q,k,a,b):
        """Sum_r ((L-r)/L) ||a.T R(-r) b - q.T R(-r) k||_F^2.

        Q, K, student A, B are [head_dim, input_dim], using R before the
        bilinear product, as in a standard RoPE attention circuit.
        """
        q,k,a,b=[np.asarray(z,dtype=np.float64) for z in (q,k,a,b)]
        if any(z.ndim!=2 or z.shape[0]!=self.d for z in (q,k,a,b)) or not (q.shape==k.shape==a.shape==b.shape):
            raise ValueError('expected equally shaped [head_dim, input_dim] projections')
        aa=a@a.T;bb=b@b.T;qq=q@q.T;kk=k@k.T
        qa=q@a.T;bk=b@k.T
        v=self._contract(aa,bb)+self._contract(qq,kk)-2*self._contract(qa,bk)
        if v < -1e-7*max(1.,np.linalg.norm(aa)*np.linalg.norm(bb)):
            raise ArithmeticError('unexpected negative RoPE Frobenius sum')
        return max(0.,v)

    def certificate(self,q,k,a,b,input_bound):
        """Upper bound on mean per-query-row attention KL, one head.

        Preconditions (not automatically checked for tensor inputs): both
        teacher/student Q/K receive the same x, ||x_i||_2 <= input_bound,
        same RoPE positions, causal softmax at scale 1/sqrt(head_dim).
        """
        if not (math.isfinite(input_bound) and input_bound>=0):
            raise ValueError('input_bound must be finite and nonnegative')
        return (input_bound**4/(4*self.d))*self.frobenius_sum(q,k,a,b)

    def explicit(self,q,k,a,b,input_bound):
        """Slow independent reference, never used to optimize bits."""
        q,k,a,b=[np.asarray(z,dtype=np.float64) for z in (q,k,a,b)]
        total=0.
        for r,w in enumerate(self.weights):
            R=rotation(self.freqs,-r)
            E=a.T@R@b-q.T@R@k
            total+=w*float(np.square(E).sum())
        return input_bound**4/(4*self.d)*total


def rmsnorm_input_bound(scale):
    """||RMSNorm_gamma(v)||_2 <= sqrt(input_width) * max(abs(gamma)).

    Requires eps>=0 and the standard RMSNorm denominator; no data needed.
    """
    g=np.asarray(scale,dtype=np.float64)
    if g.ndim!=1 or len(g)<1 or not np.isfinite(g).all():
        raise ValueError('invalid RMSNorm scale')
    return float(math.sqrt(len(g))*np.max(np.abs(g)))


def make_certified_tables(model,angles,bit_pairs,groups,q_per_k,dim,context_length,rope_theta):
    """Generate a 90x4x4-like option table directly from frozen model weights.

    No training/validation/test text is used. The resulting sum is a bound on
    the SUM of individual head attention-row KLs, not on final LM quality.
    """
    import torch
    begin_freqs=1/(float(rope_theta)**(np.arange(0,dim,2)/dim))
    metric=NormBoundRoPE(begin_freqs,context_length)
    if len(angles)<1 or len(bit_pairs)<1:
        raise ValueError('missing decision options')
    def basis(angle):
        c,s=math.cos(angle),math.sin(angle)
        eye=torch.eye(dim//2,dtype=torch.float32)
        return torch.cat((torch.cat((c*eye,-s*eye),1),torch.cat((s*eye,c*eye),1)),0)
    bases=[basis(theta) for theta in angles]
    def quant(w,b):
        n=2**(b-1)-1
        scale=w.abs().amax(1)/n
        return (w/scale.clamp_min(1e-30)[:,None]).round().clamp(-n,n)*scale[:,None]
    tables=[];layer_bounds=[]
    with torch.no_grad():
        for li,layer in enumerate(model.model.layers):
            if not hasattr(layer,'input_layernorm'):
                raise ValueError('requires input RMSNorm before both Q and K')
            if layer.self_attn.q_proj.bias is not None or layer.self_attn.k_proj.bias is not None:
                raise ValueError('requires bias-free Q/K (or affine error model)')
            radius=rmsnorm_input_bound(layer.input_layernorm.weight.detach().cpu().numpy())
            layer_bounds.append(radius)
            qw=layer.self_attn.q_proj.weight.detach().cpu().float()
            kw=layer.self_attn.k_proj.weight.detach().cpu().float()
            for g in range(groups):
                qs=[qw[(g*q_per_k+j)*dim:(g*q_per_k+j+1)*dim] for j in range(q_per_k)]
                k=kw[g*dim:(g+1)*dim]
                qns=[q.numpy().astype(float) for q in qs];kn=k.numpy().astype(float)
                choices=[]
                for u in bases:
                    row=[]
                    for qb,kb in bit_pairs:
                        as_=[quant(u@q,qb).numpy().astype(float) for q in qs]
                        bs=quant(u@k,kb).numpy().astype(float)
                        row.append(sum(metric.certificate(q,kn,a,bs,radius)
                                       for q,a in zip(qns,as_)))
                    choices.append(row)
                tables.append(choices)
    return {'tables':tables,'input_bounds_by_layer':layer_bounds,
            'horizon':context_length,'rope_theta':rope_theta,
            'score_meaning':'sum of per-head deterministic average-row attention-KL upper bounds; excludes output-LM propagation'}
