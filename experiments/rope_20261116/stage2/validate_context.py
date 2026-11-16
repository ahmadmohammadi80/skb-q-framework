from pathlib import Path
import json,time,numpy as np
from context_score import ContextScore,rotation
p=Path(__file__).parent;rng=np.random.default_rng(20261116);checks=[]
for d,n,length in [(4,7,1),(4,7,8),(8,11,17),(16,23,64)]:
 freq=1/(10000**(np.arange(0,d,2)/d));s=ContextScore(freq,length)
 for _ in range(8):
  q,k=rng.normal(size=(2,d,n));a=q+.1*rng.normal(size=q.shape);b=k+.1*rng.normal(size=k.shape);fast=s.score(q,k,a,b);direct=s.explicit(q,k,a,b);rel=abs(fast-direct)/max(1,abs(direct));assert rel<1e-10;checks.append(rel)
# Monte Carlo check includes shared query, causal masking and same-token quadratic term.
d,n,L=4,5,6;s=ContextScore(np.array([1.,.1]),L);q,k=rng.normal(size=(2,d,n));a=q+.25*rng.normal(size=q.shape);b=k+.25*rng.normal(size=k.shape)
N=120000;x=rng.normal(size=(N,L,n));values=np.zeros(N);kl=np.zeros(N)
for t in range(1,L+1):
 logits=[];errors=[]
 for j in range(t):
  rr=rotation(s.freq,j-(t-1));orig=q.T@rr@k;e=a.T@rr@b-orig
  logits.append(np.einsum('bi,ij,bj->b',x[:,t-1],orig,x[:,j])/np.sqrt(d));errors.append(np.einsum('bi,ij,bj->b',x[:,t-1],e,x[:,j])/np.sqrt(d))
 z=np.array(logits).T;e=np.array(errors).T;center=e-e.mean(axis=1,keepdims=True);values+=(center**2).sum(axis=1)/(4*L)
 def logsoftmax(z):
  z=z-z.max(axis=1,keepdims=True);return z-np.log(np.exp(z).sum(axis=1,keepdims=True))
 lp=logsoftmax(z);lq=logsoftmax(z+e);kl+=(np.exp(lp)*(lp-lq)).sum(axis=1)/L
assert np.max(kl-values)<1e-10
prediction=s.score(q,k,a,b);se=values.std(ddof=1)/np.sqrt(N);zscore=abs(values.mean()-prediction)/se;assert zscore<6
# Self term is not the independent-token Frobenius term.
e=np.diag([1.,-1.]);assert np.sum(e*e)+np.trace(e@e)+np.trace(e)**2==4
out={'direct_checks':len(checks),'maximum_relative_error':max(checks),'monte_carlo_samples':N,'analytic_score':prediction,'monte_carlo_mean':float(values.mean()),'monte_carlo_standard_error':float(se),'z_score':float(zscore),'mean_attention_KL':float(kl.mean()),'maximum_sample_KL_minus_bound':float((kl-values).max()),'all_passed':True};json.dump(out,open(p/'validation.json','w'),indent=2);print(json.dumps(out,indent=2))
