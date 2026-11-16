"""Numerical tests for the optimizer and partial-RoPE operator score."""
import json,itertools,math,numpy as np
from pathlib import Path
p=Path(__file__).parent;rng=np.random.default_rng(20261116);q,k,a,b=[rng.normal(size=(8,11)) for _ in range(4)];theta=.31;c,s=math.cos(theta),math.sin(theta);r=np.eye(8);r[:2,:2]=[[c,-s],[s,c]]
dense=np.linalg.norm(a.T@r@b-q.T@r@k)**2;gram=np.trace((a@a.T)@r@(b@b.T)@r.T)+np.trace((q@q.T)@r@(k@k.T)@r.T)-2*np.trace((q@a.T)@r@(b@k.T)@r.T);assert abs(dense-gram)<1e-10
# Independent full enumeration over both bases and bits (3 groups, 2 bases, 4 choices).
t=rng.uniform(size=(3,2,4));cost=[0,1,1,2];budget=3
brute=min(sum(t[i,a,j] for i,(a,j) in enumerate(path)) for path in itertools.product(list(itertools.product(range(2),range(4))),repeat=3) if sum(cost[j] for a,j in path)==budget)
states={0:0.}
for row in t:
 n={}
 for old,v in states.items():
  for j,d in enumerate(cost):
   if old+d<=budget:n[old+d]=min(n.get(old+d,float('inf')),v+row[:,j].min())
 states=n
assert abs(brute-states[budget])<1e-12
out={'dense_gram_absolute_error':float(abs(dense-gram)),'exhaustive_joint_objective':float(brute),'eliminated_basis_dp_objective':float(states[budget]),'all_checks_passed':True};json.dump(out,open(p/'solver_validation.json','w'),indent=2);print(out)
