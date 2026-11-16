"""Causal row-centered Gaussian logit-error score for bias-free RoPE attention.
Exact finite-context computation under the stated surrogate model, not an LLM PPL guarantee.
"""
import numpy as np

def rotation(freq,r):
 c=np.diag(np.cos(freq*r));s=np.diag(np.sin(freq*r));return np.block([[c,-s],[s,c]])

class ContextScore:
 def __init__(self,freq,length):
  self.freq=np.asarray(freq,dtype=np.float64);self.length=int(length);self.d=2*len(freq)
  if self.length<1:raise ValueError('length must be positive')
  n=len(freq);v=np.zeros((2*n,2*n),complex)
  for i in range(n):v[i,i]=v[i,i+n]=1/np.sqrt(2);v[i+n,i]=-1j/np.sqrt(2);v[i+n,i+n]=1j/np.sqrt(2)
  self.v=v;lam=np.r_[self.freq,-self.freq]
  row=1-1/np.arange(1,length+1);self.weights=np.array([row[r:].sum()/length for r in range(length)])
  r=np.arange(1,length);ph=np.exp(-1j*lam[:,None]*r[None,:]);self.kernel=(ph*self.weights[None,1:])@ph.conj().T
 def transform(self,m):return self.v.conj().T@m@self.v
 def trace_average(self,a,b):
  a=self.transform(a);b=self.transform(b)
  return float(np.real(np.einsum('ij,ji->',a,self.kernel*b)))
 def components(self,q,k,a,b):
  q,k,a,b=[np.asarray(x,dtype=np.float64) for x in (q,k,a,b)]
  aa=a@a.T;bb=b@b.T;qq=q@q.T;kk=k@k.T;qa=q@a.T;bk=b@k.T
  cross=self.trace_average(aa,bb)+self.trace_average(qq,kk)-2*self.trace_average(qa,bk)
  norm=np.trace(aa@bb)+np.trace(qq@kk)-2*np.trace(qa@bk)
  ba=b@a.T;kq=k@q.T
  trace_square=np.trace(ba@ba)+np.trace(kq@kq)-2*np.trace((b@q.T)@(k@a.T))
  trace=float(np.trace(ba)-np.trace(kq));self_second=float(norm+trace_square+trace**2)
  return {'cross':float(cross),'self_second':self_second,'self_weight':float(self.weights[0])}
 def score(self,q,k,a,b,include_self=True):
  x=self.components(q,k,a,b);return max(0,(x['cross']+(x['self_weight']*x['self_second'] if include_self else 0))/(4*self.d))
 def explicit(self,q,k,a,b):
  total=0.
  for r,w in enumerate(self.weights):
   rr=rotation(self.freq,-r);e=a.T@rr@b-q.T@rr@k
   moment=(np.sum(e*e)+np.trace(e@e)+np.trace(e)**2) if r==0 else np.sum(e*e)
   total+=w*moment
  return total/(4*self.d)

def allocate(tables,budget,increments=(0,1,3,4)):
 states={0:(0.,[])}
 for table in tables:
  table=np.asarray(table);bases=table.argmin(axis=0);vals=table[bases,np.arange(table.shape[1])];nxt={}
  for old,(value,path) in states.items():
   for j,delta in enumerate(increments):
    cost=old+delta;nv=value+float(vals[j])
    if cost<=budget and (cost not in nxt or nv<nxt[cost][0]):nxt[cost]=(nv,path+[(j,int(bases[j]))])
  states=nxt
 return states[budget]
