import json,time,numpy as np
from pathlib import Path
from context_score import ContextScore,rotation
p=Path(__file__).parent;rng=np.random.default_rng(20261118);d,n,L=64,576,512;freq=1/(100000**(np.arange(0,d,2)/d));start=time.perf_counter();score=ContextScore(freq,L);kernel_seconds=time.perf_counter()-start
start=time.perf_counter();rs=[rotation(freq,-r) for r in range(L)];rotation_seconds=time.perf_counter()-start
q,k=rng.normal(size=(2,d,n));a=q+.02*rng.normal(size=q.shape);b=k+.02*rng.normal(size=k.shape)
def explicit_gram():
 aa=a@a.T;bb=b@b.T;qq=q@q.T;kk=k@k.T;qa=q@a.T;bk=b@k.T;vals=[]
 for r in rs:vals.append(np.trace(aa@r@bb@r.T)+np.trace(qq@r@kk@r.T)-2*np.trace(qa@r@bk@r.T))
 ba=b@a.T;kq=k@q.T;self_second=vals[0]+np.trace(ba@ba)+np.trace(kq@kq)-2*np.trace((b@q.T)@(k@a.T))+(np.trace(ba)-np.trace(kq))**2
 return (score.weights[0]*self_second+np.dot(score.weights[1:],vals[1:]))/(4*d)
f=score.score(q,k,a,b);e=explicit_gram();assert abs(f-e)/max(1,abs(e))<1e-9
fast=[];slow=[]
for _ in range(7):
 t=time.perf_counter();score.score(q,k,a,b);fast.append(time.perf_counter()-t)
 t=time.perf_counter();explicit_gram();slow.append(time.perf_counter()-t)
out={'dimensions':{'head':d,'input':n,'context':L},'single_thread_blas':True,'kernel_setup_seconds':kernel_seconds,'explicit_rotation_cache_seconds':rotation_seconds,'fast_value':f,'explicit_gram_value':e,'relative_difference':abs(f-e)/abs(e),'kernel_median_seconds':float(np.median(fast)),'explicit_gram_median_seconds':float(np.median(slow)),'per_candidate_speed_ratio':float(np.median(slow)/np.median(fast)),'repetitions':7,'scope':'Same exact Gaussian population score; synthetic fixed weight matrices; CPU prototype; not end-to-end inference or calibration speedup.'};json.dump(out,open(p/'kernel_benchmark.json','w'),indent=2);print(json.dumps(out,indent=2))
