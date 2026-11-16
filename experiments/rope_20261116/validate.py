import itertools,json,math,hashlib
from pathlib import Path
import numpy as np
import torch
out=Path(__file__).parent
torch.set_num_threads(2)
src=(out/'run_full.py').read_text()
# Isolate pure definitions; do not load or evaluate the model for unit checks.
ns={'torch':torch,'math':math,'np':np}
exec(src[src.index('d=64;'):src.index('options=json')],ns)
rot,error=ns['rotation'],ns['error'];ns['rs']=[r.double() for r in ns['rs']]
gen=torch.Generator().manual_seed(17);worst=0
for _ in range(10):
 q,k,qh,kh=[torch.randn(64,13,generator=gen,dtype=torch.float64) for _ in range(4)]
 got=error(q,k,qh,kh);expected=float(torch.stack([((qh.T@r@kh-q.T@r@k)**2).sum() for r in ns['rs']]).mean())
 worst=max(worst,abs(got-expected)/expected);assert abs(got-expected)<expected*1e-12
u=rot(torch.linspace(-.7,.7,32).double());orth=float((u.T@u-torch.eye(64)).abs().max());comm=max(float((u@r-r@u).abs().max()) for r in ns['rs']);assert max(orth,comm)<1e-12
# Independent enumeration of a four-group allocation, same transition rule.
tables=json.load(open(out/'scores.json'))[:4];best=[np.min(t,axis=0) for t in tables];budget=8;delta=[0,1,3,4]
brute=min(sum(best[i][j] for i,j in enumerate(p)) for p in itertools.product(range(4),repeat=4) if sum(delta[j] for j in p)==budget)
dp={0:0.}
for row in best:
 nxt={}
 for c,v in dp.items():
  for j,dc in enumerate(delta):
   if c+dc<=budget:nxt[c+dc]=min(nxt.get(c+dc,float('inf')),v+row[j])
 dp=nxt
assert abs(dp[budget]-brute)<1e-8
# Actual byte packing and recovery for symmetric signed row quantizers.
def pack(w,bits):
 limit=2**(bits-1)-1;scale=np.max(np.abs(w),axis=1).astype(np.float32)/limit
 q=np.round(w/np.maximum(scale[:,None],1e-30)).clip(-limit,limit).astype(np.int8)
 if bits==8:payload=q.tobytes();rest=np.frombuffer(payload,np.int8).reshape(w.shape)
 else:
  z=(q.astype(np.int16)+8).astype(np.uint8);payload=(z[:,::2]|(z[:,1::2]<<4)).tobytes()
  z=np.frombuffer(payload,np.uint8).reshape(w.shape[0],-1);rest=np.stack([z&15,z>>4],-1).reshape(w.shape).astype(np.int16)-8
 assert np.array_equal(rest,q)
 return len(payload)+len(scale.tobytes())
rng=np.random.default_rng(17);q=rng.normal(size=(192,576)).astype(np.float32);k=rng.normal(size=(64,576)).astype(np.float32)
costs=[pack(q,a)+pack(k,b) for a,b in [(4,4),(4,8),(8,4),(8,8)]]
r=json.load(open(out/'results.json'));storage={}
for mode,v in r['results'].items():
 if mode=='fp32':continue
 storage[mode]=sum(costs[j] for j,b in v['allocation'])+90
assert len(set(storage.values()))==1
res={'gram_relative_error_max':worst,'orthogonality_error_max':orth,'commutation_error_max':comm,'dp_equals_enumeration':True,'row_pack_roundtrip':True,'group_option_bytes':costs,'QK_payload_scales_and_90_byte_map':storage,'scope':'Packing tested on random rows of exact model dimensions. Model inference uses dequantized float32; payload accounting is not measured RAM or latency.'}
json.dump(res,open(out/'validation.json','w'),indent=2);print(json.dumps(res,indent=2))
