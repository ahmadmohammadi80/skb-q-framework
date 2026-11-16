"""Controlled transplantation of Wang et al. 2026 Eq.3/A.2.1.
NOT a reproduction of their W4A4KV4 full-head Hadamard pipeline.
"""
from pathlib import Path
exec(compile(Path(__file__).with_name('run_full.py').read_text().split("data=load_dataset")[0],str(Path(__file__)), 'exec'))
torch.set_num_threads(2)
# Pool raw Q and K observations, with the physical 3:1 GQA head ratio.
stats=[{'n':0,'sum':torch.zeros(32,2,dtype=torch.float64),'outer':torch.zeros(32,2,2,dtype=torch.float64)} for _ in range(30)]
def hook(li):
 def collect(module,args,y):
  z=y.detach().reshape(-1,64).double();z=torch.stack([z[:,:32],z[:,32:]],-1);v=stats[li];v['n']+=len(z);v['sum']+=z.sum(0);v['outer']+=torch.einsum('nki,nkj->kij',z,z)
 return collect
handles=[]
for li,layer in enumerate(model.model.layers):
 for module in [layer.self_attn.q_proj,layer.self_attn.k_proj]:handles.append(module.register_forward_hook(hook(li)))
train=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='train');traintext='\n\n'.join(train['text']);tt=tokenizer(traintext,return_tensors='pt').input_ids
with torch.inference_mode():
 for i in range(16):model(tt[:,i*512:(i+1)*512]);print('calibrated',i,flush=True)
for h in handles:h.remove()
layerbases=[];angles=[];residual=[]
for v in stats:
 mean=v['sum']/v['n'];cov=v['outer']/v['n']-mean[:,:,None]*mean[:,None,:]
 # Eq.3 finite-position covariance, length512, before deriving the angle.
 ph=torch.arange(512,dtype=torch.float64)[:,None]*freq.double()[None,:]
 c2=(2*ph).cos().mean(0);s2=(2*ph).sin().mean(0)
 diff=cov[:,0,0]-cov[:,1,1];off=cov[:,0,1]
 a=diff*c2-2*off*s2;b=.5*diff*s2+off*c2
 angle=(.5*torch.atan2(a,2*b)+math.pi/4)%(math.pi/2)-math.pi/4
 residual.append(float((a*torch.cos(2*angle)-2*b*torch.sin(2*angle)).abs().max()))
 layerbases.append(rotation(angle.float()));angles.append(angle.tolist())
options=[]
for li,layer in enumerate(model.model.layers):
 qw=layer.self_attn.q_proj.weight.detach();kw=layer.self_attn.k_proj.weight.detach();u=layerbases[li]
 for g in range(3):
  qs=[qw[(g*3+j)*64:(g*3+j+1)*64] for j in range(3)];k=kw[g*64:(g+1)*64]
  vals=[sum(error(q,k,quant(u@q,qb),quant(u@k,kb)) for q in qs) for qb,kb in [(4,4),(4,8),(8,4),(8,8)]]
  options.append([vals]*4)
score,path=allocate('identity')
with torch.no_grad():
 for idx,(choice,bi) in enumerate(path):
  li,g=divmod(idx,3);att=model.model.layers[li].self_attn;qb,kb=[(4,4),(4,8),(8,4),(8,8)][choice];u=layerbases[li]
  for j in range(3):
   sl=slice((g*3+j)*64,(g*3+j+1)*64);att.q_proj.weight[sl].copy_(quant(u@att.q_proj.weight[sl],qb))
  sl=slice(g*64,(g+1)*64);att.k_proj.weight[sl].copy_(quant(u@att.k_proj.weight[sl],kb))
test=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='test');text='\n\n'.join(test['text']);t=tokenizer(text,return_tensors='pt').input_ids
losses=[]
res={'method':'Wang2026_position_averaged_QK_rotation_transplant','source':'https://arxiv.org/html/2608.13365v1','adaptation':'Published covariance/angle rule; QK weight-only RTN and exact storage DP substituted for original dynamic W4A4KV4 pipeline. No full-method reproduction claim.','model':name,'revision':revision,'calibration_split':'train','calibration_tokens':8192,'calibration_text_sha256':hashlib.sha256(traintext.encode()).hexdigest(),'test_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'angles':angles,'covariance_equalization_max_residual':max(residual),'surrogate':score,'allocation':path,'block_nll':losses}
with torch.inference_mode():
 for i in range(t.shape[1]//512):
  block=t[:,i*512:(i+1)*512];losses.append(float(model(block,labels=block).loss));print('published_rule',i+1,flush=True)
  res['ppl']=math.exp(float(np.mean(losses)));res['blocks']=len(losses);res['predicted_tokens']=len(losses)*511
  json.dump(res,open(out/'published_rule_results.json','w'),indent=2)
print('done',res['ppl'],flush=True)
