#!/usr/bin/env python3
"""Offline SKB-Q additive and paired-interaction validation.
Only reads supplied archives; never trains or evaluates a language model.
Post hoc exploratory comparisons (apart from pre-existing frozen Qwen Ridge).
"""
import os, zipfile, json, hashlib, csv, itertools
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from time import perf_counter
HERE=os.path.dirname(os.path.abspath(__file__)); SRC=os.environ.get('SKBQ_DATA_ROOT','.')
RNG=np.random.default_rng(20261019)
ALPHAS_R=[.001,.01,.1,1.,10.,100.,1000.];ALPHAS_L=[.00001,.0001,.0003,.001,.003,.01,.03,.1]

def sha_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  while True:
   b=f.read(1024*1024)
   if not b:break
   h.update(b)
 return h.hexdigest()

def make_X(maps,L):
 a=np.zeros((len(maps),L))
 for i,m in enumerate(maps):a[i,[int(j) for j in m]]=1
 return a

def get_y(z,names,late_name='Late'):
 late=np.mean(json.loads(z.read(f'checkpoint_{late_name}.json'))['block_mean_nll'])
 arr=[];sels=[]
 for n in names:
  d=json.loads(z.read(f'checkpoint_{n}.json'))
  assert d['blocks_complete'] == len(d['block_mean_nll']) if 'blocks_complete' in d else d['blocks_completed']==len(d['block_mean_nll'])
  sels.append(d['selected_layers']);arr.append(float(np.mean(d['block_mean_nll'])-late))
 return make_X(sels,28 if names[0].startswith(('R','H')) else 24), np.array(arr),sels,late

def choose_alpha(kind, X,y,folds=5):
 grid=ALPHAS_R if kind in ('ridge','pairwise') else ALPHAS_L
 k=min(folds,len(y));kf=KFold(n_splits=k,shuffle=True,random_state=20261016)
 losses=[]
 for a in grid:
  errs=[]
  for it,iv in kf.split(X):
   f=fit(kind,X[it],y[it],a);p=f.predict(features(kind,X[iv])); errs.extend((y[iv]-p)**2)
  losses.append(float(np.mean(errs)))
 return float(grid[int(np.argmin(losses))])

def features(kind,X):
 if kind=='pairwise':
  # Pair terms are centered but feature scale divided by sqrt(L-1).
  xc=X-np.mean(X,axis=1,keepdims=True)
  i,j=np.triu_indices(X.shape[1],k=1)
  q=xc[:,i]*xc[:,j]/np.sqrt(X.shape[1]-1)
  return np.column_stack((X,q))
 return X

def fit(kind, X,y,a=None):
 v=features(kind,X)
 if kind=='ols':return LinearRegression().fit(v,y)
 if kind=='marginal':
  # parameter-free empirical conditional contrast; not a single-layer GPU evaluation.
  m1=np.divide(X.T@y,np.sum(X,axis=0),out=np.zeros(X.shape[1]),where=np.sum(X,axis=0)>0)
  m0=np.divide((1-X).T@y,np.sum(1-X,axis=0),out=np.zeros(X.shape[1]),where=np.sum(1-X,axis=0)>0)
  beta=m1-m0
  class Marginal:
   def __init__(self,coef_,base):self.coef_=coef_;self.intercept_=base
   def predict(self,a):return self.intercept_+a@self.coef_
  return Marginal(beta,float(np.mean(y)-np.mean(X,axis=0)@beta))
 if kind=='lasso':return Lasso(alpha=a,fit_intercept=True,max_iter=300000,tol=1e-7).fit(v,y)
 if kind in ('ridge','pairwise'):return Ridge(alpha=a,fit_intercept=True).fit(v,y)
 raise ValueError(kind)

def pair_acc(y,p):
 yi,yj=np.triu_indices(len(y),1);a=np.sign(y[yi]-y[yj]);b=np.sign(p[yi]-p[yj]);return float(np.mean(a==b))

def metric(y,p):
 s=spearmanr(y,p).statistic
 i=int(np.argmin(p));j=int(np.argmin(y));e=np.asarray(y-p)
 return {'rmse':float(np.sqrt(np.mean(e**2))),'centered_rmse':float(np.std(e)), 'signed_bias_pred_minus_obs':float(np.mean(p-y)),'spearman':float(s) if np.isfinite(s) else None,'pair_accuracy':pair_acc(y,p),'retrospective_top12_regret_nats':float(y[i]-y[j]),'retrospective_predicted_best_index':i,'actual_best_index':j}

def eval_qwen():
 z=zipfile.ZipFile(os.path.join(SRC,'SKBQ_Qwen_Stage1_Development.zip'));h=zipfile.ZipFile(os.path.join(SRC,'SKBQ_Qwen_Stage2_Holdout.zip'))
 ns=[f'R{i:02d}' for i in range(1,31)]; hs=[f'H{i:02d}' for i in range(1,13)]
 X,y,maps,late=get_y(z,ns);T,v,hmaps,hlate=get_y(h,hs)
 assert X.shape==(30,28) and T.shape==(12,28) and np.all(X.sum(1)==14) and np.all(T.sum(1)==14)
 frozen=json.loads(h.read('FROZEN_PREDICTIONS_COPIED_BEFORE_EVALUATION.json'))
 results={}; details=[];stability={}
 for kind in ['constant','marginal','ols','ridge','lasso','pairwise']:
  if kind=='constant': p=np.repeat(np.mean(y),len(v)); alpha=None;f=None
  else:
   # ridge alpha must match previously frozen value; for other tunable models, use training-only KFold CV
   alpha=(float(frozen['selected_alpha']) if kind=='ridge' else choose_alpha(kind,X,y)) if kind in ('ridge','lasso','pairwise') else None
   f=fit(kind,X,y,alpha);p=f.predict(features(kind,T))
  stats=metric(v,p);stats.update({'model':kind,'alpha':alpha,'n_train':30,'n_test':12,'train_corpus':'C4 blocks 0-31','test_corpus':'C4 blocks 512-575','study':'Qwen held-out (exploratory comparisons; original Ridge frozen)'})
  results[kind]=stats
  if f is not None:
   # proposed per-method argmin is not evaluated on GPU; record for transparent follow-up only
   coeff=f.coef_[:28] if hasattr(f,'coef_') else np.zeros(28)
   if kind=='pairwise':
    # exhaustive 28 choose 14 impossible; don't assert minimax map
    selected=None
   else:selected=sorted(np.argsort(coeff,kind='stable')[:14].tolist())
   stats['unmeasured_additive_topK']=selected
  for j in range(12): details.append({'model':kind,'id':hs[j],'observed_delta_nll':v[j],'predicted_delta_nll':p[j],'error_pred_minus_obs':p[j]-v[j]})
  if kind in ('ridge','lasso','marginal','ols'):
   d=[]
   for b in range(200):
    ix=RNG.integers(0,30,size=30); fb=fit(kind,X[ix],y[ix],alpha)
    top=tuple(np.sort(np.argsort(fb.coef_[:28],kind='stable')[:14]))
    d.append(top)
   from collections import Counter
   c=Counter(d);ss=[len(set(d[0]).intersection(t)) for t in d]
   reference=tuple(np.sort(np.argsort(f.coef_[:28],kind='stable')[:14]))
   stability[kind]={'bootstrap_replicates':200,'unique_topK_maps':len(c),'modal_map_frequency':c.most_common(1)[0][1]/200,'mean_topK_overlap_with_full_data':float(np.mean([len(set(reference).intersection(t))/14 for t in d]))}
 # frozen agreement check
 ridge_coeff=np.array(frozen['coefficients_28']); fr=fit('ridge',X,y,float(frozen['selected_alpha']))
 maxdiff=float(np.max(np.abs(fr.coef_-ridge_coeff)))
 return results,details,stability,{'n_train':30,'n_test':12,'frozen_ridge_coeff_max_abs_difference':maxdiff,'frozen_selected_alpha':frozen['selected_alpha'],'dev_late_nll':late,'heldout_late_nll':hlate}

def opt_v3():
 z=zipfile.ZipFile(os.path.join(SRC,'SKBQ_VERIFIED_JSON_FOR_PAPER(1).zip'));out={};timings={}
 for name,L,K in [('opt-125m',12,8),('opt-1.3b',24,12)]:
  raw=[];dur=[];budget=[];raws=[]
  for i in range(1,31):
   nm=f'controlled_GPU/{name}/RandomGroup_{20261100+i}.json'
   d=json.loads(z.read(nm));raw.append(d)
   dur.append(d['runtime_seconds']);budget.append(d['budget']['avg_bits_linear']);raws.append(d['selected_layers'])
  X=make_X(raws,L);y=np.array([d['metrics']['mean_nll'] for d in raw])
  assert X.shape==(30,L) and np.all(X.sum(1)==K) and np.all(np.isfinite(y))
  assert max(budget)-min(budget)<1e-11
  dct={}
  for kind in ['constant','marginal','ols','ridge','lasso','pairwise']:
   pp=[]
   al=[]
   for i in range(30):
    tr=np.array([t for t in range(30) if t!=i]);te=np.array([i]);
    if kind=='constant':p=float(np.mean(y[tr]))
    else:
     a=choose_alpha(kind,X[tr],y[tr]) if kind in ('ridge','lasso','pairwise') else None
     ff=fit(kind,X[tr],y[tr],a);p=float(ff.predict(features(kind,X[te]))[0]);al.append(a)
    pp.append(p)
   s=metric(y,np.array(pp));s.update({'model':kind,'study':name+' OPT v3 leave-one-map-out exploratory','n_train_per_fold':29,'n_test_maps':30,'unique_inner_CV_alphas':sorted(set(al)) if al else []})
   dct[kind]=s
  out[name]=dct
  timings[name]={'source':'GPU checkpoint runtime_seconds in controlled_GPU/ v3 archive','gpu':raw[0].get('provenance',{}).get('gpu'), 'n_measured_maps':30,'total_seconds':float(sum(dur)),'median_per_map_s':float(np.median(dur)),'min_per_map_s':float(min(dur)),'max_per_map_s':float(max(dur)),'evaluation_blocks_each':int(raw[0]['metrics']['evaluation_blocks']),'prediction_tokens_each':int(raw[0]['metrics']['prediction_tokens']),'total_prediction_tokens':int(sum(d['metrics']['prediction_tokens'] for d in raw)),'corpus':'WikiText-2 raw test','note':'Cannot transplant this hardware/timing to Qwen development 30 maps with C4 32 blocks.'}
 return out,timings

def write_csv(path,rows):
 if not rows:return
 keys=sorted({k for r in rows for k in r})
 with open(path,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
 t=perf_counter();q,detail,stability,freeze=eval_qwen();o,cost=opt_v3()
 checksums={f:sha_file(os.path.join(SRC,f)) for f in ['SKBQ_Qwen_Stage1_Development.zip','SKBQ_Qwen_Stage2_Holdout.zip','SKBQ_VERIFIED_JSON_FOR_PAPER(1).zip','SKBQ_Qwen_Prospective_Decision_FULL.zip','SKBQ_Qwen_Baselines_STAGE_A_FREEZE.zip','SKBQ_Qwen_Baselines_STAGE_B_FULL.zip']}
 qcost={'stage1_measured_map_count':30,'stage1_late_control_count':1,'dev_blocks_per_map':32,'predicted_tokens_per_map':32*1023,'total_labeled_tokens':30*32*1023,'total_with_late_control_tokens':31*32*1023,'stage1_actual_GPU_time_seconds':None,'why_unknown':'No per-policy timing or overall GPU wall-clock in supplied Qwen Stage1 checkpoints. Do not extrapolate OPT timings.','stageA_weight_only':{'model_forward_passes_for_label_evaluation':0,'GPU_seconds':None,'operations':'FP16 weight scan; row-wise Q4/Q8 residual scoring over eligible 28 attention groups','cost_cannot_be_quantified_without_timing':True},'ridge_fitting_cpu_seconds_measured_in_this_reanalysis':None}
 full={'study_status':'post_hoc exploratory offline analysis (except previously frozen Qwen Ridge)', 'date':'2026-10-19','inputs_sha256':checksums,'qwen_30train_12heldout':q,'qwen_training_resample_stability':stability,'qwen_freeze_checks':freeze,'opt_v3_LOMO_30':o,'timing_from_opt_v3_GPU_checkpoints':cost,'qwen_stage1_label_cost':qcost,'cautions':['Post-review comparisons Lasso, marginal, paired model are retrospective on earlier 12 Qwen holdouts, NOT newly pre-registered confirmatory tests.','OPT-v3 analysis uses separate WikiText-2 243-block protocol, NOT Qwen 32/64-block C4 cross-corpus holdout.','With 30 labels and hundreds of pairwise columns, pairwise fits can overfit and no failure to improve proves true additivity.','Selected-outcome blocks are sequential; descriptive resampling is not independent document inference.','Marginal model is a conditional mean contrast from joint maps using same 30 labels; NOT a single-layer quantizer evaluation.','For Qwen, no GPU time was logged; label evaluation cost is represented by exact passes and tokens.','No heldout test outcomes are used for fitting/tuning, but the experimental alternatives were selected after research review; exploratory evidence only.']}
 out=os.path.join(HERE,'SKBQ_Additivity_Independent_Analysis.json');open(out,'w',encoding='utf-8').write(json.dumps(full,indent=2,ensure_ascii=False,default=float))
 rows=[]
 for v in q.values():rows.append(v)
 for d in o.values():rows+=list(d.values())
 write_csv(os.path.join(HERE,'SKBQ_Additivity_Comparison.csv'),rows);write_csv(os.path.join(HERE,'SKBQ_Qwen_Holdout_Per_Map_Predictions.csv'),detail)
 open(os.path.join(HERE,'INPUT_SHA256.json'),'w').write(json.dumps(checksums,indent=2))
 print('Qwen test (12 maps):')
 for k,v in q.items():print(k,'alpha',v['alpha'],'RMSE',round(v['rmse'],7),'rho',round(v['spearman'],3) if v['spearman'] is not None else None,'regret',round(v['retrospective_top12_regret_nats'],6))
 print('\nOPT LOOCV:')
 for model,rs in o.items():
  print(model,[(k,round(v['rmse'],5),round(v['spearman'],3) if v['spearman'] is not None else None) for k,v in rs.items()])
 print('Qwen coeff max diff vs frozen:',freeze['frozen_ridge_coeff_max_abs_difference'])
 print('Cost:',cost,'runtime',round(perf_counter()-t,2))
if __name__=='__main__':main()
