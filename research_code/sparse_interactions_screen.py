#!/usr/bin/env python3
"""Exploratory sparse short-range interaction screen. Not a frozen test or novelty proof."""
import json,zipfile,numpy as np,pathlib,hashlib,os
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LassoCV, ElasticNetCV
D=pathlib.Path(os.environ.get('SKBQ_DATA_ROOT','.'));OUT=pathlib.Path(__file__).resolve().parent
za=zipfile.ZipFile((OUT/'SKBQ_Qwen_Stage1_Development.zip') if (OUT/'SKBQ_Qwen_Stage1_Development.zip').exists() else (D/'SKBQ_Qwen_Stage1_Development.zip'));zb=zipfile.ZipFile((OUT/'SKBQ_Qwen_Stage2_Holdout.zip') if (OUT/'SKBQ_Qwen_Stage2_Holdout.zip').exists() else (D/'SKBQ_Qwen_Stage2_Holdout.zip'))
train=json.loads(za.read('FROZEN_PROTOCOL_COPIED_BEFORE_EVALUATION.json'))['training_maps'];dev={a['policy']:a['mean_nll'] for a in json.loads(za.read('DEVELOPMENT_RESULTS.json'))};test=json.loads(zb.read('FROZEN_PREDS_VS_HELDOUT_OBSERVATIONS.json'))
X=np.zeros((30,28));y=np.zeros(30)
for i in range(30):
 m=f'R{i+1:02d}';X[i,train[m]]=1;y[i]=dev[m]-dev['Late']
XT=np.zeros((12,28));yt=np.zeros(12)
for i,m in enumerate(test['per_map']):
 XT[i,m['selected_layers']]=1;yt[i]=m['observed_delta_nll']
assert np.all(X.sum(1)==14) and np.all(XT.sum(1)==14)

def interact(arr,radius):
 return np.column_stack([arr]+[arr[:,j]*arr[:,j+d] for d in range(1,radius+1) for j in range(28-d)])
models={
 'main_Ridge_a0.1':Ridge(alpha=.1).fit(X,y),
 'main_LassoCV':LassoCV(cv=5,alphas=60,random_state=42,max_iter=100000).fit(X,y),
 'adjacent_ElasticNetCV':ElasticNetCV(cv=5,l1_ratio=[.5,.8,.95,1],alphas=50,max_iter=100000,random_state=42).fit(interact(X,1),y),
 'radius2_ElasticNetCV':ElasticNetCV(cv=5,l1_ratio=[.5,.8,.95,1],alphas=50,max_iter=100000,random_state=42).fit(interact(X,2),y),
}
r={}
for k,m in models.items():
 xx=interact(XT,2 if 'radius2' in k else 1) if 'Elastic' in k else XT
 pr=m.predict(xx)
 c=pr-yt
 r[k]={'rho':float(spearmanr(pr,yt).statistic),'rmse':float(np.sqrt(np.mean(c*c))),'centered_rmse':float(np.sqrt(np.mean((c-c.mean())**2))),'best_predicted_policy':test['per_map'][int(np.argmin(pr))]['policy'],'best_predicted_actual_nll_regret':float(yt[np.argmin(pr)]-yt.min()),'n_nonzero_coefs':int(np.count_nonzero(m.coef_)), 'prediction_deltas':[float(v) for v in pr]}
output={'kind':'RETROSPECTIVE_REUSE_OF_SEEN_HOLDOUT_DATA_NOT_PROSPECTIVE','train_30_maps':'C4 dev 0:31','test_12_maps':'C4 holdout 512:575 (reported in earlier work)','results':r,'note':'Interactions are plausible but overlapping earlier CoopQ work; improve score alone is not novel method. These exact H01-H12 outcomes have been inspected before designing this screen.'}
(OUT/'SPARSE_PAIRWISE_SCREEN.json').write_text(json.dumps(output,indent=2))
for k,v in r.items():print(k,{z:round(v[z],6) if isinstance(v[z],float) else v[z] for z in ['rho','rmse','centered_rmse','best_predicted_policy','best_predicted_actual_nll_regret','n_nonzero_coefs']})
