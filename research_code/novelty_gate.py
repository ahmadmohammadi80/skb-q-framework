#!/usr/bin/env python3
"""Retrospectively exploratory grouped-allocation model-selection stress test.

Uses only 30 Qwen development configuration outcomes. No GPU and no outcome from
prior heldout or WikiText-2 evaluations is loaded. Not a prospective validation.
"""
import json, zipfile, numpy as np, hashlib, pathlib
from sklearn.linear_model import Ridge, LassoCV
from scipy.stats import spearmanr
P=pathlib.Path(__file__).resolve().parent
SRC=(P/'SKBQ_Qwen_Stage1_Development.zip') if (P/'SKBQ_Qwen_Stage1_Development.zip').exists() else pathlib.Path('SKBQ_Qwen_Stage1_Development.zip')
with zipfile.ZipFile(SRC) as z:
    p=json.loads(z.read('FROZEN_PROTOCOL_COPIED_BEFORE_EVALUATION.json'))
    res={q['policy']:q for q in json.loads(z.read('DEVELOPMENT_RESULTS.json'))}
    ordered=[f'R{i:02d}' for i in range(1,31)]
    X=np.zeros((30,28),float)
    y=np.zeros(30,float)
    for i,n in enumerate(ordered):
        ix=p['training_maps'][n];assert len(ix)==14
        X[i,ix]=1
        y[i]=res[n]['mean_nll']
assert np.all(X.sum(axis=1)==14)
assert np.isfinite(y).all()

def uncertainty(Xtr, x, alpha, sigma):
    xbar=Xtr.mean(axis=0)
    A=(Xtr-xbar).T@(Xtr-xbar)+alpha*np.eye(28)
    d=x-xbar
    # Regularized-design inverse uncertainty (NOT a valid predictive interval).
    return sigma*np.sqrt(np.maximum(0,np.einsum('ni,ij,nj->n',d,np.linalg.inv(A),d)))

def methods_for_split(train,test):
    xx=X[train]; yy=y[train]; xt=X[test]; yt=y[test]
    ridge=Ridge(alpha=.1,fit_intercept=True).fit(xx,yy)
    rng=np.random.default_rng(13273+int(sum(train)))
    # Purely training-fold LassoCV; no test fold labels.
    lasso=LassoCV(cv=5,random_state=43,alphas=30,max_iter=100000,fit_intercept=True).fit(xx,yy)
    pr=ridge.predict(xt); pl=lasso.predict(xt)
    # LOO residual sigma estimated on the TRAIN FOLD only.
    loo=[]
    for j in range(len(train)):
        ids=np.arange(len(train))!=j
        mdl=Ridge(alpha=.1).fit(xx[ids], yy[ids]);loo.append(yy[j]-mdl.predict(xx[j:j+1])[0])
    sigma=float(np.sqrt(np.mean(np.square(loo))))
    ru=pr+uncertainty(xx,xt,alpha=.1,sigma=sigma)
    labels=['Ridge','Lasso','LeverageGuard_k1']
    pred=[pr,pl,ru]
    oracle=np.min(yt)
    out={}
    for n,q in zip(labels,pred):
        j=int(np.argmin(q));out[n]={'regret':float(yt[j]-oracle),'selected_mean_nll':float(yt[j]),'hit':int(yt[j]<=oracle+1e-13), 'index':int(test[j]),'rmse':float(np.sqrt(np.mean((q-yt)**2)))}
    return out

rng=np.random.default_rng(20261020)
outer=300
trials=[]
for k in range(outer):
    t=rng.permutation(30); train=t[:20]; test=t[20:]
    trials.append(methods_for_split(train,test))
summary={}
for name in ['Ridge','Lasso','LeverageGuard_k1']:
    rr=np.array([q[name]['regret'] for q in trials]); hit=np.array([q[name]['hit'] for q in trials]);
    summary[name]={'mean_regret_nats':float(rr.mean()),'median_regret_nats':float(np.median(rr)),'oracle_hit_rate':float(hit.mean()),'average_rmse':float(np.mean([q[name]['rmse'] for q in trials]))}
for name in ['Ridge','Lasso']:
    d=np.array([t['LeverageGuard_k1']['regret']-t[name]['regret'] for t in trials])
    summary['LeverageGuard_minus_'+name]={'mean_paired_regret_difference_nats':float(d.mean()),'LeverageGuard_win_frac':float(np.mean(d<0)),'LeverageGuard_loss_frac':float(np.mean(d>0)),'ties_frac':float(np.mean(d==0))}
output={'study':'SKBQ-NOVELTY-STOP-GO-20261020','kind':'RETROSPECTIVE_METHOD_DEVELOPMENT_ONLY','source_zip_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'development_design':{'n_labelled':30,'features':28,'eight_bit_groups':14,'training_maps':'R01..R30','loss':'C4 development mean_nll'},'split_rule':'300 pseudorandom 20-label fit / 10 unseen-candidate folds; no independent GPU labels','methods':{'Ridge':'alpha 0.1','Lasso':'LassoCV 5fold inner training, no outer-test leakage','LeverageGuard_k1':'Ridge prediction + (train-LOO RMSE)*sqrt((x-xbar)^T(XcTXc+0.1I)^-1(x-xbar)); κ=1, no tuning'},'results':summary,'important_limits':['20 training maps -> select 1 of 10 existing held-out random maps (NOT global 28 choose 14 optimum)','300 random splits recycle the same 30 outcomes: not independent trials or statistical inference','C4 dev outcomes already explored historically, so cannot call this prospectively frozen before all outcomes','Selection on candidate pool cannot establish quality of newly constructed global allocation','Leverage penalty is an established robust-optimization motif, not automatically novel'],'n_outer_splits':outer}
(P/'NOVELTY_GATE_RESULT.json').write_text(json.dumps(output,indent=2))
print(json.dumps(summary,indent=2))
