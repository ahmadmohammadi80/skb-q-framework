#!/usr/bin/env python3
"""Verify frozen SKB-Q holdout evidence. Does not re-run GPU model inference."""
import json, zipfile, statistics, math, pathlib
HERE=pathlib.Path(__file__).resolve().parent

def read(z, name): return json.loads(z.read(name))
def ranks(x):
    return [1+sum(y<v for y in x)+(sum(y==v for y in x)-1)/2 for v in x]
def corr(x,y):
    mx=statistics.mean(x); my=statistics.mean(y)
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))**.5
def rmse(a,b):return math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/len(a))

def qwen_holdout(filename):
    with zipfile.ZipFile(filename) as z:
        r=read(z,'FROZEN_PREDS_VS_HELDOUT_OBSERVATIONS.json')
        late=statistics.mean(read(z,'checkpoint_Late.json')['block_mean_nll'])
        actual=[];pred=[];baseline=[];direct=[]
        for x in r['per_map']:
            v=read(z,'checkpoint_'+x['policy']+'.json')
            assert len(v['block_mean_nll'])==64
            assert v['freeze_sha256']==r['prediction_freeze_sha256']
            d=statistics.mean(v['block_mean_nll'])-late
            assert abs(d-x['observed_delta_nll'])<2e-6
            actual.append(d);pred.append(x['predicted_delta_nll'])
            baseline.append(x['mean_baseline']);direct.append(x['opt_zero_shot_score'])
        out={'n':len(actual),'spearman':corr(ranks(pred),ranks(actual)),
             'ridge_rmse':rmse(pred,actual),'constant_rmse':rmse(baseline,actual),
             'opt_zero_shot_spearman':corr(ranks(direct),ranks(actual))}
        assert len(actual)==12
        assert abs(out['spearman']-r['spearman_qwen_ridge'])<1e-6
        assert abs(out['ridge_rmse']-r['rmse_qwen_ridge'])<1e-6
        return out

def opt_holdout(filename):
    with zipfile.ZipFile(filename) as z:
        r=read(z,'FROZEN_PREDICTION_EVALUATION.json')['c4_prior_0_63']
        p=r['predicted_delta_nll'];a=r['actual_delta_nll']
        assert len(p)==len(a)==12
        out={'n':12,'spearman':corr(ranks(p),ranks(a)),'ridge_rmse':rmse(p,a)}
        assert abs(out['spearman']-r['spearman_rho'])<1e-6
        assert abs(out['ridge_rmse']-r['rmse_delta_nll'])<1e-6
        return out
if __name__=='__main__':
    print(json.dumps({'opt':opt_holdout(HERE/'SKBQ_Ridge_12Map_Confirmatory.zip'),
                      'qwen':qwen_holdout(HERE/'SKBQ_Qwen_Stage2_Holdout.zip')},indent=2))
    print('Internal archived metrics reproduced. No independent GPU replication.')
