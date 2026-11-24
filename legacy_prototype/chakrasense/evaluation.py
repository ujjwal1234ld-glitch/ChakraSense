import numpy as np

def average_precision_score(y,p):
    """Non-interpolated AP, with tied scores evaluated together."""
    y=np.asarray(y);p=np.asarray(p);order=np.argsort(-p,kind='stable');y=y[order];p=p[order]
    ends=np.r_[np.flatnonzero(np.diff(p)),len(p)-1];tp=np.cumsum(y)[ends]
    precision=tp/(ends+1);recall=tp/y.sum()
    return float(np.sum(np.diff(np.r_[0.,recall])*precision))

def logistic_fit(features,y,l2=1.,iterations=100):
    x=np.column_stack([np.asarray(features),np.ones(len(features))]);y=np.asarray(y,dtype=float)
    if len(np.unique(y))<2:raise ValueError('Logistic regression requires both outcomes.')
    weights=np.zeros(x.shape[1]);penalty=np.eye(x.shape[1])*l2;penalty[-1,-1]=0
    for _ in range(iterations):
        p=1/(1+np.exp(-np.clip(x@weights,-40,40)))
        gradient=x.T@(p-y)+penalty@weights
        hessian=(x.T*(p*(1-p)))@x+penalty+np.eye(x.shape[1])*1e-8
        step=np.linalg.solve(hessian,gradient)
        # Backtracking keeps Newton steps stable on nearly separated classes.
        def loss(w):
            z=x@w;return np.sum(np.logaddexp(0,z)-y*z)+.5*w@penalty@w
        scale=1.;before=loss(weights)
        while scale>1e-6 and loss(weights-scale*step)>before:scale*=.5
        weights-=scale*step
        if np.max(np.abs(scale*step))<1e-7:break
    return weights

def safe_ratio(a,b):return float(a/b) if b else None

def metrics(wind,truth,p,y,threshold,base_rate):
    wind=np.asarray(wind);truth=np.asarray(truth);p=np.asarray(p);y=np.asarray(y).astype(int)
    flag=p>=threshold;hit=int(np.sum(flag&(y==1)));miss=int(np.sum(~flag&(y==1)));false=int(np.sum(flag&(y==0)))
    bs=float(np.mean((p-y)**2));bsc=float(np.mean((base_rate-y)**2))
    return {'mae_kt':float(np.mean(np.abs(wind-truth))),'rmse_kt':float(np.sqrt(np.mean((wind-truth)**2))),
       'brier':bs,'brier_skill':float(1-bs/bsc) if bsc>0 else None,
       'detection':safe_ratio(hit,hit+miss),'false_alarm_ratio':safe_ratio(false,hit+false),
       'csi':safe_ratio(hit,hit+false+miss),'pr_auc':float(average_precision_score(y,p)) if len(np.unique(y))==2 else None,
       'hits':hit,'misses':miss,'false_alarms':false,'samples':len(y)}

def fit_calibration(logits,y):
    if len(np.unique(y))<2:raise ValueError('Calibration needs both RI and non-RI. Add whole calibration storms; do not borrow test data.')
    weights=logistic_fit(np.asarray(logits).reshape(-1,1),y)
    return {'slope':float(weights[0]),'intercept':float(weights[1])}

def calibrate(logits,c):
    z=np.clip(c['slope']*np.asarray(logits)+c['intercept'],-40,40)
    return 1/(1+np.exp(-z))

def choose_threshold(p,y,miss_cost=2.,false_cost=1.):
    if len(np.unique(y))<2:raise ValueError('Validation requires both outcomes for threshold selection.')
    choices=np.linspace(.01,.99,99)
    costs=[miss_cost*np.sum((p<t)&(y==1))+false_cost*np.sum((p>=t)&(y==0)) for t in choices]
    return float(choices[int(np.argmin(costs))])

def conformal_radius(residuals,coverage=.9):
    r=np.asarray(residuals)
    if not 0<coverage<1 or len(r)==0:raise ValueError('Invalid calibration data or coverage.')
    rank=int(np.ceil((len(r)+1)*coverage))
    if rank>len(r):raise ValueError('Too few calibration points for requested coverage.')
    return float(np.sort(np.abs(r))[rank-1])

def reliability(p,y):
    p=np.asarray(p);y=np.asarray(y);out=[]
    for lo in np.linspace(0,.8,5):
        idx=(p>=lo)&((p<lo+.2) if lo<.8 else (p<=1))
        if idx.any():out.append({'bin_lower':float(lo),'count':int(idx.sum()),'mean_probability':float(p[idx].mean()),'observed_frequency':float(y[idx].mean())})
    return out

def bootstrap_storms(pred,threshold,base_rate,repeats=300,seed=29):
    rng=np.random.default_rng(seed);ids=pred.storm_id.unique();vals={k:[] for k in ['mae_kt','rmse_kt','brier_skill','detection','false_alarm_ratio']}
    groups={sid:pred[pred.storm_id==sid] for sid in ids}
    for _ in range(repeats):
        import pandas as pd
        b=pd.concat([groups[s] for s in rng.choice(ids,len(ids),replace=True)],ignore_index=True)
        m=metrics(b.wind_pred,b.wind_kt,b.probability,b.ri,threshold,base_rate)
        for k in vals:
            if m[k] is not None:vals[k].append(m[k])
    return {k:{'lower':float(np.quantile(v,.025)),'upper':float(np.quantile(v,.975)),'valid_resamples':len(v)} if v else None for k,v in vals.items()}
