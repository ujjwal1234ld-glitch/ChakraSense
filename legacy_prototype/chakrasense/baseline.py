"""Executable lightweight baseline. This is NOT the CNN-GRU model.

Uses image statistics at each past time: ridge wind regression and logistic RI
classification. All learned transforms use training storms only.
"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from .data import validate_samples,save_json,dataset_fingerprint,sha256
from .evaluation import logistic_fit,fit_calibration,calibrate,choose_threshold,conformal_radius,metrics,reliability,bootstrap_storms

def extract(sequence,last_only=False):
    a=np.asarray(sequence,dtype=float)
    if last_only:a=a[-1:]
    return np.stack([a.mean((2,3)),a.std((2,3)),a.min((2,3)),a.max((2,3))],axis=-1).ravel()

def feature_table(rows,root,last_only=False):
    result=[]
    for r in rows.itertuples():
        with np.load(root/r.path,allow_pickle=False) as z:result.append(extract(z['bt'],last_only))
    return np.array(result)

def fit(x,wind,y):
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.
    z=(x-mean)/scale;design=np.column_stack([z,np.ones(len(z))])
    penalty=np.eye(design.shape[1])*10.;penalty[-1,-1]=0
    reg=np.linalg.solve(design.T@design+penalty,design.T@np.asarray(wind))
    clf=logistic_fit(z,y)
    return {'mean':mean,'scale':scale,'reg':reg,'clf':clf}

def infer(model,x):
    design=np.column_stack([(x-model['mean'])/model['scale'],np.ones(len(x))])
    return design@model['reg'],design@model['clf']

def train_baseline(manifest,out):
    manifest=Path(manifest).resolve();out=Path(out);start=time.perf_counter()
    if (out/'model.npz').exists():raise ValueError('Use a new run folder; do not replace a model after testing.')
    rows=validate_samples(manifest)
    for split in ['train','validation','calibration']:
        if rows[rows.split==split].ri.nunique()!=2:raise ValueError(f'{split} requires both RI outcomes.')
    tr=rows[rows.split=='train'];va=rows[rows.split=='validation'];ca=rows[rows.split=='calibration']
    m=fit(feature_table(tr,manifest.parent),tr.wind_kt,tr.ri)
    cw,cl=infer(m,feature_table(ca,manifest.parent));vw,vl=infer(m,feature_table(va,manifest.parent))
    correction=fit_calibration(cl,ca.ri);threshold=choose_threshold(calibrate(vl,correction),va.ri.to_numpy())
    radius=conformal_radius(ca.wind_kt-cw)
    out.mkdir(parents=True,exist_ok=True);np.savez(out/'model.npz',**m)
    meta={'model_type':'temporal image-statistics ridge/logistic baseline; NOT CNN-GRU',
        'kind':str(rows.kind.iloc[0]),'mode':str(rows['mode'].iloc[0]),'status':'SYNTHETIC SOFTWARE TEST ONLY' if rows.kind.iloc[0]=='synthetic' else 'RESEARCH BASELINE; NOT APPROVED FOR WARNINGS',
        'steps':int(rows.steps.iloc[0]),'hours':int(rows.hours.iloc[0]),'pixels':int(rows.pixels.iloc[0]),'width_km':float(rows.width_km.iloc[0]),
        'calibration':correction,'threshold':threshold,'wind_radius_kt':radius,'coverage_target':.9,
        'base_rate':float(tr.ri.mean()),'train_mean_wind':float(tr.wind_kt.mean()),'dataset_sha256':dataset_fingerprint(manifest),
        'model_sha256':sha256(out/'model.npz'),'training_seconds':time.perf_counter()-start,
        'split_storms':{s:int(g.storm_id.nunique()) for s,g in rows.groupby('split')},
        'threshold_policy':'Validation cost: missed RI=2, false flag=1; research choice only.',
        'limitations':['Artificial scores are not cyclone forecast skill.','Correlated windows: interval coverage is empirical, not a 90% operational guarantee.','No environment inputs, automatic storm finding, or live warnings.']}
    save_json(out/'metadata.json',meta);return meta

def load_baseline(run):
    run=Path(run);meta=json.loads((run/'metadata.json').read_text())
    if sha256(run/'model.npz')!=meta['model_sha256']:raise ValueError('Model checksum mismatch.')
    with np.load(run/'model.npz',allow_pickle=False) as z:model={k:z[k] for k in z.files}
    return model,meta

def evaluate_baseline(manifest,run):
    manifest=Path(manifest);run=Path(run)
    if (run/'evaluation.json').exists():raise ValueError('Test report already exists; preserve it.')
    rows=validate_samples(manifest);m,meta=load_baseline(run)
    if dataset_fingerprint(manifest)!=meta['dataset_sha256']:raise ValueError('Data changed after training.')
    test=rows[rows.split=='test'].copy();w,logits=infer(m,feature_table(test,manifest.parent))
    test['wind_pred']=w;test['logit']=logits;test['probability']=calibrate(logits,meta['calibration'])
    test['lower_kt']=w-meta['wind_radius_kt'];test['upper_kt']=w+meta['wind_radius_kt'];test['flag']=test.probability>=meta['threshold']
    score=metrics(w,test.wind_kt,test.probability,test.ri,meta['threshold'],meta['base_rate'])
    score['interval_coverage']=float(((test.wind_kt>=test.lower_kt)&(test.wind_kt<=test.upper_kt)).mean())
    score['interval_width_kt']=2*meta['wind_radius_kt']
    # Compare against the exact same algorithm restricted to the current frame.
    tr=rows[rows.split=='train'];va=rows[rows.split=='validation'];ca=rows[rows.split=='calibration']
    single=fit(feature_table(tr,manifest.parent,True),tr.wind_kt,tr.ri)
    _,cl=infer(single,feature_table(ca,manifest.parent,True));_,vl=infer(single,feature_table(va,manifest.parent,True))
    correction=fit_calibration(cl,ca.ri);th=choose_threshold(calibrate(vl,correction),va.ri.to_numpy())
    sw,sl=infer(single,feature_table(test,manifest.parent,True))
    report={'status':meta['status'],'model_type':meta['model_type'],'model':score,
        'single_frame_feature_baseline':metrics(sw,test.wind_kt,calibrate(sl,correction),test.ri,th,meta['base_rate']),
        'climatology':metrics(np.full(len(test),meta['train_mean_wind']),test.wind_kt,np.full(len(test),meta['base_rate']),test.ri,meta['threshold'],meta['base_rate']),
        'storm_bootstrap_95_percent':bootstrap_storms(test,meta['threshold'],meta['base_rate']),
        'reliability':reliability(test.probability,test.ri),'test_storms':int(test.storm_id.nunique()),
        'warning':'Synthetic benchmark only; no real-world skill, lead-time advantage or safety conclusion can be drawn.'}
    test.to_csv(run/'test_predictions.csv',index=False);save_json(run/'evaluation.json',report);return report

def predict_baseline(run,sequence):
    m,meta=load_baseline(run)
    with np.load(sequence,allow_pickle=False) as z:a=z['bt'];width=float(z['width_km'])
    if a.shape!=(meta['steps'],2,meta['pixels'],meta['pixels']) or width!=meta['width_km'] or not np.isfinite(a).all() or a.min()<150 or a.max()>350:raise ValueError('Invalid input or different crop contract.')
    w,l=infer(m,extract(a)[None]);p=float(calibrate(l,meta['calibration'])[0]);w=float(w[0])
    return {'wind_kt':w,'lower_kt':w-meta['wind_radius_kt'],'upper_kt':w+meta['wind_radius_kt'],'ri_probability':p,'review_flag':p>=meta['threshold'],'status':meta['status']}
