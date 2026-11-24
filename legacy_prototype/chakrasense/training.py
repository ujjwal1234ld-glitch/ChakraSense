from pathlib import Path
import json, random, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from .model import CycloneNet, StormDataset, normalise
from .data import validate_samples, save_json, sha256, dataset_fingerprint
from .evaluation import metrics,fit_calibration,calibrate,choose_threshold,conformal_radius,reliability,bootstrap_storms

def device_for(name='auto'):
    if name=='auto':return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable; use cpu or install a compatible PyTorch wheel.')
    return torch.device(name)

def get_predictions(model,manifest,split,device,batch_size=16):
    ds=StormDataset(manifest,split);out=ds.rows.copy();winds=[];logits=[]
    loader=torch.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0)
    model.eval()
    with torch.inference_mode():
        for x,_,_ in loader:
            ans=model(x.to(device));winds.extend((ans['wind']*100).cpu().tolist());logits.extend(ans['logit'].cpu().tolist())
    out['wind_pred']=winds;out['logit']=logits
    return out

def train(manifest,out,epochs=8,batch_size=16,device='auto',seed=41):
    if epochs<1 or batch_size<1:raise ValueError('Epochs and batch size must be positive.')
    manifest=Path(manifest).resolve();out=Path(out).resolve()
    if (out/'model.pt').exists():raise ValueError('Run directory already contains a model. Use a new run directory.')
    x=validate_samples(manifest)
    for split in ['train','validation','calibration']:
        g=x[x.split==split]
        if g.ri.nunique()<2:raise ValueError(f'{split} needs both outcomes; adjust predeclared whole-storm periods, not test scores.')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    dv=device_for(device);model=CycloneNet().to(dv)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    trainset=StormDataset(manifest,'train')
    loader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0,
        generator=torch.Generator().manual_seed(seed))
    regression=nn.SmoothL1Loss(beta=.1);classification=nn.BCEWithLogitsLoss()
    out.mkdir(parents=True,exist_ok=True);history=[];best=float('inf');start=time.perf_counter()
    for epoch in range(epochs):
        model.train();total=0
        for a,w,y in loader:
            optimizer.zero_grad(set_to_none=True);ans=model(a.to(dv))
            loss=regression(ans['wind'],w.to(dv))+classification(ans['logit'],y.to(dv))
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=loss.item()*len(a)
        val=get_predictions(model,manifest,'validation',dv,batch_size)
        score=float(np.mean(((val.wind_pred-val.wind_kt)/100)**2)+np.mean((1/(1+np.exp(-val.logit))-val.ri)**2))
        history.append({'epoch':epoch+1,'training_loss':total/len(trainset),'validation_selection_score':score})
        if score<best:
            best=score;torch.save(model.state_dict(),out/'model.pt')
        print(f'Epoch {epoch+1}/{epochs}: train loss {total/len(trainset):.4f}, validation score {score:.4f}',flush=True)
    model.load_state_dict(torch.load(out/'model.pt',map_location=dv,weights_only=True))
    cal=get_predictions(model,manifest,'calibration',dv,batch_size)
    val=get_predictions(model,manifest,'validation',dv,batch_size)
    correction=fit_calibration(cal.logit,cal.ri)
    # Threshold fitted to validation outputs AFTER applying frozen calibration.
    threshold=choose_threshold(calibrate(val.logit,correction),val.ri.to_numpy())
    radius=conformal_radius(cal.wind_kt-cal.wind_pred)
    metadata={'version':'0.1.0','seed':seed,'epochs':epochs,'batch_size':batch_size,'device':str(dv),
       'gpu':torch.cuda.get_device_name() if dv.type=='cuda' else None,
       'kind':str(x.kind.iloc[0]),'mode':str(x['mode'].iloc[0]),'steps':int(x.steps.iloc[0]),'hours':int(x.hours.iloc[0]),
       'pixels':int(x.pixels.iloc[0]),'width_km':float(x.width_km.iloc[0]),
       'calibration':correction,'wind_radius_kt':radius,'coverage_target':.9,'threshold':threshold,
       'threshold_policy':'Validation cost: missed RI = 2; false alert = 1. Prototype choice, NOT official policy.',
       'base_rate':float(x[x.split=='train'].ri.mean()),'train_mean_wind':float(x[x.split=='train'].wind_kt.mean()),
       'manifest_sha256':sha256(manifest),'model_sha256':sha256(out/'model.pt'),
       'dataset_sha256':dataset_fingerprint(manifest),
       'training_seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters()),
       'split_storms':{s:int(g.storm_id.nunique()) for s,g in x.groupby('split')},
       'status':'SYNTHETIC SOFTWARE TEST ONLY' if x.kind.iloc[0]=='synthetic' else 'RESEARCH MODEL NOT APPROVED FOR WARNINGS',
       'limitations':['No operational meteorological validation.','Intervals have empirical coverage; correlated storm frames do not establish exchangeability.','No external pre-training, environment fusion or automatic storm detection.']}
    save_json(out/'metadata.json',metadata);save_json(out/'history.json',history)
    return metadata

def load_run(run,device='cpu'):
    run=Path(run);meta=json.loads((run/'metadata.json').read_text())
    if sha256(run/'model.pt')!=meta['model_sha256']:raise ValueError('Model checksum mismatch.')
    dv=device_for(device);m=CycloneNet().to(dv)
    m.load_state_dict(torch.load(run/'model.pt',map_location=dv,weights_only=True));m.eval()
    return m,meta,dv

def features(rows,root):
    result=[]
    for r in rows.itertuples():
        with np.load(root/r.path,allow_pickle=False) as z:a=z['bt'][-1]
        result.append(np.concatenate([a.mean((1,2)),a.std((1,2)),a.min((1,2)),a.max((1,2))]))
    return np.asarray(result)

def evaluate(manifest,run,repeats=300):
    manifest=Path(manifest).resolve();run=Path(run)
    if (run/'evaluation.json').exists():raise ValueError('Test results already exist. Keep the original exam result; do not tune against it.')
    x=validate_samples(manifest);m,meta,dv=load_run(run)
    if sha256(manifest)!=meta['manifest_sha256']:raise ValueError('Manifest changed after training.')
    if dataset_fingerprint(manifest)!=meta['dataset_sha256']:raise ValueError('Sequence data changed after training.')
    pred=get_predictions(m,manifest,'test',dv)
    pred['probability']=calibrate(pred.logit,meta['calibration'])
    pred['lower_kt']=pred.wind_pred-meta['wind_radius_kt'];pred['upper_kt']=pred.wind_pred+meta['wind_radius_kt']
    pred['flag']=pred.probability>=meta['threshold']
    result=metrics(pred.wind_pred,pred.wind_kt,pred.probability,pred.ri,meta['threshold'],meta['base_rate'])
    result['interval_coverage']=float(((pred.wind_kt>=pred.lower_kt)&(pred.wind_kt<=pred.upper_kt)).mean())
    result['interval_width_kt']=2*meta['wind_radius_kt']
    trainrows=x[x.split=='train'];valrows=x[x.split=='validation'];calrows=x[x.split=='calibration']
    tr=features(trainrows,manifest.parent);va=features(valrows,manifest.parent);ca=features(calrows,manifest.parent);te=features(pred,manifest.parent)
    reg=make_pipeline(StandardScaler(),Ridge(alpha=10.)).fit(tr,trainrows.wind_kt)
    clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(tr,trainrows.ri)
    cc=fit_calibration(clf.decision_function(ca),calrows.ri)
    bt=choose_threshold(calibrate(clf.decision_function(va),cc),valrows.ri.to_numpy())
    baseline=metrics(reg.predict(te),pred.wind_kt,calibrate(clf.decision_function(te),cc),pred.ri,bt,meta['base_rate'])
    climatology=metrics(np.full(len(pred),meta['train_mean_wind']),pred.wind_kt,np.full(len(pred),meta['base_rate']),pred.ri,meta['threshold'],meta['base_rate'])
    # Contiguous alert episodes per storm, not an official RI-event matching definition.
    episodes=0
    for _,g in pred.groupby('storm_id'):
        g=g.sort_values('time');prev=False;prevtime=None
        for r in g.itertuples():
            t=pd.Timestamp(r.time)
            if r.flag and (not prev or prevtime is None or t-prevtime>pd.Timedelta(hours=meta['hours']*1.5)):episodes+=1
            prev=bool(r.flag);prevtime=t
    report={'status':meta['status'],'model':result,'single_frame_feature_baseline':baseline,'climatology':climatology,
       'storm_bootstrap_95_percent':bootstrap_storms(pred,meta['threshold'],meta['base_rate'],repeats),
       'reliability':reliability(pred.probability,pred.ri),'test_storms':int(pred.storm_id.nunique()),
       'contiguous_flag_episodes':episodes,'episode_note':'Descriptive count only. No independent RI-event lead-time matching or comparison against IMD warning time.',
       'comparison_note':'Synthetic scores demonstrate plumbing only. No environmental or operational persistence baseline without corresponding timed inputs.'}
    pred.to_csv(run/'test_predictions.csv',index=False);save_json(run/'evaluation.json',report)
    return report

def predict_crop(run,sequence_path):
    model,meta,dv=load_run(run)
    with np.load(sequence_path,allow_pickle=False) as z:a=z['bt'];width=float(z['width_km'])
    if a.shape!=(meta['steps'],2,meta['pixels'],meta['pixels']) or width!=meta['width_km'] or not np.isfinite(a).all():raise ValueError('Input differs from training crop contract.')
    if a.min()<150 or a.max()>350:raise ValueError('Temperature outside supported range.')
    with torch.inference_mode():ans=model(torch.from_numpy(normalise(a))[None].to(dv))
    wind=float(ans['wind'].item()*100);prob=float(calibrate([ans['logit'].item()],meta['calibration'])[0])
    return {'wind_kt':wind,'lower_kt':wind-meta['wind_radius_kt'],'upper_kt':wind+meta['wind_radius_kt'],
        'ri_probability':prob,'review_flag':prob>=meta['threshold'],'status':meta['status']}
