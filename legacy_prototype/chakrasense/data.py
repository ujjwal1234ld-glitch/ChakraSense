"""Explicit data contracts, exact future labels, whole-storm chronological splits."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

def save_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')

def utc(series):
    return pd.to_datetime(series, utc=True, errors='raise', format='mixed')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def import_ibtracs(path, out, start=2013, end=2025):
    raw=pd.read_csv(path,skiprows=[1],low_memory=False)
    for c in ['SID','ISO_TIME','NEWDELHI_WIND','NEWDELHI_LAT','NEWDELHI_LON','SEASON']:
        if c not in raw:raise ValueError(f'Missing IBTrACS column {c}')
    x=pd.DataFrame({'storm_id':raw.SID.astype(str).str.strip(), 'time':utc(raw.ISO_TIME),
        'wind_kt':pd.to_numeric(raw.NEWDELHI_WIND,errors='coerce'),
        'lat':pd.to_numeric(raw.NEWDELHI_LAT,errors='coerce'),
        'lon':pd.to_numeric(raw.NEWDELHI_LON,errors='coerce'),
        'season':pd.to_numeric(raw.SEASON,errors='coerce')})
    # Do not substitute US winds or interpolated WMO winds for missing IMD reports.
    x=x[(x.season>=start)&(x.season<=end)&x.wind_kt.between(1,250)&x.lat.between(-90,90)&x.lon.between(-180,180)].copy()
    x['agency']='NEWDELHI'; x['wind_averaging_minutes']=3; x['kind']='real'
    x=x.drop_duplicates().sort_values(['storm_id','time'])
    if x.duplicated(['storm_id','time']).any():raise ValueError('Conflicting best-track rows; resolve before importing.')
    Path(out).parent.mkdir(parents=True,exist_ok=True);x.to_csv(out,index=False)
    labels=make_labels(x)
    positives=labels[labels.ri==1]
    counts=[]
    for year,g in labels.groupby(labels.time.dt.year):
        counts.append({'year':int(year),'storms':int(g.storm_id.nunique()),'labelled_windows':len(g),'positive_windows':int(g.ri.sum()),'storms_with_ri':int(g[g.ri==1].storm_id.nunique())})
    result={'source_sha256':sha256(path),'source':'NOAA IBTrACS v04r01 NEWDELHI columns',
        'years':[start,end],'rows':len(x),'storms':int(x.storm_id.nunique()),
        'exact_24h_windows':len(labels),'positive_windows':int(labels.ri.sum()),
        'storms_with_ri':int(positives.storm_id.nunique()),
        'warning':'Overlapping positive windows are NOT independent RI episodes. Counts exclude missing NEWDELHI reports. No satellite coverage checked.',
        'by_year':counts}
    save_json(Path(out).with_suffix('.inventory.json'),result)
    return result

def read_tracks(path):
    x=pd.read_csv(path);x['time']=utc(x.time)
    if x.duplicated(['storm_id','time']).any():raise ValueError('Duplicate storm/time labels.')
    if not (x.agency=='NEWDELHI').all() or not (x.wind_averaging_minutes==3).all():
        raise ValueError('This benchmark requires NEWDELHI three-minute wind labels.')
    if not np.isfinite(x.wind_kt).all() or not x.wind_kt.between(0,250).all():raise ValueError('Invalid wind labels.')
    return x

def make_labels(tracks):
    x=tracks.copy();x['time']=utc(x.time)
    if x.duplicated(['storm_id','time']).any():raise ValueError('Duplicate storm/time labels.')
    future=x[['storm_id','time','wind_kt']].copy()
    future['time']=future.time-pd.Timedelta(hours=24)
    future=future.rename(columns={'wind_kt':'future_wind_kt'})
    x=x.merge(future,on=['storm_id','time'],how='inner',validate='one_to_one')
    x['delta_24h_kt']=x.future_wind_kt-x.wind_kt
    x['ri']=(x.delta_24h_kt>=30).astype(int)
    return x

def create_splits(tracks, validation_year, calibration_year, test_year):
    if not validation_year<calibration_year<test_year:raise ValueError('Years must be chronological.')
    # Use original storm season, not individual row years: no cross-year storm leakage.
    x=tracks.copy(); x['time']=utc(x.time)
    years=x.groupby('storm_id').time.min().dt.year
    splits={}
    for sid,year in years.items():
        splits[str(sid)]='train' if year<validation_year else 'validation' if year<calibration_year else 'calibration' if year<test_year else 'test'
    if set(splits.values())!={'train','validation','calibration','test'}:raise ValueError('Each of the four splits needs whole storms.')
    return splits

def build_sequences(frames_path, tracks_path, splits_path, out, mode='retrospective', steps=9, hours=3, tolerance_minutes=45):
    if mode not in ['retrospective','operational'] or steps<2 or hours<=0:raise ValueError('Invalid sequence settings.')
    frames_path=Path(frames_path).resolve();frames=pd.read_csv(frames_path)
    frames['observed_at']=utc(frames.observed_at)
    frames['available_at']=utc(frames.available_at)
    if (frames.available_at<frames.observed_at).any():raise ValueError('Availability cannot precede observation.')
    if frames.duplicated(['storm_id','observed_at']).any():raise ValueError('Choose one satellite per storm/time before building sequences.')
    tracks=read_tracks(tracks_path);labels=make_labels(tracks)
    splits=json.loads(Path(splits_path).read_text())
    if mode=='operational' and not {'availability_known','center_availability_known'}.issubset(frames):
        raise ValueError('Operational replay requires documented arrival times for images and centers.')
    out=Path(out).resolve()
    if (out/'samples.csv').exists():raise ValueError('Sequences already exist; use a new output folder.')
    out.mkdir(parents=True,exist_ok=True);rows=[];rejected={}
    def reject(reason):rejected[reason]=rejected.get(reason,0)+1
    grouped={sid:g.sort_values('observed_at') for sid,g in frames.groupby('storm_id')}
    for lab in labels.itertuples():
        if lab.storm_id not in splits:raise ValueError(f'Storm absent from splits: {lab.storm_id}')
        g=grouped.get(lab.storm_id)
        if g is None:reject('no_frames');continue
        seq=[];selected=[];widths=[];failed=False
        for target in pd.date_range(end=lab.time,periods=steps,freq=pd.Timedelta(hours=hours)):
            candidates=g[(g.observed_at<=target)&(g.observed_at>=target-pd.Timedelta(minutes=tolerance_minutes))]
            if mode=='operational':
                candidates=candidates[(candidates.available_at<=lab.time)&(candidates.center_source=='operational_bulletin')&
                    (candidates.availability_known==True)&(candidates.center_availability_known==True)]
                if not candidates.empty:
                    candidates=candidates[utc(candidates.center_available_at)<=lab.time]
            if candidates.empty:failed=True;reject('missing_or_unavailable_history');break
            row=candidates.iloc[-1]; fp=(frames_path.parent/str(row.frame_path)).resolve()
            with np.load(fp,allow_pickle=False) as z:
                arr=z['bt'].astype(np.float32);widths.append(float(z['width_km']) if 'width_km' in z else 1000.)
            if arr.ndim!=3 or arr.shape[0]!=2 or arr.shape[1]!=arr.shape[2] or not np.isfinite(arr).all():
                raise ValueError(f'Invalid crop {fp}')
            if arr.min()<150 or arr.max()>350:raise ValueError(f'Unexpected temperature units/range: {fp}')
            seq.append(arr);selected.append(row)
        if failed:continue
        if len({str(r.observed_at) for r in selected})!=steps:raise ValueError('Duplicate frame reused in sequence.')
        if len(set(widths))!=1:raise ValueError('Mixed physical crop widths.')
        kind=set(str(r['kind']) for r in selected)
        if len(kind)!=1 or kind!={str(lab.kind)}:raise ValueError('Mixed artificial and real data.')
        key=hashlib.sha256(f'{lab.storm_id}|{lab.time}'.encode()).hexdigest()[:20]
        dst=out/f'{key}.npz';np.savez_compressed(dst,bt=np.stack(seq),width_km=widths[0])
        rows.append({'sample_id':key,'storm_id':lab.storm_id,'time':lab.time.isoformat(),'path':dst.name,
            'split':splits[lab.storm_id],'wind_kt':float(lab.wind_kt),'ri':int(lab.ri),
            'delta_24h_kt':float(lab.delta_24h_kt),'kind':next(iter(kind)),'mode':mode,
            'latest_observed_at':selected[-1].observed_at.isoformat(),
            'latest_available_at':max(r.available_at for r in selected).isoformat(),
            'steps':steps,'hours':hours,'pixels':int(seq[0].shape[1]),'width_km':widths[0]})
    if not rows:raise ValueError(f'No usable sequences: {rejected}')
    samples=pd.DataFrame(rows);samples.to_csv(out/'samples.csv',index=False)
    save_json(out/'build_report.json',{'accepted':len(rows),'rejected':rejected,'mode':mode,
       'frames_sha256':sha256(frames_path),'labels_sha256':sha256(tracks_path),'splits_sha256':sha256(splits_path),
       'horizon':'wind(t+24h)-wind(t); t is the scheduled decision/label time in UTC'})
    return samples

def validate_samples(path):
    path=Path(path);x=pd.read_csv(path)
    required={'sample_id','storm_id','time','path','split','wind_kt','ri','kind','mode','steps','hours','pixels','width_km'}
    if not required.issubset(x.columns):raise ValueError('Incomplete sample manifest.')
    if x.empty or x.sample_id.duplicated().any():raise ValueError('Empty or duplicate samples.')
    if (x.groupby('storm_id').split.nunique()>1).any():raise ValueError('Storm leakage across splits.')
    if set(x.split)!={'train','validation','calibration','test'}:raise ValueError('Four nonempty splits required.')
    if x.kind.nunique()!=1 or x['mode'].nunique()!=1 or not x.kind.isin(['real','synthetic']).all() or not x['mode'].isin(['retrospective','operational']).all():raise ValueError('Do not mix demo/real or replay modes.')
    if not x.ri.isin([0,1]).all() or not np.isfinite(x.wind_kt).all():raise ValueError('Invalid targets.')
    if any(x[c].nunique()!=1 for c in ['steps','hours','pixels','width_km']):raise ValueError('Inconsistent sequence contract.')
    # Same crop archive cannot be reused under different storm IDs/splits.
    seen={}
    for r in x.itertuples():
        fp=(path.parent/r.path).resolve()
        if not fp.is_relative_to(path.parent.resolve()):raise ValueError('Sample path escapes dataset directory.')
        with np.load(fp,allow_pickle=False) as z:
            a=z['bt'];width=float(z['width_km'])
        if a.shape!=(r.steps,2,r.pixels,r.pixels) or width!=r.width_km:raise ValueError('Crop shape or physical width differs from manifest.')
        if not np.isfinite(a).all() or a.min()<150 or a.max()>350:raise ValueError('Invalid brightness temperatures.')
        digest=hashlib.sha256(a.astype('<f4').tobytes()).hexdigest()
        if digest in seen and seen[digest]!=r.split:raise ValueError('Duplicate sequence content across splits.')
        seen[digest]=r.split
    return x

def dataset_fingerprint(manifest):
    """Pin the manifest AND every array so changes after training are detected."""
    manifest=Path(manifest);x=pd.read_csv(manifest);h=hashlib.sha256()
    h.update(sha256(manifest).encode())
    for row in x.itertuples():h.update(sha256(manifest.parent/row.path).encode())
    return h.hexdigest()
