"""Explicit HDF5 decoding; mapping must be checked against the product handbook.

This adapter is unit-tested with fixtures, not yet validated against a user's
INSAT file. It refuses missing pixels instead of inventing cloud temperatures.
"""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from .data import save_json, sha256, utc

def inspect_h5(path):
    result={'file':str(path),'datasets':{},'attributes':{}}
    def simple(v):
        if isinstance(v,bytes):return v.decode(errors='replace')
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        return str(v)
    with h5py.File(path,'r') as f:
        result['attributes']={k:simple(v) for k,v in f.attrs.items()}
        def visit(name,obj):
            if isinstance(obj,h5py.Dataset):
                result['datasets'][name]={'shape':list(obj.shape),'dtype':str(obj.dtype),'attributes':{k:simple(v) for k,v in obj.attrs.items()}}
        f.visititems(visit)
    return result

def decoded(f,key,scale=1.,offset=0.,fill_values=()):
    raw=np.asarray(f[key][...]).squeeze()
    invalid=~np.isfinite(raw)
    for value in fill_values:invalid|=(raw==value)
    a=raw.astype(np.float64)*scale+offset
    a[invalid]=np.nan
    return a

def read_channel(f,c):
    """Scale/offset/fill are explicit, never silently inferred from variable names."""
    a=decoded(f,c['image'],fill_values=c['image_fill_values'])
    if c['units']=='counts':
        lut=decoded(f,c['lut'],c.get('lut_scale',1.),c.get('lut_offset',0.),c['lut_fill_values'])
        if lut.ndim!=1:raise ValueError('Temperature lookup table must be one-dimensional.')
        valid=np.isfinite(a)&(a>=0)&(a<len(lut))&(a==np.floor(a))
        bt=np.full(a.shape,np.nan);bt[valid]=lut[a[valid].astype(int)]
    elif c['units']=='kelvin':
        bt=a*c.get('image_scale',1.)+c.get('image_offset',0.)
    else:raise ValueError('Channel units must be counts or kelvin; Celsius/radiance needs an explicit verified conversion.')
    lat=decoded(f,c['latitude'],c['latitude_scale'],c['latitude_offset'],c['coordinate_fill_values'])
    lon=decoded(f,c['longitude'],c['longitude_scale'],c['longitude_offset'],c['coordinate_fill_values'])
    if bt.ndim!=2 or lat.shape!=bt.shape or lon.shape!=bt.shape:
        raise ValueError('Each channel needs its own matching 2-D latitude/longitude grid. Do not resize WV coordinates by assumption.')
    return bt,lat,lon

def crop_h5(path,mapping,center_lat,center_lon):
    if mapping.get('verified_against_product_manual') is not True:
        raise ValueError('Verify mapping with the actual product/manual first, then set verified_against_product_manual=true.')
    if not -40<=center_lat<=40 or not 30<=center_lon<=120:raise ValueError('Center outside supported northern Indian Ocean work area.')
    size=int(mapping['pixels']);width=float(mapping['width_km']);distance=float(mapping['max_neighbour_km'])
    if size<32 or size>256 or width<=0 or distance<=0:raise ValueError('Invalid crop settings.')
    # Local equirectangular grid, adequate as a research crop near the storm;
    # explicit fixed physical width avoids changing scale across satellites.
    axis=np.linspace(-width/2,width/2,size);xx,yy=np.meshgrid(axis,axis[::-1]);targets=np.column_stack([xx.ravel(),yy.ravel()])
    channels=[]
    with h5py.File(path,'r') as f:
        for name in ['ir','wv']:
            bt,lat,lon=read_channel(f,mapping['channels'][name])
            dx=((lon-center_lon+180)%360-180)*111.32*np.cos(np.deg2rad(center_lat));dy=(lat-center_lat)*111.32
            ok=np.isfinite(bt)&np.isfinite(dx)&np.isfinite(dy)&(np.abs(lat)<=90)&(np.abs(lon)<=180)&(bt>=150)&(bt<=350)
            ok&=(np.abs(dx)<=width/2+distance)&(np.abs(dy)<=width/2+distance)
            if not ok.any():raise ValueError(f'{name}: no valid pixels near the center.')
            d,idx=cKDTree(np.column_stack([dx[ok],dy[ok]])).query(targets,distance_upper_bound=distance)
            if not np.isfinite(d).all():raise ValueError(f'{name}: incomplete crop or wrong coordinate scale; missing pixels are not filled.')
            channels.append(bt[ok][idx].reshape(size,size).astype(np.float32))
    return np.stack(channels)

def prepare_frames(raw_manifest,mapping_path,out):
    raw_manifest=Path(raw_manifest).resolve();mapping_path=Path(mapping_path).resolve();out=Path(out).resolve()
    if (out/'frames.csv').exists():raise ValueError('Output already prepared. Use a new folder for an auditable import.')
    mapping=json.loads(mapping_path.read_text());x=pd.read_csv(raw_manifest,keep_default_na=False)
    required={'storm_id','observed_at','available_at','center_available_at','center_source','center_lat','center_lon','path','satellite'}
    if not required.issubset(x):raise ValueError('Raw manifest columns are incomplete; see config/raw_manifest_template.csv.')
    if x.empty:raise ValueError('Raw manifest contains no files.')
    if x.duplicated(['storm_id','observed_at']).any():raise ValueError('Duplicate storm/observation.')
    (out/'crops').mkdir(parents=True,exist_ok=True);rows=[];sources=[]
    for i,r in enumerate(x.itertuples()):
        p=(raw_manifest.parent/r.path).resolve();a=crop_h5(p,mapping,float(r.center_lat),float(r.center_lon))
        obs=utc(r.observed_at);known=bool(str(r.available_at).strip())
        available=utc(r.available_at) if known else obs
        center_known=bool(str(r.center_available_at).strip())
        center_time=utc(r.center_available_at) if center_known else obs
        if available<obs:raise ValueError('Arrival time cannot be before the observation.')
        if r.satellite!=mapping['satellite']:raise ValueError('Separate and verify mappings for different satellites/products.')
        if r.center_source not in ['best_track','operational_bulletin']:raise ValueError('Unknown center source.')
        name=f'crops/{i:06}.npz';np.savez_compressed(out/name,bt=a,width_km=mapping['width_km'])
        rows.append({'storm_id':r.storm_id,'observed_at':obs.isoformat(),'available_at':available.isoformat(),
           'availability_known':known,'center_available_at':center_time.isoformat(),'center_availability_known':center_known,
           'center_source':r.center_source,'frame_path':name,'kind':'real','satellite':r.satellite})
        sources.append({'file':str(p),'sha256':sha256(p)})
    pd.DataFrame(rows).to_csv(out/'frames.csv',index=False)
    save_json(out/'provenance.json',{'raw_manifest_sha256':sha256(raw_manifest),'mapping_sha256':sha256(mapping_path),
        'mapping':mapping,'sources':sources,'unknown_times':'Observation time is a storage placeholder only when availability_known is false; such frames are barred from operational replay.'})
    return {'frames':len(rows),'output':str(out/'frames.csv')}
