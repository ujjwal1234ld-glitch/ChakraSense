"""Artificial storm-like arrays: software exercise, never meteorological evidence."""
from pathlib import Path
import numpy as np
import pandas as pd
from .data import save_json,create_splits,build_sequences

def generate_demo(root,seed=20):
    root=Path(root).resolve()
    if (root/'frames.csv').exists():raise ValueError('Demo folder already exists; choose a new folder.')
    (root/'frames').mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed)
    yy,xx=np.mgrid[-1:1:64j,-1:1:64j];radius=np.sqrt(xx**2+yy**2)
    rows=[];tracks=[]
    for i in range(48):
        year=2018 if i<24 else 2022 if i<32 else 2023 if i<40 else 2024
        start=pd.Timestamp(f'{year}-05-01',tz='UTC')+pd.Timedelta(days=(i%24)*3)
        sid=f'ARTIFICIAL_{year}_{i:03}';ri=i%2==0
        slope=4.1 if ri else float(rng.uniform(-.6,1.6));base=float(rng.uniform(20,45));phase=float(rng.uniform(-3,3))
        for j in range(25):
            t=start+pd.Timedelta(hours=3*j);wind=base+slope*j
            tracks.append({'storm_id':sid,'time':t.isoformat(),'wind_kt':wind,'lat':12.,'lon':78.,'season':year,'agency':'NEWDELHI','wind_averaging_minutes':3,'kind':'synthetic'})
            if j>16:continue
            cloud=np.exp(-((radius-(.35+.002*wind))/.2)**2)
            texture=np.sin(np.arctan2(yy,xx)*3+radius*8+phase)*3
            ir=290-(30+.4*wind)*cloud+texture+rng.normal(0,1,xx.shape)
            wv=260-(18+.25*wind)*cloud+texture*.4+rng.normal(0,1,xx.shape)
            bt=np.stack([ir,wv]).astype(np.float32)
            name=f'frames/{sid}_{j:02}.npz';np.savez_compressed(root/name,bt=bt)
            rows.append({'storm_id':sid,'observed_at':t.isoformat(),'available_at':(t+pd.Timedelta(minutes=20)).isoformat(),
                'center_available_at':t.isoformat(),'center_source':'best_track','frame_path':name,'kind':'synthetic','satellite':'ARTIFICIAL'})
    pd.DataFrame(rows).to_csv(root/'frames.csv',index=False);tr=pd.DataFrame(tracks);tr.to_csv(root/'tracks.csv',index=False)
    save_json(root/'splits.json',create_splits(tr,2022,2023,2024))
    build_sequences(root/'frames.csv',root/'tracks.csv',root/'splits.json',root/'sequences')
    save_json(root/'NOTICE.json',{'kind':'synthetic','message':'Artificial data for software testing. Not INSAT observations. Never report these scores as cyclone forecast skill.'})
    return str(root/'sequences'/'samples.csv')
