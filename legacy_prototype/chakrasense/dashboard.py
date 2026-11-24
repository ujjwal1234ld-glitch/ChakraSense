"""Export a self-contained, offline HTML replay of saved test predictions."""
from pathlib import Path
import base64,io,json
import numpy as np
import pandas as pd
from PIL import Image
from .data import dataset_fingerprint

def png_channel(a):
    # Fixed 180..310 K display range. No per-image rescaling that hides changes.
    x=np.clip((a-180)/130,0,1)
    rgb=np.stack([40+180*x,75+155*x,135+100*x],axis=-1).astype('uint8')
    image=Image.fromarray(rgb).resize((256,256),Image.Resampling.NEAREST)
    buffer=io.BytesIO();image.save(buffer,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()

def export_dashboard(manifest,run,out):
    manifest=Path(manifest);run=Path(run)
    metadata=json.loads((run/'metadata.json').read_text());report=json.loads((run/'evaluation.json').read_text())
    if metadata['kind']!='synthetic' or 'image-statistics' not in metadata.get('model_type',''):
        raise ValueError('This beginner replay is for synthetic baseline runs only. Real/neural runs use their CSV and JSON reports.')
    if dataset_fingerprint(manifest)!=metadata['dataset_sha256']:raise ValueError('Dashboard data differs from the trained dataset.')
    rows=pd.read_csv(run/'test_predictions.csv');records=[]
    for r in rows.itertuples():
        with np.load(manifest.parent/r.path,allow_pickle=False) as z:a=z['bt']
        records.append({'storm':r.storm_id,'time':r.time,'observed':r.latest_observed_at,'available':r.latest_available_at,
           'wind':float(r.wind_pred),'lower':float(r.lower_kt),'upper':float(r.upper_kt),'p':float(r.probability),
           'truth':float(r.wind_kt),'ri':int(r.ri),'ir':png_channel(a[-1,0]),'wv':png_channel(a[-1,1]),
           'history':a.mean(axis=(2,3)).round(2).tolist()})
    payload=json.dumps({'meta':metadata,'report':report,'rows':records},allow_nan=False).replace('<','\\u003c')
    template=(Path(__file__).parent/'dashboard_template.html').read_text(encoding='utf-8')
    Path(out).write_text(template.replace('__PAYLOAD__',payload),encoding='utf-8')
    return {'dashboard':str(Path(out).resolve()),'records':len(records),'live_feed':False}
