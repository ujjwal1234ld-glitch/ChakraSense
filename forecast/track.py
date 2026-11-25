"""Wind-and-track alternative. Freeze on 2016-19 development before 2020-25 evaluation."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from threadpoolctl import threadpool_limits
from model import fit_pair, alternatives
from track_data import TRACK, HERE
from train import NAMES, metrics


def track_inputs(rows):
    if rows.empty or rows.columns.duplicated().any() or not set(TRACK)<=set(rows):
        raise ValueError("Provide a non-empty table with the named track predictors.")
    x=rows[TRACK].apply(pd.to_numeric,errors="raise").astype(float)
    required=["current_wind","lat","lon","land_km","season_sin","season_cos","hours_since_first",
              "recent_peak_loss","strong_wind","coastal_wind"]
    if x[required].isna().any().any() or np.isinf(x).any().any():
        raise ValueError("Missing required track measurement or nonfinite value.")
    for name,lo,hi in [("current_wind",0,200),("lat",-90,90),("lon",-180,180),("land_km",0,20000),
                       ("current_pressure",850,1050),("season_sin",-1,1),("season_cos",-1,1)]:
        if ((x[name]<lo)|(x[name]>hi)).any():
            raise ValueError(f"{name} is outside the accepted range.")
    return x


def predict_track(bundle,rows):
    wind, risks=alternatives(bundle["fit"],track_inputs(rows))
    p=risks[:,bundle["risk_index"]]
    return pd.DataFrame({"forecast_wind_kt":wind[:,bundle["wind_index"]],"ri_score":p,
                         "ri_flag":p>=bundle["threshold"]},index=rows.index)


def run(source,out):
    data=pd.read_csv(source)
    if data.duplicated(["storm_id","time"]).any() or data.groupby("storm_id").split.nunique().max()!=1:
        raise ValueError("Duplicate window or overlapping storms.")
    expected=np.select([data.SEASON<=2015,data.SEASON<=2019],["train","development"],default="test")
    if not np.array_equal(data.split,expected) or data.SEASON.max()>2025:
        raise ValueError("Unexpected temporal split.")
    if not np.array_equal(data.ri,(data.future_wind-data.current_wind>=30).astype(int)):
        raise ValueError("Incorrect target.")
    train=data[data.split=="train"]
    dev=data[data.split=="development"]
    fit=fit_pair(track_inputs(train),train)
    w,p=alternatives(fit,track_inputs(dev))
    wind_loss=np.mean(abs(w-dev.future_wind.to_numpy()[:,None]),axis=0)
    ap=[average_precision_score(dev.ri,r) for r in p.T]
    wi,pi=int(np.argmin(wind_loss)),int(np.argmax(ap))
    cut=max(np.arange(5,51)/100,key=lambda t:(f1_score(dev.ri,p[:,pi]>=t),t))
    bundle=dict(fit=fit,wind_index=wi,risk_index=pi,threshold=float(cut))
    out.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,out/"track.joblib",compress=3)
    frozen=joblib.load(out/"track.joblib")
    report=dict(wind=NAMES[wi],risk=NAMES[pi],threshold=float(cut),
                train_windows=len(train),train_storms=int(train.storm_id.nunique()),train_positives=int(train.ri.sum()),
                train_first=int(train.SEASON.min()),train_last=int(train.SEASON.max()),
                selection={n:dict(dev_mae=float(wind_loss[i]),dev_ap=float(ap[i])) for i,n in enumerate(NAMES)})
    predictions=[]
    for name in ("development","test"):
        rows=data[data.split==name].copy()
        scored=rows.join(predict_track(frozen,rows))
        report[name]=metrics(scored)
        report[name]["climatology_brier"]=float(np.mean((scored.ri-train.ri.mean())**2))
        report[name]["ri_storms"]=int(scored.loc[scored.ri==1,"storm_id"].nunique())
        report[name]["ri_storms_detected"]=int(scored.loc[(scored.ri==1)&scored.ri_flag,"storm_id"].nunique())
        predictions.append(scored)
    cols=["storm_id","time","split","SEASON","current_wind","future_wind","ri","forecast_wind_kt","ri_score","ri_flag"]
    pd.concat(predictions)[cols].to_csv(out/"track_predictions.csv",index=False,float_format="%.9g")
    (out/"track_results.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=HERE/"data/track_samples.csv")
    parser.add_argument("--out",type=Path,default=HERE/"release")
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        print(json.dumps(run(args.data,args.out),indent=2))
