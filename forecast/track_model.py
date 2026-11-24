"""Historical North Indian Ocean wind and RI forecasting. Python 3.12+.

    python forecast/track_model.py fit ibtracs.NI.list.v04r01.csv output
    python forecast/track_model.py score output/track.joblib inputs.csv forecasts.csv

fit creates a frozen estimator, named feature examples, per-window results and JSON.
On a revised NOAA archive, forecasts and split counts can differ from the milestone.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

FEATURES = ["current_wind","change_6","change_12","change_24","current_pressure","pressure_change_12",
            "lat","lon","land_km","land_change_6","latitude_change_6","longitude_change_6",
            "season_sin","season_cos","recent_peak_loss","hours_since_first","coastal_wind",
            "approach_wind","ocean_trend","strong_wind"]
REQUIRED = ["current_wind","lat","lon","land_km","season_sin","season_cos","hours_since_first",
            "recent_peak_loss","strong_wind","coastal_wind"]
NAMES = ["linear", "boosted", "equal blend"]


def build_table(path):
    cols = ["SID","SEASON","BASIN","TRACK_TYPE","ISO_TIME","IFLAG","NEWDELHI_WIND",
            "NEWDELHI_PRES","NEWDELHI_LAT","NEWDELHI_LON","DIST2LAND"]
    raw = pd.read_csv(path, usecols=cols, skiprows=[1], low_memory=False)
    raw = raw[(raw.BASIN.str.strip()=="NI") & (raw.TRACK_TYPE.str.strip()=="main") &
              raw.IFLAG.str.strip().str[5].isin(["O","V"]) & raw.SEASON.between(1982,2025)].copy()
    raw = raw.rename(columns={"SID":"storm_id","NEWDELHI_WIND":"current_wind","NEWDELHI_PRES":"current_pressure",
                              "NEWDELHI_LAT":"lat","NEWDELHI_LON":"lon","DIST2LAND":"land_km"})
    for key in ("current_wind","current_pressure","lat","lon","land_km"):
        raw[key]=pd.to_numeric(raw[key],errors="coerce")
    raw.loc[~raw.current_pressure.between(850,1050),"current_pressure"]=np.nan
    raw["time"]=pd.to_datetime(raw.ISO_TIME,utc=True)
    raw=raw[raw.current_wind.between(0,200)].sort_values(["storm_id","time"])
    if raw.duplicated(["storm_id","time"]).any():
        raise ValueError("Ambiguous storm observation")
    rows=[]
    for _,storm in raw.groupby("storm_id"):
        g=storm.set_index("time").copy()
        def lag(col, hours):
            return g[col].reindex(g.index-pd.Timedelta(hours=hours)).to_numpy()
        for h in (6,12,24):
            g[f"change_{h}"]=g.current_wind-lag("current_wind",h)
        g["pressure_change_12"]=g.current_pressure-lag("current_pressure",12)
        for dest,field in [("land_change_6","land_km"),("latitude_change_6","lat"),("longitude_change_6","lon")]:
            g[dest]=g[field]-lag(field,6)
        g["hours_since_first"]=(g.index-g.index.min()).total_seconds()/3600
        g["recent_peak_loss"]=g.current_wind.rolling("24h",closed="both").max()-g.current_wind
        g["season_sin"]=np.sin(2*np.pi*g.index.day_of_year/365.25)
        g["season_cos"]=np.cos(2*np.pi*g.index.day_of_year/365.25)
        g["coastal_wind"]=g.current_wind*np.exp(-g.land_km/120)
        projected=(g.land_km+2*g.land_change_6).clip(lower=0)
        g["approach_wind"]=g.current_wind*np.exp(-projected/120)
        g["ocean_trend"]=g.change_12*(1-np.exp(-g.land_km/120))
        g["strong_wind"]=(g.current_wind-55).clip(lower=0)
        g["future_wind"]=lag("current_wind",-24)
        g=g.dropna(subset=["future_wind","land_km","lat","lon"])
        g["ri"]=(g.future_wind-g.current_wind>=30).astype(int)
        g["split"]=np.select([g.SEASON<=2015,g.SEASON<=2019],["train","development"],default="test")
        rows.append(g.reset_index()[["storm_id","time","SEASON","split","future_wind","ri",*FEATURES]])
    result=pd.concat(rows,ignore_index=True)
    if result.groupby("storm_id").split.nunique().max()!=1:
        raise ValueError("Overlapping storms across periods")
    return result


def inputs(rows):
    if rows.empty or rows.columns.duplicated().any() or not set(FEATURES)<=set(rows):
        raise ValueError("Provide a non-empty CSV with all named predictors")
    x=rows[FEATURES].apply(pd.to_numeric,errors="raise").astype(float)
    if x[REQUIRED].isna().any().any() or np.isinf(x).any().any():
        raise ValueError("Missing required track measurement or non-finite value")
    for name,lo,hi in [("current_wind",0,200),("lat",-90,90),("lon",-180,180),("land_km",0,20000),
                       ("current_pressure",850,1050),("season_sin",-1,1),("season_cos",-1,1)]:
        if ((x[name]<lo)|(x[name]>hi)).any():
            raise ValueError(f"{name} out of range")
    return x


def fit_estimators(train):
    x=inputs(train)
    def linear(est):
        return make_pipeline(SimpleImputer(strategy="median",add_indicator=True), StandardScaler(), est)
    cfg=dict(max_iter=95,learning_rate=.045,max_leaf_nodes=7,min_samples_leaf=32,
             l2_regularization=8,early_stopping=False,random_state=42)
    wind=[linear(Ridge(alpha=90)),HistGradientBoostingRegressor(**cfg)]
    risk=[linear(LogisticRegression(C=.12,max_iter=1000)),HistGradientBoostingClassifier(**cfg)]
    for m in wind:
        m.fit(x,train.future_wind-x.current_wind)
    for m in risk:
        m.fit(x,train.ri)
    return {"wind":wind,"risk":risk}


def candidates(models, rows):
    x=inputs(rows)
    w=np.column_stack([m.predict(x)+x.current_wind for m in models["wind"]]).clip(0,200)
    p=np.column_stack([m.predict_proba(x)[:,1] for m in models["risk"]])
    return np.column_stack((w,w.mean(axis=1))),np.column_stack((p,p.mean(axis=1)))


def predict(bundle, rows):
    w,p=candidates(bundle["models"],rows)
    score=p[:,bundle["risk_index"]]
    return pd.DataFrame({"forecast_wind_kt":w[:,bundle["wind_index"]],"ri_score":score,
                         "ri_flag":score>=bundle["threshold"]},index=rows.index)


def fit(source, out):
    out.mkdir(parents=True,exist_ok=True)
    raw=build_table(source)
    raw.to_csv(out/"track_samples.csv",index=False,float_format="%.9g")
    data=pd.read_csv(out/"track_samples.csv")
    train=data[data.split=="train"]
    dev=data[data.split=="development"]
    models=fit_estimators(train)
    w,p=candidates(models,dev)
    wi=int(np.argmin(abs(w-dev.future_wind.to_numpy()[:,None]).mean(axis=0)))
    pi=int(np.argmax([average_precision_score(dev.ri,col) for col in p.T]))
    cut=max(np.arange(5,51)/100,key=lambda t:(f1_score(dev.ri,p[:,pi]>=t),t))
    bundle=dict(models=models,wind_index=wi,risk_index=pi,threshold=float(cut))
    joblib.dump(bundle,out/"track.joblib",compress=3)
    frozen=joblib.load(out/"track.joblib")
    report=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),threshold=float(cut),
                wind=NAMES[wi],risk=NAMES[pi],train_storms=int(train.storm_id.nunique()),train_windows=len(train))
    scored=[]
    for split in ("development","test"):
        g=data[data.split==split].copy()
        g=g.join(predict(frozen,g))
        err=g.forecast_wind_kt-g.future_wind
        tn,fp,fn,tp=map(int,confusion_matrix(g.ri,g.ri_flag,labels=[0,1]).ravel())
        report[split]=dict(windows=len(g),storms=int(g.storm_id.nunique()),mae=float(err.abs().mean()),
            persistence_mae=float((g.current_wind-g.future_wind).abs().mean()),rmse=float(np.sqrt((err**2).mean())),
            ap=float(average_precision_score(g.ri,g.ri_score)),brier=float(brier_score_loss(g.ri,g.ri_score)),
            tn=tn,fp=fp,fn=fn,tp=tp)
        scored.append(g)
    pd.concat(scored).to_csv(out/"track_predictions.csv",index=False,float_format="%.9g")
    data[data.split=="test"][["storm_id","time",*FEATURES]].head(4).to_csv(out/"example.csv",index=False)
    (out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    fit_parser=sub.add_parser("fit")
    fit_parser.add_argument("source",type=Path)
    fit_parser.add_argument("output",type=Path)
    score_parser=sub.add_parser("score")
    score_parser.add_argument("model",type=Path)
    score_parser.add_argument("source",type=Path)
    score_parser.add_argument("output",type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        if args.command=="fit":
            print(json.dumps(fit(args.source,args.output),indent=2))
        else:
            if args.source.resolve()==args.output.resolve():
                raise ValueError("Use a separate output file")
            rows=pd.read_csv(args.source)
            result=predict(joblib.load(args.model),rows)
            rows[[c for c in ["storm_id","time"] if c in rows]].join(result).to_csv(args.output,index=False)
