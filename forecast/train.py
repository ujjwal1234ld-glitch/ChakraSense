"""Select models using training storms only, then freeze and evaluate on later years."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits
from model import alternatives, fit_pair, forecast, predictors

HERE = Path(__file__).resolve().parent
NAMES = ["linear","boosted","equal blend"]


def metrics(rows):
    y, p, flag = rows.ri, rows.ri_score, rows.ri_flag
    tn, fp, fn, tp = map(int, confusion_matrix(y, flag, labels=[0,1]).ravel())
    error = rows.forecast_wind_kt - rows.future_wind
    def ratio(a,b):
        return float(a / b) if b else None
    return dict(windows=len(rows), storms=int(rows.storm_id.nunique()), positives=int(y.sum()),
                mae=float(error.abs().mean()), rmse=float(np.sqrt((error**2).mean())),
                persistence_mae=float((rows.current_wind - rows.future_wind).abs().mean()),
                ap=float(average_precision_score(y,p)) if y.any() else None, brier=float(brier_score_loss(y,p)),
                tn=tn, fp=fp, fn=fn, tp=tp, precision=ratio(tp,tp+fp), recall=ratio(tp,tp+fn),
                f1=ratio(2*tp,2*tp+fp+fn))


def choose(train):
    x = predictors(train)
    winds = np.full((len(train),3), np.nan)
    risks = winds.copy()
    if train.groupby("storm_id").fold.nunique().max() != 1 or set(train.fold) != {1,2,3,4}:
        raise ValueError("Invalid storm folds.")
    for fold in sorted(train.fold.unique()):
        val = (train.fold == fold).to_numpy()
        fit = fit_pair(x[~val], train[~val])
        winds[val], risks[val] = alternatives(fit, x[val])
    maes = np.mean(np.abs(winds - train.future_wind.to_numpy()[:,None]), axis=0)
    aps = [average_precision_score(train.ri, p) for p in risks.T]
    wi, pi = int(np.argmin(maes)), int(np.argmax(aps))
    scores = risks[:,pi]
    thresholds = np.arange(5, 51, 1)/100
    # Fixed grid; break equal-F1 ties in favor of fewer false flags.
    cut = max(thresholds, key=lambda t: (f1_score(train.ri,scores>=t), t))
    oof = train.assign(forecast_wind_kt=winds[:,wi],ri_score=scores,ri_flag=scores>=cut)
    audit = {n:dict(oof_mae=float(maes[i]),oof_ap=float(aps[i])) for i,n in enumerate(NAMES)}
    return dict(fit=fit_pair(x,train), wind_index=wi, risk_index=pi, threshold=float(cut)), oof, audit


def run(data, out):
    rows = pd.read_csv(data)
    if rows.sample_id.isna().any() or rows.sample_id.duplicated().any() or rows.duplicated(["storm_id","time"]).any():
        raise ValueError("Duplicate or missing sample identity.")
    if rows.groupby("storm_id").split.nunique().max()!=1:
        raise ValueError("Storm leakage across splits.")
    years = pd.to_datetime(rows.time, utc=True).dt.year
    if not ((rows.split.eq("train") & years.between(1998,2013)) | (rows.split.eq("development") & years.eq(2014)) | (rows.split.eq("test") & years.eq(2015))).all():
        raise ValueError("Unexpected split or year.")
    if not np.array_equal(rows.ri.to_numpy(), (rows.future_wind - rows.label_wind >= 30).astype(int)):
        raise ValueError("RI labels disagree with exact 24-hour best-track targets.")
    out.mkdir(parents=True,exist_ok=True)
    train = rows[rows.split=="train"].reset_index(drop=True)
    bundle,oof,audit = choose(train)
    joblib.dump(bundle, out/"forecast.joblib",compress=3)
    frozen = joblib.load(out/"forecast.joblib")
    report = dict(wind=NAMES[bundle["wind_index"]], risk=NAMES[bundle["risk_index"]],
                  threshold=bundle["threshold"],selection=audit,train_oof=metrics(oof))
    predictions = [oof]
    for split in ("development","test"):
        part=rows[rows.split==split].copy()
        scored = part.join(forecast(frozen,part))
        report[split]=metrics(scored)
        report[split+"_original_endpoints"]=metrics(scored[scored.endpoints_original])
        predictions.append(scored)
    columns=["sample_id","storm_id","time","split","satellite","endpoints_original","current_wind","label_wind","future_wind","ri","forecast_wind_kt","ri_score","ri_flag"]
    pd.concat(predictions)[columns].to_csv(out/"predictions.csv",index=False,float_format="%.9g")
    (out/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=HERE/"data/samples.csv")
    parser.add_argument("--out",type=Path,default=HERE/"release")
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        print(json.dumps(run(args.data,args.out),indent=2))
