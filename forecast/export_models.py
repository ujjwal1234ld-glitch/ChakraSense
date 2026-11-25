"""Export the selected estimators as portable JSON and check all saved holdout predictions."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from model import FEATURES, OPTIONAL
from track_data import TRACK, HERE
from predict import predict


def serialize(est):
    if hasattr(est,"steps"):
        imputer,scale,linear=(s for _,s in est.steps)
        return dict(type="linear",median=imputer.statistics_.tolist(),missing_indices=imputer.indicator_.features_.tolist(),
                    mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),coef=linear.coef_.ravel().tolist(),
                    offset=float(np.asarray(linear.intercept_).ravel()[0]))
    trees=[]
    for iteration in est._predictors:
        nodes=[]
        for n in iteration[0].nodes:
            if n["is_leaf"]:
                nodes.append([float(n["value"])])
            else:
                bound=float(n["num_threshold"])
                # A positive infinite split isolates the missing-value branch.
                if np.isneginf(bound) or np.isnan(bound):
                    raise ValueError("Unexpected numerical tree threshold")
                nodes.append([int(n["feature_idx"]),None if np.isposinf(bound) else bound,int(n["left"]),int(n["right"]),int(n["missing_go_to_left"])])
        trees.append(nodes)
    return dict(type="boost",offset=float(est._baseline_prediction.ravel()[0]),trees=trees)


def export(mode):
    satellite=mode=="satellite"
    out=HERE/"release"
    bundle=joblib.load(out/("forecast.joblib" if satellite else "track.joblib"))
    fields=FEATURES if satellite else TRACK
    optional=OPTIONAL if satellite else set(TRACK)-{"current_wind","lat","lon","land_km","season_sin","season_cos","hours_since_first","recent_peak_loss","strong_wind","coastal_wind"}
    bounds={"current_wind":[0,200],"lat":[-90,90],"lon":[-180,180],"land_km":[0,20000],"season_sin":[-1,1],
            "season_cos":[-1,1],"current_pressure":[850,1050],"core_cold_fraction":[0,1],"IR_cold_cloud_fraction":[0,1],
            "IR_std":[0,150],"IR_p10":[120,360],"IR_centre_mean":[120,360],"WV_centre_mean":[120,360]}
    bounds.update({k:[0,6] for k in fields if "wind_age" in k})
    spec=dict(mode=mode,input_fields=fields,ordered_fields=list(bundle["fit"]["wind"][0].feature_names_in_),
              optional=sorted(optional),bounds={k:v for k,v in bounds.items() if k in fields},threshold=bundle["threshold"])
    for name in ("wind","risk"):
        idx=bundle[name+"_index"]
        selected=bundle["fit"][name] if idx==2 else [bundle["fit"][name][idx]]
        spec[name]=[serialize(m) for m in selected]
    dest=out/(mode+".json")
    dest.write_text(json.dumps(spec,separators=(",",":"),allow_nan=False))
    spec=json.loads(dest.read_text())
    data=pd.read_csv(HERE/"data"/("samples.csv" if satellite else "track_samples.csv"))
    reference=pd.read_csv(out/("predictions.csv" if satellite else "track_predictions.csv"))
    keys=["storm_id","time"]
    rows=data[data.split!="train"].merge(reference[keys+["forecast_wind_kt","ri_score","ri_flag"]],on=keys,validate="one_to_one")
    if len(rows)!=sum(data.split!="train"):
        raise ValueError("Incomplete inference comparison.")
    output=np.asarray([predict(spec,r) for r in rows.to_dict("records")])
    if not np.allclose(output,rows[["forecast_wind_kt","ri_score","ri_flag"]].to_numpy(dtype=float),atol=1e-6,rtol=0):
        raise ValueError(f"Portable prediction mismatch in {mode}")
    print(f"{mode}: verified {len(rows)} predictions")
    sample=data[data.split=="test"].iloc[:4]
    sample[keys+fields].to_csv(HERE/"data"/(mode+"_example.csv"),index=False)


if __name__=="__main__":
    for mode in ["satellite","track"]:
        export(mode)
