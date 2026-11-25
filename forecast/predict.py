"""Score new, preprocessed research windows using Python's standard library."""
import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent


def vector(row, model):
    values={}
    for key in model["input_fields"]:
        if key not in row:
            raise ValueError(f"Missing input column {key}")
        raw=row[key]
        if raw is None or str(raw).strip().lower() in ("","nan","na","null"):
            if key not in model["optional"]:
                raise ValueError(f"Missing required input {key}")
            values[key]=math.nan
            continue
        number=float(raw)
        if not math.isfinite(number):
            raise ValueError(f"Invalid input {key}")
        lo,hi=model["bounds"].get(key,[-1e12,1e12])
        if not lo<=number<=hi:
            raise ValueError(f"{key} outside [{lo}, {hi}]")
        values[key]=number
    if model["mode"]=="satellite":
        for h in (6,12,24):
            if math.isnan(values[f"wind_change_{h}h"]) != math.isnan(values[f"wind_age_{h}h"]):
                raise ValueError("Missing wind change and age must be paired")
        land=values["land_km"]
        current=values["current_wind"]
        coast=math.exp(-land/120)
        approach=land+2*values["land_change_6h"]
        if math.isfinite(approach):
            approach=min(1000,max(0,approach))
        values.update(log_land=math.log1p(land),coast=coast,wind_near_coast=current*coast,
                      approach_km=approach,approach_wind=current*math.exp(-approach/120),
                      ocean_trend=values["wind_change_12h"]*(1-coast),strong_wind=max(0,current-55))
    return [values[k] for k in model["ordered_fields"]]


def component(comp,x):
    if comp["type"]=="linear":
        vals=[m if math.isnan(v) else v for v,m in zip(x,comp["median"])]
        vals.extend(float(math.isnan(x[j])) for j in comp["missing_indices"])
        return comp["offset"]+sum(w*(v-m)/s for w,v,m,s in zip(comp["coef"],vals,comp["mean"],comp["scale"]))
    total=comp["offset"]
    for nodes in comp["trees"]:
        j=0
        while len(nodes[j])>1:
            feature,bound,left,right,missing_left=nodes[j]
            v=x[feature]
            j=left if (missing_left if math.isnan(v) else bound is None or v<=bound) else right
        total+=nodes[j][0]
    return total


def predict(model,row):
    x=vector(row,model)
    current=float(row["current_wind"])
    wind=sum(min(200,max(0,current+component(c,x))) for c in model["wind"])/len(model["wind"])
    score=sum(1/(1+math.exp(-max(-700,min(700,component(c,x))))) for c in model["risk"])/len(model["risk"])
    return wind,score,score>=model["threshold"]


def score_file(source,destination,mode):
    if source.resolve()==destination.resolve():
        raise ValueError("Choose an output file distinct from the input file.")
    model=json.loads((HERE/"release"/(mode+".json")).read_text())
    with source.open(encoding="utf-8-sig",newline="") as stream, tempfile.NamedTemporaryFile(
        mode="w",newline="",encoding="utf-8",dir=destination.parent,delete=False) as out:
        temp=Path(out.name)
        try:
            rows=csv.DictReader(stream)
            if not rows.fieldnames or len(rows.fieldnames)!=len(set(rows.fieldnames)):
                raise ValueError("CSV needs unique column headers.")
            ids=[k for k in ("sample_id","storm_id","time") if k in rows.fieldnames]
            writer=csv.writer(out)
            writer.writerow(ids+["forecast_wind_kt","ri_score","ri_flag"])
            count=0
            for count,row in enumerate(rows,1):
                try:
                    w,p,flag=predict(model,row)
                except ValueError as exc:
                    raise ValueError(f"CSV row {count+1}: {exc}") from exc
                writer.writerow([row[k] for k in ids]+[round(w,6),round(p,8),int(flag)])
            if not count:
                raise ValueError("No input windows.")
        except BaseException:
            out.close()
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp,destination)
    return count


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["track","satellite"])
    parser.add_argument("input",type=Path)
    parser.add_argument("output",type=Path)
    a=parser.parse_args()
    print(f"Scored {score_file(a.input,a.output,a.mode)} retrospective windows.")
