"""Regenerate the single-file storm replay from the two frozen prediction tables."""
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
data={}
for name,filename in [("track","track_predictions.csv"),("sat","predictions.csv")]:
    data[name]=[]
    with (HERE/"release"/filename).open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"]!="test":
                continue
            data[name].append([row["storm_id"],row["time"][:16].replace("T"," "),float(row["current_wind"]),
                float(row["forecast_wind_kt"]),float(row["future_wind"]),int(row["ri"]),float(row["ri_score"]),
                row["ri_flag"].lower()=="true"])
template=(HERE/"replay_template.html").read_text(encoding="utf-8")
(HERE/"Replay.html").write_text(template.replace("/*__DATA__*/",json.dumps(data,separators=(",",":"))),encoding="utf-8")
