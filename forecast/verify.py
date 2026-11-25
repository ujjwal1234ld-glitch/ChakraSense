"""Recompute release checks with only the Python standard library."""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from predict import predict

HERE = Path(__file__).resolve().parent


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify(mode):
    prefix = "track_" if mode == "track" else ""
    release = HERE / "release"
    model = json.loads((release / f"{mode}.json").read_text())
    report = json.loads((release / f"{prefix}results.json").read_text())
    inputs = read_rows(HERE / "data" / f"{prefix}samples.csv")
    reference = read_rows(release / f"{prefix}predictions.csv")
    key = lambda row: (row["storm_id"], row["time"])
    held_out = {key(row): row for row in reference if row["split"] != "train"}
    if len(held_out) != sum(row["split"] != "train" for row in reference):
        raise ValueError("Duplicate reference forecast")
    storms = defaultdict(set)
    outcomes = Counter()
    errors, baseline_errors = [], []
    checked = set()
    for row in inputs:
        storms[row["storm_id"]].add(row["split"])
        if row["split"] == "train":
            continue
        identifier = key(row)
        if identifier in checked or identifier not in held_out:
            raise ValueError(f"Duplicate or missing forecast {identifier}")
        checked.add(identifier)
        saved = held_out[identifier]
        wind, risk, flag = predict(model, row)
        if (row["split"] != saved["split"] or
            not math.isclose(wind, float(saved["forecast_wind_kt"]), abs_tol=1e-6, rel_tol=0) or
            not math.isclose(risk, float(saved["ri_score"]), abs_tol=1e-6, rel_tol=0) or
            flag != (saved["ri_flag"].lower() == "true")):
            raise ValueError(f"Prediction mismatch at {identifier}")
        future = float(row["future_wind"])
        issue = float(row["current_wind"])
        truth = future - float(row.get("label_wind", issue)) >= 30
        if truth != bool(int(row["ri"])) or truth != bool(int(saved["ri"])):
            raise ValueError(f"RI label mismatch at {identifier}")
        if row["split"] == "test":
            outcomes[(truth, flag)] += 1
            errors.append(abs(wind - future))
            baseline_errors.append(abs(issue - future))
    if checked != set(held_out) or any(len(splits) != 1 for splits in storms.values()):
        raise ValueError("Missing evaluation rows or a storm appears in multiple splits")
    result = report["test"]
    matrix = [outcomes[(bool(y), bool(p))] for y in (0, 1) for p in (0, 1)]
    if matrix != [result[k] for k in ("tn", "fp", "fn", "tp")]:
        raise ValueError("RI counts differ from the published test report")
    for values, name in ((errors, "mae"), (baseline_errors, "persistence_mae")):
        if not math.isclose(sum(values) / len(values), result[name], abs_tol=1e-6):
            raise ValueError(f"Incorrect test {name}")
    print(f"{mode}: {len(checked)} forecasts matched, {len(errors)} test windows, MAE {result['mae']:.2f} kt")


if __name__ == "__main__":
    verify("track")
    verify("satellite")
