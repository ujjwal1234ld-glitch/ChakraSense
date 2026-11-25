"""Rebuild the keyed feature table from the two original ChakraSense backup ZIPs."""
import argparse
import csv
import io
import json
import re
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATE = re.compile(r"\.(\d{4})\.(\d{2})\.(\d{2})\.(\d{2})(\d{2})\.")
BASE = ["current_wind", "wind_change_6h", "wind_change_12h", "wind_change_24h"]
for ch in ("IR", "WV"):
    BASE.extend(ch + "_" + s for s in ("mean", "std", "min", "centre_mean", "change", "p10"))
BASE.extend(["IR_cold_cloud_fraction", "current_wind_age", "wind_age_6h", "wind_age_12h", "wind_age_24h"])


def frame_time(name):
    match = DATE.search(str(name))
    if not match:
        raise ValueError(f"Unrecognized HURSAT member {name}")
    return datetime(*map(int, match.groups()), tzinfo=timezone.utc)


def sample_features(images, issue, track, winds):
    # All lookups use the latest original observation at or before the requested time.
    a = {}
    readings = []
    for h in (0, 6, 12, 24):
        cutoff = issue - pd.Timedelta(hours=h)
        index = winds.index.searchsorted(cutoff, side="right") - 1
        age = (cutoff - winds.index[index]).total_seconds() / 3600 if index >= 0 else np.nan
        reading = winds.iloc[index] if 0 <= age <= 6 else None
        speed = float(reading.wind_kt) if reading is not None else np.nan
        readings.append(reading)
        if h == 0:
            a["current_wind"] = speed
        else:
            a[f"wind_change_{h}h"] = a["current_wind"] - speed
        a[f"wind_age_{h}h" if h else "current_wind_age"] = age if reading is not None else np.nan
    if not np.isfinite(a["current_wind"]):
        raise ValueError("Current original wind is missing or stale")
    for k, ch in enumerate(("IR", "WV")):
        first, last = images[0, k], images[-1, k]
        core = images[:, k, 40:88, 40:88]
        a.update({ch + "_mean": last.mean(), ch + "_std": last.std(),
                  ch + "_min": last.min(), ch + "_centre_mean": core[-1].mean(),
                  ch + "_change": last.mean() - first.mean(), ch + "_p10": np.percentile(last, 10)})
        a[ch + "_core_change_6h"] = core[-1].mean() - core[-3].mean()
        a[ch + "_core_change_12h"] = core[-1].mean() - core[-5].mean()
        a[ch + "_core_contrast"] = core[-1].mean() - last.mean()
    a["IR_cold_cloud_fraction"] = (images[-1, 0] < 230).mean()
    a["core_cold_fraction"] = (images[-1, 0, 40:88, 40:88] < 220).mean()
    a["core_channel_difference"] = a["IR_centre_mean"] - a["WV_centre_mean"]
    now = issue - pd.Timedelta(hours=a["current_wind_age"])
    loc = track.loc[now]
    a["lat"], a["lon"], a["land_km"] = loc.NEWDELHI_LAT, loc.NEWDELHI_LON, loc.DIST2LAND
    if readings[1] is not None:
        old = track.loc[issue - pd.Timedelta(hours=6 + a["wind_age_6h"])]
        a["land_change_6h"] = loc.DIST2LAND - old.DIST2LAND
    else:
        a["land_change_6h"] = np.nan
    a["season_sin"] = np.sin(2 * np.pi * issue.dayofyear / 365.25)
    a["season_cos"] = np.cos(2 * np.pi * issue.dayofyear / 365.25)
    return a


def rebuild(archives):
    with ExitStack() as stack:
        members = {}
        for path in archives:
            z = stack.enter_context(ZipFile(path))
            for name in z.namelist():
                if name.endswith((".npz", "combined_samples_with_splits.csv", "ibtracs.NI.list.v04r01.csv")):
                    if name in members:
                        raise ValueError(f"Duplicate backup member {name}")
                    members[name] = z
        def read(suffix):
            names = [n for n in members if n.endswith("/" + suffix)]
            if len(names) != 1:
                raise ValueError(f"Expected one backup file for {suffix}, found {len(names)}")
            return io.BytesIO(members[names[0]].read(names[0]))

        table = pd.read_csv(read("combined_samples_with_splits.csv"))
        columns = ["SID", "ISO_TIME", "NEWDELHI_LAT", "NEWDELHI_LON", "NEWDELHI_WIND", "DIST2LAND", "BASIN", "TRACK_TYPE"]
        track = pd.read_csv(read("ibtracs.NI.list.v04r01.csv"), usecols=columns, skiprows=[1], low_memory=False)
        track = track[(track.TRACK_TYPE.str.strip() == "main") & (track.BASIN.str.strip() == "NI")]
        for col in ("NEWDELHI_LAT", "NEWDELHI_LON", "NEWDELHI_WIND", "DIST2LAND"):
            track[col] = pd.to_numeric(track[col], errors="coerce")
        track["time"] = pd.to_datetime(track.ISO_TIME, utc=True)
        winds = pd.read_csv(ROOT / "research/data_metadata/original_wind_records.csv")
        winds["time"] = pd.to_datetime(winds.time, utc=True)
        if table.sample_id.duplicated().any() or table.duplicated(["storm_id", "time"]).any():
            raise ValueError("Duplicate sample identity")
        if table.groupby("storm_id").split.nunique().max() != 1:
            raise ValueError("Storm appears in multiple splits")
        rows = []
        for sid, group in table.groupby("storm_id", sort=False):
            positions = track[track.SID == sid].set_index("time").sort_index()
            history = winds[winds.storm_id == sid].set_index("time").sort_index()
            if not positions.index.is_unique or not history.index.is_unique:
                raise ValueError(f"Ambiguous storm history for {sid}")
            with np.load(read(sid + ".npz"), allow_pickle=False) as storm:
                frames, names = storm["frames"], storm["members"]
                dates = [frame_time(n) for n in names]
                for r in group.itertuples():
                    issue = pd.Timestamp(r.time)
                    ids = json.loads(r.frame_indices)
                    desired = [issue - timedelta(hours=h) for h in range(24, -1, -3)]
                    if len(ids) != 9 or any(not 0 <= i < len(frames) for i in ids):
                        raise ValueError(f"Invalid frame indices for {r.sample_id}")
                    if [dates[i] for i in ids] != desired or any(not str(names[i]).startswith(sid + ".") for i in ids):
                        raise ValueError(f"Frame chronology or identity differs for {r.sample_id}")
                    x = frames[ids]
                    if x.shape != (9, 2, 128, 128) or not np.isfinite(x).all():
                        raise ValueError(f"Invalid frames for {r.sample_id}")
                    target = positions.at[issue + timedelta(hours=24), "NEWDELHI_WIND"]
                    current = positions.at[issue, "NEWDELHI_WIND"]
                    if target != r.future_wind_kt or current != r.wind_kt or int(target-current >= 30) != r.ri:
                        raise ValueError(f"Best-track target mismatch for {r.sample_id}")
                    a = sample_features(x, issue, positions, history)
                    rows.append(dict(sample_id=r.sample_id, storm_id=sid, time=issue.isoformat(),
                                     split=r.split, label_wind=current, future_wind=target, ri=r.ri,
                                     endpoints_original=bool(r.original_endpoints), satellite=r.satellite, **a))
    output = pd.DataFrame(rows).set_index("sample_id").loc[table.sample_id].reset_index()
    folds = pd.read_csv(ROOT / "research/cross_validation/training_fold_assignments.csv",
                        usecols=["sample_id","fold","storm_id","time"])
    tr = output[output.split == "train"]
    if set(tr.sample_id) != set(folds.sample_id) or folds.sample_id.duplicated().any():
        raise ValueError("Training fold identities do not match")
    match = folds.set_index("sample_id").loc[tr.sample_id]
    if not np.array_equal(match.storm_id, tr.storm_id) or not np.array_equal(pd.to_datetime(match.time,utc=True),pd.to_datetime(tr.time,utc=True)):
        raise ValueError("Training fold metadata mismatch")
    output["fold"] = output.sample_id.map(folds.set_index("sample_id").fold).astype("Int64")
    for split, filename in [("train", "training"), ("development", "development")]:
        ref = pd.read_csv(ROOT / f"research/features/{filename}_features_v3.csv")
        actual = output.loc[output.split == split, BASE]
        if not np.allclose(actual, ref[BASE], rtol=0, atol=0.001, equal_nan=True):
            raise ValueError(f"Rebuilt features differ from archived {split} table")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"forecast/data/samples.csv")
    args = parser.parse_args()
    result = rebuild(args.archives)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format="%.8g")
    print(result.groupby("split").agg(windows=("ri","size"), storms=("storm_id","nunique")))
