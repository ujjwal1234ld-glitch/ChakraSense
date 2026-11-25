"""CPU-only model fitting and inference for retrospective 24-hour forecasts."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["current_wind","wind_change_6h","wind_change_12h","wind_change_24h","current_wind_age",
            "wind_age_6h","wind_age_12h","wind_age_24h","IR_centre_mean","IR_std","IR_p10","IR_change",
            "WV_centre_mean","WV_change","IR_cold_cloud_fraction","IR_core_change_6h","IR_core_change_12h",
            "WV_core_change_6h","IR_core_contrast","core_cold_fraction","core_channel_difference",
            "lat","lon","land_km","land_change_6h","season_sin","season_cos"]
OPTIONAL = {f"wind_{s}_{h}h" for h in (6,12,24) for s in ("age","change")} | {"land_change_6h"}


def predictors(rows):
    if rows.empty or rows.columns.duplicated().any():
        raise ValueError("Provide a non-empty table with unique field names.")
    missing = sorted(set(FEATURES) - set(rows))
    if missing:
        raise ValueError("Missing forecast fields: " + ", ".join(missing))
    x = rows[FEATURES].apply(pd.to_numeric, errors="raise").astype(float).copy()
    if np.isinf(x).any().any() or x[list(set(FEATURES) - OPTIONAL)].isna().any().any():
        raise ValueError("Required fields must be finite; only lagged observations may be absent.")
    bounds = {"current_wind": (0, 200), "lat": (-90,90), "lon": (-180,180), "land_km": (0,20000),
              "IR_cold_cloud_fraction": (0,1), "core_cold_fraction": (0,1), "season_sin": (-1,1),
              "season_cos": (-1,1), "IR_std":(0,150), "IR_centre_mean":(120,360),
              "WV_centre_mean":(120,360), "IR_p10":(120,360)}
    for column in ["current_wind_age","wind_age_6h","wind_age_12h","wind_age_24h"]:
        bounds[column] = (0,6)
    for col, (lo, hi) in bounds.items():
        if ((x[col] < lo) | (x[col] > hi)).any():
            raise ValueError(f"{col} must be between {lo} and {hi}.")
    for h in (6,12,24):
        if (x[f"wind_age_{h}h"].isna() != x[f"wind_change_{h}h"].isna()).any():
            raise ValueError(f"Indicate both missing age and missing change at {h} hours.")
    # A recent approach toward the coast may cap further strengthening.
    x["log_land"] = np.log1p(x.land_km)
    x["coast"] = np.exp(-x.land_km / 120)
    x["wind_near_coast"] = x.current_wind * x.coast
    x["approach_km"] = (x.land_km + 2*x.land_change_6h).clip(lower=0, upper=1000)
    x["approach_wind"] = x.current_wind * np.exp(-x.approach_km / 120)
    x["ocean_trend"] = x.wind_change_12h * (1 - x.coast)
    x["strong_wind"] = (x.current_wind - 55).clip(lower=0)
    return x


def fit_pair(x, rows):
    delta = rows.future_wind.to_numpy() - x.current_wind.to_numpy()
    def linear(est):
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), est)
    opts = dict(max_iter=95, learning_rate=.045, max_leaf_nodes=7, min_samples_leaf=32,
                l2_regularization=8, early_stopping=False, random_state=42)
    wind = [linear(Ridge(alpha=90)), HistGradientBoostingRegressor(**opts)]
    risk = [linear(LogisticRegression(C=.12, max_iter=1000)), HistGradientBoostingClassifier(**opts)]
    for estimator in wind:
        estimator.fit(x, delta)
    for estimator in risk:
        estimator.fit(x, rows.ri)
    return {"wind": wind, "risk": risk}


def alternatives(pair, x):
    w = np.column_stack([m.predict(x) + x.current_wind for m in pair["wind"]]).clip(0,200)
    p = np.column_stack([m.predict_proba(x)[:,1] for m in pair["risk"]])
    return (np.column_stack((w, w.mean(axis=1))), np.column_stack((p, p.mean(axis=1))))


def forecast(bundle, rows):
    wind, risk = alternatives(bundle["fit"], predictors(rows))
    p = risk[:, bundle["risk_index"]]
    return pd.DataFrame({"forecast_wind_kt": wind[:,bundle["wind_index"]], "ri_score":p,
                         "ri_flag":p >= bundle["threshold"]}, index=rows.index)
