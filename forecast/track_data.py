"""Create a separate original-observation benchmark from the NOAA North Indian Ocean CSV."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
TRACK = ["current_wind","change_6","change_12","change_24","current_pressure","pressure_change_12",
         "lat","lon","land_km","land_change_6","latitude_change_6","longitude_change_6","season_sin","season_cos",
         "recent_peak_loss","hours_since_first","coastal_wind","approach_wind","ocean_trend","strong_wind"]


def make_table(source):
    cols = ["SID","SEASON","BASIN","TRACK_TYPE","ISO_TIME","IFLAG","NEWDELHI_WIND","NEWDELHI_PRES",
            "NEWDELHI_LAT","NEWDELHI_LON","DIST2LAND"]
    raw = pd.read_csv(source,usecols=cols,skiprows=[1],low_memory=False)
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
        raise ValueError("Ambiguous storm observation.")
    result=[]
    for sid,g in raw.groupby("storm_id"):
        g=g.set_index("time").copy()
        def lag(column,h):
            return g[column].reindex(g.index-pd.Timedelta(hours=h)).to_numpy()
        for h in (6,12,24):
            g[f"change_{h}"]=g.current_wind-lag("current_wind",h)
        g["pressure_change_12"]=g.current_pressure-lag("current_pressure",12)
        for name,field in [("land_change_6","land_km"),("latitude_change_6","lat"),("longitude_change_6","lon")]:
            g[name]=g[field]-lag(field,6)
        g["hours_since_first"]=(g.index-g.index.min()).total_seconds()/3600
        g["recent_peak_loss"]=g.current_wind.rolling("24h",closed="both").max()-g.current_wind
        g["season_sin"]=np.sin(2*np.pi*g.index.day_of_year/365.25)
        g["season_cos"]=np.cos(2*np.pi*g.index.day_of_year/365.25)
        g["coastal_wind"]=g.current_wind*np.exp(-g.land_km/120)
        g["approach_wind"]=g.current_wind*np.exp(-(g.land_km+2*g.land_change_6).clip(lower=0)/120)
        g["ocean_trend"]=g.change_12*(1-np.exp(-g.land_km/120))
        g["strong_wind"]=(g.current_wind-55).clip(lower=0)
        g["future_wind"]=lag("current_wind",-24)
        g=g.dropna(subset=["future_wind","land_km","lat","lon"])
        g["ri"]=(g.future_wind-g.current_wind>=30).astype(int)
        # Assign by storm season so a New Year crossing never divides one storm.
        g["split"]=np.select([g.SEASON<=2015,g.SEASON<=2019],["train","development"],default="test")
        result.append(g.reset_index()[["storm_id","time","SEASON","split","future_wind","ri",*TRACK]])
    return pd.concat(result,ignore_index=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ibtracs",type=Path)
    parser.add_argument("--output",type=Path,default=HERE/"data/track_samples.csv")
    args=parser.parse_args()
    make_table(args.ibtracs).to_csv(args.output,index=False,float_format="%.9g")
