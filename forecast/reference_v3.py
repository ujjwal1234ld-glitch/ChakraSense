"""Re-score the frozen V3 classifier with reconstructed inputs and the original temporal rule."""
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, average_precision_score, brier_score_loss
from prepare import ROOT

def evaluate():
    path=ROOT/"models/v3/ChakraSense_V3.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest()!="ed53072c12e946b4a604c469d35a26e245c13542071a65c6c9f58db0d4ff0baf":
        raise ValueError("Archived classifier hash mismatch.")
    try:
        model=joblib.load(path)
    except ModuleNotFoundError as exc:
        if exc.name != "_loss":
            raise
        # The original Colab checkpoint names this compiled sklearn module without its package.
        from sklearn._loss import _loss
        with patch.dict(sys.modules, {"_loss": _loss}):
            model=joblib.load(path)
    rows=pd.read_csv(ROOT/"forecast/data/samples.csv").query("split == 'test'").copy()
    # Archived V3 used sample wind even where the latest original observation differed.
    inputs=rows[model["features"]].copy()
    inputs["current_wind"]=rows.label_wind
    for name,estimator in model["models"].items():
        rows[name]=estimator.predict_proba(inputs)[:,1]
    rows=rows.sort_values(["storm_id","time"])
    cut=model["threshold"]
    regular=rows.Regularized>=cut
    slow_only=(rows.Slow_Learning>=cut)&~regular
    times=pd.to_datetime(rows.loc[slow_only,"time"],utc=True)
    gaps=times.groupby(rows.loc[slow_only,"storm_id"]).diff().dt.total_seconds()/3600
    rows["flag"]=regular | (gaps.gt(0)&gaps.le(model["confirmation_hours"])).reindex(rows.index,fill_value=False)
    out=ROOT/"forecast/release"
    rows[["sample_id","storm_id","time","ri","Regularized","Slow_Learning","flag"]].to_csv(out/"v3_test_predictions.csv",index=False,float_format="%.9g")
    matrix=confusion_matrix(rows.ri,rows.flag,labels=[0,1]).tolist()
    report=dict(confusion_matrix=matrix,score_for_ap_brier="Regularized",ap=float(average_precision_score(rows.ri,rows.Regularized)),brier=float(brier_score_loss(rows.ri,rows.Regularized)))
    (out/"v3_test_results.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report)

if __name__=="__main__":
    evaluate()
