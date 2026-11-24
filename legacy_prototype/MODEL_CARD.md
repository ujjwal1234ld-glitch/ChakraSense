# Model card — included practice baseline

**Status: artificial-data software demonstration only.** Intended for learning the pipeline and testing implementation. Not for public weather decisions, evacuation, shipping, insurance or claims of cyclone forecast skill.

Model: temporal image-statistics ridge regression for current wind, logistic regression for RI, followed by held-out probability calibration. This is not the included but unexecuted CNN–GRU implementation.

Training set: 24 artificial storms, 216 sequences. Validation, calibration and test: eight artificial storms and 72 sequences each. Nine 64×64 two-channel arrays span 24 past hours per sequence. The image generator deliberately connects cloud-like patterns to the artificial wind targets, so the task is much easier than real forecasting.

Targets: current three-minute-convention wind in knots and a binary increase of at least 30 knots over the next 24 hours. The artificial data use the NEWDELHI schema to exercise the pipeline; they are not observations from NEWDELHI or any satellite.

Held-out artificial results: wind MAE 0.448 kt; Brier score 0.00245; 36 detected positive windows, zero missed positive windows, zero false flags. These are overlapping windows, not independent cyclone events. A nominal 90% wind range covered 83.3% of test examples. Full scores, baselines and bootstrap intervals are in `artifacts/demo_baseline/evaluation.json`.

Calibration: Platt scaling on calibration storms; threshold chosen from validation outcomes with a 2:1 missed/false cost. Features are standardised using training data only. Checksums pin the weight file and input dataset.

No evaluation by sex, nationality or similar personal attributes is applicable. Relevant missing evaluations include basin subregion, intensity range, storm structure, land proximity, season, satellite/product changes, missing scans, late arrivals and operational center errors. All require real data.

Training code and deterministic artificial-data generator are included. Exact package versions used for the executed practice fit are recorded in `artifacts/verification/status.json`. Raw practice intermediate frames are omitted from the ZIP; the complete model-ready practice sequences remain included.
