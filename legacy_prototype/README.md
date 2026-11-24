# ChakraSense research starter

Start with **START_HERE.md** for plain-language instructions. Open **ChakraSense_Demo.html** for the saved artificial-data demonstration. No server or account is needed to view it.

## Scope

Included: a trained NumPy temporal image-statistics baseline, a CNN–GRU implementation, exact 24-hour RI labeling, whole-storm splits, model calibration, uncertainty reporting, test-set evaluation, input fingerprinting, configurable HDF5 preprocessing, a static replay dashboard, and local CAP test files. Real NOAA/NEWDELHI labels have been imported. No real satellite image model has been trained.

The CNN–GRU and HDF5 adapter still need runtime verification in a compatible environment. See `IMPLEMENTATION_STATUS.md`; do not infer successful execution from source-code presence.

## Windows commands

Open the project folder in File Explorer. Click its address bar, type `powershell`, and press Enter. Run commands one at a time. Every command below assumes you are in this folder and **Setup_Windows.cmd** completed successfully. No PowerShell activation script or policy change is required: commands directly name the environment's Python.

Check the environment and run tests:

```powershell
.\.venv\Scripts\python.exe -m chakrasense doctor
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run the neural practice model (artificial data):

```powershell
.\.venv\Scripts\python.exe -m chakrasense train data/demo/sequences/samples.csv --out artifacts/neural_demo --epochs 8 --batch-size 16 --device auto
.\.venv\Scripts\python.exe -m chakrasense evaluate data/demo/sequences/samples.csv artifacts/neural_demo
```

Use `--batch-size 4` if GPU memory is insufficient. Use `--device cpu` only when PyTorch itself loads correctly and you intentionally want CPU execution. An Application Control DLL error needs an approved installation, not a different device flag.

Reproduce the lightweight practice run in a new folder:

```powershell
.\.venv\Scripts\python.exe -m chakrasense train-baseline data/demo/sequences/samples.csv --out artifacts/baseline_reproduction
.\.venv\Scripts\python.exe -m chakrasense evaluate-baseline data/demo/sequences/samples.csv artifacts/baseline_reproduction
.\.venv\Scripts\python.exe -m chakrasense dashboard data/demo/sequences/samples.csv artifacts/baseline_reproduction --out Baseline_Reproduction.html
```

The package already includes the original demonstration sequences and trained baseline. To generate an entirely separate artificial dataset, run `python -m chakrasense demo-data --out data/demo_new`. The standalone baseline requires only `requirements-baseline.txt`; the neural/HDF5 parts require the full requirements plus PyTorch.

## Real-data workflow

The bundled `data/real/tracks.csv` and inventory came from NOAA on 2026-11-17. To download a fresh version into a NEW snapshot:

```powershell
.\.venv\Scripts\python.exe -m chakrasense download-labels --raw data/raw/ibtracs_new.csv --out data/real_new/tracks.csv --start 2013 --end 2025
.\.venv\Scripts\python.exe -m chakrasense split data/real_new/tracks.csv --out data/real_new/splits.json --validation-year 2019 --calibration-year 2020 --test-year 2021
```

Keep the original snapshot for reproducibility. Do not replace labels during a model experiment. The planned periods are based on label availability, not on model test scores. Recheck them after satellite availability filtering. Freeze any revised plan before training.

After registering with MOSDAC and downloading the first HDF5 file:

```powershell
.\.venv\Scripts\python.exe -m chakrasense inspect-h5 "C:/ChakraSenseData/INSAT/your_actual_file.h5" --out data/real/h5_inventory.json
```

Replace `your_actual_file.h5` with your actual filename. Fill the mapping and raw manifest using `DATA_GUIDE.md`. Then:

```powershell
.\.venv\Scripts\python.exe -m chakrasense prepare data/raw/raw_manifest.csv config/insat_mapping.json --out data/processed/real_frames
.\.venv\Scripts\python.exe -m chakrasense build data/processed/real_frames/frames.csv data/real/tracks.csv data/real/splits.json --out data/processed/real_sequences --mode retrospective
```

Read `data/processed/real_sequences/build_report.json`. Plot and check crops, center alignment, accepted/missing times and RI counts. One or two sample files will not create the complete histories and four splits needed for training. Start with complete small storms, then expand.

When the dataset is ready, establish the lightweight baseline:

```powershell
.\.venv\Scripts\python.exe -m chakrasense train-baseline data/processed/real_sequences/samples.csv --out artifacts/real_baseline_v1
```

Train the neural candidate:

```powershell
.\.venv\Scripts\python.exe -m chakrasense train data/processed/real_sequences/samples.csv --out artifacts/real_neural_v1 --epochs 8 --batch-size 16 --device auto
```

Eight epochs is an initial experiment setting, not a validated optimum. Select settings using validation only. Freeze the protocol, selected model and comparison methods before running the final test commands:

```powershell
.\.venv\Scripts\python.exe -m chakrasense evaluate-baseline data/processed/real_sequences/samples.csv artifacts/real_baseline_v1
.\.venv\Scripts\python.exe -m chakrasense evaluate data/processed/real_sequences/samples.csv artifacts/real_neural_v1
```

Outputs include model weights, metadata, saved test predictions and evaluation JSON. The evaluator refuses to overwrite an existing test report; this helps preserve the original result. It cannot prevent someone creating a new folder and tuning on test outcomes, so maintain an experiment log and a separate future test period for subsequent model development.

## Prediction and alert test

`predict` uses the neural model; `predict-baseline` uses the lightweight model. Both expect a previously prepared sequence, matching the training crop width, dimensions and temperature units. They are research inference commands, not a live-data service. Sequence inference alone does not validate original arrival times; those are enforced during dataset construction.

```powershell
.\.venv\Scripts\python.exe -m chakrasense predict-baseline artifacts/demo_baseline data/demo/sequences/ACTUAL_SAMPLE_FILENAME.npz
```

Copy a real sequence filename from `samples.csv` in place of the uppercase placeholder. To create a local alert-format test after personally reviewing an example:

```powershell
.\.venv\Scripts\python.exe -m chakrasense test-cap --storm ARTIFICIAL_DEMO --probability 0.5 --reviewer "Your name" --out artifacts/local_review_test.xml
```

This writes **Test / Private** CAP XML and an audit JSON beside it. It makes no network request. The reviewer text is a local record, not authenticated identity or an official sign-off. The XML generator was validated against the [official OASIS CAP 1.2 schema](https://docs.oasis-open.org/emergency/cap/v1.2/CAP-v1.2.xsd). Its dates omit fractional seconds as that schema requires.

## Project layout

| Path | Contents |
|---|---|
| chakrasense/data.py | Label creation, splits, sequence construction and data validation |
| chakrasense/satellite.py | Explicit HDF5 decoding and common geographic crops |
| chakrasense/model.py | Small two-channel CNN + causal GRU |
| chakrasense/training.py | Neural training, inference and evaluation |
| chakrasense/baseline.py | Executed lightweight statistical model |
| chakrasense/evaluation.py | Calibration, probability metrics and storm bootstrap |
| chakrasense/dashboard.py | Offline export of saved results |
| chakrasense/alerts.py | Local test-only CAP generation |
| config | Templates with no credentials |
| data/demo/sequences | Artificial practice arrays and sample manifest |
| data/real | Real labels, split plan and satellite download plan |
| artifacts/demo_baseline | Executed baseline weights, metadata and test results |
| artifacts/verification | Test log and limitations |
| tests | Data, timing, integrity, calibration, alert and optional neural/HDF5 tests |

The ZIP omits the large Python environment, original public NOAA download, and redundant intermediate artificial frame files. It includes the practice sequences needed to train and inspect the model. It contains no MOSDAC password.

## Modeling details

CNN–GRU: three convolution blocks (16/32/64 channels), global spatial average pooling, a 48-unit causal GRU, and separate current-wind and future-RI heads. Fixed input normalisation is `(K - 250) / 50`. Wind targets are divided by 100 during training. The loss combines Smooth L1 and binary cross-entropy. AdamW, gradient clipping, and validation-based checkpoint selection are implemented. No class weighting is applied initially; add imbalance treatments only as predeclared validation experiments and recalibrate afterwards.

The executed baseline computes mean, standard deviation, minimum and maximum temperature for each channel at every past time. Training-only standardisation feeds ridge wind regression and logistic RI classification. Platt scaling is fitted on calibration storms. A threshold minimises validation cost with missed RI weighted twice a false flag; this is an explicit prototype choice, not an operational policy. The wind range uses a finite-sample residual quantile. Correlated storm windows invalidate any automatic claim of nominal conformal coverage.

Evaluation reports wind MAE/RMSE, Brier score/skill against the training rate, detection, false-alarm ratio, CSI, average precision (stored under `pr_auc`), reliability bins and actual interval coverage. Undefined ratios are null, not misleading zeros. Storm bootstrap intervals resample whole storms. Small numbers of positive storms can still produce misleadingly narrow intervals. The neural evaluator also counts contiguous flags, but does not implement independent RI-event matching or warning lead time. Real-storm operational persistence, IMD forecast and environmental baselines remain future work.

Do not claim the CNN–GRU improves on the baseline before the real comparison has been completed. Do not present artificial-demo metrics as weather forecasting accuracy.
