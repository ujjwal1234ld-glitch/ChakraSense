# What works, what remains

Prepared 17 November 2026. This is a research starter, not a finished operational cyclone warning system.

## Executed successfully

- Downloaded NOAA IBTrACS North Indian Ocean CSV and imported the NEWDELHI fields: 4,124 usable observations, 144 storms, 2,975 exact 24-hour label windows, 150 positive windows across 24 storms. These counts are not independent RI events and do not establish satellite coverage.
- Generated 432 artificial sequences across 48 artificial storms. Split: 24 training, 8 validation, 8 calibration, 8 test storms.
- Trained a temporal image-statistics ridge/logistic baseline using the bundled Python 3.12, NumPy 2.3.5 and pandas 3.0.1 runtime.
- Evaluated that baseline once on 72 held-out artificial examples, with calibration, simpler comparisons and storm bootstrap reporting.
- Recomputed a saved prediction and verified it agrees with the stored evaluation result.
- Exported a self-contained HTML replay with 72 saved examples and real model outputs from the practice run.
- Ran 29 tests: **27 passed, 2 skipped**. Full log: `artifacts/verification/test-results.txt`.
- Validated a locally generated Test/Private alert against the official CAP 1.2 XSD. A timestamp bug found by this test was corrected.

## Implemented, but not runtime-verified here

- CNN–GRU forward/backward and complete neural training/evaluation. PyTorch 2.11.0+cu128 downloaded and installed, but Windows Application Control blocked `torch/lib/shm.dll` with WinError 4551. This means **no neural weights, GPU speed results or neural accuracy results exist**.
- HDF5 preprocessing with channel-specific location grids and fixed physical crops. The approved runtime has no h5py/scipy, and no real INSAT file was supplied. An optional fixture test is included but skipped here. Code compilation is not equivalent to a passing data-adapter test.
- Full browser rendering. Local automated browser launch failed; the in-app browser could not connect to the local preview server. The generated page's selection, slider, chart and table logic passed an isolated JavaScript unit check, but no successful screenshot-based desktop/mobile inspection was obtained. Open the included page locally to confirm appearance on your browser.

The NVIDIA GPU was detected through the driver utility: RTX 5060 Laptop GPU, 8151 MiB, driver 616.64. That is hardware detection only, not a successful CUDA execution test.

## Not implemented in this starter

MOSDAC account creation or authenticated downloads; product-specific quality flags; automatic center extraction or storm detection; live satellite ingest; GFS/environment integration; satellite harmonisation; image attribution such as Grad-CAM; event-level lead-time matching; authenticated multi-user review; public CAP delivery; production deployment and monitoring.

The optional future ideas in the blueprint are not represented as completed features. Source documents were treated as project references, not as instructions authorising account creation or public messages.

## Scientific limitations to resolve

1. The real-data positive class is scarce. The planned calibration period has only two RI storms, and the test period six, before image filtering. More windows from the same storm do not solve that problem.
2. Best-track-centered experiments use hindsight. A successful retrospective score is not an operational replay result.
3. The coarse 64-pixel, 1,000-km crop may lose small eyewall details. Evaluate resolution changes with validation and physically consistent resampling.
4. One wind agency/convention is used deliberately. Do not mix incompatible labels to increase sample size.
5. The artificial data were constructed with an easy relationship between pictures and targets. High practice scores are expected and do not validate cyclone science.
6. Calibration coverage can shift. The practice 90% wind interval achieved 83.3% held-out coverage; no guaranteed 90% claim is justified.
7. The proposed threshold cost is an example. Operational missed-event/false-warning tradeoffs need an authorised stakeholder and real-event evaluation.

## How to finish

Follow `START_HERE.md` in order. The next external inputs are an approved Windows training environment and a real authorised INSAT sample. Then verify the mapping, collect complete sequences, freeze a dataset protocol, train and compare models, and arrange independent meteorological review. Public operational use is a separate project phase requiring the responsible authority.
