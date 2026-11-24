
# ChakraSense — Dataset Card

## 1. Dataset Overview

The ChakraSense dataset was developed for
experimental tropical cyclone intensity forecasting
and rapid intensification (RI) prediction.

It combines historical satellite observations
with tropical cyclone wind measurements.

The project investigates whether satellite
imagery, historical intensity trends and machine
learning can improve cyclone forecasting.

## 2. Data Sources

### HURSAT

Hurricane Satellite Data (HURSAT) provides
historical tropical cyclone satellite imagery.

The project uses:

- Infrared (IR) observations
- Water vapour (WV) observations
- Historical satellite image sequences
- Satellite calibration metadata

### IBTrACS

International Best Track Archive for Climate
Stewardship (IBTrACS) provides historical
tropical cyclone track and intensity records.

The project uses:

- Storm identification
- Observation timestamps
- Historical wind speeds
- Future wind speed targets
- IMD intensity observations
- Measurement provenance information

Historical best-track observations may differ
from information available in real time.

## 3. Satellite Systems

The following satellite systems were inspected
during the calibration audit.

| Satellite | Infrared Wavelength |
|---|---|
| GMS-5 | 10.2–11.4 micrometres |
| MET-5 | 10.6–11.8 micrometres |
| MET-7 | 10.6–12.5 micrometres |

Satellite measurements are represented using
brightness temperature in Kelvin.

## 4. Satellite Calibration

The inspected IRWIN variables contain:

- Scale factor: 0.01
- Add offset: 200.0
- Fill value: -20100
- Temperature unit: Kelvin

The packed brightness temperature values
can be decoded using:

Tb_corrected = IRWIN * 0.01 + 200.0

Missing values must be masked before decoding.

Automatic NetCDF scale and offset decoding
must not be applied twice.

The metadata also provides coefficients for
returning normalized brightness temperatures
to the ISCCP absolute calibration.

| Satellite | Cal Scale | Cal Offset |
|---|---:|---:|
| GMS-5 | 0.990055 | 3.006 |
| MET-5 | 0.990230 | 2.411 |
| MET-7 | 0.926565 | 19.841 |

The metadata conversion is:

ISCCP_Tb = (Tb_corrected - Cal_Offset) / Cal_Scale

The existence of these coefficients does not
establish that every conversion was applied
during dataset preparation.

## 5. Dataset Composition

| Parameter | Value |
|---|---:|
| Total model-ready sequences | 1505 |
| Total independent storms | 88 |
| Training sequences | 1265 |
| Development sequences | 90 |
| Independent test sequences | 150 |
| Training storms | 77 |
| Development storms | 4 |
| Independent test storms | 7 |

The dataset splits contain no detected
storm overlap.

The audited training observations span
1998-05-17 to 2013-12-11.

## 6. Image Sequence Structure

Each inspected V2 neural network sample
contains an image tensor with shape:

(9, 2, 128, 128)

Where:

- 9 represents temporal image frames
- 2 represents satellite channels
- 128 represents image height
- 128 represents image width

The channels represent infrared and
water vapour observations.

A successfully inspected batch contained:

(8, 9, 2, 128, 128)

Finite-value checks passed for the inspected
sample and batch.

## 7. Prediction Targets

The research investigates two prediction tasks.

### Wind Speed Forecasting

Predict future tropical cyclone wind speed
in knots.

The project investigates a 24-hour forecast
target.

### Rapid Intensification Prediction

Predict the possibility of a wind speed
increase of at least 30 knots over 24 hours.

RI is represented as a binary target:

- 0: Non-RI observation window
- 1: RI observation window

The exact target-generation and time-matching
implementation must be verified for
reproducibility.

## 8. Class Distribution

### Training Dataset

| Category | Count |
|---|---:|
| Total sequences | 1265 |
| RI-positive sequences | 89 |
| Non-RI sequences | 1176 |
| RI-positive storms | 16 |
| Non-RI storms | 61 |

RI-positive sequences represent approximately
7.04% of the training dataset.

### Development Dataset

| Category | Count |
|---|---:|
| Total sequences | 90 |
| RI-positive sequences | 12 |
| Non-RI sequences | 78 |

The dataset is highly imbalanced.

Independent test class counts and metrics
require verification against the saved data.

## 9. Historical Wind Reconstruction

The V3 feature preparation stage reconstructed
historical wind measurements from IBTrACS.

| Parameter | Count |
|---|---:|
| Historical wind records | 3774 |
| Storms represented | 152 |
| Original measurements (O) | 3092 |
| P-flagged measurements | 679 |
| I-flagged measurements | 3 |

The original intensity record subset contains
3092 observations across 152 storms.

Original observations were selected for
historical wind reconstruction to reduce
dependence on interpolated measurements.

The historical wind archive and the final
model-ready dataset represent different
stages of data preparation.

## 10. V3 Feature Engineering

ChakraSense V3 uses 21 engineered features.

### Historical Wind Features

- current_wind
- wind_change_6h
- wind_change_12h
- wind_change_24h

### Infrared Features

- IR_mean
- IR_std
- IR_min
- IR_centre_mean
- IR_change
- IR_p10

### Water Vapour Features

- WV_mean
- WV_std
- WV_min
- WV_centre_mean
- WV_change
- WV_p10

### Cloud Structure Features

- IR_cold_cloud_fraction

### Observation Age Features

- current_wind_age
- wind_age_6h
- wind_age_12h
- wind_age_24h

Observation age features describe the age
of the historical measurements used.

## 11. Feature Quality Improvements

| Feature | Previous Missing | V3 Missing |
|---|---:|---:|
| 6-hour wind change | 175 | 85 |
| 12-hour wind change | 322 | 144 |
| 24-hour wind change | 515 | 267 |

Historical reconstruction reduced missing
wind-history features.

Some missing values remain due to limitations
in historical observation availability.

### V3 Feature Files

| Dataset | Shape |
|---|---|
| Training | 1265 x 21 |
| Development | 90 x 21 |

The inspected training feature file contains
992 missing feature values.

The inspected development feature file
contains no missing feature values.

Missing values should not be replaced with
fabricated meteorological measurements.

## 12. Wind Consistency Audit

The V3 training wind consistency audit
produced the following results.

| Parameter | Value |
|---|---:|
| Audited samples | 1265 |
| Exact wind matches | 1236 |
| Maximum difference | 5 knots |
| Differences above 5 knots | 0 |
| Differences above 10 knots | 0 |
| Values below 15 knots | 1 |

One observation contained a wind speed
of 3 knots.

This observation requires further provenance
and data-quality verification.

## 13. Chronology Integrity

Training predictions were matched against
historical observation timestamps.

| Parameter | Value |
|---|---:|
| Training predictions | 1265 |
| Independent storms | 77 |
| Missing timestamps | 0 |
| Duplicate storm timestamps | 0 |
| Negative time gaps | 0 |
| Storms appearing in multiple folds | 0 |

### Observation Gap Distribution

| Time Gap | Count |
|---|---:|
| 3 hours | 1171 |
| 15 hours | 14 |
| 30 hours | 1 |
| 54 hours | 1 |
| 87 hours | 1 |

Temporal processing must account for
irregular observation intervals.

Temporal confirmation must not use future
observations or silently bridge large gaps.

The prediction row alignment was supported
by the performed audits, but stable sample
identifiers should be retained in future
prediction files.

## 14. Cross-Validation Strategy

Four-fold storm-grouped cross-validation
was performed.

| Fold | Training Storms | Validation Storms | Validation Sequences | RI Sequences |
|---|---:|---:|---:|---:|
| 1 | 57 | 20 | 375 | 21 |
| 2 | 58 | 19 | 358 | 32 |
| 3 | 58 | 19 | 259 | 10 |
| 4 | 58 | 19 | 273 | 26 |

No storm overlap was detected within
the training and validation fold pairs.

All sequences belonging to a storm
must remain within the same fold.

Grouped cross-validation reduces
storm-level data leakage risk.

It does not by itself establish the absence
of all possible forms of data leakage.

## 15. Repository Data Organization

The research directory contains:

### features/

- training_features_v3.csv
- development_features_v3.csv

### data_metadata/

- original_wind_records.csv

### data_quality/

- Dataset quality reports
- Processing error reports
- Year-wise reports
- Satellite channel quality reports

### cross_validation/

- Fold assignments
- Fold performance results
- Out-of-fold predictions
- Performance variation reports

### v3_analysis/

- Probability calibration
- Model optimization
- Threshold comparison
- False alarm analysis
- Storm-level diagnostics

### temporal_experiments/

- Ensemble experiments
- Temporal confirmation
- Fold robustness
- Model selection reports

## 16. External Data Storage

Large satellite archives and complete
numerical datasets are maintained separately
from this GitHub repository.

The current research backup is organized
in Google Drive under:

ChakraSense_Backup/

The repository contains selected models,
processed features, source code and
research documentation.

The GitHub repository is not yet a complete
self-contained reproduction of the dataset.

## 17. Dataset Limitations

- Limited number of independent storms.
- Severe RI class imbalance.
- Missing historical measurements.
- Irregular observation intervals.
- Historical satellite sensor differences.
- Potential best-track versus real-time
  observation differences.
- Limited environmental meteorological inputs.
- Limited modern satellite coverage.
- Potential dependence between overlapping
  forecast windows.
- Independent test provenance requires audit.

## 18. Planned Dataset Expansion

Future development will investigate:

- Sea surface temperature
- Ocean heat content
- Vertical wind shear
- Atmospheric humidity
- Atmospheric pressure
- Distance from land
- Environmental circulation
- Improved satellite image features
- Additional historical cyclones
- Modern satellite observations
- Real-time meteorological inputs

These are planned inputs and are not claimed
to be part of the current V3 dataset.

## 19. Data Attribution

Primary dataset sources:

- NOAA HURSAT
- NOAA IBTrACS
- IMD observations distributed through IBTrACS

Dataset versions, download locations,
access dates and redistribution conditions
must be documented during the final
reproducibility audit.

## 20. Intended Use

The dataset is intended for experimental
research, educational analysis and
machine learning development.

It has not been validated for operational
cyclone forecasting or public warning systems.

ChakraSense remains a research prototype.
