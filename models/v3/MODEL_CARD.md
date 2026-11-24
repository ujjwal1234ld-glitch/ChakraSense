
# ChakraSense V3 — Model Card

## 1. Model Information

| Parameter | Description |
|---|---|
| Model Name | ChakraSense V3 |
| Domain | Tropical Cyclone Forecasting |
| Status | Experimental Research Prototype |
| Model Type | Gradient Boosting Ensemble |
| Language | Python |
| Framework | Scikit-learn |
| Saved Scikit-learn Version | 1.6.1 |
| Input Features | 21 |
| RI Threshold | 0.15 |
| Temporal Confirmation | 6 hours |
| Selected Method | Temporal_6h |

## 2. Model Purpose

ChakraSense V3 investigates the prediction of
tropical cyclone rapid intensification (RI)
using historical wind observations and satellite
imagery-derived features.

The broader ChakraSense project also investigates
future cyclone wind speed forecasting.

The system is intended for experimental research,
historical analysis and model development.

It is not an operational cyclone warning system.

## 3. Model Architecture

The V3 classification pipeline incorporates two
gradient boosting classifier configurations:

1. Regularized classifier
2. Slow-learning classifier

The selected experimental method combines their
predictions with temporal confirmation.

### Processing Pipeline

Historical wind observations
        +
Satellite imagery
        |
        v
Feature extraction
        |
        v
21 engineered features
        |
        v
Regularized and Slow-learning classifiers
        |
        v
RI probability estimates
        |
        v
Temporal confirmation
        |
        v
Experimental RI alert

## 4. Input Features

### Historical Wind Features

1. current_wind
2. wind_change_6h
3. wind_change_12h
4. wind_change_24h

### Infrared Satellite Features

5. IR_mean
6. IR_std
7. IR_min
8. IR_centre_mean
9. IR_change
10. IR_p10

### Water Vapour Satellite Features

11. WV_mean
12. WV_std
13. WV_min
14. WV_centre_mean
15. WV_change
16. WV_p10

### Cloud Structure Feature

17. IR_cold_cloud_fraction

### Observation Age Features

18. current_wind_age
19. wind_age_6h
20. wind_age_12h
21. wind_age_24h

Observation age features record how old the
available historical wind measurements are.

## 5. Dataset

The project uses historical satellite imagery
and meteorological observations from HURSAT
and IBTrACS.

| Dataset Parameter | Value |
|---|---:|
| Total Sequences | 1505 |
| Independent Storms | 88 |
| RI Sequences | 114 |
| Non-RI Sequences | 1391 |
| RI-positive Storms | 20 |
| Training Sequences | 1265 |
| Development Sequences | 90 |
| Independent Test Sequences | 150 |

### Training Dataset

- Independent storms: 77
- RI-positive storms: 16
- RI-positive sequences: 89
- Non-RI sequences: 1176

The training dataset is highly imbalanced.

Approximately 7.04% of training sequences
are RI-positive.

## 6. Data Quality Improvements

Historical wind records were reconstructed
using original IMD observations from IBTrACS.

The historical observation audit identified:

- Total historical wind records: 3774
- Original intensity records: 3092
- Storms represented: 152

Missing historical wind features were reduced
during V3 feature preparation.

| Feature | Previous Missing | V3 Missing |
|---|---:|---:|
| 6-hour wind change | 175 | 85 |
| 12-hour wind change | 322 | 144 |
| 24-hour wind change | 515 | 267 |

Missing values remain in some historical
wind and observation age features.

The exact missing-value handling must be
verified in the reproducibility audit.

## 7. Evaluation Methodology

Four-fold storm-grouped cross-validation
was used during model development.

Observations from the same storm were kept
within the same validation group.

No storm overlap was detected between the
training and validation groups.

Model selection and threshold experiments
used training out-of-fold predictions.

Because multiple configurations were explored
using these predictions, the reported results
are exploratory rather than independent
estimates of final model performance.

## 8. Rapid Intensification Performance

### Fold-Averaged Classifier Comparison

| Model | AP | ROC-AUC | Brier |
|---|---:|---:|---:|
| Original | 0.2382 | 0.7797 | 0.0643 |
| Regularized | 0.2431 | 0.7883 | 0.0607 |
| Slow Learning | 0.2423 | 0.7738 | 0.0629 |
| Balanced | 0.2364 | 0.7813 | 0.1022 |

### Pooled Out-of-Fold Alert Performance

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| Regularized | 0.2532 | 0.4382 | 0.3210 |
| Max Ensemble | 0.2471 | 0.4831 | 0.3270 |
| Temporal 6h | 0.2658 | 0.4719 | 0.3401 |

Fold-averaged and pooled metrics are
different aggregation methods and should
not be treated as interchangeable.

## 9. Temporal Confirmation Experiment

| Metric | Regularized | Temporal 6h |
|---|---:|---:|
| True Positive Windows | 39 | 42 |
| False Positive Windows | 115 | 116 |
| Missed RI Windows | 50 | 47 |
| RI Storms Detected | 11 | 12 |
| RI Storms Completely Missed | 5 | 4 |
| Non-RI Storms Alerted | 17 | 17 |

The selected temporal method recovered
three additional correct RI windows and
one additional RI-positive storm, at the
cost of one additional false-positive window.

### Fold-wise F1 Performance

| Fold | Baseline | Temporal 6h |
|---|---:|---:|
| 1 | 0.4000 | 0.4000 |
| 2 | 0.3077 | 0.3333 |
| 3 | 0.2069 | 0.2667 |
| 4 | 0.3529 | 0.3529 |

The exploratory F1 score improved in two
folds and remained unchanged in two folds.

## 10. Wind Forecasting Research

The following results belong to the broader
V3 wind forecasting experiments.

| Model | MAE (knots) | RMSE (knots) |
|---|---:|---:|
| Persistence | 12.5536 | 18.6315 |
| Wind History | 11.5403 | 16.6795 |
| Gradient Boosting | 11.7566 | 17.4321 |
| Wind and Images | 11.2270 | 16.2498 |

These wind forecasting results do not establish
that the saved V3 RI classifier artifact itself
contains a wind regression model.

## 11. Probability Calibration

Regularized classifier pooled results:

- Observed RI frequency: 0.0704
- Mean predicted RI probability: 0.0671
- Brier score: 0.0609

The average predicted probability is close
to the observed RI frequency.

However, this does not establish reliable
calibration across all probability ranges.

High-probability predictions remain
insufficiently supported by the dataset.

## 12. Known Limitations

- Limited number of independent storms.
- Severe class imbalance.
- Substantial false-positive alert counts.
- Substantial missed RI detections.
- Historical satellite calibration differences.
- Missing historical wind measurements.
- Limited environmental meteorological inputs.
- Potential differences between historical
  best-track data and real-time observations.
- Limited evidence of generalisation to
  modern satellite systems.
- Temporal causality and forecast-time
  data availability require verification.
- Independent test performance has not
  yet been verified.

## 13. Independent Test Status

The independent test split contains:

- 150 sequences
- 7 independent storms
- 13 RI-positive sequences

A saved independent prediction file has
been identified in the project backup.

Its provenance, prediction alignment and
evaluation protocol require verification.

No independently verified test metrics
are claimed in this model card.

## 14. Intended Usage

Suitable for:

- Academic research
- Historical cyclone analysis
- Machine learning experimentation
- Satellite feature analysis
- Educational demonstrations

Not suitable for:

- Operational cyclone warnings
- Public safety decision-making
- Evacuation recommendations
- Replacement of official weather forecasts

## 15. Future Development

Planned improvements include:

- Sea surface temperature
- Ocean heat content
- Vertical wind shear
- Atmospheric humidity
- Land proximity information
- Improved satellite image features
- Expanded historical training dataset
- Forecast uncertainty estimation
- Independent and chronological validation
- Real-time data integration
- Comparison against established models

## 16. Saved Model Artifact

File: ChakraSense_V3.joblib

The saved artifact contains:

- Regularized classifier
- Slow-learning classifier
- Ordered list of 21 features
- RI threshold
- Temporal confirmation duration
- Scikit-learn version information

Only load trusted serialized model files.

## 17. Disclaimer

ChakraSense V3 is an experimental
research prototype.

Its current results do not establish
operational reliability.

Predictions must not replace official
meteorological forecasts or warnings.

## 18. Development Status

Research and development ongoing.
