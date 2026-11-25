
# ChakraSense

### AI-Based Tropical Cyclone Intensity Forecasting and Rapid Intensification Prediction

![Status](https://img.shields.io/badge/Status-Research%20and%20Development-orange)
![Version](https://img.shields.io/badge/Version-V3-blue)
![Python](https://img.shields.io/badge/Language-Python-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)

## November 2026 milestone

[Submission documents and form answers](submission/) · [Historical replay and model instructions](forecast/) · [Measured results](submission/Results.md)

The updated wind-and-track experiment evaluated 876 later windows from 57 storms. Mean absolute wind error was 9.05 kt against 11.76 kt for persistence. It caught 12 of 50 rapid-intensification windows and raised 45 false flags. The archived satellite V3 test detected 0 of 13 and raised one false flag. The V3 development notes below describe the earlier project stage.

Apply the released models with Python using the [four-row examples](forecast/README.md). The [offline replay](forecast/Replay.html) includes all 876 later wind-and-track forecasts and all 150 satellite test forecasts. The [form answers](submission/Form_Answers.md) identify the files to upload and the fields the team must complete.

## Original V3 project overview

ChakraSense is an experimental artificial intelligence
system designed to forecast tropical cyclone intensity
and identify the possibility of rapid intensification
(RI) using historical satellite imagery and
meteorological observations.

The project aims to investigate whether combining
satellite observations, historical wind measurements,
machine learning and temporal confirmation can improve
cyclone intensity forecasting.

## Project Objectives

- Forecast future cyclone wind speed.
- Estimate rapid intensification probability.
- Analyse infrared and water vapour satellite imagery.
- Reduce false alerts while retaining useful detections.
- Evaluate model generalisation across independent storms.
- Develop a reproducible cyclone forecasting framework.

## Dataset

The research uses historical meteorological observations
and satellite imagery from IBTrACS and HURSAT.

### Dataset Statistics

| Parameter | Value |
|---|---:|
| Total image sequences | 1505 |
| Independent storms | 88 |
| Rapid intensification sequences | 114 |
| Non-RI sequences | 1391 |
| RI-positive storms | 20 |
| Unique historical satellite frames audited | 1724 |
| Duplicate samples detected | 0 |
| Input frames per sequence | 9 |
| Satellite channels | 2 |
| Image resolution | 128 x 128 |

### Satellite Sources

- GMS-5
- MET-5
- MET-7

### Satellite Channels

- Infrared (IR)
- Water Vapour (WV)

## Dataset Splitting

Storm-level separation is used to reduce data leakage.

| Split | Storms | Sequences | RI Cases |
|---|---:|---:|---:|
| Training | 77 | 1265 | 89 |
| Development | 4 | 90 | 12 |
| Independent Test | 7 | 150 | 13 |

No storm overlap was detected between these splits.

## Model Architecture

### ChakraSense V2

The V2 architecture combines:

- Convolutional Neural Networks (CNN)
- Gated Recurrent Units (GRU)
- Historical intensity information
- Feature fusion
- Wind speed regression
- Rapid intensification classification

Total trainable parameters: 76,202.

### ChakraSense V3

The V3 research pipeline incorporates:

- 21 engineered input features
- Historical wind trends
- Infrared temperature statistics
- Water vapour temperature statistics
- Cold cloud fraction
- Observation age information
- Histogram-based gradient boosting
- Regularized classification
- Model ensembling
- Temporal alert confirmation

## Current V3 Configuration

- Regularized classifier
- Slow-learning classifier
- Experimental RI threshold: 0.15
- Temporal confirmation window: 6 hours

The threshold and confirmation configuration were
selected through exploratory storm-grouped
cross-validation.

## Research Results

### Wind Speed Prediction

Four-fold storm-grouped cross-validation:

| Model | MAE (knots) | RMSE (knots) |
|---|---:|---:|
| Persistence | 12.5536 | 18.6315 |
| Current Wind | 13.1455 | 17.9106 |
| Wind History | 11.5403 | 16.6795 |
| Gradient Boosting | 11.7566 | 17.4321 |
| Wind and Images | 11.2270 | 16.2498 |

### Rapid Intensification Prediction

Exploratory out-of-fold comparison:

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| Regularized | 0.2532 | 0.4382 | 0.3210 |
| Max Ensemble | 0.2471 | 0.4831 | 0.3270 |
| Temporal 6h | 0.2658 | 0.4719 | 0.3401 |

### Temporal Confirmation Results

| Metric | Regularized | Temporal 6h |
|---|---:|---:|
| Correct RI windows | 39 | 42 |
| False-positive windows | 115 | 116 |
| Missed RI windows | 50 | 47 |
| RI storms detected | 11 | 12 |
| Non-RI storms alerted | 17 | 17 |

These results are exploratory and must not be
interpreted as independently validated operational
forecast performance.

## Research Methodology

1. Historical meteorological data collection
2. Satellite imagery acquisition
3. Satellite calibration and quality auditing
4. Image preprocessing
5. Temporal sequence generation
6. Rapid intensification label generation
7. Storm-grouped dataset splitting
8. Baseline model development
9. CNN-GRU model training
10. Historical feature engineering
11. Grouped cross-validation
12. Probability calibration analysis
13. False alarm investigation
14. Temporal confirmation experiments
15. Independent evaluation and verification

## Repository Structure

    ChakraSense/
    |
    |-- legacy_prototype/
    |   |-- chakrasense/
    |   |-- tests/
    |   |-- docs/
    |   |-- requirements.txt
    |
    |-- models/
    |   |-- baselines/
    |   |-- v2/
    |   |-- v3/
    |
    |-- research/
    |   |-- features/
    |   |-- data_metadata/
    |   |-- data_quality/
    |   |-- cross_validation/
    |   |-- temporal_experiments/
    |   |-- v3_analysis/
    |
    |-- FILE_INVENTORY.csv
    |-- Untitled0.ipynb
    |-- README.md

## Technologies

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- Google Colab
- GitHub

## Reproducibility

The research notebook contains the experimental workflow.
Large satellite archives and additional datasets are
maintained separately in Google Drive.

The complete workflow is not yet self-contained.
Dataset paths, dependency versions, model integration
and execution instructions require further verification.

The saved V3 model records scikit-learn version 1.6.1.

## Current Development Status

- [x] Initial prototype development
- [x] Historical dataset acquisition
- [x] Real satellite image processing
- [x] Baseline machine learning models
- [x] CNN-GRU model training
- [x] V3 feature engineering
- [x] Storm-grouped cross-validation
- [x] False alarm analysis
- [x] Ensemble experiments
- [x] Temporal confirmation experiments
- [x] V3 model serialization
- [ ] Verified independent test results
- [ ] End-to-end reproducibility verification
- [ ] Environmental meteorological feature integration
- [ ] Real-time satellite data integration
- [ ] Operational forecasting pipeline
- [ ] External scientific validation

## Future Development

Planned research includes:

- Sea surface temperature integration
- Vertical wind shear analysis
- Atmospheric humidity features
- Ocean heat content integration
- Improved satellite feature extraction
- Improved rapid intensification detection
- Forecast uncertainty estimation
- Real-time data processing
- Interactive monitoring dashboard
- Comparison with established forecasting systems

## Data Sources

- NOAA IBTrACS
  https://www.ncei.noaa.gov/products/international-best-track-archive

- NOAA HURSAT
  https://www.ncei.noaa.gov/products/hurricane-satellite-data

Dataset usage and redistribution conditions must be
reviewed before public release.

## Limitations and Disclaimer

ChakraSense is currently an experimental research
prototype and is not an operational cyclone warning
system.

The model has limited historical training data,
class imbalance and substantial missed detections
and false alarms.

Its predictions must not replace official forecasts
or warnings issued by meteorological authorities.

## Project Status

Active Research and Development.
