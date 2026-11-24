# ChakraSense milestone evaluation

## Question and design

Predict the maximum sustained NEWDELHI wind 24 hours after an issue time for an identified North Indian Ocean cyclone. Mark rapid intensification (RI) when the increase is at least 30 kt. All results below are retrospective. The baseline carries the wind at issue forward by 24 hours.

## Data and implementation

The satellite experiment reconstructed 1,505 nine-frame windows from 88 storms. Each window uses two channels on a 128 x 128 grid at three-hour intervals. The reconstruction checked storm IDs, exact frame times, 24-hour future winds and agreement with the archived training and development features. Extra predictors described six- and twelve-hour changes in core cloud temperatures, location, distance to land and wind history. Training used 77 storms. Wind candidates were ridge regression, histogram gradient boosting and their equal-weight average. RI candidates were logistic regression, histogram gradient boosting and their average. Training folds selected the lowest wind MAE, highest RI average precision, and then maximum F1 on a fixed threshold grid from 0.05 to 0.50.

The separate wind-and-track experiment used NOAA IBTrACS North Indian Ocean v04r01, main tracks, original or verified NEWDELHI wind observations, exact 24-hour endpoints and same-agency positions. Training used 2,997 windows from 250 storms in 1982–2015. Model and threshold selection used 818 windows from 40 storms in 2016–2019. The locked test contained 876 windows from 57 storms in 2020–2025. Features included historical wind change, central pressure and pressure trend when available, storm position, distance and recent approach to land, season and recent weakening. Missing optional history was handled within model fitting. Both candidate pipelines selected averaged linear and boosted wind forecasts and a boosted RI classifier. The satellite threshold was 0.23 and the track threshold was 0.16. Neither uses a language model or inference prompt.

## Measured results

| Test | Windows / storms | Wind MAE | Persistence MAE | RI detected | Missed | False flags |
|---|---:|---:|---:|---:|---:|---:|
| Archived V3, 2015 | 150 / 7 | — | — | 0 / 13 | 13 | 1 |
| New satellite candidate, 2014 development | 90 / 4 | 15.67 kt | 23.38 kt | 6 / 12 | 6 | 10 |
| New satellite candidate, 2015 test | 150 / 7 | 14.62 kt | 14.54 kt | 0 / 13 | 13 | 3 |
| Wind and track, 2016–2019 development | 818 / 40 | 9.29 kt | 13.41 kt | 24 / 53 | 29 | 41 |
| Wind and track, 2020–2025 test | 876 / 57 | 9.05 kt | 11.76 kt | 12 / 50 | 38 | 45 |

The later track test reduced average absolute wind error by 2.71 kt (23.1%) relative to persistence. RI precision was 21.1%, recall 24.0%, F1 0.224, average precision 0.170 and Brier score 0.0523. The constant training-frequency score had Brier 0.0539. Correct RI flags occurred in 3 of 8 RI-positive storms. Each forecast window is a separate prediction; overlapping windows from a storm are correlated.

On 16 May 2020 at 15:00 UTC, storm 2020136N10088 had 40 kt winds. The track forecast was 58.8 kt, the later observation was 75 kt and its RI score of 0.170 crossed the fixed 0.16 threshold. The mean wind estimate and RI decision come from separate estimators. This is a detected example, and the full dataset contains 38 misses.

## Interpretation and next experiment

The strongest supported result is improved average wind forecasting on the later track benchmark. Satellite RI generalization is still poor. Fine-tuning to the 2015 labels would weaken their value as a test. Historical best tracks can be revised after an event, so operational latency and reliability remain unknown. A future evaluation should freeze model and thresholds before receiving additional storms, compare with archived operational forecasts, and investigate water temperature, shear and humidity using issue-time availability. Public warnings belong to the responsible forecasting agencies.

## Reproduction sources

- Archived V3 code and recorded test matrix: final executed cell in `Untitled0.ipynb`.
- Repository feature order and storm folds: `research/features/` and `research/cross_validation/`.
- Historical raw archive SHA-256 used for the track experiment: `56e080f34cd6b4e7308e95b095e9c1d8356d31c930b413dba07dda53e04f6253`.
- NOAA IBTrACS v04r01, https://doi.org/10.25921/82ty-9e16 and https://www.ncei.noaa.gov/products/international-best-track-archive
- NOAA HURSAT-B1, https://www.ncei.noaa.gov/products/hurricane-satellite-data
