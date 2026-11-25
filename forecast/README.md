# Run ChakraSense

## Replay the test storms

Download [Replay.html](Replay.html) and open it in a browser. The file works offline and contains every held-out window from both experiments. Choose a model, storm ID and UTC issue time. Select **Reveal observation** to see the next day's wind and the absolute error in knots. The RI outcome filter lets you inspect missed events and false flags.

To replay the example in the reel, select **Wind and track**, storm **2020136N10088**, and **2020-05-16 15:00**. Issue wind was 40 kt, the forecast was 58.8 kt and the later observation was 75 kt.

## Score the examples

Download the complete project ZIP from GitHub and extract it. Open a terminal in the extracted folder that contains this README's parent folder and the original notebook. Python's standard library is enough for inference.

```shell
python forecast/predict.py track forecast/data/track_example.csv track_forecasts.csv
python forecast/predict.py satellite forecast/data/satellite_example.csv satellite_forecasts.csv
```

Each input row represents a preprocessed storm window at its issue time. Keep the example column names and units. The output contains the predicted wind in knots, a score from 0 to 1 and an RI flag. Both estimators predict 24 hours ahead. The fixed decision thresholds are 0.16 for wind and track and 0.23 for satellite. The example files contain predictors only. The complete training tables also contain evaluation labels in separate columns.

Run python forecast/verify.py to independently recalculate all released development and test forecasts, split checks, wind errors and test RI counts with standard Python.

## Reproduce the models

Use Python 3.14 and the pinned numerical packages. Run these commands from the project root. Training and export use the included feature tables and take no GPU.

```shell
python -m pip install -r forecast/requirements.txt
python forecast/train.py
python forecast/track.py
python forecast/export_models.py
python forecast/build_replay.py
```

Training writes checkpoints and forecast tables under `forecast/release/`. The export checks the dependency-free models against the scikit-learn outputs on 240 satellite and 1,694 track development and test windows. The dashboard reads the two test subsets. [Measured results](../submission/Results.md)

| Files | What they contain |
|---|---|
| `data/samples.csv` and `data/track_samples.csv` | Storm ID, timestamp, split, future label and issue-time predictors for each experiment |
| `release/satellite.json` and `release/track.json` | Standalone model parameters and the required input fields, bounds and decision thresholds |
| `release/results.json` and `release/track_results.json` | Model selection, wind errors and rapid-intensification results |
| `release/predictions.csv` and `release/track_predictions.csv` | Forecast, observation, score and decision for each evaluation window |
| `release/v3_test_predictions.csv` and `release/v3_test_results.json` | Independent 2015 results from the original satellite classifier |

## Rebuild features from source

The included CSVs are sufficient to reproduce training and scoring. To reconstruct satellite features from the project's two original backup ZIPs, run `python forecast/prepare.py path/to/part1.zip path/to/part2.zip`. It checks timestamps, wind labels, storm folds and archived features. The multi-gigabyte raw satellite frames remain in the original archives.

To reconstruct the track table, run `python forecast/track_data.py path/to/ibtracs.NI.list.v04r01.csv`. Use the original NOAA North Indian Ocean snapshot for identical results. Its SHA-256 and citation are in the [results](../submission/Results.md).

The archived V3 classifier requires its original scikit-learn 1.6.1 environment. In that separate environment, run `python forecast/reference_v3.py` or see the final executed cell of [the original notebook](../Untitled0.ipynb). Historical best-track records can receive later revisions. Live ingestion and prospective validation remain future work.
