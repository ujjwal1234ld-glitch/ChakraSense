# Try ChakraSense

Download `Replay.html` and open it in a browser. It works offline. The viewer includes all 1,265 earlier satellite training-fold cases plus five selected track-model examples. The latter are selected detections, and the dashboard shows totals across all 876 later test windows. Use the reveal button to compare the forecast with the future observation. The test report includes missed events and false flags.

## Train the wind-and-track model

1. Use Python 3.14 and create a virtual environment.
2. Install the tested numerical packages with `python -m pip install -r forecast/requirements.txt`.
3. Use the North Indian Ocean IBTrACS v04r01 CSV from the existing ChakraSense backup, or download the corresponding North Indian Ocean CSV from [NOAA](https://www.ncei.noaa.gov/products/international-best-track-archive). The milestone snapshot checksum is in [Results.md](../submission/Results.md). NOAA revisions may change the results.
4. Run `python forecast/track_model.py fit ibtracs.NI.list.v04r01.csv output`. The program fits candidate estimators, selects using development storms, saves a frozen model, and writes a test row for each forecast.
5. Run `python forecast/track_model.py score output/track.joblib output/example.csv output/example_predictions.csv`. Load only your own generated checkpoint.

## Evaluate the result

Open `output/metrics.json` for split counts, wind MAE, persistence MAE, RMSE, average precision, Brier score and the RI confusion counts. Open `output/track_predictions.csv` to inspect individual development and test forecasts, including all false positives and missed windows. `output/track_samples.csv` retains exact target labels and the named predictor columns.

For the original V3 independent-test workflow, see the final executed cell of `Untitled0.ipynb`. It uses the storm backup and frozen ensemble. The old notebook requires its documented Colab paths and scikit-learn 1.6.1. The 2015 matrix was also reproduced locally from reconstructed inputs.

## Scope

The replay and forecast tables use historical data. They provide no public warnings or live ingestion. Uncertainty intervals and modern operational-satellite performance require further work.
