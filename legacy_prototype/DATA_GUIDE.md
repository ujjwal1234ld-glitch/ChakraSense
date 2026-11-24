# Data guide

## Inputs and units

The first implementation uses two brightness-temperature channels: infrared first, water vapour second. Each model example is an array shaped `[9, 2, 64, 64]`: nine times, two channels, 64 rows and 64 columns. The default crop covers 1,000 km by 1,000 km. This is a deliberately coarse starter, not a high-resolution eyewall system. A higher resolution requires a new dataset and a separately trained model.

Nine frames at three-hour intervals cover the previous 24 hours. A decision at 12:00 uses targets 12:00 yesterday, 15:00, …, 12:00 today. The reader chooses an observation at or before each target, at most 45 minutes older. It never selects a later scan. The future target is the wind exactly 24 hours after the decision time.

All times are UTC, written like `2020-05-16T12:00:00Z`. IST is UTC + 5 hours 30 minutes. Keep all stored times in UTC.

## Fill the raw manifest

Save a copy as `data/raw/raw_manifest.csv`. One row describes one downloaded file at one observation time for one storm.

| Column | Meaning |
|---|---|
| storm_id | Exact ID from `data/real/tracks.csv` or the download plan; not just the storm name |
| observed_at | Actual scan/observation time from the product metadata |
| available_at | When the file could actually have been received by a forecasting system; blank if unknown |
| center_available_at | When the center information became available; blank if unknown |
| center_source | `best_track` for historical research, or `operational_bulletin` for genuinely timestamped operational information |
| center_lat | Center latitude in degrees north; south is negative |
| center_lon | Center longitude in degrees east |
| path | Absolute file path, or path relative to this CSV; use forward slashes inside CSV paths |
| satellite | Exact same identifier used in the mapping file |

Use the `lat` and `lon` columns in `tracks.csv` for retrospective centers when observation times match. Do not silently use the center from tomorrow. If interpolation is required, document it and keep the result labelled retrospective. The code does not currently interpolate centers or join external bulletins automatically.

Leave unknown arrival times blank rather than inventing them. The prepared manifest marks these as unknown; its internal storage placeholder equals the observation time. Operational replay rejects them. A download time recorded today does not prove that an old file was available at the original forecast time.

## Map one actual HDF5 product

The mapping template is intentionally not prefilled with guessed dataset names. `inspect-h5` prints names, shapes, data types and attributes. Match every mapping field against the actual file and official product documentation.

- For raw counts, provide a one-dimensional count-to-temperature lookup table and its fill values/scale/offset.
- For temperatures already in kelvin, set `units` to `kelvin` and provide any image scale/offset.
- For each channel, latitude and longitude must be two-dimensional arrays with the same shape as that channel.
- Explicitly list missing values before scaling. Do not assume zero means missing: latitude/longitude zero can be a real location.
- The crop uses a local equirectangular coordinate approximation with nearest-neighbour sampling. Every output point must have a valid source within the configured distance, default 12 km. Check the geometric error and resolution on your actual product before trusting it.
- All cropped temperatures must be finite and within the implementation's 150–350 K screening range. A rejection may indicate a valid extreme, an unsupported product or a decoding error. Investigate; do not simply clip raw data to pass.
- Check image orientation, center location, scan timestamp interpretation, quality flags and cloud patterns against the official preview. Product-specific quality-flag decoding is not yet implemented; exclude bad scans manually until it is.

Prepare different satellite/product mappings separately. The initial benchmark should use one consistent product family. Combining satellites needs overlap checks and a documented harmonisation method; matching array sizes alone is insufficient.

## Labels and splits

The importer uses only `NEWDELHI_WIND`, `NEWDELHI_LAT` and `NEWDELHI_LON` in NOAA IBTrACS v04r01. It does not substitute US one-minute winds or generic WMO fields. It removes missing/invalid agency values and keeps years 2013–2025 by default. The best-track labels are retrospective truth, not real-time predictors.

RI = 1 when `wind(t + 24h) - wind(t) >= 30 knots`. If an exact future record is missing, that decision has no RI label and is excluded. It is not assigned RI = 0. This operational definition must be clearly stated when reporting results; thresholds based on other wind-averaging conventions are not automatically equivalent.

A storm gets its split from its earliest retained record. Whole storms stay together even across New Year. The default split plan and counts are saved in `data/real/split_plan.json`. Some of the 144 imported storms have no exact 24-hour label pair; the split summary includes only storms that do.

The sample validator checks storm separation, temperature/shape consistency, duplicate sequence content across splits, file paths and dataset integrity. Fingerprints detect altered input files after training. It cannot prove there are no mislabeled storms, duplicate individual source scans or incorrect product metadata; those still need dataset review.

## Retrospective versus operational

`--mode retrospective` allows historical best-track centers and observations without verified original arrival times. It measures an easier historical task and must be named as such.

`--mode operational` additionally requires actual known image and center availability by the decision time, and centers labelled `operational_bulletin`. It fails closed when those facts are missing. This mode is an input filter, not a complete live feed or an approved forecasting service.

No environmental variables, terrain-aware modeling, storm discovery, GFS ingest, ERA5 ingest, multi-satellite harmonisation, Grad-CAM, event-level warning lead-time matching, live delivery, authentication system or monitoring service is implemented. The earlier blueprint contains future design ideas as well as the narrower starter implemented here.

## Sources

- [NOAA IBTrACS data](https://www.ncei.noaa.gov/products/international-best-track-archive)
- [IBTrACS column documentation](https://www.ncei.noaa.gov/sites/default/files/2025-09/IBTrACS_v04r01_column_documentation.pdf)
- [MOSDAC download manual](https://mosdac.gov.in/downloadapi-manual)
- [INSAT-3DS operational products document](https://www.mosdac.gov.in/docs/INSAT-3DS_Operational_Products_V1.pdf)
- [NOAA GFS description](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast)

Record the download date, product version, file hashes and licence/attribution terms for any new source. Retain access credentials privately.
