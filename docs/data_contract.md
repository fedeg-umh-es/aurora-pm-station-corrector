# Aligned forecast data contract

The experiment consumes one row per station, pollutant, forecast cycle and lead time.

## Required columns

| Column | Type | Meaning |
|---|---|---|
| `station_id` | str | Stable station identifier |
| `pollutant` | str | `PM10` or `PM2.5` |
| `cycle_time` | datetime (UTC) | Forecast issue time (cycle start) |
| `available_time` | datetime (UTC) | When this forecast row became operationally available; `cycle_time + dissemination_lag` |
| `valid_time` | datetime (UTC) | Verification time in UTC |
| `lead_hours` | int | `valid_time − cycle_time` in hours |
| `target_available_time` | datetime (UTC) | When the target observation was published; used as the train admissibility boundary |
| `y_true` | float | Station observation at `valid_time`, µg m⁻³ (NaN if not yet published) |
| `local_latest` | float | Latest station value operationally available at `available_time` |
| `local_lag_12h` | float | Station value available 12 h before `available_time` |
| `local_lag_24h` | float | Station value available 24 h before `available_time` |
| `local_mean_24h` | float | Trailing 24 h mean using only data available at `available_time` |
| `local_std_24h` | float | Trailing 24 h std using only data available at `available_time` |
| `aurora_forecast` | float | Aurora Air Pollution forecast at station, µg m⁻³ |
| `cams_forecast` | float | CAMS forecast at station, same cycle and lead time, µg m⁻³ |
| `aurora_grid_lat` | float | Grid latitude used for Aurora extraction |
| `aurora_grid_lon` | float | Grid longitude used for Aurora extraction |
| `cams_grid_lat` | float | Grid latitude used for CAMS extraction |
| `cams_grid_lon` | float | Grid longitude used for CAMS extraction |
| `spatial_operator` | str | `bilinear` or `nearest_neighbour` |
| `aurora_checkpoint` | str | Aurora model identifier (HuggingFace ID or local path) |
| `cams_dataset` | str | CAMS dataset identifier |
| `source_file_hash` | str | SHA-256 of source GRIB/NetCDF files for reproducibility |

Calendar features (`issue_hour_sin`, `issue_hour_cos`, `issue_doy_sin`, `issue_doy_cos`) are derived from `cycle_time` and may be precomputed at feature-build time.

## Invariants

- `valid_time > cycle_time` for all rows.
- `lead_hours` equals the integer UTC hour difference.
- Keys `(station_id, pollutant, cycle_time, lead_hours)` are unique.
- All local features use only observations with `observation_available_time <= available_time`.
- `target_available_time` must be present; see `pseudo_operational_assumption` in `configs/experiment_registered.yaml` for the fallback when the column is absent.
- Aurora and CAMS represent forecasts available at `cycle_time`. Reanalysis or analysis fields must be labelled separately and cannot support an operational claim.
- Both forecast columns are in µg m⁻³; Aurora input to the model must have been converted from kg m⁻³.

## Train admissibility

A row is admissible as a training sample at rolling-origin `O` iff:

```
available_time < O  AND  target_available_time <= O
```

The second condition prevents using rows whose target observation was not yet published at the evaluation origin.

## Missingness

Do not impute `y_true` across the train/test boundary. Missing predictors may be imputed inside each training fold only. Rows missing `y_true`, `cycle_time`, `valid_time`, `station_id`, or `pollutant` are invalid and must be dropped before splitting.

## Provenance

Each generated aligned dataset must have a companion metadata file recording:
- source products (Aurora checkpoint, CAMS dataset, station network);
- cycle timestamps and download timestamps;
- unit conversions applied;
- grid-to-point method and spatial operator used;
- known latency assumptions for `available_time` and `target_available_time`.
