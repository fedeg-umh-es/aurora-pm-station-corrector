# Aligned forecast data contract

The experiment consumes one row per station, pollutant, forecast issue and lead time.

## Required columns

| Column | Meaning |
|---|---|
| `station_id` | Stable station identifier |
| `pollutant` | `PM10` or `PM25` |
| `issue_time` | Forecast issue time in UTC |
| `valid_time` | Verification time in UTC |
| `lead_hours` | `valid_time - issue_time` in hours |
| `y_true` | Station observation at valid time, µg m-3 |
| `local_latest` | Latest station value operationally available at issue time |
| `local_lag_12` | Station value available 12 h before issue |
| `local_lag_24` | Station value available 24 h before issue |
| `local_mean_24h` | Trailing 24 h mean computed only from data available at issue time |
| `local_std_24h` | Trailing 24 h standard deviation computed only from available data |
| `aurora_pm` | Aurora Air Pollution forecast aligned to issue and valid time |
| `cams_pm` | CAMS forecast aligned to the same issue and valid time |

Calendar features may be precomputed from `issue_time`: `issue_hour_sin`, `issue_hour_cos`, `issue_doy_sin`, `issue_doy_cos`.

## Invariants

- `valid_time > issue_time`.
- `lead_hours` equals the UTC time difference.
- Keys `(station_id, pollutant, issue_time, lead_hours)` are unique.
- Local features use no observation recorded after `issue_time`.
- Aurora and CAMS must represent forecasts available at the stated issue time. Reanalysis or analysis fields must be labelled separately and cannot support an operational claim.
- Units and pollutant definitions must be harmonised before alignment.

## Missingness

Do not interpolate targets across the train/test boundary. Missing predictors may be imputed inside each training fold only. Rows missing `y_true`, `issue_time`, `valid_time`, station or pollutant identifiers are invalid.

## Provenance

Each generated aligned dataset should have a companion metadata file recording source products, cycle/version, download timestamp, station network, unit conversions, grid-to-point method and known latency assumptions.
