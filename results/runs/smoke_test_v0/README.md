# Smoke test v0 report

## Inputs
- Predictions file: `results/runs/smoke_test_v0/persistence_predictions.csv`
- Metrics file: `results/runs/smoke_test_v0/persistence_metrics.csv`

## Basic counts
- Prediction rows: 6
- Metric rows: 4
- Leads evaluated: [1, 2]
- Pollutants: ['PM10', 'PM25']

## Persistence metrics

| station_id | pollutant | lead_time | model | n | mae | rmse |
|---|---|---:|---|---:|---:|---:|
| TBD_001 | PM10 | 1 | persistence | 2 | 5.9500 | 6.1745 |
| TBD_001 | PM10 | 2 | persistence | 1 | 3.3000 | 3.3000 |
| TBD_001 | PM25 | 1 | persistence | 2 | 2.1000 | 2.2136 |
| TBD_001 | PM25 | 2 | persistence | 1 | 1.4000 | 1.4000 |

## Status
- Smoke test persistence baseline completed successfully.