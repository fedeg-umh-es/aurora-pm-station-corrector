# Registered experiment protocol

## Objective

Estimate whether frozen Aurora Air Pollution forecasts add station-level predictive information beyond local concentration history and CAMS for PM10 and PM2.5.

The contribution is an attribution and verification study. It is not a new post-processing algorithm, a causal analysis, or a novelty claim about rolling-origin validation.

## Required model ladder

All learned models must use the same estimator family, outer folds, inner tuning rule and target rows.

1. `persistence`: latest observation available at issue time.
2. `aurora_raw`: Aurora PM value at the station/grid match.
3. `cams_raw`: CAMS PM value at the same issue and valid time.
4. `local_only`: local history and calendar variables.
5. `local_aurora`: local history plus Aurora.
6. `local_cams`: local history plus CAMS.
7. `local_aurora_cams`: local history plus both global predictors.

Primary contrasts:

- `local_aurora - local_only`: incremental Aurora value beyond station memory.
- `local_cams - local_only`: incremental CAMS value beyond station memory.
- `local_aurora_cams - local_cams`: Aurora information not already represented by CAMS.
- `local_aurora - local_cams`: downstream comparison under the same corrector.

Improvement over `aurora_raw` alone is not sufficient evidence of Aurora value.

## Temporal validation

Use outer rolling-origin evaluation by issue time. A training row is admissible only when its target `valid_time` is no later than the first issue time of the test block. This prevents multi-horizon target leakage.

All transformations, imputation, scaling, threshold estimation and hyperparameter selection must be fitted using training data only. Hyperparameter tuning uses a final chronological block inside each outer training fold.

Report each horizon independently: 12, 24, ..., 120 hours. Do not make the main claim from a metric pooled over horizons.

## Metrics

Continuous verification:

- MAE as primary loss;
- RMSE and bias;
- skill relative to persistence and `local_only`;
- paired fold-wise loss differences with uncertainty intervals before publication.

Amplitude and tail verification:

- predicted/observed variance ratio;
- Q05-Q95 amplitude ratio;
- bias conditional on observed Q90 and Q95;
- peak bias;
- probability of detection, false-alarm ratio, precision and F1 for a train-derived event threshold.

Thresholds must be estimated within each training fold. A global threshold computed from all observations is leakage.

## Decision rules

Aurora has incremental station-level value only when `local_aurora` improves over `local_only` across a coherent range of horizons and stations. Aurora has value beyond CAMS only when `local_aurora_cams` improves over `local_cams`.

A mean-error gain does not establish recovery of extremes when variance ratio, tail bias or event detection deteriorate. Negative results remain admissible when the protocol is sufficiently broad and the failure mode is clearly diagnosed.

## Scope controls

Do not introduce Aurora 1.5 meteorology, satellite predictors, emission inventories, probabilistic calibration or Aurora fine-tuning in the primary experiment. They may be added only after the core attribution question is resolved.
