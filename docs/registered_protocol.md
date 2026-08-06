# Registered Protocol — Aurora PM Station Corrector

**Registration date:** 2026-08-06  
**Repository:** fedeg-umh-es/aurora-pm-station-corrector  
**Branch:** claude/aurora-pm-station-corrector-2hg6vo  
**Status:** Phase A — registration only, no results available

---

## 1. Scientific Questions

### 1.1 Primary question

Do frozen Aurora Air Pollution forecasts contain predictive information for PM10
and PM2.5 at station scale beyond the information already contained in CAMS
forecasts and in the local observational history available at emission time?

### 1.2 Secondary (operational) question

Does that incremental predictive content — if it exists — improve the detection,
magnitude, duration, and timing of elevated-PM episodes relative to the
reference systems?

### 1.3 Scope note

African dust is analysed as an atmospheric regime (conditioning variable), not
as a primary scientific claim.

---

## 2. Background and Motivation

Aurora Air Pollution was fine-tuned using CAMS products. Any apparent
improvement of Aurora over CAMS may therefore reflect shared training signal
rather than independent physical content. The primary question is formulated in
conditional terms — incremental content *given* CAMS and local history — to
account for this dependence from the outset. No causal mechanism will be
attributed.

---

## 3. Unit of Analysis

Each experimental row represents exactly one forecast-observation pair with the
following mandatory fields:

| Field | Definition |
|---|---|
| `station_id` | Monitoring station identifier |
| `pollutant` | PM10 or PM2.5 |
| `cycle_time` | NWP/Aurora initialisation time (UTC) |
| `available_time` | When the forecast became available at the forecast decision time (cycle_time + dissemination lag) |
| `valid_time` | Time of the target observation (UTC) |
| `target_available_time` | When the target observation was published/validated (valid_time + measurement/publication lag) |
| `lead_hours` | `valid_time − cycle_time` in hours |
| `obs` | Observed PM concentration at valid_time |
| `aurora_fc` | Aurora raw forecast for valid_time |
| `cams_fc` | CAMS raw forecast for valid_time |
| `local_lags` | PM observations with `observation_available_time <= available_time` |
| `provenance` | Run-level metadata hash (see Section 9) |

### 3.1 Hard row invariant

`valid_time = cycle_time + timedelta(hours=lead_hours)` for every row.  
Automated test fails (non-zero exit) if this is violated.

### 3.2 Train admissibility invariant

A row may enter a training set at decision origin O only if:

```
available_time < O                (forecast was issued before O)
target_available_time <= O        (target observation was published before O)
```

This is stricter than filtering on `valid_time <= O` alone, because in practice
station observations are published with a measurement/validation delay (typically
0.5–2 h). Automated tests enforce this per split.

### 3.3 Lag feature invariant

Each local lag observation used as a feature must satisfy:

```
observation_available_time <= available_time
```

where `available_time` is the forecast decision time. The lag builder enforces
this row by row; automated tests verify it on synthetic data with known latency.

### 3.4 Pseudo-operational assumption

When `target_available_time` is absent from the data, the splitter falls back
to `valid_time` (zero latency). This assumption:

- must be registered in provenance metadata as `pseudo_operational: true`;
- requires sensitivity analysis with 1-h and 2-h latency before final reporting;
- produces a `UserWarning` at runtime.

### 3.5 Threshold invariant

All thresholds used for binary event classification and tail-bias computation
must be estimated from train data only at each rolling-origin step.
`fit_tail_threshold(obs_train, quantile)` is the only sanctioned API;
`np.quantile` applied directly to test observations is prohibited.

---

## 4. Forecast Horizons

Evaluated separately. Horizons must not be pooled during evaluation or in
primary conclusions.

12 h · 24 h · 36 h · 48 h · 60 h · 72 h · 84 h · 96 h · 108 h · 120 h

---

## 5. Models and Baselines

All nine variants below must be implemented and evaluated. Variants 6–9 use the
same estimator family wherever feasible to isolate the effect of the predictor
set.

| ID | Name | Description |
|---|---|---|
| 1 | `seasonal_climatology` | Hour-of-day × month climatology from train data |
| 2 | `persistence` | Last observation at available_time |
| 3 | `sarima_or_sarimax` | Seasonal ARIMA fitted to local series; SARIMAX with CAMS if identifiable |
| 4 | `aurora_raw` | Aurora forecast without post-processing |
| 5 | `cams_raw` | CAMS forecast without post-processing |
| 6 | `local_only` | Estimator trained on local lags only |
| 7 | `local_plus_aurora` | Estimator trained on local lags + Aurora forecast |
| 8 | `local_plus_cams` | Estimator trained on local lags + CAMS forecast |
| 9 | `local_plus_cams_plus_aurora` | Estimator trained on local lags + CAMS + Aurora |

### 5.1 Primary contrasts

- `local_plus_cams_plus_aurora` vs `local_plus_cams` — does Aurora add beyond CAMS?
- `local_plus_aurora` vs `local_only` — does Aurora add beyond local history?

### 5.2 Secondary contrasts

- `aurora_raw` vs `cams_raw` — raw head-to-head
- corrected models vs `persistence`
- corrected models vs `sarima_or_sarimax`

---

## 6. Validation Protocol

### 6.1 Design

Rolling-origin (or expanding-window) validation. Both are acceptable; the
chosen variant must be documented in provenance metadata of every run.

### 6.2 Within-origin procedure

For each origin O:

1. Collect all rows with `available_time < O`.
2. Split that set into internal train and internal validation (temporal, not
   random) for hyperparameter tuning only.
3. All imputation, scaling, lag selection, and feature engineering are fitted
   exclusively on rows from step 2.
4. Generate out-of-sample predictions for rows with `available_time` in the
   first block after O.
5. Record exact date ranges of train, validation, and test for every origin.
6. Append row-level out-of-sample predictions with their origin label.

### 6.3 Leakage test

Automated CI test: for each origin O, `max(train.target_available_time) <= O`.
The test must fail hard (non-zero exit) if violated. Filtering on `valid_time`
alone is insufficient — it ignores the observational publication delay.

### 6.4 Threshold derivation

Event detection thresholds are estimated from train rows only, at each origin.
No test observations may influence threshold selection.

---

## 7. Metrics

### 7.1 Continuous performance (primary)

| Metric | Note |
|---|---|
| MAE | Primary loss |
| RMSE | |
| Bias (mean error) | |
| Skill vs persistence | `1 − MAE_model / MAE_persistence` |
| Skill vs SARIMA/SARIMAX | |
| Skill vs `local_only` | |
| Skill vs `local_plus_cams` | |

### 7.2 Variability

| Metric | Note |
|---|---|
| Variance ratio (predicted / observed) | |
| IQR ratio Q05–Q95 | |
| Regression slope (predicted on observed) | OLS via `np.polyfit`; avoids ddof mismatch |
| Tail bias at Q90 | `tail_bias(obs_test, pred_test, threshold=fit_tail_threshold(obs_train, 0.90))` |
| Tail bias at Q95 | `tail_bias(obs_test, pred_test, threshold=fit_tail_threshold(obs_train, 0.95))` |
| Maximum bias (mean error at observed maximum) | `max_bias(obs_test, pred_test)` |

### 7.3 Event-based (episode detection)

Thresholds derived from train only; applied at each rolling-origin step.

| Metric |
|---|
| Probability of Detection (POD) |
| False Alarm Ratio (FAR) |
| Critical Success Index (CSI) |
| Precision |
| Recall |
| F1 |
| Temporal error of maximum (hours) |
| Duration error (hours) |
| Peak concentration error |
| Accumulated exceedance above threshold |

---

## 8. Statistical Inference

### 8.1 Uncertainty

- Bootstrap confidence intervals using temporal block bootstrap (block length
  ≥ autocorrelation length of residuals; documented per run).

### 8.2 Significance tests

- Diebold–Mariano (DM) test, one test per triplet (pollutant, lead_hours,
  contrast pair).
- HAC covariance estimator (Newey–West) when residuals show autocorrelation.
- Multiplicity correction (Bonferroni or Benjamini–Hochberg) applied when the
  total number of DM tests across all contrasts, horizons, and pollutants
  exceeds 10; the correction method and adjusted p-values must be documented.

### 8.3 Prohibited uses

- DM must not be computed on pooled residuals from multiple horizons.
- Aggregate skill scores across horizons must not be presented as primary
  evidence.

---

## 9. Reproducibility Requirements

Every run must record the following provenance fields in a JSON sidecar:

```
config_hash          SHA-256 of the experiment YAML
git_commit           HEAD commit hash
aurora_checkpoint    Model checkpoint identifier
cams_dataset         Dataset name and cycle
cams_download_date   ISO-8601
aurora_cycle         Initialisation cycle time
available_time       As used in this run
dependency_versions  {package: version, ...}
seed                 Integer seed for all random operations
device               CPU/GPU identifier
spatial_operator     Interpolation or nearest-neighbour specification
stations             [station_id, ...]
excluded_rows        [{row_id, reason}, ...]
run_duration_s       Elapsed seconds
run_status           "ok" | "partial" | "failed"
```

### 9.1 Idempotency and resumability

- Downloads: skip if file exists and checksum matches; verify SHA-256 or MD5
  of each product.
- Model fits: skip if output artefact exists and provenance matches.
- Results: never overwrite existing valid outputs; write to timestamped
  subdirectory instead.

---

## 10. Phase Gates

| Phase | Scope | Gate criterion |
|---|---|---|
| A | Registration | This document committed and reviewed |
| B | Single cycle | 1 cycle, 4 horizons, 3 stations; leakage test passes; all 9 models produce output |
| C | Pilot | 30 days, 5–10 stations, all horizons; DM tests run; no leakage violations |
| D | Full scale | Only after A–C documented as PASS or CONDITIONAL PASS |

Full hindcast data must not be downloaded before Phase B gate is passed.

---

## 11. Out-of-Scope Elements (Phase A)

The following are not implemented in Phase A and must not be claimed:

- Results of any kind (no data, no metrics, no forecasts)
- Fine-tuning or modification of Aurora
- CAMS data acquisition
- Real station observations
- Causal attribution of skill improvement to any mechanism

---

## 12. Deliverables — Phase A Checklist

- [x] `docs/registered_protocol.md` — this document
- [x] `docs/claim_boundaries.md`
- [x] `configs/experiment_registered.yaml`
- [x] `src/validation/rolling_origin.py` — split-level leakage guards with `target_available_time`
- [x] `src/metrics/basic.py` — provisional metric primitives
- [x] `src/features/lag_builder.py` — observational availability enforcement
- [x] `tests/test_no_future_leak.py` — 22 automated tests (all pass)
- [x] `.github/workflows/tests.yml` — CI gate
- [ ] Phase B: single-cycle execution (gate: this document merged and reviewed)

## 13. Phase A Decision

**Phase A — CONDITIONAL PASS**

Pending: external review confirmation. Phase B is not permitted until this
protocol document is reviewed and merged into main.

Resolved issues:
- Availability semantics corrected: `target_available_time` replaces `valid_time`
  as the train admissibility criterion.
- Threshold API corrected: `tail_bias` requires pre-fitted threshold from train.
- `regression_slope` ddof mismatch fixed via `np.polyfit`.
- `duration_error` renamed to `exceedance_count_error`; `episode_duration_error`
  added with proper episode segmentation.
- 22 integrated tests cover: row invariant, observational latency (1-h and 2-h),
  pseudo-operational fallback warning, lag builder with real availability filter,
  multi-pollutant isolation, multi-horizon and multi-station splits, metric
  correctness, train-only threshold enforcement.
- CI workflow added.

Remaining open item before Phase D:
- `estimator_family: TBD` must be resolved before first rolling-origin corrector
  training run.
