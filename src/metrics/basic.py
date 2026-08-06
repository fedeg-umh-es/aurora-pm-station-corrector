"""
Metric implementations for the Aurora PM station corrector.

All functions operate on 1-D array-likes of equal length (no NaNs expected;
callers must drop NaN before calling). Threshold-based metrics require the
threshold to have been derived from train data only — callers are responsible
for this constraint.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Continuous performance
# ---------------------------------------------------------------------------

def mae(obs: np.ndarray, pred: np.ndarray) -> float:
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    return float(np.mean(np.abs(obs - pred)))


def rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def bias(obs: np.ndarray, pred: np.ndarray) -> float:
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    return float(np.mean(pred - obs))


def skill(mae_model: float, mae_reference: float) -> float:
    """1 - MAE_model / MAE_reference; NaN if reference is zero."""
    if mae_reference == 0.0:
        return float("nan")
    return float(1.0 - mae_model / mae_reference)


# ---------------------------------------------------------------------------
# Variability
# ---------------------------------------------------------------------------

def variance_ratio(obs: np.ndarray, pred: np.ndarray) -> float:
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    var_obs = np.var(obs)
    if var_obs == 0.0:
        return float("nan")
    return float(np.var(pred) / var_obs)


def iqr_ratio(obs: np.ndarray, pred: np.ndarray, q_low: float = 0.05, q_high: float = 0.95) -> float:
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    iqr_obs = np.quantile(obs, q_high) - np.quantile(obs, q_low)
    if iqr_obs == 0.0:
        return float("nan")
    iqr_pred = np.quantile(pred, q_high) - np.quantile(pred, q_low)
    return float(iqr_pred / iqr_obs)


def regression_slope(obs: np.ndarray, pred: np.ndarray) -> float:
    """OLS slope of pred ~ obs (obs as regressor)."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    if np.var(obs) == 0.0:
        return float("nan")
    return float(np.cov(obs, pred)[0, 1] / np.var(obs))


def quantile_bias(obs: np.ndarray, pred: np.ndarray, quantile: float) -> float:
    """Mean prediction bias at rows where obs >= quantile(obs, q)."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    threshold = np.quantile(obs, quantile)
    mask = obs >= threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(pred[mask] - obs[mask]))


def max_bias(obs: np.ndarray, pred: np.ndarray, top_k: int = 1) -> float:
    """Mean error at the top_k observed maxima."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    idx = np.argsort(obs)[-top_k:]
    return float(np.mean(pred[idx] - obs[idx]))


# ---------------------------------------------------------------------------
# Event / episode metrics
# ---------------------------------------------------------------------------

def contingency(obs_binary: np.ndarray, pred_binary: np.ndarray) -> dict[str, int]:
    obs_b = np.asarray(obs_binary, bool)
    pred_b = np.asarray(pred_binary, bool)
    tp = int(np.sum(obs_b & pred_b))
    fp = int(np.sum(~obs_b & pred_b))
    fn = int(np.sum(obs_b & ~pred_b))
    tn = int(np.sum(~obs_b & ~pred_b))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def pod(obs_binary: np.ndarray, pred_binary: np.ndarray) -> float:
    ct = contingency(obs_binary, pred_binary)
    denom = ct["TP"] + ct["FN"]
    return float(ct["TP"] / denom) if denom > 0 else float("nan")


def far(obs_binary: np.ndarray, pred_binary: np.ndarray) -> float:
    ct = contingency(obs_binary, pred_binary)
    denom = ct["TP"] + ct["FP"]
    return float(ct["FP"] / denom) if denom > 0 else float("nan")


def csi(obs_binary: np.ndarray, pred_binary: np.ndarray) -> float:
    ct = contingency(obs_binary, pred_binary)
    denom = ct["TP"] + ct["FP"] + ct["FN"]
    return float(ct["TP"] / denom) if denom > 0 else float("nan")


def precision(obs_binary: np.ndarray, pred_binary: np.ndarray) -> float:
    ct = contingency(obs_binary, pred_binary)
    denom = ct["TP"] + ct["FP"]
    return float(ct["TP"] / denom) if denom > 0 else float("nan")


def recall(obs_binary: np.ndarray, pred_binary: np.ndarray) -> float:
    return pod(obs_binary, pred_binary)


def f1(obs_binary: np.ndarray, pred_binary: np.ndarray) -> float:
    p = precision(obs_binary, pred_binary)
    r = recall(obs_binary, pred_binary)
    if np.isnan(p) or np.isnan(r) or (p + r) == 0:
        return float("nan")
    return float(2 * p * r / (p + r))


def accumulated_exceedance(obs: np.ndarray, pred: np.ndarray, threshold: float) -> float:
    """Sum of predicted values above threshold minus sum of observed above threshold."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    return float(np.sum(np.maximum(pred - threshold, 0)) - np.sum(np.maximum(obs - threshold, 0)))


# ---------------------------------------------------------------------------
# Temporal episode metrics (operate on event time series)
# ---------------------------------------------------------------------------

def temporal_error_of_maximum(
    obs_times: np.ndarray, obs_values: np.ndarray,
    pred_times: np.ndarray, pred_values: np.ndarray,
) -> float:
    """
    Signed difference in hours: time of predicted max minus time of observed max.
    obs_times and pred_times are numeric (hours since epoch) or datetime64.
    """
    obs_times = np.asarray(obs_times)
    pred_times = np.asarray(pred_times)
    obs_values = np.asarray(obs_values, float)
    pred_values = np.asarray(pred_values, float)

    obs_peak_idx = int(np.argmax(obs_values))
    pred_peak_idx = int(np.argmax(pred_values))

    obs_t = obs_times[obs_peak_idx]
    pred_t = pred_times[pred_peak_idx]

    if np.issubdtype(obs_times.dtype, np.datetime64):
        delta = (pred_t - obs_t) / np.timedelta64(1, "h")
    else:
        delta = float(pred_t) - float(obs_t)
    return float(delta)


def duration_error(
    obs_binary: np.ndarray, pred_binary: np.ndarray,
) -> float:
    """Signed difference in number of time steps above threshold (pred - obs)."""
    obs_b = np.asarray(obs_binary, bool)
    pred_b = np.asarray(pred_binary, bool)
    return float(pred_b.sum() - obs_b.sum())
