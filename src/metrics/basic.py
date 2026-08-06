"""
Provisional metric primitives for the Aurora PM station corrector.

Design contract
---------------
* All functions operate on 1-D array-likes of equal length.
* Callers must drop NaN before calling; behaviour on NaN inputs is undefined.
* Threshold-based metrics (tail_bias, event metrics) require a threshold that
  was fitted on TRAIN data only. The API enforces this: callers must pass the
  threshold explicitly — it is never derived internally from the input arrays.
  Use fit_tail_threshold() on train obs to obtain it.
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
    """Mean signed error: positive = over-prediction."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    return float(np.mean(pred - obs))


def skill(mae_model: float, mae_reference: float) -> float:
    """1 - MAE_model / MAE_reference. NaN when reference is zero."""
    if mae_reference == 0.0:
        return float("nan")
    return float(1.0 - mae_model / mae_reference)


# ---------------------------------------------------------------------------
# Variability
# ---------------------------------------------------------------------------

def variance_ratio(obs: np.ndarray, pred: np.ndarray) -> float:
    """var(pred) / var(obs). NaN when obs variance is zero."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    var_obs = np.var(obs)
    if var_obs == 0.0:
        return float("nan")
    return float(np.var(pred) / var_obs)


def iqr_ratio(
    obs: np.ndarray,
    pred: np.ndarray,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> float:
    """IQR(pred) / IQR(obs) at the given quantile bounds. NaN when obs IQR is zero."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    iqr_obs = np.quantile(obs, q_high) - np.quantile(obs, q_low)
    if iqr_obs == 0.0:
        return float("nan")
    return float((np.quantile(pred, q_high) - np.quantile(pred, q_low)) / iqr_obs)


def regression_slope(obs: np.ndarray, pred: np.ndarray) -> float:
    """
    OLS slope of pred ~ obs (obs as regressor, pred as response).

    Uses np.polyfit to avoid the ddof mismatch that arises when mixing
    np.cov (sample norm, ddof=1) and np.var (population norm, ddof=0).
    """
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    if np.var(obs) == 0.0:
        return float("nan")
    coeffs = np.polyfit(obs, pred, 1)
    return float(coeffs[0])


# ---------------------------------------------------------------------------
# Threshold fitting (train-only) and tail metrics
# ---------------------------------------------------------------------------

def fit_tail_threshold(obs_train: np.ndarray, quantile: float) -> float:
    """
    Compute a percentile threshold from TRAINING observations only.

    This is the only sanctioned way to derive Q90/Q95 thresholds for
    tail_bias and binary event classification. Never call np.quantile
    on test observations for this purpose.
    """
    return float(np.quantile(np.asarray(obs_train, float), quantile))


def tail_bias(
    obs: np.ndarray,
    pred: np.ndarray,
    threshold: float,
) -> float:
    """
    Mean prediction bias at rows where obs >= threshold.

    Parameters
    ----------
    obs, pred:
        Test observations and predictions.
    threshold:
        Pre-fitted threshold from train data (e.g. from fit_tail_threshold).
        Must NOT be computed from obs or pred inside this function.
    """
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
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

def contingency(
    obs_binary: np.ndarray, pred_binary: np.ndarray
) -> dict[str, int]:
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


def accumulated_exceedance(
    obs: np.ndarray, pred: np.ndarray, threshold: float
) -> float:
    """Sum of predicted exceedance minus sum of observed exceedance above threshold."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    return float(
        np.sum(np.maximum(pred - threshold, 0))
        - np.sum(np.maximum(obs - threshold, 0))
    )


def exceedance_count_error(
    obs_binary: np.ndarray, pred_binary: np.ndarray
) -> float:
    """
    Signed difference in total number of time steps above threshold:
    predicted count - observed count.

    Note: this is a count of steps, not a duration in hours. To convert to
    hours, multiply by the time-step length. For episode-level duration
    (contiguous segments), use episode_duration_error().
    """
    obs_b = np.asarray(obs_binary, bool)
    pred_b = np.asarray(pred_binary, bool)
    return float(pred_b.sum() - obs_b.sum())


def episode_duration_error(
    obs_binary: np.ndarray,
    pred_binary: np.ndarray,
) -> float:
    """
    Signed difference in total episode duration (steps above threshold).

    An episode is a maximal contiguous run of True values. This function
    sums all episode lengths in predicted and observed sequences and returns
    predicted_total - observed_total (in time steps).

    Numerically identical to exceedance_count_error for single-episode series;
    differs when multiple episodes are present because it operates on
    contiguous-run structure rather than raw counts.
    """
    def total_episode_length(binary: np.ndarray) -> int:
        total = 0
        in_episode = False
        for val in binary:
            if val:
                total += 1
                in_episode = True
            else:
                in_episode = False
        return total

    obs_b = np.asarray(obs_binary, bool)
    pred_b = np.asarray(pred_binary, bool)
    return float(total_episode_length(pred_b) - total_episode_length(obs_b))


# ---------------------------------------------------------------------------
# Temporal episode metrics (operate on event time series)
# ---------------------------------------------------------------------------

def temporal_error_of_maximum(
    obs_times: np.ndarray,
    obs_values: np.ndarray,
    pred_times: np.ndarray,
    pred_values: np.ndarray,
) -> float:
    """
    Signed difference: time of predicted maximum minus time of observed maximum,
    in hours if times are datetime64, otherwise in raw numeric units.
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
        return float((pred_t - obs_t) / np.timedelta64(1, "h"))
    return float(pred_t) - float(obs_t)
