"""Deterministic, amplitude and threshold-event verification metrics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _arrays(y_true: Iterable[float], y_pred: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    mask = np.isfinite(truth) & np.isfinite(pred)
    if not mask.any():
        raise ValueError("No finite paired observations")
    return truth[mask], pred[mask]


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    truth, pred = _arrays(y_true, y_pred)
    error = pred - truth
    obs_variance = float(np.var(truth, ddof=1)) if truth.size > 1 else np.nan
    pred_variance = float(np.var(pred, ddof=1)) if pred.size > 1 else np.nan
    variance_ratio = pred_variance / obs_variance if obs_variance > 0 else np.nan
    obs_iqr = float(np.quantile(truth, 0.95) - np.quantile(truth, 0.05))
    pred_iqr = float(np.quantile(pred, 0.95) - np.quantile(pred, 0.05))
    amplitude_ratio = pred_iqr / obs_iqr if obs_iqr > 0 else np.nan

    result: dict[str, float] = {
        "n": float(truth.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "variance_ratio": float(variance_ratio),
        "amplitude_ratio_q05_q95": float(amplitude_ratio),
        "peak_bias": float(np.max(pred) - np.max(truth)),
    }
    for quantile in (0.90, 0.95):
        threshold = float(np.quantile(truth, quantile))
        tail = truth >= threshold
        result[f"tail_bias_q{int(quantile * 100)}"] = (
            float(np.mean(pred[tail] - truth[tail])) if tail.any() else np.nan
        )
    return result


def event_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    threshold: float | Iterable[float],
) -> dict[str, float]:
    truth, pred = _arrays(y_true, y_pred)
    if np.isscalar(threshold):
        thresholds = np.full(truth.shape, float(threshold))
    else:
        thresholds = np.asarray(list(threshold), dtype=float)
        if thresholds.shape != truth.shape:
            raise ValueError("threshold must be scalar or match y_true shape")

    observed = truth >= thresholds
    forecast = pred >= thresholds
    tp = int(np.sum(observed & forecast))
    fp = int(np.sum(~observed & forecast))
    fn = int(np.sum(observed & ~forecast))
    tn = int(np.sum(~observed & ~forecast))
    pod = tp / (tp + fn) if tp + fn else np.nan
    far = fp / (tp + fp) if tp + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * precision * pod / (precision + pod) if precision + pod > 0 else np.nan

    return {
        "event_threshold_mean": float(np.mean(thresholds)),
        "observed_event_rate": float(np.mean(observed)),
        "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
        "pod": float(pod), "far": float(far), "precision": float(precision), "f1": float(f1),
    }


def skill_score(model_loss: float, reference_loss: float) -> float:
    """Return 1 - model/reference; positive values indicate improvement."""
    if not np.isfinite(reference_loss) or reference_loss <= 0:
        return np.nan
    return float(1.0 - model_loss / reference_loss)
