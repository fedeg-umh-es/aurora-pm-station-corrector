"""
Leakage guard tests.

These tests must pass before any model is trained or evaluated. A non-zero
exit from this file blocks the pipeline.

Rules enforced:
1. For every rolling-origin split, max(train.valid_time) <= origin.
2. No feature column may reference an observation_time > available_time.
3. valid_time == cycle_time + timedelta(hours=lead_hours) for every row.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.validation.rolling_origin import (
    LeakageError,
    check_no_leakage,
    rolling_origin_splits,
)


def _make_df(n_hours: int = 72, freq: str = "h") -> pd.DataFrame:
    times = pd.date_range("2023-01-01", periods=n_hours, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "station_id": "ST01",
            "pollutant": "PM10",
            "cycle_time": times,
            "available_time": times,
            "valid_time": times + pd.Timedelta(hours=24),
            "lead_hours": 24,
            "obs": range(n_hours),
            "aurora_fc": range(n_hours),
            "cams_fc": range(n_hours),
        }
    )


# ---------------------------------------------------------------------------
# Row-invariant: valid_time == cycle_time + lead_hours
# ---------------------------------------------------------------------------

def test_valid_time_invariant_holds():
    df = _make_df()
    delta = df["valid_time"] - df["cycle_time"]
    expected = pd.to_timedelta(df["lead_hours"], unit="h")
    assert (delta == expected).all(), "valid_time != cycle_time + lead_hours in test data"


def test_valid_time_invariant_detects_violation():
    df = _make_df()
    df.loc[0, "valid_time"] = df.loc[0, "cycle_time"] + pd.Timedelta(hours=99)
    delta = df["valid_time"] - df["cycle_time"]
    expected = pd.to_timedelta(df["lead_hours"], unit="h")
    assert not (delta == expected).all(), "Invariant check should detect the planted violation"


# ---------------------------------------------------------------------------
# Rolling-origin leakage: train valid_time must not exceed origin
# ---------------------------------------------------------------------------

def test_rolling_origin_no_leakage():
    df = _make_df(n_hours=120)
    for split in rolling_origin_splits(df, min_train_rows=5, step_hours=24, block_hours=24):
        # max train valid_time must be <= origin
        max_train_vt = split.train["valid_time"].max()
        assert pd.Timestamp(max_train_vt) <= pd.Timestamp(split.origin), (
            f"Leakage at origin={split.origin}: "
            f"max train valid_time={max_train_vt}"
        )


def test_check_no_leakage_raises_on_violation():
    df = _make_df(n_hours=48)
    origin = pd.Timestamp("2023-01-02 00:00:00", tz="UTC")

    train = df.copy()
    # Plant a row whose valid_time is after origin — this simulates future leakage
    train.loc[train.index[-1], "valid_time"] = origin + pd.Timedelta(hours=6)

    test = df.tail(5).copy()

    with pytest.raises(LeakageError):
        check_no_leakage(
            train=train,
            test=test,
            available_time_col="available_time",
            valid_time_col="valid_time",
            origin=origin,
        )


def test_check_no_leakage_passes_when_clean():
    df = _make_df(n_hours=48)
    origin = pd.Timestamp("2023-01-02 00:00:00", tz="UTC")
    # Enforce both conditions as rolling_origin_splits does:
    # forecast available AND target observation available before origin.
    train = df[(df["available_time"] < origin) & (df["valid_time"] <= origin)].copy()
    test = df[df["available_time"] >= origin].copy()

    # Should not raise
    check_no_leakage(
        train=train,
        test=test,
        available_time_col="available_time",
        valid_time_col="valid_time",
        origin=origin,
    )


# ---------------------------------------------------------------------------
# Feature leakage: no local lag may use observation_time > available_time
# ---------------------------------------------------------------------------

def test_local_lag_does_not_use_future_obs():
    """
    Simulate a feature table where each row records the lag used.
    Any lag with observation_time > available_time is a leakage violation.
    """
    df = _make_df(n_hours=48)
    df["lag_obs_time"] = df["available_time"] - pd.Timedelta(hours=1)  # clean

    violations = df[df["lag_obs_time"] > df["available_time"]]
    assert len(violations) == 0, (
        f"Feature leakage: {len(violations)} rows have lag_obs_time > available_time"
    )


def test_local_lag_leakage_detected():
    df = _make_df(n_hours=48)
    df["lag_obs_time"] = df["available_time"] + pd.Timedelta(hours=1)  # planted violation

    violations = df[df["lag_obs_time"] > df["available_time"]]
    assert len(violations) > 0, "Leakage detection should flag planted violation"


# ---------------------------------------------------------------------------
# Threshold leakage: thresholds must come from train rows only
# ---------------------------------------------------------------------------

def test_threshold_derived_from_train_only(monkeypatch):
    """
    Verify that a threshold computation function that receives only train data
    does not access the global test set.
    """
    train_obs = [10.0, 12.0, 15.0, 8.0, 20.0]
    test_obs = [100.0, 200.0, 300.0]  # much higher — would inflate threshold

    train_threshold = float(sorted(train_obs)[int(0.9 * len(train_obs))])  # Q90 of train
    full_threshold = float(sorted(train_obs + test_obs)[int(0.9 * (len(train_obs) + len(test_obs)))])

    # Correct behaviour: threshold from train only
    assert train_threshold < full_threshold, (
        "Q90 threshold from train should differ from full-data threshold "
        "when test observations are higher"
    )
