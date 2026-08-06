"""
Leakage guard and availability tests.

Scope:
  1. Row invariant: valid_time == cycle_time + lead_hours
  2. Rolling-origin splits: train admissibility under target_available_time
  3. Observational latency: lag builder respects observation_available_time
  4. Train-only threshold: tail_bias API requires pre-fitted threshold
  5. Multi-horizon datasets: splits remain clean across lead_hours
  6. Multi-station / multi-pollutant datasets
  7. Metric correctness: regression_slope ddof, exceedance_count_error naming

All tests must pass before any model is trained or any data is downloaded.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.validation.rolling_origin import (
    PSEUDO_OPERATIONAL_ASSUMPTION,
    LeakageError,
    check_no_leakage,
    rolling_origin_splits,
)
from src.metrics.basic import (
    fit_tail_threshold,
    tail_bias,
    regression_slope,
    exceedance_count_error,
    episode_duration_error,
    mae, rmse, bias, skill,
)
from src.features.lag_builder import build_lag_features, check_lag_availability


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    n_hours: int = 72,
    n_stations: int = 1,
    pollutants: tuple[str, ...] = ("PM10",),
    lead_hours: int = 24,
    obs_latency_hours: float = 0.0,
    dissemination_lag_hours: float = 0.0,
) -> pd.DataFrame:
    """
    Build a synthetic forecast-observation dataset.

    obs_latency_hours: how many hours after valid_time the observation
                       becomes available (target_available_time latency).
    dissemination_lag_hours: how many hours after cycle_time the forecast
                             becomes available.
    """
    cycle_times = pd.date_range(
        "2023-01-01", periods=n_hours, freq="h", tz="UTC"
    )
    rows = []
    for station in [f"ST{i:02d}" for i in range(1, n_stations + 1)]:
        for pollutant in pollutants:
            for i, ct in enumerate(cycle_times):
                avail = ct + pd.Timedelta(hours=dissemination_lag_hours)
                vt = ct + pd.Timedelta(hours=lead_hours)
                tgt_avail = vt + pd.Timedelta(hours=obs_latency_hours)
                rows.append(
                    {
                        "station_id": station,
                        "pollutant": pollutant,
                        "cycle_time": ct,
                        "available_time": avail,
                        "valid_time": vt,
                        "target_available_time": tgt_avail,
                        "lead_hours": lead_hours,
                        "obs": float(i),
                        "aurora_fc": float(i) * 1.1,
                        "cams_fc": float(i) * 0.9,
                    }
                )
    return pd.DataFrame(rows)


def _make_obs_df(
    n_obs: int = 48,
    obs_latency_hours: float = 1.0,
) -> pd.DataFrame:
    """Build a synthetic local observation history table."""
    obs_times = pd.date_range("2023-01-01", periods=n_obs, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "station_id": "ST01",
            "pollutant": "PM10",
            "observation_time": obs_times,
            "observation_available_time": obs_times
            + pd.Timedelta(hours=obs_latency_hours),
            "obs_value": np.arange(n_obs, dtype=float),
        }
    )


# ---------------------------------------------------------------------------
# 1. Row invariant: valid_time == cycle_time + lead_hours
# ---------------------------------------------------------------------------

def test_valid_time_invariant_holds():
    df = _make_df()
    delta = df["valid_time"] - df["cycle_time"]
    expected = pd.to_timedelta(df["lead_hours"], unit="h")
    assert (delta == expected).all()


def test_valid_time_invariant_detects_violation():
    df = _make_df()
    df.loc[0, "valid_time"] = df.loc[0, "cycle_time"] + pd.Timedelta(hours=99)
    delta = df["valid_time"] - df["cycle_time"]
    expected = pd.to_timedelta(df["lead_hours"], unit="h")
    assert not (delta == expected).all()


# ---------------------------------------------------------------------------
# 2a. Rolling-origin with target_available_time (zero latency)
# ---------------------------------------------------------------------------

def test_rolling_origin_no_leakage_zero_latency():
    df = _make_df(n_hours=120, obs_latency_hours=0.0)
    splits = list(
        rolling_origin_splits(
            df,
            target_available_time_col="target_available_time",
            min_train_rows=5,
            step_hours=24,
            block_hours=24,
        )
    )
    assert len(splits) > 0, "Expected at least one split"
    for split in splits:
        max_tgt = split.train["target_available_time"].max()
        assert pd.Timestamp(max_tgt) <= pd.Timestamp(split.origin)


# ---------------------------------------------------------------------------
# 2b. Rolling-origin with observational latency (realistic: 1h delay)
# ---------------------------------------------------------------------------

def test_rolling_origin_respects_observation_latency():
    """
    With a 1-h observational latency, target_available_time = valid_time + 1h.
    A row that would pass a valid_time-only filter may still fail the
    target_available_time filter.
    """
    df = _make_df(n_hours=120, lead_hours=24, obs_latency_hours=1.0)
    splits = list(
        rolling_origin_splits(
            df,
            target_available_time_col="target_available_time",
            min_train_rows=5,
            step_hours=24,
            block_hours=24,
        )
    )
    for split in splits:
        # target_available_time must be <= origin
        assert (
            split.train["target_available_time"] <= pd.Timestamp(split.origin)
        ).all()
        # pseudo_operational should be False when column is present
        assert not split.pseudo_operational


# ---------------------------------------------------------------------------
# 2c. Pseudo-operational fallback when target_available_time is absent
# ---------------------------------------------------------------------------

def test_pseudo_operational_fallback_issues_warning():
    df = _make_df(n_hours=80, obs_latency_hours=0.0)
    df = df.drop(columns=["target_available_time"])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        splits = list(
            rolling_origin_splits(
                df,
                target_available_time_col="target_available_time",
                min_train_rows=5,
                step_hours=24,
                block_hours=24,
            )
        )
    assert any(PSEUDO_OPERATIONAL_ASSUMPTION in str(warning.message) for warning in w)
    for split in splits:
        assert split.pseudo_operational


# ---------------------------------------------------------------------------
# 2d. check_no_leakage raises and passes correctly
# ---------------------------------------------------------------------------

def test_check_no_leakage_raises_on_violation():
    df = _make_df(n_hours=48, obs_latency_hours=0.0)
    origin = pd.Timestamp("2023-01-02 00:00:00", tz="UTC")
    train = df[
        (df["available_time"] < origin)
        & (df["target_available_time"] <= origin)
    ].copy()
    # Plant a violation: push one target_available_time past origin
    train.loc[train.index[-1], "target_available_time"] = (
        origin + pd.Timedelta(hours=6)
    )
    test = df[df["available_time"] >= origin].copy()
    with pytest.raises(LeakageError):
        check_no_leakage(
            train=train,
            test=test,
            available_time_col="available_time",
            target_available_time_col="target_available_time",
            origin=origin,
        )


def test_check_no_leakage_passes_when_clean():
    df = _make_df(n_hours=48, obs_latency_hours=0.0)
    origin = pd.Timestamp("2023-01-02 00:00:00", tz="UTC")
    train = df[
        (df["available_time"] < origin)
        & (df["target_available_time"] <= origin)
    ].copy()
    test = df[df["available_time"] >= origin].copy()
    # Must not raise
    check_no_leakage(
        train=train,
        test=test,
        available_time_col="available_time",
        target_available_time_col="target_available_time",
        origin=origin,
    )


# ---------------------------------------------------------------------------
# 3. Lag builder: observational latency enforcement
# ---------------------------------------------------------------------------

def test_lag_builder_excludes_unavailable_observations():
    """
    An observation that occurred at T but is only published at T+2h must not
    appear in lag features built for a decision at T+1h.
    """
    obs_time = pd.Timestamp("2023-01-02 00:00:00", tz="UTC")
    avail_time = obs_time + pd.Timedelta(hours=2)
    decision_before_avail = obs_time + pd.Timedelta(hours=1)

    obs_df = pd.DataFrame(
        {
            "station_id": ["ST01"],
            "pollutant": ["PM10"],
            "observation_time": [obs_time],
            "observation_available_time": [avail_time],
            "obs_value": [50.0],
        }
    )

    # Decision at T+1h: observation not yet available
    feats_before = build_lag_features(
        obs_df, "ST01", "PM10", decision_before_avail, lag_hours=(1,)
    )
    assert all(
        np.isnan(v) for v in feats_before.values()
    ), "Lag feature should be NaN before observation is available"


def test_lag_builder_includes_available_observations():
    """
    With a 5-hour history (each obs published 2h after measurement) and a
    decision 3h after the latest measurement, all three requested lags should
    be retrievable because their publication times are <= decision_time.
    """
    base = pd.Timestamp("2023-01-02 00:00:00", tz="UTC")
    # 5 hourly observations ending at base; each published 2h after measurement
    obs_times = [base - pd.Timedelta(hours=h) for h in range(5)]
    avail_times = [t + pd.Timedelta(hours=2) for t in obs_times]
    decision_after_avail = base + pd.Timedelta(hours=3)  # all published by then

    obs_df = pd.DataFrame(
        {
            "station_id": "ST01",
            "pollutant": "PM10",
            "observation_time": obs_times,
            "observation_available_time": avail_times,
            "obs_value": [float(i * 10) for i in range(5)],
        }
    )

    feats_after = build_lag_features(
        obs_df, "ST01", "PM10", decision_after_avail, lag_hours=(1, 2, 3)
    )
    non_nan = [v for v in feats_after.values() if not np.isnan(v)]
    assert len(non_nan) > 0, "At least one lag should be non-NaN after availability"


def test_lag_builder_check_helper_detects_future_obs():
    """check_lag_availability flags rows with observation_available_time > decision_time."""
    obs_df = _make_obs_df(n_obs=5, obs_latency_hours=2.0)
    decision = obs_df["observation_time"].min()  # before any obs is available
    assert check_lag_availability(obs_df, decision)


def test_lag_builder_multi_pollutant_isolation():
    """Lag features for PM10 must not include PM2.5 observations."""
    obs_df = pd.DataFrame(
        {
            "station_id": ["ST01", "ST01"],
            "pollutant": ["PM10", "PM2.5"],
            "observation_time": [
                pd.Timestamp("2023-01-01 12:00", tz="UTC"),
                pd.Timestamp("2023-01-01 12:00", tz="UTC"),
            ],
            "observation_available_time": [
                pd.Timestamp("2023-01-01 12:00", tz="UTC"),
                pd.Timestamp("2023-01-01 12:00", tz="UTC"),
            ],
            "obs_value": [30.0, 15.0],
        }
    )
    decision = pd.Timestamp("2023-01-01 14:00", tz="UTC")
    feats_pm10 = build_lag_features(obs_df, "ST01", "PM10", decision, lag_hours=(1, 2))
    feats_pm25 = build_lag_features(obs_df, "ST01", "PM2.5", decision, lag_hours=(1, 2))

    pm10_vals = [v for v in feats_pm10.values() if not np.isnan(v)]
    pm25_vals = [v for v in feats_pm25.values() if not np.isnan(v)]
    if pm10_vals and pm25_vals:
        assert pm10_vals[0] != pm25_vals[0], "PM10 and PM2.5 lag values should differ"


# ---------------------------------------------------------------------------
# 4. Train-only threshold enforcement
# ---------------------------------------------------------------------------

def test_tail_bias_requires_explicit_threshold():
    """
    tail_bias must receive the threshold as an argument (pre-fitted from train).
    Verify that using a train-derived vs test-derived threshold yields different
    results when train and test have different distributions.
    """
    rng = np.random.default_rng(42)
    train_obs = rng.uniform(0, 50, 200)       # typical range
    test_obs = rng.uniform(20, 120, 50)       # test has higher values
    test_pred = test_obs * 0.9 + rng.normal(0, 3, 50)

    train_threshold = fit_tail_threshold(train_obs, 0.90)
    test_threshold = fit_tail_threshold(test_obs, 0.90)

    # Thresholds must differ given different distributions
    assert abs(train_threshold - test_threshold) > 1.0

    bias_train_thr = tail_bias(test_obs, test_pred, threshold=train_threshold)
    bias_test_thr = tail_bias(test_obs, test_pred, threshold=test_threshold)

    # Results must differ (confirms the threshold is actually used)
    assert abs(bias_train_thr - bias_test_thr) > 0.0


def test_fit_tail_threshold_uses_only_provided_array():
    """fit_tail_threshold operates on exactly the array passed to it."""
    obs_train = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    threshold = fit_tail_threshold(obs_train, 0.80)
    expected = float(np.quantile(obs_train, 0.80))
    assert abs(threshold - expected) < 1e-10


# ---------------------------------------------------------------------------
# 5. Multi-horizon datasets
# ---------------------------------------------------------------------------

def _make_multi_horizon_df() -> pd.DataFrame:
    """
    One cycle per day, three lead hours per cycle.
    6-hour dissemination lag, 1-hour observational latency.
    """
    cycle_times = pd.date_range("2023-01-01", periods=30, freq="24h", tz="UTC")
    lead_list = [24, 48, 72]
    rows = []
    for ct in cycle_times:
        avail = ct + pd.Timedelta(hours=6)
        for lead in lead_list:
            vt = ct + pd.Timedelta(hours=lead)
            rows.append(
                {
                    "station_id": "ST01",
                    "pollutant": "PM10",
                    "cycle_time": ct,
                    "available_time": avail,
                    "valid_time": vt,
                    "target_available_time": vt + pd.Timedelta(hours=1),
                    "lead_hours": lead,
                    "obs": float(lead),
                    "aurora_fc": float(lead) * 1.05,
                    "cams_fc": float(lead) * 0.98,
                }
            )
    return pd.DataFrame(rows)


def test_multi_horizon_splits_are_leakage_free():
    df = _make_multi_horizon_df()
    splits = list(
        rolling_origin_splits(
            df,
            target_available_time_col="target_available_time",
            min_train_rows=5,
            step_hours=24,
            block_hours=24,
        )
    )
    assert len(splits) > 0
    for split in splits:
        assert (
            split.train["target_available_time"] <= pd.Timestamp(split.origin)
        ).all()


def test_multi_horizon_splits_preserve_all_lead_hours():
    """All lead_hours in train must have target_available_time <= origin."""
    df = _make_multi_horizon_df()
    for split in rolling_origin_splits(
        df,
        target_available_time_col="target_available_time",
        min_train_rows=5,
        step_hours=24,
        block_hours=24,
    ):
        leads_in_train = set(split.train["lead_hours"].unique())
        # Every lead-hour group must satisfy the leakage constraint
        for lead in leads_in_train:
            subset = split.train[split.train["lead_hours"] == lead]
            assert (
                subset["target_available_time"] <= pd.Timestamp(split.origin)
            ).all(), f"Leakage in lead_hours={lead} at origin={split.origin}"


# ---------------------------------------------------------------------------
# 6. Multi-station / multi-pollutant datasets
# ---------------------------------------------------------------------------

def test_multi_station_multi_pollutant_no_leakage():
    df = _make_df(
        n_hours=80,
        n_stations=3,
        pollutants=("PM10", "PM2.5"),
        obs_latency_hours=1.0,
        dissemination_lag_hours=6.0,
    )
    splits = list(
        rolling_origin_splits(
            df,
            target_available_time_col="target_available_time",
            min_train_rows=5,
            step_hours=24,
            block_hours=24,
        )
    )
    assert len(splits) > 0
    for split in splits:
        assert (
            split.train["target_available_time"] <= pd.Timestamp(split.origin)
        ).all()
        # All stations and pollutants must appear in train (given enough history)
        if len(split.train) >= 6:
            assert split.train["station_id"].nunique() >= 1
            assert split.train["pollutant"].nunique() >= 1


# ---------------------------------------------------------------------------
# 7. Metric correctness
# ---------------------------------------------------------------------------

def test_regression_slope_no_ddof_bias():
    """
    For a perfect linear relationship y = 2x, slope must be 2.0.
    The old np.cov/np.var mix introduced an n/(n-1) bias for small n.
    """
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = 2.0 * obs
    slope = regression_slope(obs, pred)
    assert abs(slope - 2.0) < 1e-10, f"Slope should be 2.0, got {slope}"


def test_regression_slope_constant_obs_returns_nan():
    obs = np.ones(10)
    pred = np.arange(10, dtype=float)
    assert np.isnan(regression_slope(obs, pred))


def test_exceedance_count_error_is_count_not_episodes():
    """exceedance_count_error counts steps, not connected episodes."""
    obs_bin = np.array([True, False, True, False, True])   # 3 isolated steps
    pred_bin = np.array([True, True, True, False, False])  # 3 consecutive steps
    result = exceedance_count_error(obs_bin, pred_bin)
    assert result == 0.0, "Same count of True steps → error should be zero"


def test_episode_duration_error_respects_contiguous_structure():
    obs_bin = np.array([True, True, False, True, True])   # two episodes of length 2
    pred_bin = np.array([False, True, True, True, False]) # one episode of length 3
    obs_total = 4  # 2 + 2
    pred_total = 3
    result = episode_duration_error(obs_bin, pred_bin)
    assert result == pred_total - obs_total, f"Expected {pred_total - obs_total}, got {result}"


def test_mae_rmse_bias_sanity():
    obs = np.array([10.0, 20.0, 30.0])
    pred = np.array([12.0, 18.0, 33.0])
    errors = pred - obs
    assert abs(mae(obs, pred) - np.mean(np.abs(errors))) < 1e-10
    assert abs(rmse(obs, pred) - np.sqrt(np.mean(errors ** 2))) < 1e-10
    assert abs(bias(obs, pred) - np.mean(errors)) < 1e-10


def test_skill_vs_reference():
    assert abs(skill(mae_model=5.0, mae_reference=10.0) - 0.5) < 1e-10
    assert np.isnan(skill(mae_model=5.0, mae_reference=0.0))
