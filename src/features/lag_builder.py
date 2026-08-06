"""
Lag feature builder with strict observational availability enforcement.

For each (station, pollutant, decision_time) triple, builds lag features
using only observations whose observation_available_time <= decision_time.
Rows violating this constraint are silently excluded — they represent data
that did not yet exist from the decision-maker's perspective.

Relationship to the timing model:
  decision_time == available_time of the forecast row being featurised.
  observation_available_time is when each historical observation was published.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_lag_features(
    obs_df: pd.DataFrame,
    station_id: str,
    pollutant: str,
    decision_time: pd.Timestamp,
    obs_time_col: str = "observation_time",
    obs_avail_col: str = "observation_available_time",
    obs_value_col: str = "obs_value",
    station_col: str = "station_id",
    pollutant_col: str = "pollutant",
    lag_hours: tuple[int, ...] = (1, 2, 3, 6, 12, 24),
    tolerance_minutes: int = 30,
) -> dict[str, float]:
    """
    Return lag features for one (station, pollutant, decision_time) triple.

    Parameters
    ----------
    obs_df:
        Long-format observation table. Required columns:
        station_id, pollutant, observation_time,
        observation_available_time, obs_value.
    station_id, pollutant:
        Group keys.
    decision_time:
        The forecast decision time (== available_time of the forecast row).
        No observation with observation_available_time > decision_time is used.
    lag_hours:
        Lag offsets in hours relative to the most recent eligible observation.
    tolerance_minutes:
        Matching tolerance around each target lag timestamp. Handles irregular
        sub-hourly observation schedules.

    Returns
    -------
    Dict mapping "lag_XXXh" -> float (NaN when no eligible observation found).
    """
    decision_ts = pd.Timestamp(decision_time)
    if decision_ts.tzinfo is None:
        decision_ts = decision_ts.tz_localize("UTC")

    eligible = obs_df[
        (obs_df[station_col] == station_id)
        & (obs_df[pollutant_col] == pollutant)
        & (pd.to_datetime(obs_df[obs_avail_col], utc=True) <= decision_ts)
    ].copy()

    eligible[obs_time_col] = pd.to_datetime(eligible[obs_time_col], utc=True)

    features: dict[str, float] = {}

    if eligible.empty:
        for lag in lag_hours:
            features[f"lag_{lag:03d}h"] = np.nan
        return features

    latest_time = eligible[obs_time_col].max()
    tol = pd.Timedelta(minutes=tolerance_minutes)

    for lag in lag_hours:
        target_time = latest_time - pd.Timedelta(hours=lag)
        window = eligible[
            (eligible[obs_time_col] >= target_time - tol)
            & (eligible[obs_time_col] <= target_time + tol)
        ]
        if len(window) > 0:
            features[f"lag_{lag:03d}h"] = float(window.iloc[-1][obs_value_col])
        else:
            features[f"lag_{lag:03d}h"] = np.nan

    return features


def check_lag_availability(
    obs_df: pd.DataFrame,
    decision_time: pd.Timestamp,
    obs_avail_col: str = "observation_available_time",
) -> bool:
    """
    Return True if obs_df contains any row that violates the availability
    constraint (observation_available_time > decision_time). Used in tests.
    """
    decision_ts = pd.Timestamp(decision_time)
    if decision_ts.tzinfo is None:
        decision_ts = decision_ts.tz_localize("UTC")
    avail = pd.to_datetime(obs_df[obs_avail_col], utc=True)
    return bool((avail > decision_ts).any())
