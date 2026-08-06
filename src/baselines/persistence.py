"""
Persistence baseline: predict the last observed value available at available_time.

This is a stateless baseline — no fitting required. The function operates on
a DataFrame that already contains the correct temporal structure (one row per
station × pollutant × cycle_time × lead_hours).
"""
from __future__ import annotations

import pandas as pd


def persistence_forecast(
    df: pd.DataFrame,
    obs_col: str = "obs",
    available_time_col: str = "available_time",
    station_col: str = "station_id",
    pollutant_col: str = "pollutant",
    last_obs_col: str = "last_obs_at_available_time",
) -> pd.Series:
    """
    Return a Series of persistence forecasts aligned to df.index.

    The persistence forecast for each row is the last observed value with
    observation_time <= available_time, grouped by station and pollutant.

    Parameters
    ----------
    df:
        Must contain `last_obs_at_available_time` (pre-computed from local
        lags) or `obs` shifted appropriately. Callers must ensure no future
        observations are included.

    Returns
    -------
    pd.Series with the same index as df.
    """
    if last_obs_col in df.columns:
        return df[last_obs_col].copy()

    # Fallback: use same-row obs shifted within the group (for dummy data only).
    # In production, callers must supply last_obs_at_available_time from the
    # local lag table; this fallback is intentionally conservative.
    raise ValueError(
        f"Column '{last_obs_col}' not found. "
        "Provide the last observed value at available_time as a separate column. "
        "Never derive it from the test observation (that would be leakage)."
    )


def add_persistence_column(
    df: pd.DataFrame,
    obs_col: str = "obs",
    available_time_col: str = "available_time",
    valid_time_col: str = "valid_time",
    station_col: str = "station_id",
    pollutant_col: str = "pollutant",
    lead_hours: int = 1,
    out_col: str = "persistence_fc",
) -> pd.DataFrame:
    """
    Compute persistence by shifting the obs column back by `lead_hours` rows
    within each (station, pollutant) group, assuming uniform hourly spacing.

    Only valid for aligned time series without gaps. For production use with
    real data use `persistence_forecast` with a precomputed last_obs column.

    This function is provided for the smoke-test pipeline where the series is
    already sorted and gap-free.
    """
    df = df.copy().sort_values([station_col, pollutant_col, valid_time_col])
    df[out_col] = (
        df.groupby([station_col, pollutant_col])[obs_col]
        .shift(lead_hours)
    )
    return df
