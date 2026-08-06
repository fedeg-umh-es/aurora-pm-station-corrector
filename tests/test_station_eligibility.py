"""
Tests for the station eligibility pipeline (P1).

Uses a synthetic series with known gaps, duplicates, and negative values
to verify that all metrics are computed correctly and that no silent
imputation occurs.
"""
from __future__ import annotations

import io
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.p1_01_compute_station_eligibility import (
    analyse_station_pollutant_year,
    compute_longest_gap_days,
    compute_station_summary,
    expected_hourly_records,
    regional_decision,
)

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "station_eligibility.yaml"


@pytest.fixture()
def cfg():
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hourly_series(year: int, coverage: float = 1.0, neg_count: int = 0) -> pd.DataFrame:
    """
    Build a synthetic hourly PM10 series for one year with controlled coverage.
    """
    exp = expected_hourly_records(year)
    idx = pd.date_range(f"{year}-01-01", periods=exp, freq="h", tz="UTC")
    values = np.random.default_rng(42).uniform(5, 80, exp).astype(float)

    # Introduce missing values (evenly distributed)
    n_missing = int(exp * (1 - coverage))
    missing_idx = np.linspace(0, exp - 1, n_missing, dtype=int)
    values[missing_idx] = np.nan

    # Introduce some negative values
    if neg_count > 0:
        values[:neg_count] = -1.0

    df = pd.DataFrame({"value": values}, index=idx)
    df.index.name = "datetime"
    return df


def _series_with_gap(year: int, gap_days: int, gap_start_day: int = 60) -> pd.DataFrame:
    """Build a series with a single contiguous gap of gap_days starting at gap_start_day."""
    exp = expected_hourly_records(year)
    idx = pd.date_range(f"{year}-01-01", periods=exp, freq="h", tz="UTC")
    values = np.ones(exp, dtype=float) * 20.0

    start = gap_start_day * 24
    end = start + gap_days * 24
    values[start:end] = np.nan

    return pd.DataFrame({"value": values}, index=idx)


# ---------------------------------------------------------------------------
# expected_hourly_records
# ---------------------------------------------------------------------------

def test_expected_records_non_leap():
    assert expected_hourly_records(2023) == 8760


def test_expected_records_leap():
    assert expected_hourly_records(2024) == 8784


# ---------------------------------------------------------------------------
# compute_longest_gap_days
# ---------------------------------------------------------------------------

def test_longest_gap_no_missing():
    s = pd.Series([1.0, 2.0, 3.0])
    assert compute_longest_gap_days(s) == 0.0


def test_longest_gap_known_gap():
    # 48 NaN → 2 days
    values = [1.0] * 24 + [np.nan] * 48 + [1.0] * 24
    s = pd.Series(values)
    assert abs(compute_longest_gap_days(s) - 2.0) < 1e-9


def test_longest_gap_multiple_gaps():
    # Gap of 24h and gap of 72h → longest is 3 days
    values = [np.nan] * 24 + [1.0] * 10 + [np.nan] * 72 + [1.0] * 10
    s = pd.Series(values)
    assert abs(compute_longest_gap_days(s) - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# analyse_station_pollutant_year
# ---------------------------------------------------------------------------

def test_full_coverage_year(cfg):
    df = _hourly_series(2023, coverage=1.0)
    result = analyse_station_pollutant_year(df, "ST01", "PM10", 2023, cfg, [])
    assert result["coverage"] == 1.0
    assert result["eligible_pollutant_year"] is True
    assert result["exclusion_reason"] == ""


def test_low_coverage_year(cfg):
    df = _hourly_series(2023, coverage=0.80)
    result = analyse_station_pollutant_year(df, "ST01", "PM10", 2023, cfg, [])
    assert result["coverage"] < 0.90
    assert result["eligible_pollutant_year"] is False
    assert "coverage" in result["exclusion_reason"]


def test_large_gap_triggers_ineligibility(cfg):
    df = _series_with_gap(2023, gap_days=35)
    result = analyse_station_pollutant_year(df, "ST01", "PM10", 2023, cfg, [])
    assert result["longest_missing_gap_days"] > 30
    assert result["eligible_pollutant_year"] is False
    assert "gap" in result["exclusion_reason"]


def test_negative_values_are_counted_not_removed(cfg):
    df = _hourly_series(2023, neg_count=10)
    result = analyse_station_pollutant_year(df, "ST01", "PM10", 2023, cfg, [])
    assert result["negative_value_count"] == 10
    # Coverage should still count negatives as valid records (they are present)
    # The protocol says: do NOT silently convert negatives to missing
    assert result["valid_records"] > 0


def test_duplicate_timestamps_are_counted(cfg):
    df = _hourly_series(2023)
    # Inject duplicates: repeat the first 5 timestamps
    dup = df.iloc[:5].copy()
    df_with_dups = pd.concat([df, dup])
    df_with_dups.index.name = "datetime"
    result = analyse_station_pollutant_year(df_with_dups, "ST01", "PM10", 2023, cfg, [])
    assert result["duplicate_timestamp_count"] == 5


def test_no_imputation_zero_coverage_returns_zeros(cfg):
    """A completely empty series should give coverage=0 and ineligible, not imputed."""
    df = pd.DataFrame({"value": [np.nan] * 8760},
                      index=pd.date_range("2023-01-01", periods=8760, freq="h", tz="UTC"))
    result = analyse_station_pollutant_year(df, "ST01", "PM10", 2023, cfg, [])
    assert result["coverage"] == 0.0
    assert result["eligible_pollutant_year"] is False


# ---------------------------------------------------------------------------
# regional_decision
# ---------------------------------------------------------------------------

def test_regional_strong(cfg):
    assert regional_decision(25, cfg) == "REGIONAL_STRONG"


def test_regional_bounded(cfg):
    assert regional_decision(15, cfg) == "REGIONAL_BOUNDED"


def test_pilot_only(cfg):
    assert regional_decision(7, cfg) == "PILOT_ONLY"


def test_stop_or_expand(cfg):
    assert regional_decision(3, cfg) == "STOP_OR_EXPAND_REGION"


def test_boundary_exactly_20(cfg):
    assert regional_decision(20, cfg) == "REGIONAL_STRONG"


def test_boundary_exactly_10(cfg):
    assert regional_decision(10, cfg) == "REGIONAL_BOUNDED"


def test_boundary_exactly_5(cfg):
    assert regional_decision(5, cfg) == "PILOT_ONLY"
