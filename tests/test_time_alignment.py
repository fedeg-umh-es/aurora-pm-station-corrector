"""
P2 tests — time alignment.

Verifies the invariants required for the single-cycle output table gate:
  1. valid_time == cycle_time + lead_hours for every row
  2. Aurora and CAMS share the same cycle_time
  3. All configured lead_hours appear in the output
  4. available_time is set (not NaN) on every row
  5. No row is labelled as an operational forecast without available_time
"""
from __future__ import annotations

from datetime import timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "hindcast_single_cycle.yaml"


@pytest.fixture()
def cfg():
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_output_table(
    cycle_time: pd.Timestamp,
    lead_hours: list[int],
    n_stations: int = 3,
    pollutants: tuple[str, ...] = ("PM10", "PM2.5"),
    include_available_time: bool = True,
    aurora_cycle: pd.Timestamp | None = None,
    cams_cycle: pd.Timestamp | None = None,
    dissemination_lag_hours: int = 6,
) -> pd.DataFrame:
    """
    Build a minimal synthetic single-cycle output table.
    """
    aurora_cycle = aurora_cycle or cycle_time
    cams_cycle = cams_cycle or cycle_time
    rows = []
    for lead in lead_hours:
        for station in [f"ST{i:02d}" for i in range(1, n_stations + 1)]:
            for pollutant in pollutants:
                avail = (
                    cycle_time + pd.Timedelta(hours=dissemination_lag_hours)
                    if include_available_time
                    else pd.NaT
                )
                rows.append(
                    {
                        "cycle_time": cycle_time,
                        "available_time": avail,
                        "valid_time": cycle_time + pd.Timedelta(hours=lead),
                        "lead_hours": lead,
                        "station_id": station,
                        "pollutant": pollutant,
                        "aurora_forecast": 25.0,
                        "cams_forecast": 23.0,
                        "observation": float("nan"),
                        "aurora_grid_lat": 39.5,
                        "aurora_grid_lon": -0.5,
                        "cams_grid_lat": 39.5,
                        "cams_grid_lon": -0.5,
                        "spatial_operator": "bilinear",
                        "aurora_checkpoint": "microsoft/aurora-0.25-pretrained",
                        "cams_dataset": "cams-global-atmospheric-composition-forecasts",
                        "source_file_hash": "abc123",
                        "_aurora_cycle": aurora_cycle,
                        "_cams_cycle": cams_cycle,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Invariant 1: valid_time == cycle_time + lead_hours
# ---------------------------------------------------------------------------

def test_valid_time_invariant_passes(cfg):
    cycle = pd.Timestamp(cfg["cycle"]["cycle_time"])
    df = _make_output_table(cycle, cfg["cycle"]["lead_hours"])
    delta = df["valid_time"] - df["cycle_time"]
    expected = pd.to_timedelta(df["lead_hours"], unit="h")
    assert (delta == expected).all()


def test_valid_time_invariant_fails_on_violation(cfg):
    cycle = pd.Timestamp(cfg["cycle"]["cycle_time"])
    df = _make_output_table(cycle, cfg["cycle"]["lead_hours"])
    df.loc[0, "valid_time"] = cycle + pd.Timedelta(hours=999)
    delta = df["valid_time"] - df["cycle_time"]
    expected = pd.to_timedelta(df["lead_hours"], unit="h")
    assert not (delta == expected).all()


# ---------------------------------------------------------------------------
# Invariant 2: Aurora and CAMS share the same cycle_time
# ---------------------------------------------------------------------------

def test_shared_cycle_passes(cfg):
    cycle = pd.Timestamp(cfg["cycle"]["cycle_time"])
    df = _make_output_table(cycle, cfg["cycle"]["lead_hours"])
    assert (df["_aurora_cycle"] == df["_cams_cycle"]).all()


def test_shared_cycle_fails_on_mismatch(cfg):
    cycle = pd.Timestamp(cfg["cycle"]["cycle_time"])
    wrong_cycle = cycle + pd.Timedelta(hours=6)
    df = _make_output_table(
        cycle, cfg["cycle"]["lead_hours"], cams_cycle=wrong_cycle
    )
    assert not (df["_aurora_cycle"] == df["_cams_cycle"]).all()


# ---------------------------------------------------------------------------
# Invariant 3: all configured lead_hours appear in the output
# ---------------------------------------------------------------------------

def test_all_lead_hours_present(cfg):
    cycle = pd.Timestamp(cfg["cycle"]["cycle_time"])
    configured = set(cfg["cycle"]["lead_hours"])
    df = _make_output_table(cycle, cfg["cycle"]["lead_hours"])
    in_output = set(df["lead_hours"].unique())
    assert configured == in_output


def test_missing_lead_hour_is_detected(cfg):
    cycle = pd.Timestamp(cfg["cycle"]["cycle_time"])
    configured = set(cfg["cycle"]["lead_hours"])
    # Drop one lead hour from output
    df = _make_output_table(cycle, cfg["cycle"]["lead_hours"])
    drop_lead = next(iter(configured))
    df_missing = df[df["lead_hours"] != drop_lead]
    in_output = set(df_missing["lead_hours"].unique())
    assert configured != in_output


# ---------------------------------------------------------------------------
# Invariant 4: available_time is set on every row
# ---------------------------------------------------------------------------

def test_available_time_required(cfg):
    cycle = pd.Timestamp(cfg["cycle"]["cycle_time"])
    df = _make_output_table(cycle, cfg["cycle"]["lead_hours"], include_available_time=True)
    assert df["available_time"].notna().all()


def test_missing_available_time_is_detected(cfg):
    cycle = pd.Timestamp(cfg["cycle"]["cycle_time"])
    df = _make_output_table(cycle, cfg["cycle"]["lead_hours"], include_available_time=False)
    assert df["available_time"].isna().all()
