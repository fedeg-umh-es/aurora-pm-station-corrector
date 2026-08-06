"""
Rolling-origin (expanding-window) validation with strict leakage enforcement.

Timing model
------------
Each row represents a forecast-observation pair:

  cycle_time                  NWP/Aurora initialisation time (UTC)
  available_time              When the forecast became available to a user
                              (cycle_time + dissemination lag)
  valid_time                  Time of the target observation (UTC)
  target_available_time       When the target observation became available
                              (valid_time + measurement/publication lag)
                              If absent, falls back to valid_time — this is the
                              PSEUDO_OPERATIONAL_ASSUMPTION (documented below).
  lead_hours                  valid_time - cycle_time in hours

Train admissibility at decision origin O:
  available_time < O              (forecast was issued before O)
  target_available_time <= O      (target observation was published before O)

For local lag features the analogous rule (enforced by the lag builder):
  observation_available_time <= decision_time

Pseudo-operational assumption
------------------------------
When target_available_time is absent from the DataFrame, the splitter falls
back to valid_time as a proxy. This assumes station observations are
immediately available when measured — not true in practice (typical reporting
delay: 0.5–2 h). Any result obtained under this assumption must:
  1. Be labelled PSEUDO_OPERATIONAL in provenance metadata.
  2. Be subjected to sensitivity analysis with a 1-h and a 2-h latency before
     final reporting.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, Optional

import pandas as pd


PSEUDO_OPERATIONAL_ASSUMPTION = (
    "target_available_time absent; falling back to valid_time as proxy. "
    "Assumes zero measurement/publication delay — sensitivity analysis "
    "with 1-h and 2-h latency required before final reporting."
)


class LeakageError(RuntimeError):
    """Raised when a training row contains a target that post-dates the origin."""


@dataclass
class OriginSplit:
    origin: datetime
    train: pd.DataFrame
    test: pd.DataFrame
    train_date_range: tuple[datetime, datetime]
    test_date_range: tuple[datetime, datetime]
    pseudo_operational: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def rolling_origin_splits(
    df: pd.DataFrame,
    available_time_col: str = "available_time",
    valid_time_col: str = "valid_time",
    target_available_time_col: Optional[str] = "target_available_time",
    min_train_rows: int = 30,
    step_hours: int = 24,
    block_hours: int = 24,
) -> Iterator[OriginSplit]:
    """
    Yield OriginSplit objects for all valid rolling origins.

    A training row is admitted if and only if:
      1. Its forecast was available before the origin:
         available_time < origin
      2. Its target observation was available before the origin:
         target_available_time <= origin   (or valid_time if column absent)

    Parameters
    ----------
    df:
        Full dataset with required columns.
    available_time_col:
        Column marking when the forecast became available.
    valid_time_col:
        Column marking the time of the target observation.
    target_available_time_col:
        Column marking when the target observation became publicly available.
        If None or not present in df, falls back to valid_time with a warning.
    min_train_rows:
        Minimum train rows for a split to be yielded.
    step_hours:
        Hours between consecutive origins.
    block_hours:
        Width of the test window after each origin.
    """
    df = df.copy()
    df[available_time_col] = pd.to_datetime(df[available_time_col], utc=True)
    df[valid_time_col] = pd.to_datetime(df[valid_time_col], utc=True)

    pseudo_operational = False
    if target_available_time_col and target_available_time_col in df.columns:
        df[target_available_time_col] = pd.to_datetime(
            df[target_available_time_col], utc=True
        )
        tgt_avail_col = target_available_time_col
    else:
        warnings.warn(PSEUDO_OPERATIONAL_ASSUMPTION, UserWarning, stacklevel=2)
        tgt_avail_col = valid_time_col
        pseudo_operational = True

    times = sorted(df[available_time_col].unique())
    if len(times) < 2:
        return

    step = pd.Timedelta(hours=step_hours)
    block = pd.Timedelta(hours=block_hours)
    origin = times[0] + step

    while origin <= times[-1]:
        origin_ts = pd.Timestamp(origin)

        # Condition 1: forecast available before origin
        # Condition 2: target observation available at or before origin
        train_mask = (df[available_time_col] < origin_ts) & (
            df[tgt_avail_col] <= origin_ts
        )
        train = df[train_mask].copy()

        test = df[
            (df[available_time_col] >= origin_ts)
            & (df[available_time_col] < origin_ts + block)
        ].copy()

        if len(train) >= min_train_rows and len(test) > 0:
            check_no_leakage(
                train=train,
                test=test,
                available_time_col=available_time_col,
                target_available_time_col=tgt_avail_col,
                origin=origin_ts,
            )
            yield OriginSplit(
                origin=origin,
                train=train,
                test=test,
                train_date_range=(
                    train[valid_time_col].min().to_pydatetime(),
                    train[valid_time_col].max().to_pydatetime(),
                ),
                test_date_range=(
                    test[valid_time_col].min().to_pydatetime(),
                    test[valid_time_col].max().to_pydatetime(),
                ),
                pseudo_operational=pseudo_operational,
                metadata={
                    "step_hours": step_hours,
                    "block_hours": block_hours,
                    "pseudo_operational": pseudo_operational,
                    "target_available_time_col": tgt_avail_col,
                },
            )

        origin = origin_ts + step


def check_no_leakage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    available_time_col: str,
    target_available_time_col: str,
    origin: datetime,
) -> None:
    """
    Enforce: max(train[target_available_time_col]) <= origin.

    Raises LeakageError with a full diagnostic on violation.
    """
    if train.empty:
        return
    max_tgt_avail = train[target_available_time_col].max()
    if pd.Timestamp(max_tgt_avail) > pd.Timestamp(origin):
        raise LeakageError(
            f"Leakage at origin={origin}: "
            f"max(train.{target_available_time_col})={max_tgt_avail} > origin. "
            "The target observation used for training post-dates the decision origin."
        )


def provenance_hash(config: dict[str, Any]) -> str:
    """SHA-256 of the serialised config dict (sorted keys)."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()
