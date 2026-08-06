"""
Rolling-origin (expanding-window) validation with strict leakage enforcement.

Each origin O produces:
  - train rows:  available_time < O
  - test rows:   available_time in the first block after O

All preprocessing (imputation, scaling, lag selection) must be fitted
exclusively on train rows. The leakage check is run automatically and raises
LeakageError on violation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator

import pandas as pd


class LeakageError(RuntimeError):
    """Raised when a test row has a feature derived from post-available_time data."""


@dataclass
class OriginSplit:
    origin: datetime
    train: pd.DataFrame
    test: pd.DataFrame
    train_date_range: tuple[datetime, datetime]
    test_date_range: tuple[datetime, datetime]
    metadata: dict[str, Any] = field(default_factory=dict)


def rolling_origin_splits(
    df: pd.DataFrame,
    available_time_col: str = "available_time",
    valid_time_col: str = "valid_time",
    min_train_rows: int = 30,
    step_hours: int = 24,
    block_hours: int = 24,
) -> Iterator[OriginSplit]:
    """
    Yield OriginSplit objects over all valid origins.

    Parameters
    ----------
    df:
        Full dataset with at minimum `available_time_col` and `valid_time_col`.
    min_train_rows:
        Minimum number of training rows required before an origin is yielded.
    step_hours:
        Hours between consecutive origins.
    block_hours:
        Width of the test window after each origin.
    """
    df = df.copy()
    df[available_time_col] = pd.to_datetime(df[available_time_col], utc=True)
    df[valid_time_col] = pd.to_datetime(df[valid_time_col], utc=True)

    times = sorted(df[available_time_col].unique())
    if len(times) < 2:
        return

    step = pd.Timedelta(hours=step_hours)
    block = pd.Timedelta(hours=block_hours)
    origin = times[0] + step

    while origin <= times[-1]:
        train = df[df[available_time_col] < origin].copy()
        test = df[
            (df[available_time_col] >= origin)
            & (df[available_time_col] < origin + block)
        ].copy()

        # A training row requires both the forecast (available_time < origin)
        # AND the target observation (valid_time <= origin) to be non-leaking.
        train = train[train[valid_time_col] <= pd.Timestamp(origin)].copy()

        if len(train) >= min_train_rows and len(test) > 0:
            check_no_leakage(train, test, available_time_col, valid_time_col, origin)
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
                metadata={"step_hours": step_hours, "block_hours": block_hours},
            )

        origin += step


def check_no_leakage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    available_time_col: str,
    valid_time_col: str,
    origin: datetime,
) -> None:
    """
    Enforce: max(train.valid_time) <= origin (available_time of first test row).

    Raises LeakageError with full diagnostic if violated.
    """
    if train.empty:
        return
    max_train_valid = train[valid_time_col].max()
    if pd.Timestamp(max_train_valid) > pd.Timestamp(origin):
        raise LeakageError(
            f"Leakage detected at origin={origin}: "
            f"max train valid_time={max_train_valid} > origin={origin}. "
            "Check that no train row uses future observations."
        )


def provenance_hash(config: dict[str, Any]) -> str:
    """SHA-256 of the serialised config dict (sorted keys)."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()
