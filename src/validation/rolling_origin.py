"""Leakage-safe rolling-origin splits for station-level forecast evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RollingOriginFold:
    """Indices and temporal boundaries for one outer evaluation fold."""

    fold_id: int
    train_index: np.ndarray
    test_index: np.ndarray
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def iter_rolling_origin_splits(
    frame: pd.DataFrame,
    *,
    min_train_issues: int,
    test_issues: int,
    step_issues: int,
    gap_issues: int = 0,
    train_window_issues: int | None = None,
    issue_time_col: str = "issue_time",
    valid_time_col: str = "valid_time",
) -> Iterator[RollingOriginFold]:
    """Yield chronological folds without using targets unavailable at test issue time."""

    if min_train_issues <= 0 or test_issues <= 0 or step_issues <= 0:
        raise ValueError("min_train_issues, test_issues and step_issues must be positive")
    if gap_issues < 0:
        raise ValueError("gap_issues cannot be negative")
    if train_window_issues is not None and train_window_issues < min_train_issues:
        raise ValueError("train_window_issues must be >= min_train_issues")
    if issue_time_col not in frame or valid_time_col not in frame:
        raise KeyError(f"Missing {issue_time_col!r} or {valid_time_col!r}")

    issue_time = pd.to_datetime(frame[issue_time_col], utc=True, errors="raise")
    valid_time = pd.to_datetime(frame[valid_time_col], utc=True, errors="raise")
    unique_issues = pd.DatetimeIndex(issue_time.drop_duplicates().sort_values())

    first_test_position = min_train_issues + gap_issues
    last_start = len(unique_issues) - test_issues
    if first_test_position > last_start:
        raise ValueError(
            "Not enough issue times for one fold: "
            f"found={len(unique_issues)}, required>={first_test_position + test_issues}"
        )

    fold_id = 0
    for test_position in range(first_test_position, last_start + 1, step_issues):
        train_stop = test_position - gap_issues
        train_start = 0 if train_window_issues is None else max(0, train_stop - train_window_issues)
        train_issues = unique_issues[train_start:train_stop]
        test_block = unique_issues[test_position : test_position + test_issues]
        test_start = test_block[0]
        test_end = test_block[-1]

        # A multi-horizon target is trainable only after its valid time has passed.
        train_mask = issue_time.isin(train_issues) & (valid_time <= test_start)
        test_mask = issue_time.isin(test_block)
        train_index = np.flatnonzero(train_mask.to_numpy())
        test_index = np.flatnonzero(test_mask.to_numpy())

        if train_index.size == 0 or test_index.size == 0:
            continue
        if valid_time.iloc[train_index].max() > test_start:
            raise AssertionError("Training targets extend beyond the test origin")
        if issue_time.iloc[train_index].max() >= test_start:
            raise AssertionError("Training issues are not strictly before the test origin")

        yield RollingOriginFold(
            fold_id=fold_id,
            train_index=train_index,
            test_index=test_index,
            test_start=test_start,
            test_end=test_end,
        )
        fold_id += 1
