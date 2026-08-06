import pandas as pd

from src.validation import iter_rolling_origin_splits


def test_multihorizon_targets_are_available_before_test_origin():
    issue = pd.date_range("2025-01-01", periods=12, freq="12h", tz="UTC")
    rows = []
    for current in issue:
        for lead in (12, 24):
            rows.append({"issue_time": current, "valid_time": current + pd.Timedelta(hours=lead)})
    frame = pd.DataFrame(rows)

    folds = list(
        iter_rolling_origin_splits(
            frame,
            min_train_issues=5,
            test_issues=2,
            step_issues=2,
        )
    )

    assert folds
    for fold in folds:
        train = frame.iloc[fold.train_index]
        test = frame.iloc[fold.test_index]
        assert pd.to_datetime(train["valid_time"], utc=True).max() <= fold.test_start
        assert pd.to_datetime(train["issue_time"], utc=True).max() < fold.test_start
        assert pd.to_datetime(test["issue_time"], utc=True).min() == fold.test_start
