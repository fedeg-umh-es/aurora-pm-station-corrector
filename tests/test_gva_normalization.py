from __future__ import annotations

import pandas as pd
import pytest

from scripts.p1_00b_normalize_gva import coord_status, parse_gva_month


def _frame(rows):
    return pd.DataFrame(
        rows, columns=["COD_ESTACION", "NOM_ESTACION", "FECHA", "HORA", "PM10", "PM2.5"]
    )


def test_decimal_comma_and_dash_missing():
    df = _frame([("03009006", "X", "2023-01-01", "00", "12,5", "-")])
    tidy, _ = parse_gva_month(df)
    assert list(tidy["pollutant"]) == ["PM10"]
    assert tidy["value"].iloc[0] == pytest.approx(12.5)


def test_timestamp_is_utc_interval_end():
    df = _frame([("1", "X", "2023-07-15", "12", "10", "5")])
    tidy, _ = parse_gva_month(df)
    assert set(tidy["datetime"]) == {pd.Timestamp("2023-07-15T12:00:00Z")}


def test_hora_zero_ends_at_midnight_of_fecha():
    df = _frame([("1", "X", "2023-01-01", "00", "10", "5")])
    tidy, _ = parse_gva_month(df)
    assert tidy["datetime"].iloc[0] == pd.Timestamp("2023-01-01T00:00:00Z")


def test_negative_values_are_kept_not_dropped():
    df = _frame([("1", "X", "2023-01-01", "01", "-3,0", "-")])
    tidy, _ = parse_gva_month(df)
    assert tidy["value"].iloc[0] == pytest.approx(-3.0)


def test_unparsed_tokens_are_reported_not_silently_dropped():
    df = _frame([("1", "X", "2023-01-01", "02", "n/d", "4")])
    tidy, stats = parse_gva_month(df)
    assert stats["unparsed_tokens"]["PM10"] == {"n/d": 1}
    assert list(tidy["pollutant"]) == ["PM2.5"]


def test_hora_out_of_range_raises():
    df = _frame([("1", "X", "2023-01-01", "24", "1", "1")])
    with pytest.raises(ValueError):
        parse_gva_month(df)


def test_missing_required_column_raises():
    df = _frame([("1", "X", "2023-01-01", "00", "1", "1")]).drop(columns=["PM2.5"])
    with pytest.raises(ValueError):
        parse_gva_month(df)


def test_station_id_keeps_leading_zeros():
    df = _frame([("03009006", "X", "2023-01-01", "00", "1", "1")])
    tidy, _ = parse_gva_month(df)
    assert set(tidy["station_id"]) == {"03009006"}


def test_coord_status_flags():
    dup = pd.Series({(39.0, -0.4): 3, (38.4, -0.5): 1})
    row = lambda la, lo: pd.Series({"latitude": la, "longitude": lo})  # noqa: E731
    assert coord_status(row(38.4, -0.5), dup) == "ok"
    assert coord_status(row(39.0, -0.4), dup) == "duplicated_in_catalogue"
    assert coord_status(row(50.0, -0.4), dup) == "outside_region"
    assert coord_status(row(float("nan"), -0.4), dup) == "no_coordinates"
