"""Canonical OHLCV columns must not depend on the input column order."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest
from openquant import data


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": ["2024-01-02"],
            "symbol": ["ABC"],
            "open": [10.0],
            "high": [12.0],
            "low": [9.0],
            "close": [11.0],
            "volume": [100.0],
            "adj_close": [10.5],
        }
    )


@pytest.mark.parametrize(
    ("canonical", "alias", "alias_value", "expected"),
    [
        ("ts", "date", "2020-01-01", datetime(2024, 1, 2)),
        ("symbol", "ticker", "WRONG", "ABC"),
        ("open", "OPEN", 99.0, 10.0),
        ("high", "HIGH", 99.0, 12.0),
        ("low", "LOW", 99.0, 9.0),
        ("close", "CLOSE", 99.0, 11.0),
        ("volume", "VOLUME", 99.0, 100.0),
        ("adj_close", "adjusted close", 99.0, 10.5),
    ],
)
@pytest.mark.parametrize("alias_first", [False, True])
def test_canonical_column_wins_regardless_of_alias_order(
    canonical, alias, alias_value, expected, alias_first
):
    frame = _frame().with_columns(pl.Series(alias, [alias_value]))
    if alias_first:
        frame = frame.select(alias, *_frame().columns)
    cleaned = data.clean_ohlcv(frame)
    assert cleaned[canonical].to_list() == [expected]
    assert cleaned.columns == data.CANONICAL_OHLCV_COLUMNS


def test_first_alias_is_used_when_no_canonical_column_exists():
    frame = _frame().rename({"ts": "date"}).with_columns(pl.lit("2020-01-01").alias("timestamp"))
    assert data.clean_ohlcv(frame)["ts"].to_list() == [datetime(2024, 1, 2)]


def test_all_aliased_columns_are_still_normalized():
    frame = _frame().rename({"ts": "date", "symbol": "ticker", "adj_close": "adjusted close"})
    result = data.clean_ohlcv(frame)
    assert result["ts"].to_list() == [datetime(2024, 1, 2)]
    assert result["symbol"].to_list() == ["ABC"]
    assert result["adj_close"].to_list() == [10.5]


@pytest.mark.parametrize("stamps", [["invalid"], ["", "not a date", None]])
def test_unparseable_timestamps_are_reported_as_nulls(stamps):
    frame = pl.concat([_frame()] * len(stamps)).with_columns(pl.Series("ts", stamps))
    cleaned, report = data.clean_ohlcv(frame, return_report=True)
    assert cleaned.height == 0
    assert cleaned.schema["ts"] == pl.Datetime("us")
    assert report["row_count"] == 0
    assert report["null_counts"]["ts"] == len(stamps)
    assert report["ts_min"] is None
    assert report["ts_max"] is None
    assert data.data_quality_report(frame)["null_counts"]["ts"] == len(stamps)


@pytest.mark.parametrize(
    ("stamps", "expected"),
    [
        ([], []),
        ([None], []),
        (["bad", "2024-01-02"], [datetime(2024, 1, 2)]),
        (["2024/01/02"], [datetime(2024, 1, 2)]),
        (["02/01/2024"], [datetime(2024, 1, 2)]),
        (["2024-01-02T13:30:00"], [datetime(2024, 1, 2, 13, 30)]),
        (
            ["2024-01-02 13:30:00", "2024-01-02 13:30:00.123456"],
            [datetime(2024, 1, 2, 13, 30), datetime(2024, 1, 2, 13, 30, 0, 123456)],
        ),
    ],
)
def test_timestamp_fallback_keeps_supported_formats_and_frame_length(stamps, expected):
    frame = pl.concat([_frame()] * len(stamps)) if stamps else _frame().head(0)
    frame = frame.with_columns(pl.Series("ts", stamps, dtype=pl.String))
    cleaned = data.clean_ohlcv(frame)
    assert cleaned["ts"].to_list() == expected
