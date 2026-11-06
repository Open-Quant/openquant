"""Timezone-aware inputs normalize to the existing UTC-naive OHLCV schema."""

from datetime import UTC, datetime

import polars as pl
import pytest
from openquant import data


def _bars(timestamps):
    n = len(timestamps)
    return pl.DataFrame(
        {
            "ts": timestamps,
            "symbol": ["A"] * n,
            "open": [10.0] * n,
            "high": [12.0] * n,
            "low": [9.0] * n,
            "close": [10.0 + i for i in range(n)],
            "volume": [1.0] * n,
        }
    )


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        ("2024-01-01T12:30:00Z", datetime(2024, 1, 1, 12, 30)),
        ("2024-01-01T12:30:00+05:30", datetime(2024, 1, 1, 7)),
        ("2024-01-01T12:30:00-05:00", datetime(2024, 1, 1, 17, 30)),
        ("2024-01-01T01:30:00+14:00", datetime(2023, 12, 31, 11, 30)),
        ("2024-01-01T23:30:00-12:00", datetime(2024, 1, 2, 11, 30)),
        ("2024-01-01T12:30:00.123456789Z", datetime(2024, 1, 1, 12, 30, 0, 123456)),
    ],
)
def test_offset_strings_preserve_the_utc_instant(timestamp, expected):
    cleaned, report = data.clean_ohlcv(_bars([timestamp]), return_report=True)
    assert cleaned.schema["ts"] == pl.Datetime("us")
    assert cleaned["ts"].to_list() == [expected]
    assert report["null_counts"]["ts"] == 0
    assert report["ts_min"] == expected.isoformat(sep=" ")


@pytest.mark.parametrize("zone", ["UTC", "America/New_York", "Asia/Kolkata"])
def test_aware_datetime_columns_preserve_distinct_dst_instants(zone):
    # In New York these are both 01:30, on opposite sides of the autumn clock change.
    instants = [datetime(2024, 11, 3, 5, 30, tzinfo=UTC), datetime(2024, 11, 3, 6, 30, tzinfo=UTC)]
    ts = pl.Series("ts", instants).dt.convert_time_zone(zone)
    cleaned = data.clean_ohlcv(_bars(ts))
    assert cleaned.schema["ts"] == pl.Datetime("us")
    assert cleaned["ts"].to_list() == [value.replace(tzinfo=None) for value in instants]
    assert cleaned["close"].to_list() == [10.0, 11.0]


@pytest.mark.parametrize(("keep", "close"), [("first", 10.0), ("last", 11.0)])
def test_equivalent_offset_strings_share_one_duplicate_key(keep, close):
    raw = _bars(["2024-01-01T12:30:00Z", "2024-01-01T07:30:00-05:00"])
    cleaned, report = data.clean_ohlcv(raw, dedupe_keep=keep, return_report=True)
    assert cleaned["ts"].to_list() == [datetime(2024, 1, 1, 12, 30)]
    assert cleaned["close"].to_list() == [close]
    assert report["rows_removed_by_deduplication"] == 1


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_load_accepts_offset_aware_timestamps(tmp_path, suffix):
    raw = _bars(["2024-01-01T12:30:00+05:30"])
    path = tmp_path / f"bars.{suffix}"
    if suffix == "csv":
        raw.write_csv(path)
    else:
        raw.write_parquet(path)
    assert data.load_ohlcv(path)["ts"].to_list() == [datetime(2024, 1, 1, 7)]


def test_calendar_alignment_uses_elapsed_utc_hours_across_dst():
    raw = _bars(["2024-11-03T00:30:00-04:00", "2024-11-03T02:30:00-05:00"])
    aligned = data.align_calendar(raw, interval="1h")
    assert aligned["ts"].to_list() == [datetime(2024, 11, 3, hour, 30) for hour in range(4, 8)]
    assert aligned["is_missing_bar"].to_list() == [False, True, True, False]


@pytest.mark.parametrize("timestamp", ["2024-01-01", "2024-01-01T00:00:00", "2024-01-01 00:00:00"])
def test_naive_timestamp_values_and_schema_are_unchanged(timestamp):
    cleaned = data.clean_ohlcv(_bars([timestamp]))
    assert cleaned.schema["ts"] == pl.Datetime("us")
    assert cleaned["ts"].to_list() == [datetime(2024, 1, 1)]
