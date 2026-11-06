"""Cleaning without a report must not execute report-only source queries."""

import polars as pl
import pytest
from openquant import data
from polars.testing import assert_frame_equal


def _input():
    return pl.DataFrame(
        {
            "ts": ["2024-01-02", "2024-01-01", "2024-01-02", "invalid", "2024-01-03"],
            "symbol": ["ABC"] * 5,
            "open": [10.0] * 5,
            "high": [15.0] * 5,
            "low": [9.0] * 5,
            "close": [12.0, 10.0, 14.0, 99.0, None],
            "volume": [100.0] * 5,
        }
    )


def test_no_report_executes_the_input_plan_only_once(monkeypatch):
    original_cast = data._cast_ohlcv_lf
    executions = []

    def counted_cast(frame):
        def record(batch):
            executions.append(batch.height)
            return batch

        return original_cast(frame).map_batches(
            record,
            predicate_pushdown=False,
            projection_pushdown=False,
            slice_pushdown=False,
        )

    monkeypatch.setattr(data, "_cast_ohlcv_lf", counted_cast)
    cleaned = data.clean_ohlcv(_input())
    assert cleaned["close"].to_list() == [10.0, 14.0]
    assert executions == [5]


@pytest.mark.parametrize(("keep", "last_close"), [("first", 12.0), ("last", 14.0)])
def test_optional_report_preserves_cleaned_rows_and_deduplication_counts(keep, last_close):
    raw = _input()
    without_report = data.clean_ohlcv(raw, dedupe_keep=keep)
    with_report, report = data.clean_ohlcv(raw, dedupe_keep=keep, return_report=True)
    assert_frame_equal(without_report, with_report)
    assert with_report["close"].to_list() == [10.0, last_close]
    assert report["row_count"] == 2
    assert report["rows_removed_by_deduplication"] == 1
    assert report["duplicate_key_count"] == 0
    assert report["null_counts"]["ts"] == 1
    assert report["null_counts"]["close"] == 1
