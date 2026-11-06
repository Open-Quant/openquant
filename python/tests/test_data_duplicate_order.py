"""First/last duplicate selection must follow input order, before key sorting."""

from datetime import datetime, timedelta
from random import Random

import polars as pl
import pytest
from openquant import data
from polars.testing import assert_frame_equal


def _duplicate_bars():
    keys = [i % 8 for i in range(128)]
    Random(8).shuffle(keys)
    return pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1) + timedelta(days=k % 4) for k in keys],
            "symbol": [f"S{k // 4}" for k in keys],
            "open": [100.0 + i for i in range(len(keys))],
            "high": [101.0 + i for i in range(len(keys))],
            "low": [99.0 + i for i in range(len(keys))],
            "close": [100.5 + i for i in range(len(keys))],
            "volume": [float(i + 1) for i in range(len(keys))],
            "adj_close": [100.25 + i for i in range(len(keys))],
        }
    )


def _input_order_oracle(raw, keep):
    selected = {}
    for row in raw.iter_rows(named=True):
        key = row["symbol"], row["ts"]
        if keep == "last" or key not in selected:
            selected[key] = row
    return pl.DataFrame([selected[key] for key in sorted(selected)])


@pytest.mark.parametrize("keep", ["first", "last"])
@pytest.mark.parametrize("return_report", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
def test_clean_duplicate_rows_follow_input_order(keep, return_report, chunked):
    raw = _duplicate_bars()
    if chunked:
        raw = pl.concat([raw.slice(i, 32) for i in range(0, raw.height, 32)], rechunk=False)
        assert raw.n_chunks() == 4
    expected = _input_order_oracle(raw, keep)
    result = data.clean_ohlcv(raw, dedupe_keep=keep, return_report=return_report)
    if return_report:
        cleaned, report = result
        assert report["row_count"] == 8
        assert report["rows_removed_by_deduplication"] == 120
        assert report["duplicate_key_count"] == 0
    else:
        cleaned = result
    assert_frame_equal(cleaned, expected)


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
@pytest.mark.parametrize("return_report", [False, True])
def test_load_keeps_the_last_input_revision(tmp_path, suffix, return_report):
    raw = _duplicate_bars()
    path = tmp_path / f"bars.{suffix}"
    if suffix == "csv":
        raw.write_csv(path)
    else:
        raw.write_parquet(path)
    result = data.load_ohlcv(path, return_report=return_report)
    cleaned = result[0] if return_report else result
    assert_frame_equal(cleaned, _input_order_oracle(raw, "last"))


@pytest.mark.parametrize("return_report", [False, True])
def test_calendar_uses_the_last_input_revision(return_report):
    raw = _duplicate_bars()
    result = data.align_calendar(raw, return_report=return_report)
    aligned = result[0] if return_report else result
    assert not aligned["is_missing_bar"].any()
    assert_frame_equal(aligned.drop("is_missing_bar"), _input_order_oracle(raw, "last"))
