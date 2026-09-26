from __future__ import annotations

import hashlib
import json
from pathlib import Path

import openquant
import polars as pl
import pytest


def _fixture_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "ohlcv_us_equities.csv"


def _digest_frame(df: pl.DataFrame) -> str:
    normalized = df.with_columns(pl.col("ts").dt.strftime("%Y-%m-%d %H:%M:%S"))
    payload = json.dumps(normalized.to_dict(as_series=False), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_load_ohlcv_enforces_canonical_schema_and_determinism():
    p = _fixture_path()
    out1, report1 = openquant.data.load_ohlcv(p, return_report=True)
    out2 = openquant.data.load_ohlcv(p)

    assert out1.columns == [
        "ts",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj_close",
    ]
    assert out1.equals(out2)
    assert report1["row_count"] == 5
    assert report1["rows_removed_by_deduplication"] == 1
    assert report1["gap_interval_count"] == 1

    digest_1 = _digest_frame(out1)
    digest_2 = _digest_frame(openquant.data.load_ohlcv(p))
    assert digest_1 == digest_2


def test_load_ohlcv_symbol_argument_required_when_missing():
    df = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.0, 10.5],
            "close": [10.5, 11.4],
            "volume": [1000, 1100],
        }
    )
    p = _fixture_path().with_name("ohlcv_no_symbol.csv")
    df.write_csv(p)
    try:
        try:
            openquant.data.load_ohlcv(p)
            assert False, "expected load_ohlcv to raise without symbol"
        except ValueError:
            pass

        out = openquant.data.load_ohlcv(p, symbol="SPY")
        assert set(out["symbol"].to_list()) == {"SPY"}
    finally:
        p.unlink(missing_ok=True)


def test_align_calendar_marks_missing_bars():
    raw = openquant.data.load_ohlcv(_fixture_path())
    aligned = openquant.data.align_calendar(raw)

    msft = aligned.filter(pl.col("symbol") == "MSFT")
    assert msft.height == 3
    # 2024-01-02 is absent in fixture for MSFT and should be represented.
    missing = msft.filter(pl.col("is_missing_bar"))
    assert missing.height == 1
    assert str(missing["ts"][0]).startswith("2024-01-02")


def _raw_df_frame() -> pl.DataFrame:
    day_us = 86_400 * 1_000_000
    base = 1_704_067_200 * 1_000_000  # 2024-01-01 00:00:00 UTC
    return pl.DataFrame(
        {
            "symbol": ["MSFT", "AAPL", "AAPL", "MSFT"],
            "ts_us": [base + 2 * day_us, base + day_us, base + day_us, base],
            "open": [372.2, 185.0, 185.05, 370.1],
            "high": [373.1, 186.2, 186.3, 372.0],
            "low": [371.4, 184.7, 184.6, 369.9],
            "close": [372.0, 185.9, 186.0, 371.5],
            "volume": [5500.0, 7000.0, 7100.0, 6000.0],
            "adj_close": [371.9, 185.8, 185.9, 371.3],
        }
    )


def test_core_dataframe_bindings_accept_polars_frames():
    # These take a polars DataFrame through pyo3-polars'
    # `FromPyObject for PyDataFrame`, the code `vendor/pyo3-polars` patches.
    # Unpatched 0.20.0 passes an integer `compat_level` to `Series.to_arrow`,
    # which Python polars >= 1.32.3 rejects with a TypeError.
    # See vendor/README.md.
    from openquant import _core

    raw = _raw_df_frame()

    cleaned, report = _core.data.clean_ohlcv_df(raw, True)
    assert isinstance(cleaned, pl.DataFrame)
    assert cleaned["symbol"].to_list() == ["AAPL", "MSFT", "MSFT"]
    assert cleaned["close"].to_list() == [186.0, 371.5, 372.0]
    assert report["rows_removed_by_deduplication"] == 1

    quality = _core.data.quality_report_df(raw)
    assert quality["row_count"] == 4

    aligned = _core.data.align_calendar_df(cleaned, 86_400)
    msft = aligned.filter(pl.col("symbol") == "MSFT")
    assert msft.height == 3
    assert msft["is_missing_bar"].to_list() == [False, True, False]


def _daily_frame(stamps: list[str], symbol: str = "AAA") -> pl.DataFrame:
    n = len(stamps)
    return pl.DataFrame(
        {
            "ts": stamps,
            "symbol": [symbol] * n,
            "open": [1.0] * n,
            "high": [1.0] * n,
            "low": [1.0] * n,
            "close": [float(i) for i in range(n)],
            "volume": [1.0] * n,
        }
    )


def test_gap_count_follows_the_bar_frequency():
    # Mirrors crates/openquant/tests/data_processing.rs::gap_count_follows_the_bar_frequency
    # (#168). September 2024: the 2nd is a Monday.
    week = [f"2024-09-{d:02d}" for d in (2, 3, 4, 5, 6, 9)]
    report = openquant.data.data_quality_report(_daily_frame(week))
    assert report["inferred_interval_us"] == 86_400 * 1_000_000
    assert report["gap_interval_count"] == 0  # the weekend is not a gap

    missing = [f"2024-09-{d:02d}" for d in (2, 3, 5, 6, 10)]
    assert openquant.data.data_quality_report(_daily_frame(missing))["gap_interval_count"] == 2

    hourly = [f"2024-09-02 {h:02d}:00:00" for h in (9, 10, 11, 13, 14, 15)]
    report = openquant.data.data_quality_report(_daily_frame(hourly))
    assert report["inferred_interval_us"] == 3_600 * 1_000_000
    assert report["gap_interval_count"] == 1

    # The Rust core agrees.
    from openquant import _core

    for stamps, gaps in ((week, 0), (missing, 2), (hourly, 1)):
        ts_us = (
            _daily_frame(stamps)
            .with_columns(pl.col("ts").str.to_datetime())
            .get_column("ts")
            .dt.timestamp("us")
            .to_list()
        )
        ones = [1.0] * len(ts_us)
        core = _core.data.quality_report(
            ts_us, ["AAA"] * len(ts_us), ones, ones, ones, ones, ones, ones
        )
        assert core["gap_interval_count"] == gaps


def test_align_calendar_reports_off_grid_bars():
    # Mirrors crates/openquant/tests/data_processing.rs::align_calendar_reports_off_grid_bars
    # (#168): a bar that is not on the grid used to vanish without a trace.
    raw = _daily_frame(
        ["2024-09-02 00:00:00", "2024-09-03 00:00:00", "2024-09-04 16:00:00", "2024-09-05 00:00:00"]
    )
    aligned, report = openquant.data.align_calendar(raw, return_report=True)
    assert aligned["is_missing_bar"].to_list() == [False, False, True, False]
    assert report["off_grid_bar_count"] == 1
    assert report["rows_removed_by_deduplication"] == 0
    assert report["off_grid_bars"]["ts"].dt.strftime("%Y-%m-%d %H:%M").to_list() == [
        "2024-09-04 16:00"
    ]

    with pytest.warns(UserWarning, match="dropped 1 bar"):
        openquant.data.align_calendar(raw)

    from openquant import _core

    core_in = raw.with_columns(
        pl.col("ts").str.to_datetime().dt.timestamp("us").alias("ts_us"),
        pl.col("close").alias("adj_close"),
    ).drop("ts")
    frame, core_report = _core.data.align_calendar_df(core_in, 86_400, return_report=True)
    assert frame["is_missing_bar"].to_list() == [False, False, True, False]
    assert core_report["off_grid_bar_count"] == 1
    assert core_report["off_grid_bars"] == [("AAA", core_in["ts_us"][2])]
    assert isinstance(_core.data.align_calendar_df(core_in, 86_400), pl.DataFrame)

    cols = [core_in[c].to_list() for c in ("ts_us", "symbol", "open", "high", "low", "close")]
    cols += [core_in["volume"].to_list(), core_in["adj_close"].to_list()]
    out, col_report = _core.data.align_calendar(*cols, 86_400, return_report=True)
    assert out[-1] == [False, False, True, False]
    assert col_report == core_report
    assert len(_core.data.align_calendar(*cols, 86_400)) == 9


def test_quality_report_counts_nulls():
    # Issue #194: null_counts was always 0, because the rows with nulls were dropped first.
    frame = _daily_frame([f"2024-09-{d:02d}" for d in (2, 3, 4, 5, 6)]).with_columns(
        pl.Series("close", [1.0, None, 3.0, None, 5.0]),
        pl.Series("volume", [1.0, 1.0, None, 1.0, 1.0]),
        pl.Series("ts", ["2024-09-02", "2024-09-03", "2024-09-04", "2024-09-05", "not a date"]),
    )
    want = {
        "ts": 1,  # a timestamp that does not parse
        "symbol": 0,
        "open": 0,
        "high": 0,
        "low": 0,
        "close": 2,
        "volume": 1,
        "adj_close": 2,  # filled from close
    }
    report = openquant.data.data_quality_report(frame)
    assert report["null_counts"] == want
    assert report["row_count"] == 1  # only 2024-09-02 has no null
    assert openquant.data.quality_failures(report) == [
        f"nulls in { {k: v for k, v in want.items() if v} }"
    ]

    _, cleaned_report = openquant.data.clean_ohlcv(frame, return_report=True)
    assert cleaned_report["null_counts"] == want
    assert cleaned_report["row_count"] == 1

    clean = openquant.data.data_quality_report(_daily_frame(["2024-09-02", "2024-09-03"]))
    assert set(clean["null_counts"].values()) == {0}
