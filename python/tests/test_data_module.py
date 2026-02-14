from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

import openquant


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
