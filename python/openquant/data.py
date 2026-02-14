from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from . import _core


CANONICAL_OHLCV_COLUMNS = [
    "ts",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
]

_COLUMN_ALIASES = {
    "ts": "ts",
    "timestamp": "ts",
    "datetime": "ts",
    "date": "ts",
    "symbol": "symbol",
    "ticker": "symbol",
    "asset": "symbol",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "adj_close": "adj_close",
    "adjusted_close": "adj_close",
    "adjclose": "adj_close",
    "adjusted close": "adj_close",
    "adj close": "adj_close",
}


def _normalize_column_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _canonicalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    rename_map: dict[str, str] = {}
    used_targets: set[str] = set()
    for col in df.columns:
        key = _normalize_column_name(col)
        if key in _COLUMN_ALIASES:
            target = _COLUMN_ALIASES[key]
            if target in used_targets and col != target:
                continue
            rename_map[col] = target
            used_targets.add(target)
    if rename_map:
        return df.rename(rename_map)
    return df


def _validate_required_columns(df: pl.DataFrame) -> None:
    required = {"ts", "symbol", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"missing required OHLCV columns: {', '.join(missing)}")


def _cast_and_order(df: pl.DataFrame) -> pl.DataFrame:
    casted = df.with_columns(
        pl.col("ts").cast(pl.Utf8).str.strptime(pl.Datetime, strict=False),
        pl.col("symbol").cast(pl.Utf8),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    )
    if "adj_close" in casted.columns:
        casted = casted.with_columns(pl.col("adj_close").cast(pl.Float64))
    else:
        casted = casted.with_columns(pl.col("close").alias("adj_close"))
    return casted.select(CANONICAL_OHLCV_COLUMNS)


def _to_core_vectors(df: pl.DataFrame) -> tuple[list[str], list[str], list[float], list[float], list[float], list[float], list[float], list[float]]:
    return (
        [str(x) for x in df["ts"].to_list()],
        [str(x) for x in df["symbol"].to_list()],
        [float(x) for x in df["open"].to_list()],
        [float(x) for x in df["high"].to_list()],
        [float(x) for x in df["low"].to_list()],
        [float(x) for x in df["close"].to_list()],
        [float(x) for x in df["volume"].to_list()],
        [float(x) for x in df["adj_close"].to_list()],
    )


def _rows_to_frame(rows: list[tuple[str, str, float, float, float, float, float, float]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            {
                "ts": [],
                "symbol": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
                "adj_close": [],
            }
        )
    return pl.DataFrame(
        {
            "ts": [r[0] for r in rows],
            "symbol": [r[1] for r in rows],
            "open": [r[2] for r in rows],
            "high": [r[3] for r in rows],
            "low": [r[4] for r in rows],
            "close": [r[5] for r in rows],
            "volume": [r[6] for r in rows],
            "adj_close": [r[7] for r in rows],
        }
    ).with_columns(pl.col("ts").str.strptime(pl.Datetime, strict=False))


def _interval_to_seconds(interval: str) -> int:
    s = interval.strip().lower()
    if s.endswith("d"):
        return int(s[:-1]) * 24 * 3600
    if s.endswith("h"):
        return int(s[:-1]) * 3600
    if s.endswith("m"):
        return int(s[:-1]) * 60
    if s.endswith("s"):
        return int(s[:-1])
    raise ValueError(f"unsupported interval format: {interval}")


def clean_ohlcv(
    df: pl.DataFrame,
    *,
    dedupe_keep: str = "last",
    return_report: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, dict[str, Any]]:
    out = _canonicalize_columns(df)
    _validate_required_columns(out)
    out = _cast_and_order(out)
    ts, symbol, open_, high, low, close, volume, adj_close = _to_core_vectors(out)
    rows, report = _core.data.clean_ohlcv(
        ts,
        symbol,
        open_,
        high,
        low,
        close,
        volume,
        adj_close,
        dedupe_keep == "last",
    )
    frame = _rows_to_frame(rows).sort(["symbol", "ts"])
    report = dict(report)
    report["null_counts"] = {
        "ts": 0,
        "symbol": 0,
        "open": 0,
        "high": 0,
        "low": 0,
        "close": 0,
        "volume": 0,
        "adj_close": 0,
    }
    if return_report:
        return frame, report
    return frame


def data_quality_report(df: pl.DataFrame) -> dict[str, Any]:
    out = _canonicalize_columns(df)
    _validate_required_columns(out)
    out = _cast_and_order(out).sort(["symbol", "ts"])
    ts, symbol, open_, high, low, close, volume, adj_close = _to_core_vectors(out)
    report = dict(
        _core.data.quality_report(
            ts,
            symbol,
            open_,
            high,
            low,
            close,
            volume,
            adj_close,
        )
    )
    report["null_counts"] = {
        "ts": 0,
        "symbol": 0,
        "open": 0,
        "high": 0,
        "low": 0,
        "close": 0,
        "volume": 0,
        "adj_close": 0,
    }
    return report


def load_ohlcv(
    path: str | Path,
    *,
    symbol: str | None = None,
    return_report: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"file not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        raw = pl.read_csv(file_path)
    elif suffix in {".parquet", ".pq"}:
        raw = pl.read_parquet(file_path)
    else:
        raise ValueError(f"unsupported file type: {suffix}")

    raw = _canonicalize_columns(raw)
    if "symbol" not in raw.columns:
        if symbol is None:
            raise ValueError("symbol column missing and no symbol argument provided")
        raw = raw.with_columns(pl.lit(symbol).alias("symbol"))
    return clean_ohlcv(raw, return_report=return_report)


def align_calendar(
    df: pl.DataFrame,
    *,
    interval: str = "1d",
) -> pl.DataFrame:
    clean = clean_ohlcv(df)
    ts, symbol, open_, high, low, close, volume, adj_close = _to_core_vectors(clean)
    rows = _core.data.align_calendar(
        ts,
        symbol,
        open_,
        high,
        low,
        close,
        volume,
        adj_close,
        _interval_to_seconds(interval),
    )
    if not rows:
        return pl.DataFrame(
            {
                "ts": [],
                "symbol": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
                "adj_close": [],
                "is_missing_bar": [],
            }
        )
    return pl.DataFrame(
        {
            "ts": [r[0] for r in rows],
            "symbol": [r[1] for r in rows],
            "open": [r[2] for r in rows],
            "high": [r[3] for r in rows],
            "low": [r[4] for r in rows],
            "close": [r[5] for r in rows],
            "volume": [r[6] for r in rows],
            "adj_close": [r[7] for r in rows],
            "is_missing_bar": [r[8] for r in rows],
        }
    ).with_columns(
        pl.col("ts").str.strptime(pl.Datetime, strict=False),
    ).sort(["symbol", "ts"])
