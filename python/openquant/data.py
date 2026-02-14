from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


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

_ZERO_NULL_COUNTS = {
    "ts": 0,
    "symbol": 0,
    "open": 0,
    "high": 0,
    "low": 0,
    "close": 0,
    "volume": 0,
    "adj_close": 0,
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


def _prepare_ohlcv_lf(df: pl.DataFrame) -> pl.LazyFrame:
    frame = _canonicalize_columns(df)
    _validate_required_columns(frame)

    lf = frame.lazy().with_columns(
        pl.col("ts").cast(pl.Utf8).str.strptime(pl.Datetime, strict=False),
        pl.col("symbol").cast(pl.Utf8),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    )
    if "adj_close" in frame.columns:
        lf = lf.with_columns(pl.col("adj_close").cast(pl.Float64))
    else:
        lf = lf.with_columns(pl.col("close").alias("adj_close"))
    return lf.select(CANONICAL_OHLCV_COLUMNS).drop_nulls(CANONICAL_OHLCV_COLUMNS)


def _format_ts(v: Any) -> str | None:
    if v is None:
        return None
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    return str(v)


def _gap_expr(symbol_expr: pl.Expr, ts_us_expr: pl.Expr, threshold_seconds: int = 24 * 3600) -> pl.Expr:
    threshold_us = int(threshold_seconds) * 1_000_000
    return (
        (symbol_expr == symbol_expr.shift(1))
        & ((ts_us_expr - ts_us_expr.shift(1)) > threshold_us)
    )


def _build_quality_report(sorted_df: pl.DataFrame, rows_removed_by_deduplication: int) -> dict[str, Any]:
    if sorted_df.height == 0:
        return {
            "row_count": 0,
            "symbol_count": 0,
            "duplicate_key_count": 0,
            "gap_interval_count": 0,
            "ts_min": None,
            "ts_max": None,
            "rows_removed_by_deduplication": rows_removed_by_deduplication,
            "null_counts": dict(_ZERO_NULL_COUNTS),
        }

    summary = (
        sorted_df.lazy()
        .select(
            pl.len().alias("row_count"),
            pl.col("symbol").n_unique().alias("symbol_count"),
            (
                ((pl.col("symbol") == pl.col("symbol").shift(1)) & (pl.col("ts_us") == pl.col("ts_us").shift(1)))
                .cast(pl.UInt32)
                .sum()
            ).alias("duplicate_key_count"),
            _gap_expr(pl.col("symbol"), pl.col("ts_us")).cast(pl.UInt32).sum().alias("gap_interval_count"),
            pl.col("ts").min().alias("ts_min"),
            pl.col("ts").max().alias("ts_max"),
        )
        .collect()
        .row(0, named=True)
    )
    return {
        "row_count": int(summary["row_count"]),
        "symbol_count": int(summary["symbol_count"]),
        "duplicate_key_count": int(summary["duplicate_key_count"]),
        "gap_interval_count": int(summary["gap_interval_count"]),
        "ts_min": _format_ts(summary["ts_min"]),
        "ts_max": _format_ts(summary["ts_max"]),
        "rows_removed_by_deduplication": rows_removed_by_deduplication,
        "null_counts": dict(_ZERO_NULL_COUNTS),
    }


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
    if dedupe_keep not in {"first", "last"}:
        raise ValueError("dedupe_keep must be 'first' or 'last'")

    base_lf = _prepare_ohlcv_lf(df).with_columns(pl.col("ts").dt.timestamp(time_unit="us").alias("ts_us"))
    sorted_lf = base_lf.sort(["symbol", "ts_us"])

    duplicate_key_count = int(
        sorted_lf
        .select(
            (
                ((pl.col("symbol") == pl.col("symbol").shift(1)) & (pl.col("ts_us") == pl.col("ts_us").shift(1)))
                .cast(pl.UInt32)
                .sum()
            ).alias("duplicate_key_count")
        )
        .collect()
        .item(0, 0)
    )

    cleaned = (
        sorted_lf.unique(
            subset=["symbol", "ts_us"],
            keep=dedupe_keep,
            maintain_order=True,
        )
        .sort(["symbol", "ts"])
        .collect()
    )

    frame = cleaned.select(CANONICAL_OHLCV_COLUMNS)
    if not return_report:
        return frame

    report = _build_quality_report(cleaned, rows_removed_by_deduplication=duplicate_key_count)
    report["duplicate_key_count"] = 0
    return frame, report


def data_quality_report(df: pl.DataFrame) -> dict[str, Any]:
    sorted_df = (
        _prepare_ohlcv_lf(df)
        .with_columns(pl.col("ts").dt.timestamp(time_unit="us").alias("ts_us"))
        .sort(["symbol", "ts_us"])
        .collect()
    )
    return _build_quality_report(sorted_df, rows_removed_by_deduplication=0)


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
    interval_seconds = _interval_to_seconds(interval)
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be > 0")

    clean = clean_ohlcv(df).lazy()
    bounds = clean.group_by("symbol").agg(
        pl.col("ts").min().alias("ts_min"),
        pl.col("ts").max().alias("ts_max"),
    )
    calendar = (
        bounds.with_columns(
            pl.datetime_ranges(
                "ts_min",
                "ts_max",
                interval=f"{interval_seconds}s",
                closed="both",
            ).alias("ts")
        )
        .explode("ts")
        .select(["symbol", "ts"])
    )

    out = (
        calendar.join(clean, on=["symbol", "ts"], how="left")
        .with_columns(pl.col("open").is_null().alias("is_missing_bar"))
        .select(CANONICAL_OHLCV_COLUMNS + ["is_missing_bar"])
        .sort(["symbol", "ts"])
        .collect()
    )
    return out
