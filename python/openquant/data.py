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


def data_quality_report(df: pl.DataFrame) -> dict[str, Any]:
    _validate_required_columns(df)
    key_cols = ["symbol", "ts"]
    row_count = df.height
    duplicate_keys = (
        df.group_by(key_cols)
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    gap_count = 0
    for sym in df["symbol"].unique().to_list():
        ts_values = (
            df.filter(pl.col("symbol") == sym)
            .sort("ts")
            .select("ts")
            .to_series()
            .to_list()
        )
        for prev, cur in zip(ts_values, ts_values[1:]):
            if prev is None or cur is None:
                continue
            if (cur - prev).total_seconds() > 24 * 3600:
                gap_count += 1

    report = {
        "row_count": row_count,
        "symbol_count": int(df.select(pl.col("symbol").n_unique()).item()),
        "duplicate_key_count": duplicate_keys,
        "gap_interval_count": gap_count,
        "null_counts": {
            col: int(df.select(pl.col(col).null_count()).item())
            for col in CANONICAL_OHLCV_COLUMNS
            if col in df.columns
        },
        "ts_min": str(df.select(pl.col("ts").min()).item()),
        "ts_max": str(df.select(pl.col("ts").max()).item()),
    }
    return report


def clean_ohlcv(
    df: pl.DataFrame,
    *,
    dedupe_keep: str = "last",
    return_report: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, dict[str, Any]]:
    out = _canonicalize_columns(df)
    _validate_required_columns(out)
    out = _cast_and_order(out)
    out = out.drop_nulls(subset=["ts", "symbol"])
    pre_rows = out.height
    out = out.unique(subset=["symbol", "ts"], keep=dedupe_keep).sort(["symbol", "ts"])
    removed = pre_rows - out.height
    report = data_quality_report(out)
    report["rows_removed_by_deduplication"] = removed
    if return_report:
        return out, report
    return out


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
    out = clean_ohlcv(raw, return_report=return_report)
    return out


def align_calendar(
    df: pl.DataFrame,
    *,
    interval: str = "1d",
) -> pl.DataFrame:
    clean = clean_ohlcv(df)
    symbols = clean["symbol"].unique().to_list()
    aligned: list[pl.DataFrame] = []
    for sym in symbols:
        sdf = clean.filter(pl.col("symbol") == sym).sort("ts")
        start = sdf.select(pl.col("ts").min()).item()
        end = sdf.select(pl.col("ts").max()).item()
        calendar = pl.DataFrame(
            {"ts": pl.datetime_range(start, end, interval=interval, eager=True)}
        ).with_columns(pl.lit(sym).alias("symbol"))
        merged = calendar.join(sdf, on=["symbol", "ts"], how="left")
        merged = merged.with_columns(
            pl.col("open").is_null().alias("is_missing_bar"),
            pl.col("adj_close").fill_null(pl.col("close")),
        )
        aligned.append(merged)
    return pl.concat(aligned, how="vertical").sort(["symbol", "ts"])
