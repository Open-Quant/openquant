from __future__ import annotations

import math

import polars as pl

from . import data


def _prepare(df: pl.DataFrame) -> pl.DataFrame:
    return data.clean_ohlcv(df)


def _aggregate(df: pl.DataFrame) -> pl.DataFrame:
    out = (
        df.group_by(["symbol", "bar_id"])
        .agg(
            pl.col("ts").min().alias("start_ts"),
            pl.col("ts").max().alias("end_ts"),
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("adj_close").last().alias("adj_close"),
            pl.col("volume").sum().alias("volume"),
            pl.len().alias("n_obs"),
        )
        .sort(["symbol", "end_ts", "bar_id"])
        .drop("bar_id")
        .rename({"end_ts": "ts"})
    )
    return out.select(
        [
            "ts",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adj_close",
            "start_ts",
            "n_obs",
        ]
    )


def build_time_bars(df: pl.DataFrame, *, interval: str = "1d") -> pl.DataFrame:
    clean = _prepare(df)
    grouped = clean.with_columns(pl.col("ts").dt.truncate(interval).alias("bar_id"))
    return _aggregate(grouped)


def build_tick_bars(df: pl.DataFrame, *, ticks_per_bar: int = 50) -> pl.DataFrame:
    if ticks_per_bar <= 0:
        raise ValueError("ticks_per_bar must be > 0")
    clean = _prepare(df)
    grouped = clean.with_columns(
        (pl.int_range(0, pl.len()).over("symbol") // ticks_per_bar)
        .cast(pl.Int64)
        .alias("bar_id")
    )
    return _aggregate(grouped)


def build_volume_bars(df: pl.DataFrame, *, volume_per_bar: float = 100_000.0) -> pl.DataFrame:
    if volume_per_bar <= 0:
        raise ValueError("volume_per_bar must be > 0")
    clean = _prepare(df)
    eps = volume_per_bar * 1e-9
    grouped = (
        clean.with_columns(pl.col("volume").cum_sum().over("symbol").alias("cum_volume"))
        .with_columns(
            (((pl.col("cum_volume") - eps).clip(lower_bound=0.0)) / volume_per_bar)
            .floor()
            .cast(pl.Int64)
            .alias("bar_id")
        )
        .drop("cum_volume")
    )
    return _aggregate(grouped)


def build_dollar_bars(
    df: pl.DataFrame,
    *,
    dollar_value_per_bar: float = 5_000_000.0,
) -> pl.DataFrame:
    if dollar_value_per_bar <= 0:
        raise ValueError("dollar_value_per_bar must be > 0")
    clean = _prepare(df)
    eps = dollar_value_per_bar * 1e-9
    grouped = (
        clean.with_columns((pl.col("close") * pl.col("volume")).alias("dollar_value"))
        .with_columns(pl.col("dollar_value").cum_sum().over("symbol").alias("cum_dollar"))
        .with_columns(
            (((pl.col("cum_dollar") - eps).clip(lower_bound=0.0)) / dollar_value_per_bar)
            .floor()
            .cast(pl.Int64)
            .alias("bar_id")
        )
        .drop(["dollar_value", "cum_dollar"])
    )
    return _aggregate(grouped)


def _lag1_autocorr(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    x = values[:-1]
    y = values[1:]
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return cov / (sx * sy)


def bar_diagnostics(df: pl.DataFrame) -> dict[str, float]:
    clean = _prepare(df).sort(["symbol", "ts"])
    returns = (
        clean.with_columns(
            (
                (pl.col("close") - pl.col("close").shift(1).over("symbol"))
                / pl.col("close").shift(1).over("symbol")
            ).alias("ret")
        )
        .drop_nulls(subset=["ret"])
        .select("ret")
        .to_series()
        .to_list()
    )
    if len(returns) < 3:
        return {
            "n_bars": float(clean.height),
            "lag1_return_autocorr": 0.0,
            "lag1_sq_return_autocorr": 0.0,
            "return_std": 0.0,
        }
    sq = [r * r for r in returns]
    mean_r = sum(returns) / len(returns)
    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1))
    return {
        "n_bars": float(clean.height),
        "lag1_return_autocorr": _lag1_autocorr(returns),
        "lag1_sq_return_autocorr": _lag1_autocorr(sq),
        "return_std": std_r,
    }
