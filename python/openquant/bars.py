from __future__ import annotations

import math
from typing import Callable

import polars as pl

from . import _core
from . import data


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


def _rows_to_frame(symbol: str, rows: list[tuple[str, str, float, float, float, float, float, float, int]]) -> pl.DataFrame:
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
                "start_ts": [],
                "n_obs": [],
                "dollar_value": [],
            }
        )
    return pl.DataFrame(
        {
            "start_ts": [r[0] for r in rows],
            "ts": [r[1] for r in rows],
            "open": [r[2] for r in rows],
            "high": [r[3] for r in rows],
            "low": [r[4] for r in rows],
            "close": [r[5] for r in rows],
            "volume": [r[6] for r in rows],
            "dollar_value": [r[7] for r in rows],
            "n_obs": [r[8] for r in rows],
        }
    ).with_columns(
        pl.lit(symbol).alias("symbol"),
        pl.col("start_ts").str.strptime(pl.Datetime, strict=False),
        pl.col("ts").str.strptime(pl.Datetime, strict=False),
        pl.col("close").alias("adj_close"),
    ).select(
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
            "dollar_value",
        ]
    )


def _build_by_symbol(
    df: pl.DataFrame,
    rust_builder: Callable[[list[str], list[float], list[float], float | int], list[tuple[str, str, float, float, float, float, float, float, int]]],
    param: float | int,
) -> pl.DataFrame:
    clean = data.clean_ohlcv(df).sort(["symbol", "ts"])
    out_frames: list[pl.DataFrame] = []
    for symbol in clean["symbol"].unique().to_list():
        sdf = clean.filter(pl.col("symbol") == symbol).sort("ts")
        rows = rust_builder(
            [str(x) for x in sdf["ts"].to_list()],
            [float(x) for x in sdf["close"].to_list()],
            [float(x) for x in sdf["volume"].to_list()],
            param,
        )
        out_frames.append(_rows_to_frame(symbol, rows))
    if not out_frames:
        return _rows_to_frame("", [])
    return pl.concat(out_frames, how="vertical").sort(["symbol", "ts"])


def build_time_bars(df: pl.DataFrame, *, interval: str = "1d") -> pl.DataFrame:
    return _build_by_symbol(df, _core.bars.build_time_bars, _interval_to_seconds(interval))


def build_tick_bars(df: pl.DataFrame, *, ticks_per_bar: int = 50) -> pl.DataFrame:
    if ticks_per_bar <= 0:
        raise ValueError("ticks_per_bar must be > 0")
    return _build_by_symbol(df, _core.bars.build_tick_bars, ticks_per_bar)


def build_volume_bars(df: pl.DataFrame, *, volume_per_bar: float = 100_000.0) -> pl.DataFrame:
    if volume_per_bar <= 0:
        raise ValueError("volume_per_bar must be > 0")
    return _build_by_symbol(df, _core.bars.build_volume_bars, volume_per_bar)


def build_dollar_bars(
    df: pl.DataFrame,
    *,
    dollar_value_per_bar: float = 5_000_000.0,
) -> pl.DataFrame:
    if dollar_value_per_bar <= 0:
        raise ValueError("dollar_value_per_bar must be > 0")
    return _build_by_symbol(df, _core.bars.build_dollar_bars, dollar_value_per_bar)


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
    clean = data.clean_ohlcv(df).sort(["symbol", "ts"])
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
