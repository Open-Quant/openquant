from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TypeVar

import polars as pl

from . import _core, data

# The bar-size parameter: an int for tick and time bars, a float for volume and dollar bars.
_Param = TypeVar("_Param", int, float)


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


def _rows_to_frame(
    symbol: str, rows: list[tuple[str, str, float, float, float, float, float, float, int]]
) -> pl.DataFrame:
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
    return (
        pl.DataFrame(
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
        )
        .with_columns(
            pl.lit(symbol).alias("symbol"),
            data._parse_ts(pl.col("start_ts")),
            data._parse_ts(pl.col("ts")),
            pl.col("close").alias("adj_close"),
        )
        .select(
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
    )


def _build_by_symbol(
    df: pl.DataFrame,
    rust_builder: Callable[
        [Sequence[str], Sequence[float], Sequence[float], _Param],
        list[tuple[str, str, float, float, float, float, float, float, int]],
    ],
    param: _Param,
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
    """Resample an OHLCV frame into time bars per symbol (AFML section 2.3.1.1).

    The input goes through `openquant.data.clean_ohlcv` first (column aliases resolved, rows
    with nulls dropped, duplicate `(symbol, ts)` keys removed) and is then processed one
    symbol at a time. Each input row is treated as one trade at its `close` price with its
    `volume`; the input `open`, `high`, `low` and `adj_close` are ignored, so the output
    `high`/`low` are the extremes of the input closes.

    A bar closes on the first row at least `interval` after the bar's first row, and that
    row belongs to the closing bar; bars are anchored to each bar's first row, not to the
    wall clock. A trailing partial bar is kept as a final, shorter bar.

    Parameters
    ----------
    df : polars.DataFrame
        Long OHLCV frame with columns `ts`, `symbol`, `open`, `high`, `low`, `close` and
        `volume` (or aliases accepted by `clean_ohlcv`).
    interval : str, default "1d"
        Bar length: an integer followed by `d`, `h`, `m` or `s` (for example `"4h"`,
        `"15m"`); case-insensitive.

    Returns
    -------
    polars.DataFrame
        Columns `ts` (time of the bar's last row), `symbol`, `open`, `high`, `low`, `close`,
        `volume`, `adj_close` (equal to `close`), `start_ts` (time of the bar's first row),
        `n_obs` (rows in the bar) and `dollar_value` (sum of `close * volume`), sorted by
        `symbol` then `ts`.

    Raises
    ------
    ValueError
        If `interval` does not have a supported form or is not positive, or `df` lacks a
        required column.
    """
    return _build_by_symbol(df, _core.bars.build_time_bars, _interval_to_seconds(interval))


def build_tick_bars(df: pl.DataFrame, *, ticks_per_bar: int = 50) -> pl.DataFrame:
    """Group an OHLCV frame into tick bars of a fixed row count per symbol (AFML 2.3.1.2).

    The input goes through `openquant.data.clean_ohlcv` first (column aliases resolved, rows
    with nulls dropped, duplicate `(symbol, ts)` keys removed) and is then processed one
    symbol at a time. Each input row is treated as one trade at its `close` price with its
    `volume`; the input `open`, `high`, `low` and `adj_close` are ignored, so the output
    `high`/`low` are the extremes of the input closes.

    A bar closes once `ticks_per_bar` rows have accumulated; a trailing partial bar is
    dropped.

    Parameters
    ----------
    df : polars.DataFrame
        Long OHLCV frame with columns `ts`, `symbol`, `open`, `high`, `low`, `close` and
        `volume` (or aliases accepted by `clean_ohlcv`).
    ticks_per_bar : int, default 50
        Number of rows per bar.

    Returns
    -------
    polars.DataFrame
        Columns `ts` (time of the bar's last row), `symbol`, `open`, `high`, `low`, `close`,
        `volume`, `adj_close` (equal to `close`), `start_ts` (time of the bar's first row),
        `n_obs` (rows in the bar) and `dollar_value` (sum of `close * volume`), sorted by
        `symbol` then `ts`.

    Raises
    ------
    ValueError
        If `ticks_per_bar <= 0`, or `df` lacks a required column.
    """
    if ticks_per_bar <= 0:
        raise ValueError("ticks_per_bar must be > 0")
    return _build_by_symbol(df, _core.bars.build_tick_bars, ticks_per_bar)


def build_volume_bars(df: pl.DataFrame, *, volume_per_bar: float = 100_000.0) -> pl.DataFrame:
    """Group an OHLCV frame into volume bars per symbol (AFML section 2.3.1.3).

    The input goes through `openquant.data.clean_ohlcv` first (column aliases resolved, rows
    with nulls dropped, duplicate `(symbol, ts)` keys removed) and is then processed one
    symbol at a time. Each input row is treated as one trade at its `close` price with its
    `volume`; the input `open`, `high`, `low` and `adj_close` are ignored, so the output
    `high`/`low` are the extremes of the input closes.

    A bar closes when cumulative `volume` reaches `volume_per_bar`; the row that crosses the
    threshold belongs to the bar it closes, so bars overshoot. A trailing partial bar is
    dropped.

    Parameters
    ----------
    df : polars.DataFrame
        Long OHLCV frame with columns `ts`, `symbol`, `open`, `high`, `low`, `close` and
        `volume` (or aliases accepted by `clean_ohlcv`).
    volume_per_bar : float, default 100_000.0
        Cumulative volume that closes a bar.

    Returns
    -------
    polars.DataFrame
        Columns `ts` (time of the bar's last row), `symbol`, `open`, `high`, `low`, `close`,
        `volume`, `adj_close` (equal to `close`), `start_ts` (time of the bar's first row),
        `n_obs` (rows in the bar) and `dollar_value` (sum of `close * volume`), sorted by
        `symbol` then `ts`.

    Raises
    ------
    ValueError
        If `volume_per_bar` is not positive and finite, or `df` lacks a required column.
    """
    if volume_per_bar <= 0:
        raise ValueError("volume_per_bar must be > 0")
    return _build_by_symbol(df, _core.bars.build_volume_bars, volume_per_bar)


def build_dollar_bars(
    df: pl.DataFrame,
    *,
    dollar_value_per_bar: float = 5_000_000.0,
) -> pl.DataFrame:
    """Group an OHLCV frame into dollar bars per symbol (AFML section 2.3.1.4).

    The input goes through `openquant.data.clean_ohlcv` first (column aliases resolved, rows
    with nulls dropped, duplicate `(symbol, ts)` keys removed) and is then processed one
    symbol at a time. Each input row is treated as one trade at its `close` price with its
    `volume`; the input `open`, `high`, `low` and `adj_close` are ignored, so the output
    `high`/`low` are the extremes of the input closes.

    A bar closes when cumulative `close * volume` reaches `dollar_value_per_bar`; the row
    that crosses the threshold belongs to the bar it closes, so bars overshoot. A trailing
    partial bar is dropped.

    Parameters
    ----------
    df : polars.DataFrame
        Long OHLCV frame with columns `ts`, `symbol`, `open`, `high`, `low`, `close` and
        `volume` (or aliases accepted by `clean_ohlcv`).
    dollar_value_per_bar : float, default 5_000_000.0
        Cumulative traded value that closes a bar.

    Returns
    -------
    polars.DataFrame
        Columns `ts` (time of the bar's last row), `symbol`, `open`, `high`, `low`, `close`,
        `volume`, `adj_close` (equal to `close`), `start_ts` (time of the bar's first row),
        `n_obs` (rows in the bar) and `dollar_value` (sum of `close * volume`), sorted by
        `symbol` then `ts`.

    Raises
    ------
    ValueError
        If `dollar_value_per_bar` is not positive and finite, or `df` lacks a required
        column.
    """
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
    """Summarize the statistical properties of a bar series' close-to-close returns.

    Useful for comparing bar types (AFML section 2.3 argues that activity-based bars have
    returns closer to IID normal than time bars). Simple returns are computed within each
    symbol after `clean_ohlcv`, then pooled across symbols in (symbol, ts) order; the lag-1
    autocorrelations are taken over that pooled list, so with several symbols they include
    the pairs that straddle a symbol boundary.

    Parameters
    ----------
    df : polars.DataFrame
        Long OHLCV frame with columns `ts`, `symbol`, `open`, `high`, `low`, `close` and
        `volume` (or aliases accepted by `clean_ohlcv`).

    Returns
    -------
    dict[str, float]
        `n_bars` (rows after cleaning), `lag1_return_autocorr`, `lag1_sq_return_autocorr`
        (lag-1 autocorrelation of squared returns, a volatility-clustering gauge) and
        `return_std` (sample standard deviation). With fewer than 3 returns, every entry but
        `n_bars` is 0.0; an autocorrelation is also 0.0 when either side has zero variance.

    Raises
    ------
    ValueError
        If `df` lacks a required column.
    """
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
