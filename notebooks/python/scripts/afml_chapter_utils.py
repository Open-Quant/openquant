from __future__ import annotations

import csv
import io
import math
import urllib.request
from typing import Iterable

import polars as pl

SYMBOL_MAP = {
    "USO": "uso.us",
    "BNO": "bno.us",
    "XLE": "xle.us",
    "GLD": "gld.us",
    "UNG": "ung.us",
}


def fetch_stooq_ohlcv(symbol: str) -> pl.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    txt = urllib.request.urlopen(url, timeout=30).read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(txt)))
    if not rows:
        raise RuntimeError(f"No data for {symbol}")
    return pl.DataFrame(rows).select(
        pl.col("Date").alias("date"),
        pl.col("Open").cast(pl.Float64).alias("open"),
        pl.col("High").cast(pl.Float64).alias("high"),
        pl.col("Low").cast(pl.Float64).alias("low"),
        pl.col("Close").cast(pl.Float64).alias("close"),
        pl.col("Volume").cast(pl.Float64).alias("volume"),
    )


def fetch_panel(window: int = 900) -> pl.DataFrame:
    joined = None
    for name, sym in SYMBOL_MAP.items():
        df = fetch_stooq_ohlcv(sym).rename({"close": name, "volume": f"{name}_volume"})
        keep = ["date", name, f"{name}_volume"]
        df = df.select(keep)
        joined = df if joined is None else joined.join(df, on="date", how="inner")
    joined = joined.sort("date").tail(window).drop_nulls()
    return joined


def simple_returns(series: list[float]) -> list[float]:
    out = [0.0]
    for i in range(1, len(series)):
        out.append(series[i] / series[i - 1] - 1.0)
    return out


def probs_and_sides_from_momentum(close: list[float], lookback: int = 5) -> tuple[list[float], list[float]]:
    ret = simple_returns(close)
    probs, sides = [], []
    for i in range(len(ret)):
        chunk = ret[max(0, i - lookback) : i + 1]
        edge = sum(chunk) / max(len(chunk), 1)
        p = 0.5 + max(min(edge * 18.0, 0.2), -0.2)
        probs.append(min(max(p, 0.05), 0.95))
        sides.append(1.0 if edge >= 0 else -1.0)
    return probs, sides


def timestamps_from_dates(dates: Iterable[str]) -> list[str]:
    return [f"{d} 00:00:00" for d in dates]


def lag_corr(x: list[float], y: list[float], lag: int) -> float:
    if lag <= 0 or lag >= len(x):
        return float("nan")
    x_l = x[:-lag]
    y_t = y[lag:]
    mx = sum(x_l) / len(x_l)
    my = sum(y_t) / len(y_t)
    cov = sum((a - mx) * (b - my) for a, b in zip(x_l, y_t))
    vx = sum((a - mx) ** 2 for a in x_l)
    vy = sum((b - my) ** 2 for b in y_t)
    den = math.sqrt(vx * vy)
    return cov / den if den > 0 else float("nan")


def fracdiff_ffd(series: list[float], d: float = 0.4, thresh: float = 1e-5) -> list[float]:
    weights = [1.0]
    k = 1
    while k < len(series):
        w = -weights[-1] * (d - k + 1.0) / k
        if abs(w) < thresh:
            break
        weights.append(w)
        k += 1
    weights = list(reversed(weights))
    width = len(weights) - 1
    out = [float("nan")] * len(series)
    for i in range(width, len(series)):
        loc0 = i - width
        out[i] = sum(weights[j] * series[loc0 + j] for j in range(len(weights)))
    return out
