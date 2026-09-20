"""Figures for the AFML Chapter 2 module pages. Run from the repo root:

    .venv/bin/python docs-site/scripts/figures/ch2_figures.py

The data is the same seeded synthetic series the pages' Python examples build, so the pictures
show exactly what the printed output describes.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

import polars as pl

from _svg import MONO, THEMES, Chart
from openquant import bars, filters


def ticks():
    rng = random.Random(7)
    t, price, rows = datetime(2024, 1, 2, 9, 30), 100.0, []
    for i in range(6000):
        burst = 2500 <= i < 3500
        t += timedelta(seconds=1 if burst else rng.randint(2, 7))
        price *= math.exp(rng.gauss(0, 0.0004 if burst else 0.00015))
        rows.append((t, "ES", price, float(rng.randint(1, 8) * (3 if burst else 1))))
    return rows


def bars_figure():
    rows = ticks()
    frame = pl.DataFrame(rows, schema=["ts", "symbol", "price", "volume"], orient="row")
    ohlcv = frame.select("ts", "symbol", open="price", high="price", low="price", close="price", volume="volume")
    time_bars = bars.build_time_bars(ohlcv, interval="5m")["ts"].to_list()
    dollar_bars = bars.build_dollar_bars(ohlcv, dollar_value_per_bar=60_000.0)["ts"].to_list()

    t0, t1 = rows[0][0], rows[-1][0]
    secs = lambda t: (t - t0).total_seconds()
    prices = [r[2] for r in rows]
    minutes = (rows[3500][0] - rows[2500][0]).total_seconds() / 60
    share = sum(r[3] for r in rows[2500:3500]) / sum(r[3] for r in rows)
    for name, theme in THEMES.items():
        ch = Chart(760, 330, theme, "Time bars and dollar bars over one synthetic trading day")
        x = ch.scale(0, secs(t1), 56, 740)
        y = ch.scale(min(prices), max(prices), 190, 34)
        ch.band(x(secs(rows[2500][0])), x(secs(rows[3500][0])), 24, 300)
        ch.label(x(secs(rows[2500][0])) + 6, 40, f"{minutes:.0f} min · {share:.0%} of volume", "muted")
        ch.line([(x(secs(r[0])), y(r[2])) for r in rows[::6]])
        ch.label(56, 222, f"time bars · {len(time_bars)}")
        for t in time_bars:
            ch.rule(x(secs(t)), 230, x(secs(t)), 250, "muted", 1.1)
        ch.label(56, 272, f"dollar bars · {len(dollar_bars)}", "accent")
        for t in dollar_bars:
            ch.rule(x(secs(t)), 280, x(secs(t)), 300, "accent", 1.1)
        for hour in range(10, 17):
            hx = x(secs(datetime(2024, 1, 2, hour)))
            ch.text(hx, 322, f"{hour}:00", size=11, anchor="middle", family=MONO)
        for v in (min(prices), max(prices)):
            ch.text(48, y(v) + 4, f"{v:.1f}", size=11, anchor="end", family=MONO)
        print(ch.save("ch2-bars", name))


def cusum_figure():
    rng = random.Random(11)
    close, p = [], 100.0
    for i in range(600):
        p *= math.exp(rng.gauss(0, 0.004 if 300 <= i < 400 else 0.001))
        close.append(p)
    events = filters.cusum_filter_indices(close, 0.01)

    for name, theme in THEMES.items():
        ch = Chart(760, 300, theme, "CUSUM filter events on a synthetic minute series")
        x = ch.scale(0, 599, 56, 740)
        y = ch.scale(min(close), max(close), 250, 34)
        ch.band(x(300), x(400), 24, 262)
        ch.label(x(300) + 6, 40, "volatility ×4")
        ch.line([(x(i), y(v)) for i, v in enumerate(close)])
        for i in events:
            ch.dot(x(i), y(close[i]))
        ch.label(56, 40, f"{len(events)} events · h = 1%", "accent")
        for i in range(0, 601, 100):
            ch.text(x(min(i, 599)), 284, str(i), size=11, anchor="middle", family=MONO)
        for v in (min(close), max(close)):
            ch.text(48, y(v) + 4, f"{v:.1f}", size=11, anchor="end", family=MONO)
        print(ch.save("ch2-cusum", name))


if __name__ == "__main__":
    bars_figure()
    cusum_figure()
