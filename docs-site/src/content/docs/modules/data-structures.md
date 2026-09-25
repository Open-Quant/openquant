---
title: "data_structures"
description: "Time, tick, volume, dollar, run and imbalance bars built from a stream of trades."
status: authored
last_authored: '2026-09-24'
audience:
  - quant-dev
  - platform-engineering
module: "data_structures"
api_surface: "both"
afml_chapter:
  - "2"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 2, §2.3.1 Standard Bars; §2.3.2 Information-Driven Bars."
  - "Mandelbrot, B. and Taylor, H. (1967). On the distribution of stock price differences. Operations Research 15(6), 1057–1062."
  - "Clark, P. K. (1973). A subordinated stochastic process model with finite variance for speculative prices. Econometrica 41(1), 135–155."
  - "Ané, T. and Geman, H. (2000). Order flow, transaction clock, and normality of asset returns. Journal of Finance 55(5), 2259–2284."
rust_api:
  - "standard_bars"
  - "time_bars"
  - "run_bars"
  - "imbalance_bars"
  - "Trade"
  - "StandardBar"
  - "StandardBarType"
  - "ImbalanceBarType"
python_api:
  - "bars.build_time_bars"
  - "bars.build_tick_bars"
  - "bars.build_volume_bars"
  - "bars.build_dollar_bars"
  - "bars.bar_diagnostics"
sidebar:
  badge: Module
---

A market does not produce information at a constant rate per minute. It produces it per
trade, per share, per dollar. Sampling a price every five minutes therefore oversamples the
quiet hours and undersamples the open, the close and every news release — and the returns of
such a series are heteroskedastic, serially correlated and fat-tailed for reasons that have
nothing to do with the asset (AFML §2.3.1). The alternative, going back to Mandelbrot and
Taylor (1967) and Clark (1973), is to let *activity* be the clock: close a bar every $N$
trades, every $V$ units of volume, or every $D$ dollars exchanged.

This module builds those bars from a slice of trades. It is the first step of every workflow
on this site; [`filters`](/modules/filters/) and [`labeling`](/modules/labeling/) consume its
output.

## What each bar type does

| Builder | A bar closes when | AFML |
| --- | --- | --- |
| `time_bars` | the first trade at least `interval` after the bar's first trade arrives | §2.3.1.1 |
| `standard_bars(…, Tick)` | `threshold` trades have accumulated | §2.3.1.2 |
| `standard_bars(…, Volume)` | cumulative volume reaches `threshold` | §2.3.1.3 |
| `standard_bars(…, Dollar)` | cumulative `price × volume` reaches `threshold` | §2.3.1.4 |
| `imbalance_bars` | the absolute signed imbalance reaches `threshold` | §2.3.2.1–2, simplified |
| `run_bars` | `threshold` consecutive trades move the same way | §2.3.2.3, simplified |

Three behaviours are common to all of them. The trade that crosses the threshold belongs to
the bar it closes, so bars overshoot the threshold rather than undershoot it. A trailing
partial bar is dropped — except by `time_bars`, which emits whatever is left as a final,
shorter bar. And time bars are anchored to the first trade of each bar, not to the wall
clock: a five-minute bar that opens at 09:30:04 closes on the first trade at or after
09:35:04.

For a dollar bar the closing condition is

$$
\sum_{i=t_0}^{t} p_i v_i \;\ge\; \theta
$$

with $t_0$ the first trade after the previous bar closed. AFML prefers dollar bars to tick
and volume bars for a practical reason: a stock that doubles in price halves the number of
shares a given dollar amount buys, and corporate actions change share counts outright, so
only the dollar value exchanged is comparable across years (§2.3.1.4).

### The information-driven bars are simplified

`imbalance_bars` and `run_bars` are **not** the adaptive bars of §2.3.2. In the book the
threshold is an expectation that is re-estimated as bars form — the expected imbalance
$E_0[T]\,\lvert 2P[b_t=1]-1\rvert$, updated by exponentially weighted averages of previous
bars. Here `threshold` is a constant you supply. `imbalance_bars` accumulates the tick-rule
sign $b_t$ (weighted by nothing, volume or dollar value) and closes a bar when
$\lvert\sum b_t\rvert$ reaches it; `run_bars` closes a bar after `threshold` consecutive
same-direction trades, which is a simpler condition than AFML's count of the dominant side
within the bar. Use them as fixed-threshold variants, and do not cite them as the book's
bars.

## The point, on data

The example builds one synthetic session in which seventeen minutes carry far more trading
than the rest, then counts how many bars of each kind close inside those minutes.

```python
import math
import random
from datetime import datetime, timedelta

import polars as pl
from openquant import bars

# One synthetic session of 6,000 trades. Trades 2,500-3,499 are a burst:
# one per second instead of one every 2-7, three times the size, more volatile.
rng = random.Random(7)
t, price, rows = datetime(2024, 1, 2, 9, 30), 100.0, []
for i in range(6000):
    burst = 2500 <= i < 3500
    t += timedelta(seconds=1 if burst else rng.randint(2, 7))
    price *= math.exp(rng.gauss(0, 0.0004 if burst else 0.00015))
    rows.append((t, "ES", price, float(rng.randint(1, 8) * (3 if burst else 1))))
burst_start, burst_end = rows[2500][0], rows[3500][0]

trades = pl.DataFrame(rows, schema=["ts", "symbol", "price", "volume"], orient="row")
# The builders take OHLCV rows; a trade is a row whose four prices are equal.
ohlcv = trades.select(
    "ts", "symbol", open="price", high="price", low="price", close="price", volume="volume"
)

in_burst = trades.filter(pl.col("ts").is_between(burst_start, burst_end, closed="left"))
minutes = (burst_end - burst_start).total_seconds() / 60
session = (rows[-1][0] - rows[0][0]).total_seconds() / 60
print(f"burst: {minutes:.0f} of {session:.0f} minutes ({minutes / session:.0%}), "
      f"{in_burst.height / trades.height:.0%} of trades, "
      f"{in_burst['volume'].sum() / trades['volume'].sum():.0%} of volume")

for name, frame in [
    ("time 5m", bars.build_time_bars(ohlcv, interval="5m")),
    ("tick 75", bars.build_tick_bars(ohlcv, ticks_per_bar=75)),
    ("volume 600", bars.build_volume_bars(ohlcv, volume_per_bar=600.0)),
    ("dollar 60k", bars.build_dollar_bars(ohlcv, dollar_value_per_bar=60_000.0)),
]:
    closes = frame.filter(pl.col("ts").is_between(burst_start, burst_end, closed="right"))
    share = closes.height / frame.height
    print(f"{name:11s} {frame.height:3d} bars, {closes.height:2d} close inside the burst ({share:.0%})")
```

```text
burst: 17 of 394 minutes (4%), 17% of trades, 38% of volume
time 5m      78 bars,  4 close inside the burst (5%)
tick 75      80 bars, 13 close inside the burst (16%)
volume 600   59 bars, 22 close inside the burst (37%)
dollar 60k   59 bars, 22 close inside the burst (37%)
```

Each bar type spends its observations in proportion to its own clock. Time bars give the
burst 5% of their bars because it is 4% of the session; tick bars give it 16% because it holds
17% of the trades; volume and dollar bars give it 37% because it holds 38% of the volume. A
model trained on the time bars sees four observations of the only interesting quarter-hour
of the day.

<figure>
<img class="dark:sl-hidden" src="/figures/ch2-bars-light.svg" alt="Price over one synthetic session with the closing times of 78 five-minute bars and 59 dollar bars marked beneath it. The time-bar marks are evenly spaced; the dollar-bar marks crowd into a shaded seventeen-minute burst of trading." />
<img class="light:sl-hidden" src="/figures/ch2-bars-dark.svg" alt="Price over one synthetic session with the closing times of 78 five-minute bars and 59 dollar bars marked beneath it. The time-bar marks are evenly spaced; the dollar-bar marks crowd into a shaded seventeen-minute burst of trading." />
<figcaption>Where the bars close. Same trades, same session; only the clock differs.</figcaption>
</figure>

## Choosing a threshold

The threshold sets how many bars you get, and that is a modelling decision, not a default to
accept. A common starting point is the number of bars you would have had with time bars:
divide the session's total dollar volume by that count. The example traded about 3.6 million
dollars, so a threshold of 60,000 gives roughly the sixty bars that five-minute sampling gives
in a quieter session. The Python defaults (`ticks_per_bar=50`, `volume_per_bar=100_000`,
`dollar_value_per_bar=5_000_000`) suit no instrument in particular. Pick a number, then check
the bar count and `bars.bar_diagnostics` before trusting what is built on top.

Because thresholds are in absolute units, a dollar threshold fixed in 2015 produces several
times as many bars on the same instrument in 2024. Over long histories, re-derive the
threshold per period or scale it by a rolling average of daily dollar volume.

## From Rust

```rust
use chrono::{Duration, NaiveDate};
use openquant::data_structures::{standard_bars, time_bars, StandardBarType, Trade};

let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 30, 0).unwrap();
let trades: Vec<Trade> = (0..1_000)
    .map(|i| Trade {
        timestamp: open + Duration::seconds(i * 3),
        price: 100.0 + (i as f64 * 0.01).sin(),
        volume: 1.0 + (i % 5) as f64,
    })
    .collect();

let dollar = standard_bars(&trades, 25_000.0, StandardBarType::Dollar)?;
let five_minute = time_bars(&trades, Duration::minutes(5))?;

// Every bar carries OHLC, volume, dollar value, tick count and both of its timestamps.
let first = &dollar[0];
assert!(first.dollar_value >= 25_000.0);
assert!(first.start_timestamp <= first.timestamp);
// 1,000 trades three seconds apart: nine full five-minute bars and the partial one left over.
assert_eq!(five_minute.len(), 10);
```

A non-positive threshold or interval is an `InputError`, not a panic. An empty slice of
trades is valid and yields no bars.

## What to watch for

- **Identical timestamps collapse in Python.** Sub-second stamps are kept (microseconds
  through `openquant.bars`, which is the resolution of a Python `datetime`). But the Python
  builders first pass the frame through `data.clean_ohlcv`, which removes rows that repeat a
  `(symbol, ts)` pair, so two trades with exactly the same stamp collapse into one. If your
  feed stamps several trades identically, build bars in Rust.
- **The Python builders take OHLCV rows, not trades.** They read `close` as the trade price
  and `volume` as the trade size. Feeding them one-minute OHLCV bars works and is sometimes
  what you want, but the result is a bar of bars: its high and low come from minute closes,
  not from the minute highs and lows.
- **Run and imbalance bars have no Python wrapper.** `openquant._core.bars` exposes
  `build_run_bars` and `build_imbalance_bars`; `openquant.bars` does not wrap them in a frame.
- **Bars overshoot.** One block trade can be many thresholds wide, and it still closes exactly
  one bar. `dollar_value` on the bar records what was actually accumulated.

## Related modules

- [`filters`](/modules/filters/) — sample events from the bars built here.
- [`labeling`](/modules/labeling/) — label those events.
- [`microstructural-features`](/modules/microstructural-features/) — per-bar liquidity and
  order-flow features from the same trades.
- [`data`](/modules/data/) — load and clean the OHLCV frame the Python builders expect.
