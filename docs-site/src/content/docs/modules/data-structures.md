---
title: "data_structures"
description: "Time, tick, volume, dollar, run and imbalance bars built from a stream of trades."
status: authored
last_authored: '2026-11-12'
audience:
  - quant-dev
  - platform-engineering
module: "data_structures"
python_module: "bars"
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

Build OHLC bars by grouping observations according to elapsed time, trade count,
volume or traded value. Activity-based bars use trading activity as the clock:
close a bar every $N$ trades, $V$ units of volume or $D$ units of price × volume.
AFML §2.3.1 discusses why these can be useful alternatives to regularly spaced
bars. They do **not** guarantee independent, normal or constant-variance returns;
measure those properties on the data and sampling policy you actually use.

You do not need to rebuild bars if your input already has the sampling definition
your study needs. [`filters`](/modules/filters/) and
[`labeling`](/modules/labeling/) can consume an appropriate existing price series.

## Choose the input path before building bars

| Input | Path | Important limit |
| --- | --- | --- |
| OHLCV-like Polars frame with one observation per symbol/timestamp | Public Python `openquant.bars` wrappers | Each row becomes a trade at **close** with its volume; input high/low are not aggregated |
| Raw trades, including legitimate equal-timestamp prints | Native list builders (`openquant._core.bars`) or Rust `Trade` builders | Caller supplies valid inputs in the correct order; see the [retained-trades check](#check-which-trades-cleaning-retained) |
| Already aggregated candles whose full OHLC range must survive | Keep those candles or use an explicit OHLC resampler | These trade builders cannot reconstruct an intrabar path from a candle's close |

The Python frame wrappers clean and deduplicate `(symbol, ts)` before building
bars. An exchange trade ID is not part of that key. A tick feed can therefore
lose genuine trades even if its rows are otherwise valid. The low-level `_core`
namespace avoids that cleaning but is not a drop-in frame API: check the exact
signature, output tuple and validation responsibilities before using it.

For a first audit, [calculate a small bar by hand](#audit-a-small-bar-by-hand).
For a file split into chunks, read [the batching example](#decide-where-aggregation-state-resets)
and the caller contract below before assuming separate calls equal one pass.

## What each bar type does

| Builder | A bar closes when | AFML |
| --- | --- | --- |
| `time_bars` | the first trade at least `interval` after the bar's first trade arrives | §2.3.1.1 |
| `standard_bars(…, Tick)` | `threshold` trades have accumulated | §2.3.1.2 |
| `standard_bars(…, Volume)` | cumulative volume reaches `threshold` | §2.3.1.3 |
| `standard_bars(…, Dollar)` | cumulative `price × volume` reaches `threshold` | §2.3.1.4 |
| `imbalance_bars` | the absolute signed imbalance reaches `threshold` | §2.3.2.1–2, simplified |
| `run_bars` | `threshold` consecutive signed ticks have the same sign, including unchanged prices that inherit it | §2.3.2.3, simplified |

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

Their direction is the tick rule: up-ticks are +1, down-ticks −1, and unchanged prices
retain the previous sign. The first trade establishes the price reference and contributes
no direction; unchanged initial prices remain unsigned. After each completed bar, direction
state resets and the next trade establishes a fresh reference. A run counts signed ticks,
including unchanged prices that retain a sign, rather than strictly changing prices only.

### Trace direction across a completed bar

The reset applies at **every completed bar**, not just between calls. A flat price
after a reset has no direction until a new up-tick or down-tick establishes one.
The unsigned trades still belong to the bar: they contribute to its OHLC, volume,
dollar value and observation count, but not its signed stopping statistic.

For these eight prices, both a run threshold of two and a tick-imbalance threshold
of two close at rows 2 and 6:

| Row | Price | Sign within the current bar | Result |
| --- | --- | --- | --- |
| 0 | 100 | 0 | Establish reference |
| 1 | 101 | +1 | First signed tick |
| 2 | 101 | +1 | Flat price inherits +1; close first bar |
| 3 | 101 | 0 | Fresh reference; previous sign is discarded |
| 4 | 101 | 0 | Still unsigned |
| 5 | 102 | +1 | First signed tick of second bar |
| 6 | 102 | +1 | Close second bar |
| 7 | 102 | 0 | Unfinished tail is dropped |

```python
from openquant import _core

stamps = [f"2026-01-02 09:30:{i:02d}" for i in range(8)]
prices = [100.0, 101.0, 101.0, 101.0, 101.0, 102.0, 102.0, 102.0]
volumes = [1.0] * len(prices)
outputs = {
    "run": _core.bars.build_run_bars(stamps, prices, volumes, 2),
    "tick imbalance": _core.bars.build_imbalance_bars(
        stamps, prices, volumes, 2.0, "tick"
    ),
}
for name, rows in outputs.items():
    # Native tuple: start, end, open, high, low, close, volume, dollar value, count.
    assert [(row[0], row[1]) for row in rows] == [
        (stamps[0], stamps[2]), (stamps[3], stamps[6])
    ]
    assert [row[2:] for row in rows] == [
        (100.0, 101.0, 100.0, 101.0, 3.0, 302.0, 3),
        (101.0, 102.0, 101.0, 102.0, 4.0, 406.0, 4),
    ]
    print(name, "counts:", [row[8] for row in rows])
```

```text
run counts: [3, 4]
tick imbalance counts: [3, 4]
```

Carrying +1 from the first bar would instead let the next two flat trades close
a bar. That is a different sampling policy, not an equivalent optimization.
[#279](https://github.com/Open-Quant/openquant/issues/279) tracks the choice of
cross-bar direction policy. This example records the implemented reset behavior;
it does not resolve that design question or establish parity with another library.

### Separate consecutive runs from net imbalance

The two builders agree above only because every signed tick within each completed
bar is positive. In this second path the signs are `0, +1, +1, -1, +1, +1`.
Net imbalance reaches three, but the longest uninterrupted run is only two:

```python
from openquant import _core

stamps = [f"2026-01-02 09:30:{i:02d}" for i in range(6)]
prices, volumes = [100.0, 101.0, 102.0, 101.0, 102.0, 103.0], [1.0] * 6
runs = _core.bars.build_run_bars(stamps, prices, volumes, 3)
imbalances = _core.bars.build_imbalance_bars(stamps, prices, volumes, 3.0, "tick")
assert runs == []
assert len(imbalances) == 1
assert imbalances[0] == (
    stamps[0], stamps[5], 100.0, 103.0, 100.0, 103.0, 6.0, 609.0, 6
)
print("run bars:", len(runs), "tick-imbalance bars:", len(imbalances))
```

```text
run bars: 0 tick-imbalance bars: 1
```

Neither the number of price changes nor the dominant side's total count is the
run stopping rule. Conversely, an opposite sign subtracts from imbalance; it does
not discard all earlier signed accumulation.

### Distinguish signed activity from reported totals

A large first trade is unsigned even in a volume- or dollar-imbalance bar.
For the following path the signed totals after all four rows are one tick,
three volume units (`2 - 1 + 2`), and 306 dollars (`202 - 100 + 204`).
The reported totals include **all** trades, including the first 1,000-unit trade
and the negatively signed one. They are volume 1,005 and dollar value 100,506.

```python
from openquant import _core

stamps = [f"2026-01-02 09:30:{i:02d}" for i in range(4)]
prices, volumes = [100.0, 101.0, 100.0, 102.0], [1000.0, 2.0, 1.0, 2.0]
for kind, threshold in [("tick", 2.0), ("volume", 3.0), ("dollar", 300.0)]:
    rows = _core.bars.build_imbalance_bars(stamps, prices, volumes, threshold, kind)
    if kind == "tick":
        assert rows == []
    else:
        assert rows == [(
            stamps[0], stamps[3], 100.0, 102.0, 100.0, 102.0,
            1005.0, 100506.0, 4
        )]
    print(kind, "counts:", [row[8] for row in rows])
```

```text
tick counts: []
volume counts: [4]
dollar counts: [4]
```

The tuple does not return the signed imbalance or run length. Reconstruct that
statistic from the input prices, sizes and reset policy when auditing a boundary;
do not infer it from `volume`, `dollar_value` or `tick_count` alone. Keep the threshold's
units with the chosen imbalance type. These tiny paths test mechanics, not whether
any of the three thresholds improves a research result.

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
divide a completed reference session's total dollar volume by that count. The example traded about 3.6 million
dollars, so a threshold of 60,000 gives roughly the sixty bars that five-minute sampling gives
in a quieter session. The Python defaults (`ticks_per_bar=50`, `volume_per_bar=100_000`,
`dollar_value_per_bar=5_000_000`) suit no instrument in particular. Pick a number, then check
the bar count and `bars.bar_diagnostics` before trusting what is built on top.

Because thresholds are in absolute units, a fixed dollar threshold can produce different
bar counts as traded dollar volume changes. Over long histories, a policy may re-derive the
threshold per period or scale it by a rolling average of daily dollar volume. Use completed
past periods for a causal sampling policy; choosing a threshold from the entire future
evaluation session leaks its total activity. Record the estimation window and when the
threshold becomes effective, and include alternative policies in the trial inventory.

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
  feed stamps several trades identically, the Rust functions or list-based
  `openquant._core.bars` builders preserve those rows; supply them in chronological order.
- **The Python builders take OHLCV rows, not trades.** They read `close` as the trade price
  and `volume` as the trade size. Feeding them one-minute OHLCV bars works and is sometimes
  what you want, but the result is a bar of bars: its high and low come from minute closes,
  not from the minute highs and lows.
- **Run and imbalance bars have no Python wrapper.** `openquant._core.bars` exposes
  `build_run_bars` and `build_imbalance_bars`; `openquant.bars` does not wrap them in a frame.
- **Bars overshoot.** One block trade can be many thresholds wide, and it still closes exactly
  one bar. `dollar_value` on the bar records what was actually accumulated.
- **Partial symbols do not block complete bars.** Python tick, volume and dollar builders
  omit symbols that never reach the threshold and keep complete bars from other symbols.
  When no bars remain (including empty input or rows all dropped by cleaning), the output
  still has the same columns and types as a populated result: microsecond datetimes for `ts` and
  `start_ts`, a string `symbol`, `Int64` `n_obs`, and `Float64` prices, volume and dollar value.
  Time bars still keep each symbol's trailing partial bar.

- **Order, cleaning and sessions are caller responsibilities at the native boundary.**
  Rust and `_core.bars` process the supplied order without sorting or validating trade
  prices/volumes. Frame wrappers sort each symbol and remove canonical-column nulls, but
  do not turn `NaN` or infinity into valid observations. Validate finite positive prices
  and finite nonnegative sizes yourself. There is no automatic daily/session reset.
- **Calls do not carry unfinished bars forward.** Splitting one symbol by date or arbitrary
  row chunks resets accumulation, drops unfinished tick/volume/dollar bars, and emits extra
  partial time bars. Processing complete symbol histories separately preserves the
  per-symbol behavior; processing separate sessions is an explicit change of bar definition.
- **Inspect diagnostics per symbol.** `bar_diagnostics` computes returns within symbols,
  then pools them for lag-1 correlations. Multiple symbols therefore introduce artificial
  adjacent pairs at symbol boundaries. With fewer than three pooled returns, all reported
  statistics except `n_bars` are zero, which does not establish low volatility or independence.

## Audit a small bar by hand

Use a short input whose expected result you can calculate without the library before
running a large feed. These five rows deliberately have different input highs/lows,
so the example also catches the mistake of treating the frame wrapper as an OHLC
resampler. The wrappers see only the closes `100, 101, 102, 103, 104` and volumes
`2, 2, 7, 1, 1` after cleaning.

| Builder | First completed bar | What happens to the last rows |
| --- | --- | --- |
| Three ticks | Rows 0–2, volume 11, traded value 1,116 | Rows 3–4 are dropped |
| Volume 5 | Rows 0–2, volume 11: the seven-unit row is not split | Rows 3–4 are dropped; the overshoot is not carried forward |
| Dollar value 500 | Rows 0–2, traded value 1,116 | Rows 3–4 are dropped; no second bar is created from the overshoot |
| One minute | Rows 0–2, because row 2 is exactly one minute after row 0 | Rows 3–4 are retained as a partial bar |

The first bar's open/high/low/close are `100, 102, 100, 102`, independent of the
input `open=90`, `high=110` and `low=80`. Its traded value is
`100 × 2 + 101 × 2 + 102 × 7 = 1,116`, not the closing price times total volume.

```python
from datetime import datetime, timedelta

import polars as pl
from openquant import bars

start = datetime(2026, 1, 2, 9, 30)
frame = pl.DataFrame({
    "ts": [start + timedelta(seconds=30 * i) for i in range(5)],
    "symbol": ["X"] * 5,
    "open": [90.0] * 5, "high": [110.0] * 5, "low": [80.0] * 5,
    "close": [100.0, 101.0, 102.0, 103.0, 104.0],
    "volume": [2.0, 2.0, 7.0, 1.0, 1.0],
})
columns = ["open", "high", "low", "close", "volume", "dollar_value", "n_obs"]
# Literal arithmetic above, not expected values generated by another OpenQuant call.
expected_first = (100.0, 102.0, 100.0, 102.0, 11.0, 1116.0, 3)
outputs = {
    "tick": bars.build_tick_bars(frame, ticks_per_bar=3),
    "volume": bars.build_volume_bars(frame, volume_per_bar=5.0),
    "dollar": bars.build_dollar_bars(frame, dollar_value_per_bar=500.0),
    "time": bars.build_time_bars(frame, interval="1m"),
}
for name, result in outputs.items():
    assert result.select(columns).row(0) == expected_first
    assert result["start_ts"][0] == start
    assert result["ts"][0] == start + timedelta(minutes=1)
    assert result["adj_close"].to_list() == result["close"].to_list()
    print(name, "n_obs:", result["n_obs"].to_list())

assert outputs["time"].select(columns).row(1) == (
    103.0, 104.0, 103.0, 104.0, 2.0, 207.0, 2
)
assert outputs["time"]["start_ts"][1] == start + timedelta(seconds=90)
assert outputs["time"]["ts"][1] == start + timedelta(seconds=120)
```

```text
tick n_obs: [3]
volume n_obs: [3]
dollar n_obs: [3]
time n_obs: [3, 2]
```

This checks boundary membership, aggregation and trailing-row policy. It does not
show that the chosen threshold is suitable for a market, that raw trades were
preserved by cleaning, or that the resulting returns are independent.

## Decide where aggregation state resets

A call boundary is a reset, not a calendar session boundary recognized by the
builder. The following four trades straddle midnight. Tick, volume and dollar
bars each form one three-trade bar when called once. Two calls with two rows each
form none. One-minute time bars instead retain each call's partial tail: the
single pass groups rows as `[3, 1]`, while the split calls group them as `[2, 2]`.
The time-bar count is two in both cases, so comparing only counts misses the change.

```python
from openquant import _core

stamps = [
    "2026-01-02 23:59:00", "2026-01-02 23:59:30",
    "2026-01-03 00:00:00", "2026-01-03 00:00:30",
]
prices, volumes = [100.0] * 4, [1.0] * 4
builders = [
    ("tick", _core.bars.build_tick_bars, 3),
    ("volume", _core.bars.build_volume_bars, 3.0),
    ("dollar", _core.bars.build_dollar_bars, 300.0),
    ("time", _core.bars.build_time_bars, 60),
]
for name, build, threshold in builders:
    whole = build(stamps, prices, volumes, threshold)
    split = [row for lo, hi in [(0, 2), (2, 4)]
             for row in build(stamps[lo:hi], prices[lo:hi], volumes[lo:hi], threshold)]
    # Native bar tuples end with the observation count, at position 8.
    whole_counts = [row[8] for row in whole]
    split_counts = [row[8] for row in split]
    assert whole_counts == ([3, 1] if name == "time" else [3])
    assert split_counts == ([2, 2] if name == "time" else [])
    print(name, "whole:", whole_counts, "split:", split_counts)
```

```text
tick whole: [3] split: []
volume whole: [3] split: []
dollar whole: [3] split: []
time whole: [3, 1] split: [2, 2]
```

For a continuous definition, process each symbol's complete ordered history in one
call. If you intentionally reset at sessions, split by the venue's explicit session
calendar before calling and record that rule; splitting at UTC midnight is not a
universal exchange-session policy. A state-preserving streaming implementation would
need its own single-pass equivalence tests. None of these wrappers exposes carry state.

## Check which trades cleaning retained

The frame wrappers invoke `data.clean_ohlcv` before aggregation. The native list
builders do not. Here two different trades have the same timestamp. The cleaner
retains one `(symbol, ts)` row, so the frame wrapper cannot complete a three-tick
bar. The native call retains all three and can complete it. The example avoids
assuming which duplicate wins: if both trades matter, either outcome loses data.

```python
import polars as pl
from openquant import _core, bars, data

stamps = ["2026-01-02 09:30:00", "2026-01-02 09:30:00", "2026-01-02 09:30:01"]
prices, volumes = [100.0, 101.0, 102.0], [1.0, 1.0, 1.0]
raw = pl.DataFrame({
    "ts": stamps, "symbol": ["X"] * 3,
    "open": prices, "high": prices, "low": prices, "close": prices,
    "volume": volumes,
})
cleaned = data.clean_ohlcv(raw)
wrapped = bars.build_tick_bars(raw, ticks_per_bar=3)
native = _core.bars.build_tick_bars(stamps, prices, volumes, 3)
assert cleaned.height == 2 and wrapped.height == 0
assert len(native) == 1 and native[0][8] == 3
assert native[0][2:8] == (100.0, 102.0, 100.0, 102.0, 3.0, 303.0)
print("input rows:", raw.height, "cleaned rows:", cleaned.height)
print("three-tick bars: frame", wrapped.height, "native", len(native))
```

```text
input rows: 3 cleaned rows: 2
three-tick bars: frame 0 native 1
```

For a raw trade feed, preserve a sequence identifier and confirm the timestamp
resolution and duplicate policy before choosing the frame wrapper. Do not add
invented timestamp offsets to force trades through the cleaner: those offsets
change ordering and time-bar membership. A native call requires you to supply the
correct order and validate the prices and sizes yourself. If the feed does not
establish an order among equal timestamps, record that ambiguity.

### A bar-construction handoff

A reproducible result needs more than `interval="1m"` or a threshold:

- **Input meaning:** trades versus pre-aggregated OHLCV rows, price and volume units,
  symbol, time representation and ordering among equal timestamps.
- **Retained observations:** raw and cleaned counts, duplicate/null treatment, and
  whether timestamp conversion reduced precision. Use the [cleaning report](/modules/data/#cleaning-and-provenance-contracts)
  and keep the input identity with it.
- **Sampling rule:** exact builder/namespace and threshold, per-symbol grouping,
  session/reset policy, and the trailing-partial policy.
  For run/imbalance bars, also record the fixed-threshold variant, imbalance weight
  (`tick`, `volume` or `dollar`), and the within-bar tie and cross-bar direction rules.
- **Output evidence:** compare timestamps, OHLC, volume, traded value and observation
  count on a hand-calculated sample, not only the total number of bars. Record the
  loaded source/build as described in the [agent guide](/agents/#resolve-the-source-of-truth).
- **Research boundary:** choose and fit sampling policies using information available
  at the intended decision time. A threshold derived from a complete future test
  session uses future activity even though each threshold crossing uses only rows
  accumulated so far. A trailing time bar is emitted at end of input, so its final
  completeness is not established merely by its last trade timestamp.

The examples above are deterministic contract checks against
[`python/openquant/bars.py`](https://github.com/Open-Quant/openquant/blob/main/python/openquant/bars.py),
[`crates/pyopenquant/src/bars.rs`](https://github.com/Open-Quant/openquant/blob/main/crates/pyopenquant/src/bars.rs)
and [`crates/openquant/src/data_structures.rs`](https://github.com/Open-Quant/openquant/blob/main/crates/openquant/src/data_structures.rs).
They validate the displayed cases, not provider data quality or a trading result.

## Related modules

- [`filters`](/modules/filters/) — sample events from the bars built here.
- [`labeling`](/modules/labeling/) — label those events.
- [`microstructural-features`](/modules/microstructural-features/) — per-bar liquidity and
  order-flow features from the same trades.
- [`data`](/modules/data/) — load and clean the OHLCV frame the Python builders expect.
