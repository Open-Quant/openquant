---
title: "microstructural_features"
description: "Spread, price-impact, order-flow and entropy features estimated from bars or from trades."
status: authored
last_authored: '2026-09-24'
audience:
  - quant-dev
  - platform-engineering
module: "microstructural_features"
api_surface: "both"
afml_chapter:
  - "18"
  - "19"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 19: §19.3.1 The Tick Rule; §19.3.2 The Roll Model; §19.3.3 High-Low Volatility Estimator; §19.3.4 Corwin and Schultz (Snippets 19.1–19.2); §19.4.1 Kyle's Lambda; §19.4.2 Amihud's Lambda; §19.4.3 Hasbrouck's Lambda; §19.5.2 Volume-Synchronized Probability of Informed Trading. Chapter 18: §18.2 Shannon's Entropy; §18.3 The Plug-in Estimator (Snippet 18.1); §18.4 Lempel-Ziv Estimators (Snippets 18.2–18.4); §18.5 Encoding Schemes."
  - "Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market. Journal of Finance 39(4), 1127–1139."
  - "Corwin, S. A. and Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. Journal of Finance 67(2), 719–760."
  - "Kyle, A. S. (1985). Continuous auctions and insider trading. Econometrica 53(6), 1315–1335."
  - "Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects. Journal of Financial Markets 5(1), 31–56."
  - "Easley, D., López de Prado, M. and O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. Review of Financial Studies 25(5), 1457–1493."
rust_api:
  - "get_roll_measure"
  - "get_roll_impact"
  - "get_corwin_schultz_estimator"
  - "get_bekker_parkinson_vol"
  - "get_bar_based_kyle_lambda"
  - "get_bar_based_amihud_lambda"
  - "get_bar_based_hasbrouck_lambda"
  - "get_trades_based_kyle_lambda"
  - "get_trades_based_amihud_lambda"
  - "get_trades_based_hasbrouck_lambda"
  - "get_vpin"
  - "get_bvc_buy_volume"
  - "vwap"
  - "get_avg_tick_size"
  - "encode_tick_rule_array"
  - "quantile_mapping"
  - "sigma_mapping"
  - "encode_array"
  - "get_shannon_entropy"
  - "get_plug_in_entropy"
  - "get_lempel_ziv_entropy"
  - "get_konto_entropy"
  - "MicrostructuralFeaturesGenerator"
  - "MicrostructuralError"
python_api:
  - "microstructural.get_roll_measure"
  - "microstructural.get_roll_impact"
  - "microstructural.get_corwin_schultz_estimator"
  - "microstructural.get_bekker_parkinson_vol"
  - "microstructural.get_bar_based_kyle_lambda"
  - "microstructural.get_bar_based_amihud_lambda"
  - "microstructural.get_bar_based_hasbrouck_lambda"
  - "microstructural.get_trades_based_kyle_lambda"
  - "microstructural.get_trades_based_amihud_lambda"
  - "microstructural.get_trades_based_hasbrouck_lambda"
  - "microstructural.get_vpin"
  - "microstructural.get_bvc_buy_volume"
  - "microstructural.vwap"
  - "microstructural.get_avg_tick_size"
  - "microstructural.encode_tick_rule_array"
  - "microstructural.quantile_mapping"
  - "microstructural.sigma_mapping"
  - "microstructural.encode_array"
  - "microstructural.get_shannon_entropy"
  - "microstructural.get_plug_in_entropy"
  - "microstructural.get_lempel_ziv_entropy"
  - "microstructural.get_konto_entropy"
sidebar:
  badge: Module
---

Most price data carries no record of the things a trader most wants to know about a market:
how wide the spread was, how far a given order would have moved the price, whether the flow
was one-sided. Market microstructure theory offers ways to *infer* them from what is
recorded, and AFML's Chapter 19 collects those estimators as features. The ones that need
only bars go back decades and work on any history; the ones that need signed trades are
sharper and need tick data. Chapter 18's entropy estimators, which measure how predictable a
sequence of price moves is, live here too. The Python module is `openquant.microstructural`.

## The estimators

**Spread.** The **Roll measure** (1984) uses the fact that prices bouncing between bid and
ask are negatively autocorrelated. If trades hit either side at random around an efficient
price, the spread is $2\sqrt{-\operatorname{cov}(\Delta p_t,\Delta p_{t-1})}$.
**Corwin–Schultz** (2012) needs only highs and lows: the high is usually a buy and the low a
sell, so the high–low range contains the spread once, while volatility grows with the
horizon; comparing one-bar with two-bar ranges separates them. It returns a *relative* spread.
**Bekker–Parkinson** volatility is the high–low volatility estimate with that spread taken
out.

**Price impact**, the $\lambda$ in "price moves $\lambda$ per unit of flow". **Kyle's
lambda** relates the price change to signed volume; **Amihud's** relates the absolute return
to dollar volume and needs no trade signs; **Hasbrouck's** relates the return to the signed
square root of dollar volume. Each comes in two forms. The `bar_based` functions take closes
and volumes, infer the sign from the direction of the bar, and return a rolling mean of
per-bar ratios. The `trades_based` functions take the trades of one bar, with their aggressor
flags, and fit a regression through the origin.

**Order flow.** `get_bvc_buy_volume` is **bulk volume classification**: with no trade signs,
attribute the share $\Phi(\Delta p/\sigma_{\Delta p})$ of a bar's volume to buyers.
`get_vpin` is the **volume-synchronised probability of informed trading** of Easley, López de
Prado and O'Hara (2012), the rolling mean absolute imbalance between buy and sell volume as a
share of volume.

All rolling functions return a vector as long as the input, with `NaN` until a full window of
valid values is available.

## Checked against a market whose answers are known

The simulation fixes a spread and a price impact, generates trades, and asks each estimator
to recover them, first from the trades and then from 50-trade bars.

```python
import random

from openquant import microstructural as ms

# 60,000 trades around an efficient price that follows a random walk. Every trade prints
# half a spread above or below it: the true spread is 0.10 on a price near 100, or 10 bp.
# Each trade also moves the efficient price by 0.002 per unit of signed volume.
rng = random.Random(1)
mid, spread, impact = 100.0, 0.10, 0.002
prices, volumes, sides = [], [], []
for _ in range(60_000):
    side, size = rng.choice((-1.0, 1.0)), rng.uniform(1, 10)
    mid += impact * side * size + rng.gauss(0, 0.01)
    prices.append(mid + side * spread / 2)
    volumes.append(size)
    sides.append(side)

changes = [0.0] + [b - a for a, b in zip(prices, prices[1:])]
print(f"Roll spread            {ms.get_roll_measure(prices[-10_000:], 5000)[-1]:.4f}   (true 0.1000)")
print(f"Kyle lambda, by trade  {ms.get_trades_based_kyle_lambda(changes, volumes, sides):.5f}  (true 0.00200 + bounce)")

# The same trades as 1,200 bars of 50: only high, low, close and volume survive.
bars = [prices[i:i + 50] for i in range(0, len(prices), 50)]
high, low = [max(b) for b in bars], [min(b) for b in bars]
cs = [v for v in ms.get_corwin_schultz_estimator(high, low, 20) if v == v]
print(f"Corwin-Schultz spread  {sum(cs) / len(cs):.5f}  (true 0.00100 relative); "
      f"{sum(v == 0 for v in cs) / len(cs):.0%} of bars floored at zero")

# Order-flow toxicity from bar data alone: bulk volume classification, then VPIN.
close = [b[-1] for b in bars]
bar_volume = [sum(volumes[i:i + 50]) for i in range(0, len(volumes), 50)]
true_buys = [sum(v for v, s in zip(volumes[i:i + 50], sides[i:i + 50]) if s > 0)
             for i in range(0, len(volumes), 50)]
bvc = ms.get_bvc_buy_volume(close, bar_volume, 20)
pairs = [(e / v, t / v) for e, t, v in zip(bvc, true_buys, bar_volume) if e == e]
mx, my = sum(a for a, _ in pairs) / len(pairs), sum(b for _, b in pairs) / len(pairs)
corr = (sum((a - mx) * (b - my) for a, b in pairs)
        / (sum((a - mx) ** 2 for a, _ in pairs) * sum((b - my) ** 2 for _, b in pairs)) ** 0.5)
print(f"BVC buy share vs true buy share: correlation {corr:.2f}")
```

```text
Roll spread            0.1127   (true 0.1000)
Kyle lambda, by trade  0.00942  (true 0.00200 + bounce)
Corwin-Schultz spread  0.00128  (true 0.00100 relative); 25% of bars floored at zero
BVC buy share vs true buy share: correlation 0.67
```

Three of these are good news with a caveat each. Roll recovers the spread to within 13%; it
runs high here because price impact adds negative autocorrelation of its own, which the
model attributes to the spread. Corwin–Schultz gets the right order of magnitude from highs
and lows alone, but a quarter of its values are exactly zero: the estimator goes negative
whenever a window's volatility swamps the spread and is floored, so its *mean* is usable and
its individual values are not. Bulk volume classification, given nothing but closes and
volume, correlates 0.67 with the true buy share.

Kyle's lambda is the instructive failure: 0.0094 against a true 0.0020. The regression is of
*transaction* price changes on signed volume, and a buy prints at the ask, so every trade
carries half a spread in the direction of its sign. The estimate is impact plus bounce.
Regress mid-price changes if you have quotes; otherwise read the trades-based lambda as a
cost of trading, not as permanent impact.

## Entropy of a price sequence

Chapter 18's idea is that a market in which prices are unpredictable produces incompressible
sequences, and that a drop in entropy — patterns appearing — marks inefficiency worth
investigating. The route has two steps.

**Encode** the series as a string. `encode_tick_rule_array` maps tick signs to letters
(`1 → a`, `−1 → b`, `0 → c`). For returns or volumes, `quantile_mapping(values, n_letters)`
builds a codebook with equally populated bins and `sigma_mapping(values, step)` one with bins
of fixed width; `encode_array` applies either.

**Estimate** the entropy of the string, in bits per symbol:

| Function | Estimator |
| --- | --- |
| `get_shannon_entropy(msg)` | $-\sum p\log_2 p$ over single symbols; ignores order entirely |
| `get_plug_in_entropy(msg, w)` | Shannon entropy of overlapping words of length `w`, divided by `w` |
| `get_lempel_ziv_entropy(msg)` | size of the Lempel–Ziv dictionary divided by message length; lower means more repetition |
| `get_konto_entropy(msg, window)` | Kontoyiannis' estimator from longest-match lengths; `window = 0` uses an expanding window |

Only the last three can see *structure*. The string `abababab…` has a Shannon entropy of
exactly 1 bit, the same as a fair coin, because it has as many `a`s as `b`s; its plug-in
entropy with three-letter words is 0.33 and its Kontoyiannis entropy 0.21, against 1.00 and
0.87 for a random string of the same length (400 symbols, a Kontoyiannis window of 20).

## From Rust

```rust
use openquant::microstructural_features::{
    encode_tick_rule_array, get_lempel_ziv_entropy, get_plug_in_entropy, get_shannon_entropy,
    get_trades_based_kyle_lambda, get_vpin, vwap,
};

// Regression through the origin of price change on signed volume: sum(xy) / sum(x^2).
let lambda = get_trades_based_kyle_lambda(&[0.2, -0.1, 0.4], &[10.0, 5.0, 20.0], &[1.0, -1.0, 1.0])?;
assert!((lambda - 0.02).abs() < 1e-12);

// VPIN over 3 bars of equal volume: mean |buys - sells| / volume. NaN until the window fills.
let vpin = get_vpin(&[100.0; 5], &[80.0, 20.0, 50.0, 90.0, 10.0], 3)?;
assert!(vpin[0].is_nan() && vpin[1].is_nan());
assert!((vpin[2] - 0.4).abs() < 1e-12);

assert!((vwap(&[1000.0, 2000.0], &[10.0, 10.0])? - 150.0).abs() < 1e-12);

// A perfectly periodic message: maximal Shannon entropy, low entropy by every other measure.
assert_eq!(encode_tick_rule_array(&[1, 1, -1, 0])?, "aabc");
assert_eq!(get_shannon_entropy("abababab"), 1.0);
assert_eq!(get_plug_in_entropy("abababab", 2)?, 0.5);
assert_eq!(get_lempel_ziv_entropy("abababab"), 0.5);
```

`MicrostructuralFeaturesGenerator::new_from_csv(path, tick_counts, volume_encoding,
pct_encoding)` streams a three-column trades file — timestamp, price, volume — and emits the
trades-based features and entropies for bars of the given tick counts. It is Rust-only.

## What to watch for

- **`get_trades_based_hasbrouck_lambda` changed in a fix.** Earlier versions regressed the
  *absolute* return on signed volume, so buys and sells cancelled and the estimate was near
  zero under balanced flow. It now regresses the signed return, as in AFML: on 5,000 simulated
  trades with a true $\lambda$ of 1e-5 it returns 1.0001e-5, where it returned −4.2e-8
  ([#105](https://github.com/Open-Quant/openquant/issues/105)). Results computed before the
  fix are not comparable.
- **The Roll measure takes the absolute value of the covariance.** Roll's formula needs a
  *negative* autocovariance and is undefined otherwise. Following mlfinlab, this function
  uses $2\sqrt{\lvert\operatorname{cov}\rvert}$, so a trending series with *positive*
  autocovariance also reports a "spread". Check the sign of the autocovariance yourself on
  anything that is not a liquid, mean-reverting tick series.
- **VPIN divides by the current bar's volume**, not the window's. It is the published measure
  only on **volume bars**, where every bar has the same volume. On time bars a quiet bar
  inflates it and a busy bar deflates it.
- **Bar-based lambdas are means of ratios, not regressions.** One bar with tiny volume
  contributes an enormous $\Delta p/V$ and dominates the window. Winsorise, or use volume or
  dollar bars so that the denominators are comparable.
- **Codebooks start at character 0.** `quantile_mapping` and `sigma_mapping` assign letters
  from the start of the 256-character table, so encoded strings begin with control characters
  including NUL. They are fine as input to the entropy functions and unsafe to print, log, or
  pass through anything that treats a string as C text. Alphabets are capped at 256 symbols.
- **Entropy estimates are biased on short messages**, downward for plug-in with long words
  (most words are seen once) and erratically for Lempel–Ziv. Compare messages of equal length
  and encoding; the level means little, the change over time is the feature.
- **The Lempel–Ziv and Kontoyiannis estimators are quadratic or worse** in message length.
  Keep messages to a bar's worth of ticks, not a day's.

## Related modules

- [`data-structures`](/modules/data-structures/) — the bars these features are computed on;
  volume and dollar bars suit most of them better than time bars.
- [`streaming-hpc`](/modules/streaming-hpc/) — incremental VPIN and concentration for live
  data.
- [`util-volatility`](/modules/util-volatility/) — range-based volatility estimators related
  to Bekker–Parkinson.
- [`structural-breaks`](/modules/structural-breaks/) — the other family of regime features.
