---
title: "fracdiff"
description: "Fractional differentiation: make a price series stationary while keeping as much of its memory as possible."
status: authored
last_authored: '2026-09-25'
audience:
  - quant-dev
  - platform-engineering
module: "fracdiff"
api_surface: "both"
afml_chapter:
  - "5"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 5: §5.2 The Stationarity vs. Memory Dilemma; §5.4 The Method (Snippet 5.1); §5.5.1 Expanding Window (Snippet 5.2); §5.5.2 Fixed-Width Window Fracdiff (Snippet 5.3); §5.6 Stationarity with Maximum Memory Preservation (Snippet 5.4)."
  - "Hosking, J. R. M. (1981). Fractional differencing. Biometrika 68(1), 165–176."
rust_api:
  - "get_weights"
  - "get_weights_ffd"
  - "frac_diff"
  - "frac_diff_ffd"
python_api:
  - "fracdiff.get_weights"
  - "fracdiff.get_weights_ffd"
  - "fracdiff.frac_diff"
  - "fracdiff.frac_diff_ffd"
sidebar:
  badge: Module
---

Supervised learning wants stationary features: a model cannot map a price of 4,000 to an
outcome if it was trained when prices were near 1,000. The usual fix is to take returns, and
returns are stationary because they forget everything — each one is the difference between
two adjacent prices and carries no trace of the level or the path. AFML §5.2 calls this the
stationarity-versus-memory dilemma, and points out that the choice is not binary. Returns are
the series differenced $d=1$ times. Differencing it $d=0.4$ times is also defined, is often
already stationary, and keeps most of the level's information.

## The weights

Write $B$ for the backshift operator, $B^k x_t = x_{t-k}$. Integer differencing is
$(1-B)^d$, and the binomial series extends it to real $d$ (Hosking, 1981):

$$
(1-B)^d \;=\; \sum_{k=0}^{\infty} w_k B^k,
\qquad
w_0 = 1,
\qquad
w_k \;=\; -\,w_{k-1}\,\frac{d-k+1}{k}
$$

so the differenced value is $\tilde x_t=\sum_k w_k\,x_{t-k}$. For integer $d$ the weights hit
zero after $d$ lags: $d=1$ gives $\{1,-1,0,0,\dots\}$, the ordinary first difference. For
fractional $d$ they never reach zero but decay like $k^{-(1+d)}$, which is where the memory
comes from — every past value still contributes, a little.

<figure>
<img class="dark:sl-hidden" src="/figures/ch5-fracdiff-weights-light.svg" alt="Fractional differencing weights at lags 1 to 8 for four values of d. At lag 1 the weight equals minus d. For d = 1 every later weight is exactly zero; for d = 0.2, 0.5 and 0.8 the weights stay slightly negative and decay slowly toward zero." />
<img class="light:sl-hidden" src="/figures/ch5-fracdiff-weights-dark.svg" alt="Fractional differencing weights at lags 1 to 8 for four values of d. At lag 1 the weight equals minus d. For d = 1 every later weight is exactly zero; for d = 0.2, 0.5 and 0.8 the weights stay slightly negative and decay slowly toward zero." />
<figcaption>The first lag carries −<em>d</em>. What distinguishes a fractional <em>d</em> is the tail that follows.</figcaption>
</figure>

`get_weights(d, size)` returns the first `size` weights. **They come back oldest lag first**,
so the last element is $w_0=1$; that is the order in which they are multiplied against a
window of the series.

## Two windows

An infinite sum has to be cut somewhere, and the two functions cut it differently.

**`frac_diff(series, d, thresh)` — expanding window** (Snippet 5.2). Every output uses all the
history available at that point, so early outputs are built from fewer weights than late
ones. The first outputs, where the weights that *would* have applied to missing history add
up to more than `thresh` of the total, are returned as `NaN`. The drawback is structural:
because the number of terms grows with $t$, the output picks up a drift that comes from the
window and not from the data (§5.5.1).

**`frac_diff_ffd(series, d, thresh)` — fixed-width window** (Snippet 5.3). Weights are
generated until one falls below `thresh` in absolute value, and that fixed set is applied at
every point. Every output is the same function of the same number of lags, so there is no
window-induced drift. This is the one to use for features. The first `width − 1` outputs are
`NaN`.

`thresh` means different things in the two functions: a share of cumulative weight in the
first, a floor on a single weight in the second.

## Choosing d

AFML's procedure (§5.6) is to sweep $d$ from 0 to 1, run an ADF test on each output, and keep
the **smallest** $d$ that rejects a unit root. This crate has no ADF test, so the example
reports two plain statistics instead: the correlation between the differenced series and the
original log price (memory kept) and the differenced series' lag-1 autocorrelation (a value
near 1 is what a unit root looks like).

```python
import math
import random
from statistics import correlation

from openquant import fracdiff

# A random-walk log price: 2,000 bars, 1% volatility. Non-stationary by construction.
rng = random.Random(5)
log_price, level = [], math.log(100.0)
for _ in range(2000):
    level += rng.gauss(0.0002, 0.01)
    log_price.append(level)

def lag1(xs):
    return correlation(xs[:-1], xs[1:])

print(" d   window   corr with price   lag-1 autocorr")
for d in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
    out = fracdiff.frac_diff_ffd(log_price, d, 1e-4)
    kept = [(p, v) for p, v in zip(log_price, out) if not math.isnan(v)]
    prices, values = zip(*kept)
    width = len(log_price) - len(kept) + 1
    memory = correlation(prices, values) if d > 0 else 1.0
    print(f"{d:.1f}   {width:6d}   {memory:15.3f}   {lag1(values):14.3f}")
```

```text
 d   window   corr with price   lag-1 autocorr
0.0        1             1.000            0.994
0.2      497             0.919            0.970
0.4      282             0.698            0.859
0.6      140             0.382            0.577
0.8       64             0.177            0.257
1.0        2             0.055            0.008
```

The last row is returns: no autocorrelation and almost no relationship to the price level.
The rows between are the point of the chapter. At $d=0.4$ the series still correlates 0.70
with the level while its autocorrelation has dropped well away from 1. Whether 0.4 is
*enough* is what the ADF test decides. In theory a pure random walk differenced $d$ times is
stationary only for $d>0.5$, so this synthetic series would not pass at 0.4; real prices are
not pure random walks, and AFML reports most liquid futures passing at $d<0.6$ (§5.6). Run
the sweep with `statsmodels.tsa.stattools.adfuller` on your own data, per instrument; $d$
does not transfer between them.

The `window` column is the cost. At `thresh = 1e-4`, $d=0.2$ needs 497 bars of history before
it produces its first value. A lower $d$ keeps more memory *because* it uses a longer window.

## From Rust

```rust
use openquant::fracdiff::{frac_diff_ffd, get_weights, get_weights_ffd};

// Oldest lag first; the last weight is w_0 = 1 and the one before it is -d.
let w = get_weights(0.5, 4);
assert_eq!(w, vec![-0.0625, -0.125, -0.5, 1.0]);

// d = 1 is the first difference: two weights, then an exact zero ends the expansion.
assert_eq!(get_weights_ffd(1.0, 1e-5, 100), vec![-1.0, 1.0]);

let series: Vec<f64> = (1..=10).map(f64::from).collect();
let diffed = frac_diff_ffd(&series, 1.0, 1e-5);
assert!(diffed[0].is_nan());
assert!(diffed[1..].iter().all(|v| *v == 1.0));

// A smaller threshold keeps more weights, so more leading values are NaN.
let width = |thresh| get_weights_ffd(0.5, thresh, 10_000).len();
assert_eq!((width(1e-2), width(1e-4)), (10, 200));
```

The third argument of `get_weights_ffd` caps the number of weights, for every value: `lim = 0`
gives no weights and `lim = 1` gives just `[1.0]`. The cap is what ends the expansion when
the threshold cannot: for a fractional $d$ the weights never reach zero, so a `thresh` of
zero or below runs to exactly `lim` weights. `frac_diff_ffd` sets the cap to the series
length, so a threshold too small for the data yields a window as long as the series and a
single non-`NaN` output.

```rust
use openquant::fracdiff::get_weights_ffd;

assert_eq!(get_weights_ffd(0.5, 0.0, 1), vec![1.0]);
assert_eq!(get_weights_ffd(0.5, 0.0, 4).len(), 4);
```

## What to watch for

- **Difference the log price, or the price — not returns.** The input is a level. Feeding
  returns differences them a second time.
- **Nothing is validated.** These functions return `Vec<f64>`, not `Result`. A negative `d`
  is accepted and *integrates* the series instead of differencing it; its weights decay so
  slowly that the window runs to the cap and only the last output is a number. A `thresh` of
  zero does the same. A `NaN` in the input poisons every output whose window covers it;
  AFML's snippet skips such points, and this port does not. Clean the series first.
- **The window is look-back only, but `d` is not.** Each output uses values at or before its
  own bar. A $d$ chosen by an ADF sweep over the full history, though, has seen the test
  period. Choose $d$ on the training span.
- **`frac_diff` with `thresh = 1.0` skips nothing**, so its first output is just
  `series[0]`. That is the mlfinlab default and rarely what you want; AFML uses 0.01.
- **Cost grows with the window.** `frac_diff_ffd` is $O(n\cdot\text{width})$ and `frac_diff`
  is $O(n^2)$. Tick-level series with a `thresh` of `1e-5` and a small $d$ mean windows in
  the thousands.
- **Stationarity is per series and per regime.** A $d$ that passed on ten years of data can
  fail on the last two.

## Related modules

- [`structural-breaks`](/modules/structural-breaks/) — SADF and related tests, the nearest
  thing in this crate to a unit-root test.
- [`feature-importance`](/modules/feature-importance/) — check that a fractionally
  differenced feature earns its place.
- [`data-structures`](/modules/data-structures/) — build the bars first; differencing a
  time-bar series inherits its heteroskedasticity.
