---
title: "fracdiff"
description: "Fractional differentiation: make a price series stationary while keeping as much of its memory as possible."
status: authored
last_authored: '2026-11-07'
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

A price-level feature can change its relationship with an outcome as the price level
moves. Differencing is one way to reduce this dependence, but neither prices nor returns
are guaranteed stationary. A first difference of log prices is a log return; a first
difference of raw prices is an absolute price change. AFML §5.2 describes the trade-off
between stationarity and retaining information about past levels. Fractional differencing
lets you explore intermediate orders, commonly $0 \le d \le 1$, and test the resulting
features on the training sample.

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

so the differenced value is $\tilde x_t=\sum_k w_k\,x_{t-k}$. For non-negative integer $d$ the weights hit
zero after $d$ lags: $d=1$ gives $\{1,-1,0,0,\dots\}$, the ordinary first difference. For
$0<d<1$ they never reach zero in exact arithmetic but decay like $k^{-(1+d)}$, which is where the memory
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
ones. To determine the leading `NaN` count, the implementation generates `len(series)`
weights in oldest-first order, forms their normalized cumulative absolute sum, and counts
entries strictly greater than `thresh`. That count also depends on the supplied batch
length. The drawback of the expanding window is structural:
because the number of terms grows with $t$, the output picks up a drift that comes from the
window and not from the data (§5.5.1).

**`frac_diff_ffd(series, d, thresh)` — fixed-width window** (Snippet 5.3). Weights are
generated until one falls below `thresh` in absolute value, and that fixed set is applied at
every eligible point within that call. Every such output uses the same number of lags,
avoiding the expanding window's changing filter. This is the one to use for features. The first `width − 1` outputs are
`NaN`.

`thresh` means different things in the two functions: a share of cumulative weight in the
first, a floor on a single weight in the second.

## Choosing d

AFML's procedure (§5.6) is to sweep $d$ from 0 to 1, run an ADF test on each output, and keep
the **smallest** $d$ that rejects a unit root. This crate has no ADF test, so the example
reports two plain statistics instead: the correlation between the differenced series and the
original log price (memory kept) and the differenced series' lag-1 autocorrelation (a value
near 1 can indicate persistence, but is not a unit-root test).

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

The last row is log returns, with near-zero sample autocorrelation in this simulation.
At $d=0.4$ the output still correlates 0.70 with the level while lag-1 autocorrelation has
fallen to 0.859. Neither statistic establishes stationarity. An ADF rejection is evidence
against its unit-root null under the selected test specification, not a guarantee of future
stationarity. Run `statsmodels.tsa.stattools.adfuller` on training data and record its lag,
trend and significance choices; do not infer a finite-sample test outcome from $d$ alone.

The finite-width implementation also differs from the theoretical infinite fractional
filter: truncating the weights generally leaves a nonzero weight sum, so a random-walk
component can remain. Test the actual feature and window you plan to use.

The `window` column is the cost. At `thresh = 1e-4`, $d=0.2$ needs 497 observations for its
first value. Window length is not universally decreasing in $d$: $d=0$ uses just one weight.

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

The missing-value and batch-cap behavior is visible on small inputs:

```python
import math
from openquant import fracdiff

series = [1.0, float("nan"), 3.0, 4.0, 5.0]
expanding = fracdiff.frac_diff(series, 1.0, 1.0)
fixed = fracdiff.frac_diff_ffd(series, 1.0, 1e-4)
assert all(math.isnan(x) for x in expanding[1:])
assert fixed[3:] == [1.0, 1.0]

short = fracdiff.frac_diff_ffd([1.0, 2.0], 0.5, 1e-4)
longer = fracdiff.frac_diff_ffd([1.0, 2.0, 3.0], 0.5, 1e-4)
assert short[1] == 1.5 and math.isnan(longer[1])  # width changed from 2 to 3
```

## What to watch for

- **Difference the log price, or the price — not returns.** The input is a level. Feeding
  returns differences them a second time.
- **Numeric inputs are not validated.** Rust returns `Vec<f64>`, not `Result`; Python
  delegates the arithmetic after argument conversion. A negative `d` integrates rather
  than differences, and weights need not decay. A nonpositive or `NaN` FFD threshold never
  satisfies the stopping comparison and runs to the cap, including zero integer-order
  weights. Use finite parameters and validate the chosen window.
- **Missing values are not skipped.** A `NaN` poisons every dot product whose window
  contains it, even when its weight is zero (`0 * NaN` remains `NaN`). Thus an expanding
  window can remain missing forever; FFD recovers only after the missing value leaves the
  window. Clean or explicitly model gaps before differencing.
- **The cap depends on batch length.** Appending data can change the FFD width when the
  previous call hit its series-length cap, changing earlier warm-up/output availability.
  Do not equate separate chunk calls with one continuous transform. Fix and record the
  intended weights and supply their required history before evaluating held-out rows.
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
