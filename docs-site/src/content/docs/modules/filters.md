---
title: "filters"
description: "The symmetric CUSUM filter and a rolling z-score filter, for sampling events from a price series."
status: authored
last_authored: '2026-09-25'
audience:
  - quant-dev
  - platform-engineering
module: "filters"
api_surface: "both"
afml_chapter:
  - "2"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 2, §2.5.2 Event-Based Sampling; §2.5.2.1 The CUSUM Filter; Snippet 2.4."
  - "Page, E. S. (1954). Continuous inspection schemes. Biometrika 41(1/2), 100–115."
  - "Lam, K. and Yam, H. C. (1997). CUSUM techniques for technical trading in financial markets. Financial Engineering and the Japanese Markets 4(3), 257–274."
rust_api:
  - "cusum_filter_indices"
  - "cusum_filter_timestamps"
  - "z_score_filter_indices"
  - "z_score_filter_timestamps"
  - "Threshold"
  - "FilterError"
python_api:
  - "filters.cusum_filter_indices"
  - "filters.cusum_filter_timestamps"
  - "filters.z_score_filter_indices"
  - "filters.z_score_filter_timestamps"
sidebar:
  badge: Module
---

Even on well-built bars, most bars are uneventful, and a classifier trained on every one of
them spends its capacity learning that nothing happened. AFML's answer (§2.5.2) is to sample:
keep only the bars at which something measurable occurred, and label those. The filter that
decides *which* bars is the first modelling choice in the pipeline, and everything downstream
— labels, sample weights, cross-validation — inherits its timestamps.

## The symmetric CUSUM filter

CUSUM is a quality-control method (Page, 1954) for detecting that the mean of a process has
shifted away from a target. Applied to returns with a target of zero, it accumulates upward
and downward drift separately and floors each accumulator at zero:

$$
\begin{aligned}
S_t^{+} &= \max\bigl(0,\; S_{t-1}^{+} + r_t\bigr) \\
S_t^{-} &= \min\bigl(0,\; S_{t-1}^{-} + r_t\bigr)
\end{aligned}
$$

where $r_t=\ln(p_t/p_{t-1})$. Bar $t$ is an event when $S_t^{+} > h$ or $S_t^{-} < -h$, and
the accumulator that fired is reset to zero. This is AFML's Snippet 2.4 with one difference:
the snippet differences whatever series it is given, and this implementation always takes
log returns of `close`. Pass prices, not returns.

The floor is what makes the filter useful. A price that wanders up 0.9% and back down does
not trigger it and leaves no residue, whereas a Bollinger-band rule fires repeatedly while a
price hovers at the band. CUSUM needs a full run of length $h$ from the last reset, in one
direction net of reversals, so it fires once per move.

```python
import math
import random
from datetime import datetime, timedelta
from statistics import median

from openquant import filters

# 600 one-minute closes. Minutes 300-399 are four times as volatile as the rest.
rng = random.Random(11)
start, price, close, stamps = datetime(2024, 1, 2, 9, 30), 100.0, [], []
for i in range(600):
    price *= math.exp(rng.gauss(0, 0.004 if 300 <= i < 400 else 0.001))
    close.append(price)
    stamps.append((start + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S"))

h = 0.01  # fire when the cumulative log return since the last event reaches 1%
events = filters.cusum_filter_indices(close, h)
volatile = [i for i in events if 300 <= i < 400]
print(f"{len(events)} events from {len(close)} bars; {len(volatile)} of them in the volatile 100")

gaps = [(a, b - a) for a, b in zip(events, events[1:])]
print("median bars between events: "
      f"calm {median(g for a, g in gaps if not 300 <= a < 400):.0f}, "
      f"volatile {median(g for a, g in gaps if 300 <= a < 400):.0f}")

print("first three:", filters.cusum_filter_timestamps(close, stamps, h)[:3])
```

```text
22 events from 600 bars; 13 of them in the volatile 100
median bars between events: calm 44, volatile 6
first three: ['2024-01-02 10:32:00', '2024-01-02 10:55:00', '2024-01-02 11:30:00']
```

A sixth of the bars produce more than half the events, which is the behaviour you want from a
sampler and also the problem with a fixed $h$: in the volatile stretch an event arrives every
six bars, and their labels will overlap heavily.

<figure>
<img class="dark:sl-hidden" src="/figures/ch2-cusum-light.svg" alt="A synthetic one-minute price series of 600 bars with 22 CUSUM events marked on it. Thirteen of the markers fall inside a shaded stretch of 100 bars where volatility is four times higher." />
<img class="light:sl-hidden" src="/figures/ch2-cusum-dark.svg" alt="A synthetic one-minute price series of 600 bars with 22 CUSUM events marked on it. Thirteen of the markers fall inside a shaded stretch of 100 bars where volatility is four times higher." />
<figcaption>Events at <em>h</em> = 1%. The filter is quiet while the price drifts and busy when it moves.</figcaption>
</figure>

## Choosing h

$h$ is in log-return units and means "a move worth labelling". Two practical anchors:

- **Tie it to the label.** If the triple-barrier profit target is 1.5%, an $h$ near 1.5% asks
  the classifier about moves of the size it is being paid to predict. An $h$ far below the
  barrier width mostly samples noise that never reaches a barrier.
- **Tie it to volatility.** A constant $h$ fires too often in turbulent regimes and rarely in
  calm ones. Pass one threshold per bar instead — typically a multiple of
  [`get_daily_vol`](/modules/util-volatility/) — and bar $t$ is compared with element $t$.
  In Rust that is `Threshold::Dynamic`; a vector shorter than the series is a
  `FilterError::MissingDynamicThreshold`, not a silent truncation. In Python, pass a list,
  NumPy array or pandas Series as `threshold` instead of a number.

From Python, a per-bar `threshold` must have exactly one value per price, or it is a
`ValueError`. Element 0 is never read (the first return is at bar 1), so it may be `NaN`, as
the first value of a volatility estimate usually is. Every other element, and a scalar
threshold, must be finite and non-negative: a `NaN` threshold would silently never fire and a
negative one would fire on every bar. Fill a volatility estimate's warm-up before passing it.

```python
import math
import random

import numpy as np
from openquant import filters

# The same 600 closes as above: minutes 300-399 are four times as volatile.
rng = random.Random(11)
price, close = 100.0, []
for i in range(600):
    price *= math.exp(rng.gauss(0, 0.004 if 300 <= i < 400 else 0.001))
    close.append(price)

# h_t = 5 x the trailing 50-bar standard deviation of log returns; 1% until it is defined.
returns = np.diff(np.log(close), prepend=np.nan)
vol = np.array([returns[t - 49 : t + 1].std(ddof=1) if t >= 50 else np.nan for t in range(600)])
h = np.where(np.isnan(vol), 0.01, 5.0 * vol)

events = filters.cusum_filter_indices(close, h)
volatile = [i for i in events if 300 <= i < 400]
print(f"{len(events)} events from {len(close)} bars; {len(volatile)} of them in the volatile 100")
print("a constant array is the scalar filter:",
      filters.cusum_filter_indices(close, [0.01] * 600) == filters.cusum_filter_indices(close, 0.01))
```

```text
29 events from 600 bars; 6 of them in the volatile 100
a constant array is the scalar filter: True
```

The threshold rises with the turbulence, so the volatile stretch yields 6 events instead of 13
and the calm stretches more: the sampler now fires on moves that are large *for their
regime*, which is what a volatility-scaled triple barrier will then label.

Whichever you use, count the events and look at their spacing before labelling. Events closer
together than the label horizon produce overlapping labels; [`sampling`](/modules/sampling/)
measures that overlap and [`sample-weights`](/modules/sample-weights/) corrects for it, but
neither removes it.

## The z-score filter

`z_score_filter_indices(close, mean_window, std_window, threshold)` marks bar $t$ when

$$
p_t \;\ge\; \bar p_{t}^{(m)} + k\,\sigma_{t}^{(s)}
$$

with $\bar p^{(m)}$ the rolling mean over `mean_window` bars, $\sigma^{(s)}$ the rolling sample
standard deviation (ddof = 1) over `std_window` bars, both including bar $t$, and $k$ the
`threshold`. It is not from AFML; it is ported from mlfinlab. Three properties to know before
using it:

- It is **one-sided**. Only upward excursions are events; a crash three deviations below the
  mean is not. Negate the series to sample the downside.
- It works on **price levels**, not returns, so on a trending series the price sits above its
  trailing mean and the filter fires in runs rather than once per move.
- It has no reset, so consecutive bars above the band are consecutive events.

## From Rust

```rust
use openquant::filters::{cusum_filter_indices, FilterError, Threshold};

let close = vec![100.0, 100.4, 100.9, 101.3, 101.0, 100.2, 99.6, 99.9];

// Fixed 1% threshold: one upward event, then one downward.
let events = cusum_filter_indices(&close, Threshold::Scalar(0.01))?;
assert_eq!(events, vec![3, 5]);

// Per-bar thresholds must cover the series.
let too_short = cusum_filter_indices(&close, Threshold::Dynamic(vec![0.01; 3]));
assert!(matches!(too_short, Err(FilterError::MissingDynamicThreshold { index: 3, available: 3 })));
```

`cusum_filter_timestamps` and `z_score_filter_timestamps` return the timestamps at the event
positions instead, and report a timestamp slice that is too short as
`FilterError::TimestampIndexOutOfBounds`.

## What to watch for

- **No look-ahead, but no warm-up either.** The filter uses only returns up to bar $t$, so its
  events are safe to label. Its accumulators start at zero at the first bar, so the first
  event on a series that starts mid-move comes late; drop a burn-in stretch if that matters.
- **Both accumulators can be non-zero at once.** Only the side that fires is reset. After an
  upward event, $S^{-}$ keeps whatever downward drift it had.
- **Events are bars, not trades.** An event at bar $t$ is known at the *close* of bar $t$.
  Entering at that close is optimistic; [`labeling`](/modules/labeling/) measures outcomes
  from the event bar's close, so any execution delay is yours to model.

## Related modules

- [`data-structures`](/modules/data-structures/) — the bars this filter samples from.
- [`labeling`](/modules/labeling/) — triple-barrier labels at the sampled events.
- [`util-volatility`](/modules/util-volatility/) — the volatility estimate behind a dynamic
  threshold.
- [`structural-breaks`](/modules/structural-breaks/) — CUSUM *tests* for a change in regime,
  which share the name and the statistic but answer a different question.
