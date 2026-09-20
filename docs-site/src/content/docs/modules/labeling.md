---
title: "labeling"
description: "Triple-barrier labels and meta-labels: what happened after each event, measured in units of that moment's volatility."
status: authored
last_authored: '2026-09-20'
audience:
  - quant-dev
  - platform-engineering
module: "labeling"
api_surface: "both"
afml_chapter:
  - "3"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 3: §3.2 The Fixed-Time Horizon Method; §3.3 Computing Dynamic Thresholds (Snippet 3.1); §3.4 The Triple-Barrier Method (Snippet 3.2); §3.5 Learning Side and Size (Snippets 3.3–3.5); §3.6 Meta-Labeling (Snippets 3.6–3.7); §3.9 Dropping Unnecessary Labels (Snippet 3.8)."
rust_api:
  - "triple_barrier_events"
  - "triple_barrier_labels"
  - "meta_labels"
  - "add_vertical_barrier"
  - "get_events"
  - "get_bins"
  - "drop_labels"
  - "Event"
  - "LabeledEvent"
  - "TripleBarrierConfig"
python_api:
  - "labeling.add_vertical_barrier"
  - "labeling.get_events"
  - "labeling.get_bins"
  - "labeling.triple_barrier_events"
  - "labeling.triple_barrier_labels"
  - "labeling.meta_labels"
  - "labeling.drop_labels"
sidebar:
  badge: Module
---

The usual way to label financial data is to look a fixed number of bars ahead and take the
sign of the return. AFML §3.2 lists what is wrong with that. The same 1% threshold is a large
move in a quiet market and noise in a turbulent one; and the label ignores the path, so a
position that would have been stopped out on day two is labelled a winner because the price
recovered by day five. No trader holds through a stop they have set.

The triple-barrier method (§3.4) labels an observation by which of three barriers the price
reaches first: a profit target above, a stop below, or a time limit. The two horizontal
barriers are set in multiples of a volatility estimate taken at the event, so "a large move"
means the same thing in every regime.

<figure>
<img class="dark:sl-hidden" src="/figures/ch3-triple-barrier-light.svg" alt="Two price paths in units of the volatility target, each inside a box whose top and bottom are the profit and stop barriers at plus and minus 1.5 and whose right edge is the 48-hour time limit. The first path touches the top barrier after 37 hours and is labelled plus one. The second wanders inside the box for 48 hours, ends 0.06 targets below zero, and is labelled minus one." />
<img class="light:sl-hidden" src="/figures/ch3-triple-barrier-dark.svg" alt="Two price paths in units of the volatility target, each inside a box whose top and bottom are the profit and stop barriers at plus and minus 1.5 and whose right edge is the 48-hour time limit. The first path touches the top barrier after 37 hours and is labelled plus one. The second wanders inside the box for 48 hours, ends 0.06 targets below zero, and is labelled minus one." />
<figcaption>Two events from the example below. The second is the case to worry about: a label of −1 earned by a return of −0.06 targets.</figcaption>
</figure>

## How an event is resolved

For an event at bar $t_0$ with target $\sigma_{t_0}$, profit multiple $m_{pt}$, stop multiple
$m_{sl}$ and side $s\in\{-1,+1\}$ (taken as $+1$ when no side is given), the path return and
the first touch are

$$
r_t = s\left(\frac{p_t}{p_{t_0}}-1\right), \qquad
t_1=\min\Bigl\{\,t>t_0 : r_t > m_{pt}\sigma_{t_0} \;\text{ or }\; r_t < -m_{sl}\sigma_{t_0}\Bigr\} \wedge t_{\text{vertical}}
$$

and the label is $\operatorname{sign}(r_{t_1})$. The details that decide edge cases:

- **Returns are simple**, $p_t/p_{t_0}-1$, measured from the event bar's close.
- **A barrier is touched when the return goes strictly beyond it**, as in Snippet 3.2. Sitting
  exactly on a barrier is not a touch.
- **A multiple of zero disables that barrier.** `pt_sl = (0.0, 1.5)` is a stop and a time
  limit with no profit target.
- **The label is the sign of the return at $t_1$, including at the vertical barrier.** This is
  AFML's Snippet 3.5. It is *not* the variant, common elsewhere, that assigns 0 when time runs
  out. A label of 0 appears only when the return is exactly zero.
- **Events are dropped, not defaulted.** An event whose target is NaN or not above `min_ret`
  is discarded. An event with no vertical barrier that never touches a horizontal one has no
  outcome yet; it is kept in the events with an empty end time and produces no label.
- **Only closes are checked.** A bar whose high pierced the target and whose close did not
  does not count as a touch. On coarse bars this understates how often stops are hit.

## The whole path, on data

```python
import math
import random
from collections import Counter
from datetime import datetime, timedelta

from openquant import filters, labeling, volatility

# 90 days of hourly closes.
rng = random.Random(3)
start, price, stamps, close = datetime(2024, 1, 1), 100.0, [], []
for i in range(24 * 90):
    price *= math.exp(rng.gauss(0, 0.004))
    stamps.append((start + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"))
    close.append(price)

# 1. The unit of risk: an EWMA of daily returns (Snippet 3.1). The first row is NaN.
vol = [(t, v) for t, v in volatility.get_daily_vol(stamps, close, 100) if not math.isnan(v)]
vol_stamps, vol_values = [t for t, _ in vol], [v for _, v in vol]

# 2. Where to look: CUSUM events at roughly one day's volatility.
events = filters.cusum_filter_timestamps(close, stamps, 0.02)

# 3. How long to wait: a vertical barrier two days after each event.
vertical = labeling.add_vertical_barrier(events, stamps, close, 2, 0, 0, 0)

# 4. Barriers at 1.5 daily vols either side; ignore events whose target is under 0.5%.
found = labeling.get_events(stamps, close, events, (1.5, 1.5), vol_stamps, vol_values, 0.005,
                            vertical_barrier_times=vertical)
bins = labeling.get_bins(found, stamps, close)
print(f"{len(events)} events -> {len(found)} with a usable target -> {len(bins)} labels")
print("labels:", dict(sorted(Counter(b[3] for b in bins).items())))

# Which barrier ended each event? A return beyond 1.5 targets means a horizontal one did.
def exit_of(ret, target):
    return "profit" if ret > 1.5 * target else "stop" if ret < -1.5 * target else "time"

for kind in ("profit", "stop", "time"):
    rows = [b for b in bins if exit_of(b[1], b[2]) == kind]
    ups = sum(b[3] == 1 for b in rows)
    size = sum(abs(b[1]) / b[2] for b in rows) / len(rows)
    print(f"{kind:6s} {len(rows):3d} events, {ups:2d} labelled +1, mean |return| = {size:.2f} targets")

# Meta-labeling: give each event a side (here: follow the last day's move) and ask only
# whether acting on it paid. Labels become {0, 1}; returns are signed by the side.
index = {t: i for i, t in enumerate(stamps)}
sides = [1.0 if close[index[t]] > close[index[t] - 24] else -1.0 for t in events]
meta = labeling.meta_labels(stamps, close, events, vol_stamps, vol_values, list(zip(events, sides)),
                            pt=1.5, sl=1.5, min_ret=0.005, vertical_barrier_times=vertical)
print("meta-labels:", dict(sorted(Counter(m[3] for m in meta).items())))
```

```text
111 events -> 109 with a usable target -> 108 labels
labels: {-1: 52, 1: 56}
profit  35 events, 35 labelled +1, mean |return| = 1.67 targets
stop    32 events,  0 labelled +1, mean |return| = 1.64 targets
time    41 events, 21 labelled +1, mean |return| = 0.54 targets
meta-labels: {0: 54, 1: 54}
```

Read the three exit rows. Every profit exit is labelled +1 and every stop exit −1, by
construction, and they ended about 1.65 targets away from the entry. The 41 events that ran
out of time are split 21 to 20 and ended, on average, half a target from where they started:
**their labels are close to coin flips, and they are 38% of the training set.** That is
the price of taking the sign at the vertical barrier. The remedies are to give those
observations less weight (weight by absolute return — see
[`sample-weights`](/modules/sample-weights/)), to widen the time limit, or to relabel them 0
yourself and train a three-class model.

The meta-labels come out 54 to 54 because the side rule — follow yesterday's move — has no
edge on a random walk, which is what this series is. On real data that ratio is the base rate
the secondary model has to beat.

## Meta-labeling

Give `get_events` a side for each event and the question changes from "which way?" to "was
acting on that call worth it?" (§3.6). The return is multiplied by the side, the barriers
become the side's profit target and stop, and the label is 1 if the signed return is positive
and 0 otherwise. A primary model — a rule, a discretionary signal, another classifier — sets
the side; a secondary model learns from the {0, 1} labels when to act on it, and its
probability sizes the bet ([`bet-sizing`](/modules/bet-sizing/)). The primary model can then
be tuned for recall, because the secondary one is there to restore precision.

`meta_labels` is `triple_barrier_labels` restricted to events that carry a side.

## Choosing the parameters

- **Target.** [`get_daily_vol`](/modules/util-volatility/) is Snippet 3.1: an exponentially
  weighted standard deviation of one-day returns. Its first value is NaN and early values
  rest on few observations, so expect to lose the first events. Any positive series aligned to
  the event timestamps will do; what matters is that it is known *at* the event.
- **Multiples.** Symmetric barriers make the label a statement about direction. Asymmetric
  ones build a payoff ratio into the label, and the class balance shifts with them — check it
  before training. `pt_sl` is `(profit, stop)` in that order.
- **`min_ret`.** Events whose target is tiny produce barriers inside the bid-ask spread and
  labels that cost more to trade than they are worth.
- **Time limit.** `add_vertical_barrier` looks up the first bar at or after the event time
  plus the offset. An event too close to the end of the series to have one gets no vertical
  barrier at all, rather than a shortened one.
- **Rare classes.** `drop_labels(bins, min_pct)` removes the rarest class while it holds no
  more than `min_pct` of the observations and at least three classes remain (Snippet 3.8).

## From Rust

```rust
use chrono::{Duration, NaiveDate};
use openquant::labeling::{add_vertical_barrier, get_bins, get_events};

let t0 = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
let day = |d: i64| t0 + Duration::days(d);
let prices = [100.0, 100.5, 101.0, 103.5, 102.0, 101.0, 100.0, 99.0];
let close: Vec<_> = prices.iter().enumerate().map(|(i, p)| (day(i as i64), *p)).collect();

// One event on day 0; a constant 2% target; barriers one target wide; a five-day limit.
let events = vec![day(0)];
let target = vec![(day(0), 0.02)];
let vertical = add_vertical_barrier(&events, &close, 5, 0, 0, 0);
let found = get_events(&close, &events, (1.0, 1.0), &target, 0.0, 1, Some(&vertical), None);
let bins = get_bins(&found, &close);

// +1.0% on day 2 is inside the barrier; +3.5% on day 3 is beyond it.
assert_eq!(found[0].1.t1, Some(day(3)));
let (_, ret, _, label, side) = bins[0];
assert!((ret - 0.035).abs() < 1e-12);
assert_eq!((label, side), (1, None));
```

## What to watch for

- **Labels overlap.** An event's label depends on prices up to $t_1$, and events fired close
  together share most of that window. The observations are not independent, and an ordinary
  K-fold split leaks. Measure the overlap with [`sampling`](/modules/sampling/), weight with
  [`sample-weights`](/modules/sample-weights/), and split with
  [`cross-validation`](/modules/cross-validation/), passing each event's $(t_0, t_1)$.
- **Timestamps must match exactly.** Events, targets and sides are joined to `close` by
  timestamp. An event whose timestamp is not a bar in `close` is skipped without an error.
- **Whole seconds only from Python.** Timestamps cross the binding as strings at one-second
  resolution ([#87](https://github.com/Open-Quant/openquant/issues/87)).
- **`num_threads` does nothing.** It is kept so that mlfinlab call sites port unchanged.

## Related modules

- [`filters`](/modules/filters/) — where the events come from.
- [`util-volatility`](/modules/util-volatility/) — the target.
- [`sampling`](/modules/sampling/), [`sample-weights`](/modules/sample-weights/),
  [`cross-validation`](/modules/cross-validation/) — what the overlap between labels requires.
- [`bet-sizing`](/modules/bet-sizing/) — from a meta-label probability to a position.
