---
title: "labeling"
description: "Triple-barrier event labeling and metadata generation."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "labeling"
api_surface: "both"
afml_chapters:
  - 3
risk_notes:
  - "Label stability is dominated by event quality and volatility-target quality; calibrate these before tuning ML models."
  - "Always audit class balance and average holding time after labeling; both drive downstream model behavior."
  - "In meta-labeling, side alignment and timestamp joins are a frequent hidden bug source."
rust_api:
  - "add_vertical_barrier"
  - "get_events"
  - "get_bins"
  - "drop_labels"
  - "Event"
sidebar:
  badge: Module
---

## Concept Overview

The triple-barrier method (AFML Chapter 3) replaces fixed-horizon labeling with a path-dependent approach. Instead of asking "did the price go up in 10 days?", it asks "which barrier did the price hit first — a profit-taking ceiling, a stop-loss floor, or a maximum holding horizon?"

This matters because fixed-horizon labels create artifacts: a trade that hits +5% then reverses to -1% at the horizon gets labeled as a loss. Triple-barrier labels capture the actual trade outcome under realistic exit rules.

**Meta-labeling** is a two-stage extension: a primary model predicts direction (side), while a secondary model learns *when to act* on that signal. The secondary model's label is binary (1 = the primary model was correct, 0 = it wasn't). This separation lets you combine a simple directional model with a sophisticated sizing/filtering model.

Barrier widths are scaled by a volatility target (typically EWMA of returns), making them adaptive across regimes. Events are sourced from structural filters like CUSUM rather than calendar time.

## When to Use

Use this module immediately after event detection (CUSUM/z-score filters) and volatility estimation. It sits at the start of the ML pipeline: raw price events go in, labeled training examples come out.

**Prerequisites**: A price series with timestamps, filtered event timestamps, and a volatility target series.

**Alternatives**: Fixed-horizon labeling (simpler but regime-blind), or trend-scanning labels for continuous-valued targets instead of classification.

## Mathematical Foundations

### Triple-Barrier Event Time

$$\tau=\min\left(\tau_{pt},\tau_{sl},t_1\right),\quad\tau_{pt}=\inf\{u>t:r_{t,u}\ge pt\cdot\sigma_t\},\quad\tau_{sl}=\inf\{u>t:r_{t,u}\le-sl\cdot\sigma_t\}$$

### Labeling Rule

$$y_t=\begin{cases}1,&r_{t,\tau}>0\\0,&r_{t,\tau}=0\\-1,&r_{t,\tau}<0\end{cases},\qquad\text{meta label: }y_t^{meta}=\mathbf 1\{\operatorname{side}_t\cdot r_{t,\tau}>0\}$$

### Target Volatility Scaling

$$\sigma_t=\operatorname{EWMA}\big(|r_t|\big),\qquad\text{barrier widths }\propto \sigma_t$$

## Key Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `pt` | `f64` | Profit-taking barrier multiplier (× volatility target) | 1.0 |
| `sl` | `f64` | Stop-loss barrier multiplier (× volatility target) | 1.0 |
| `min_ret` | `f64` | Minimum return threshold; events with smaller absolute returns are labeled 0 | 0.0 |
| `vertical_barrier_times` | `Option<Vec>` | Maximum holding period timestamps; events expire if neither profit nor stop barrier is hit | None |
| `side_prediction` | `Option<Vec<f64>>` | Primary model side forecasts (+1/−1) for meta-labeling mode | None |

## Usage Examples

### Python

#### Triple-barrier labels from price series

```python
from openquant._core import labeling, filters

# 1) Detect events with CUSUM filter
timestamps = ["2024-01-01T09:30:00", "2024-01-01T09:31:00", ...]
close = [100.0, 100.1, 99.9, 100.2, 100.05, 100.3, ...]
event_ts = filters.cusum_filter_timestamps(close, timestamps, 0.02)

# 2) Estimate target volatility (use your own EWMA or rolling std)
target_ts = event_ts
target_vals = [0.02] * len(event_ts)  # simplified constant target

# 3) Compute triple-barrier labels
labels = labeling.triple_barrier_labels(
    close_timestamps=timestamps,
    close_prices=close,
    t_events=event_ts,
    target_timestamps=target_ts,
    target_values=target_vals,
    pt=1.0, sl=1.0, min_ret=0.005,
)
# Each label: (event_ts, return, target, label_int, touch_ts)
```

#### Meta-labeling: learn when to act on a primary signal

```python
from openquant._core import labeling

# Primary model gives side predictions (+1 or -1) at each event
side_prediction = [1.0, -1.0, 1.0, 1.0, -1.0, ...]

meta_labels = labeling.meta_labels(
    close_timestamps=timestamps,
    close_prices=close,
    t_events=event_ts,
    target_timestamps=target_ts,
    target_values=target_vals,
    side_prediction=side_prediction,
    pt=1.0, sl=1.0, min_ret=0.005,
)
# Train a secondary classifier on meta_labels to filter false signals
```

### Rust

#### End-to-end: Event Filter -> Vertical Barrier -> Triple Barrier Labels

```rust
use chrono::NaiveDateTime;
use openquant::filters::{cusum_filter_timestamps, Threshold};
use openquant::labeling::{add_vertical_barrier, get_events, get_bins};
use openquant::util::volatility::get_daily_vol;

// 1) price series and timestamps
let close: Vec<(NaiveDateTime, f64)> = /* load bars */ vec![];
let prices: Vec<f64> = close.iter().map(|(_, p)| *p).collect();
let ts: Vec<NaiveDateTime> = close.iter().map(|(t, _)| *t).collect();

// 2) detect candidate events via CUSUM filter
let events = cusum_filter_timestamps(&prices, &ts, Threshold::Scalar(0.02));

// 3) estimate target volatility and add max-holding horizon
let target = get_daily_vol(&close, 100);
let vbars = add_vertical_barrier(&events, &close, 1, 0, 0, 0);

// 4) compute barrier touches and labels
let ev = get_events(&close, &events, (1.0, 1.0), &target, 0.005, 3, Some(&vbars), None);
let bins = get_bins(&ev, &close);
assert!(!bins.is_empty());
```

#### Meta-Labeling Workflow with Side Signal

```rust
use chrono::NaiveDateTime;
use openquant::labeling::{get_events, get_bins};

let close: Vec<(NaiveDateTime, f64)> = /* bars */ vec![];
let events: Vec<NaiveDateTime> = /* primary event timestamps */ vec![];
let target: Vec<(NaiveDateTime, f64)> = /* vol target */ vec![];
let vbars: Vec<(NaiveDateTime, NaiveDateTime)> = /* horizon */ vec![];

// Primary model side forecast (+1 / -1)
let side: Vec<(NaiveDateTime, f64)> = events.iter().map(|t| (*t, 1.0)).collect();

let meta_events = get_events(
    &close,
    &events,
    (1.0, 1.0),
    &target,
    0.005,
    3,
    Some(&vbars),
    Some(&side),
);
let meta_bins = get_bins(&meta_events, &close);
// Use meta_bins to train a second-stage filter (take/skip decision)
assert!(!meta_bins.is_empty());
```

## Common Pitfalls

- Setting symmetric barriers (pt=sl=1) when the strategy has asymmetric payoff — calibrate each barrier width independently.
- Using calendar-time vertical barriers with information-driven bars — the holding period should match bar frequency, not wall time.
- Ignoring class imbalance after labeling: if 80% of events hit the vertical barrier, the model learns to predict 'no movement' and the labels need recalibration.
- Forgetting that meta-labeling requires aligned timestamps between the primary model's side predictions and the event set — off-by-one joins silently corrupt labels.

## API Reference

### Python API

- `labeling.triple_barrier_labels`
- `labeling.triple_barrier_events`
- `labeling.meta_labels`

### Rust API

- `add_vertical_barrier`
- `get_events`
- `get_bins`
- `drop_labels`
- `Event`

## Implementation Notes

- Label stability is dominated by event quality and volatility-target quality; calibrate these before tuning ML models.
- Always audit class balance and average holding time after labeling; both drive downstream model behavior.
- In meta-labeling, side alignment and timestamp joins are a frequent hidden bug source.

## Related Modules

- [`filters`](/modules/filters/)
- [`sample-weights`](/modules/sample-weights/)
- [`sampling`](/modules/sampling/)
- [`bet-sizing`](/modules/bet-sizing/)
