---
title: "labeling"
description: "Triple-barrier event labeling and metadata generation."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "labeling"
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

## Subject

**Event-Driven Data and Labeling**

## Why This Module Exists

Converts event outcomes into ML labels with controlled horizon and risk barriers.

## Key Public APIs

- `add_vertical_barrier`
- `get_events`
- `get_bins`
- `drop_labels`
- `Event`

## Mathematical Definitions

### Triple-Barrier Event Time

$$\tau=\min\left(\tau_{pt},\tau_{sl},t_1\right),\quad\tau_{pt}=\inf\{u>t:r_{t,u}\ge pt\cdot\sigma_t\},\quad\tau_{sl}=\inf\{u>t:r_{t,u}\le-sl\cdot\sigma_t\}$$

### Labeling Rule

$$y_t=\begin{cases}1,&r_{t,\tau}>0\\0,&r_{t,\tau}=0\\-1,&r_{t,\tau}<0\end{cases},\qquad\text{meta label: }y_t^{meta}=\mathbf 1\{\operatorname{side}_t\cdot r_{t,\tau}>0\}$$

### Target Volatility Scaling

$$\sigma_t=\operatorname{EWMA}\big(|r_t|\big),\qquad\text{barrier widths }\propto \sigma_t$$

## Implementation Examples

### End-to-end: Event Filter -> Vertical Barrier -> Triple Barrier Labels

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

### Meta-Labeling Workflow with Side Signal

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

## Implementation Notes

- Label stability is dominated by event quality and volatility-target quality; calibrate these before tuning ML models.
- Always audit class balance and average holding time after labeling; both drive downstream model behavior.
- In meta-labeling, side alignment and timestamp joins are a frequent hidden bug source.
