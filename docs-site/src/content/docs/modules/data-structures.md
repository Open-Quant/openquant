---
title: "data_structures"
description: "Constructs standard/time/run/imbalance bars from trade streams."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "data_structures"
risk_notes:
  - "Threshold selection controls bar frequency and noise level."
  - "Keep OHLCV semantics consistent across downstream features."
rust_api:
  - "standard_bars"
  - "time_bars"
  - "run_bars"
  - "imbalance_bars"
  - "Trade"
  - "StandardBar"
sidebar:
  badge: Module
---

## Subject

**Event-Driven Data and Labeling**

## Why This Module Exists

Event-based bars reduce heteroskedasticity and improve stationarity versus fixed-time sampling.

## Key Public APIs

- `standard_bars`
- `time_bars`
- `run_bars`
- `imbalance_bars`
- `Trade`
- `StandardBar`

## Mathematical Definitions

### Dollar Bar Trigger

$$\sum_{i=t_0}^{t} p_i v_i \ge \theta$$

### Imbalance Trigger

$$\left|\sum b_i\right| \ge E[|\sum b_i|]$$

## Implementation Examples

### Build time bars

```rust
use chrono::Duration;
use openquant::data_structures::{time_bars, Trade};

let trades: Vec<Trade> = vec![];
let bars = time_bars(&trades, Duration::minutes(5));
```

## Implementation Notes

- Threshold selection controls bar frequency and noise level.
- Keep OHLCV semantics consistent across downstream features.
