---
title: "data_structures"
description: "Constructs standard/time/run/imbalance bars from trade streams."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "data_structures"
api_surface: "both"
afml_chapters:
  - 2
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

## Concept Overview

Traditional financial data uses fixed-time bars (1-minute, daily), but these sample uniformly regardless of market activity. During quiet periods you get noise; during volatile periods you under-sample important information.

Information-driven bars (AFML Chapter 2) sample based on market activity instead of clock time. **Dollar bars** trigger a new bar when cumulative traded dollar volume reaches a threshold, producing roughly equal-information observations. **Volume bars** trigger on cumulative share volume. **Tick bars** trigger on trade count.

**Imbalance bars** go further: they detect when the net signed trade flow (buy minus sell) exceeds its expected magnitude, capturing points where informed trading pressure shifts. **Run bars** detect runs of same-signed trades exceeding expectations.

The key insight is that information-driven bars produce returns that are closer to IID normal, which makes downstream ML models (labeling, feature importance, cross-validation) better behaved. All AFML workflows assume information-driven bars as input.

## When to Use

This is the first module in any AFML pipeline. Raw tick or trade data goes in; structured OHLCV bars come out. Everything downstream — labeling, features, sampling — consumes these bars.

**Prerequisites**: Raw trade or tick data with timestamps, prices, and volumes.

**Alternatives**: Standard time bars if your data is already aggregated. For pre-aggregated OHLCV data, use the `data` module's `load_ohlcv` and `clean_ohlcv` functions instead.

## Mathematical Foundations

### Dollar Bar Trigger

$$\sum_{i=t_0}^{t} p_i v_i \ge \theta$$

### Imbalance Trigger

$$\left|\sum b_i\right| \ge E[|\sum b_i|]$$

## Key Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `threshold` | `f64` | Bar trigger threshold; interpretation depends on bar type (dollar notional, volume, tick count) | — |
| `trades` | `Vec<Trade>` | Input trade stream with timestamp, price, volume, and optional side | — |

## Usage Examples

### Python

#### Build dollar bars from Python

```python
from openquant._core import data_structures

# Trade data: list of (timestamp_str, price, volume, side)
trades = [
    ("2024-01-02T09:30:00", 100.0, 150.0, 1),
    ("2024-01-02T09:30:01", 100.1, 200.0, 1),
    ("2024-01-02T09:30:02", 99.9, 300.0, -1),
    # ... more trades
]

# Dollar bars: each bar aggregates ~$50k of notional
bars = data_structures.dollar_bars(trades, threshold=50_000.0)
# Each bar: (timestamp, open, high, low, close, volume, dollar_volume)
```

### Rust

#### Build time bars

```rust
use chrono::Duration;
use openquant::data_structures::{time_bars, Trade};

let trades: Vec<Trade> = vec![];
let bars = time_bars(&trades, Duration::minutes(5));
```

## Common Pitfalls

- Using time bars when your data has highly variable activity — dollar or volume bars will produce more stationary returns.
- Setting the threshold too low, creating extremely noisy high-frequency bars, or too high, losing intraday resolution.
- Forgetting to assign trade direction (buy/sell sign) before constructing imbalance or run bars — these require signed volume.
- Mixing bar types across train and inference: if you train on dollar bars, your live pipeline must also use dollar bars with the same threshold.

## API Reference

### Python API

- `data_structures.dollar_bars`
- `data_structures.volume_bars`
- `data_structures.tick_bars`
- `data_structures.imbalance_bars`
- `data_structures.run_bars`
- `data_structures.time_bars`

### Rust API

- `standard_bars`
- `time_bars`
- `run_bars`
- `imbalance_bars`
- `Trade`
- `StandardBar`

## Implementation Notes

- Threshold selection controls bar frequency and noise level.
- Keep OHLCV semantics consistent across downstream features.

## Related Modules

- [`filters`](/modules/filters/)
- [`labeling`](/modules/labeling/)
- [`fracdiff`](/modules/fracdiff/)
