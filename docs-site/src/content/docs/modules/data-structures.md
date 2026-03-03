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
  - "Run bars and imbalance bars are available via bars.build_run_bars and bars.build_imbalance_bars."
  - "`bar_diagnostics` is Python-only; use it to verify low return autocorrelation after bar construction."
rust_api:
  - "standard_bars"
  - "time_bars"
  - "run_bars"
  - "imbalance_bars"
  - "Trade"
  - "StandardBar"
  - "StandardBarType"
  - "ImbalanceBarType"
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
| `dollar_value_per_bar` | `float` | Dollar notional threshold for dollar bars (Python) | 5_000_000.0 |
| `volume_per_bar` | `float` | Cumulative volume threshold for volume bars (Python) | 100_000.0 |
| `ticks_per_bar` | `int` | Trade count threshold for tick bars (Python) | 50 |
| `interval` | `str` | Time interval for time bars, e.g. '1d', '5m', '1h' (Python) | '1d' |
| `threshold` | `f64` | Bar trigger threshold for standard_bars, run_bars, imbalance_bars (Rust) | — |
| `bar_type` | `StandardBarType` | Tick, Volume, or Dollar — selects accumulation metric (Rust) | — |

## Usage Examples

### Python

#### Build dollar bars from a Polars DataFrame

```python
from openquant.bars import build_dollar_bars, bar_diagnostics
import polars as pl

# Input: Polars DataFrame with ts, symbol, open, high, low, close, volume columns
df = pl.read_parquet("trades.parquet")

# Dollar bars: each bar aggregates ~$5M of notional
bars = build_dollar_bars(df, dollar_value_per_bar=5_000_000.0)
# Returns: Polars DataFrame with ts, symbol, open, high, low, close, volume, adj_close, start_ts, n_obs, dollar_value

# Check bar quality: low autocorrelation = good
diag = bar_diagnostics(bars)
print(diag)  # {"n_bars": 482.0, "lag1_return_autocorr": -0.02, ...}
```

#### Build tick and volume bars

```python
from openquant.bars import build_tick_bars, build_volume_bars, build_time_bars

tick_bars = build_tick_bars(df, ticks_per_bar=50)
vol_bars = build_volume_bars(df, volume_per_bar=100_000.0)
time_bars = build_time_bars(df, interval="5m")
```

### Rust

#### Build bars from Rust

```rust
use chrono::Duration;
use openquant::data_structures::{
    standard_bars, time_bars, run_bars, imbalance_bars,
    Trade, StandardBarType, ImbalanceBarType,
};

let trades: Vec<Trade> = vec![];

// Fixed-time bars
let t_bars = time_bars(&trades, Duration::minutes(5));

// Dollar bars via standard_bars
let d_bars = standard_bars(&trades, 50_000.0, StandardBarType::Dollar);

// Run bars
let r_bars = run_bars(&trades, 100);

// Tick imbalance bars
let ib = imbalance_bars(&trades, 500.0, ImbalanceBarType::Tick);
```

## Common Pitfalls

- Using time bars when your data has highly variable activity — dollar or volume bars will produce more stationary returns.
- Setting the threshold too low, creating extremely noisy high-frequency bars, or too high, losing intraday resolution.
- Forgetting to assign trade direction (buy/sell sign) before constructing imbalance or run bars — these require signed volume.
- Mixing bar types across train and inference: if you train on dollar bars, your live pipeline must also use dollar bars with the same threshold.
- Run bars and imbalance bars are available in Python via `bars.build_run_bars` and `bars.build_imbalance_bars`.

## API Reference

### Python API

- `bars.build_time_bars`
- `bars.build_tick_bars`
- `bars.build_volume_bars`
- `bars.build_dollar_bars`
- `bars.build_run_bars`
- `bars.build_imbalance_bars`
- `bars.bar_diagnostics`

### Rust API

- `standard_bars`
- `time_bars`
- `run_bars`
- `imbalance_bars`
- `Trade`
- `StandardBar`
- `StandardBarType`
- `ImbalanceBarType`

## Implementation Notes

- Threshold selection controls bar frequency and noise level.
- Keep OHLCV semantics consistent across downstream features.
- Run bars and imbalance bars are available via `bars.build_run_bars` and `bars.build_imbalance_bars`.
- `bar_diagnostics` is Python-only; use it to verify low return autocorrelation after bar construction.

## Related Modules

- [`filters`](/modules/filters/)
- [`labeling`](/modules/labeling/)
- [`fracdiff`](/modules/fracdiff/)
