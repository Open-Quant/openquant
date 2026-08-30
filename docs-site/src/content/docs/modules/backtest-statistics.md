---
title: "backtest_statistics"
description: "Performance diagnostics for strategy returns and position trajectories."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-30'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "backtest_statistics"
api_surface: "both"
risk_notes:
  - "Use annualization constants consistent with your bar frequency."
  - "Deflated Sharpe is useful when strategy mining many variants."
rust_api:
  - "sharpe_ratio"
  - "deflated_sharpe_ratio"
  - "probabilistic_sharpe_ratio"
  - "drawdown_and_time_under_water"
  - "average_holding_period"
sidebar:
  badge: Module
---

## Subject

**Portfolio Construction and Risk**

## Why This Module Exists

Turns raw PnL/returns into risk-adjusted diagnostics used in model selection and production monitoring.

## Mathematical Foundations

### Sharpe

$$S=\frac{\mu-r_f}{\sigma}$$

### Information Ratio

$$IR=\frac{\mu-r_b}{\sigma_{(r-r_b)}}$$

## Usage Examples

### Rust

#### Compute Sharpe and drawdown

```rust
use chrono::{Duration, NaiveDateTime};
use openquant::backtest_statistics::{drawdown_and_time_under_water, sharpe_ratio};

let returns = vec![0.01, -0.005, 0.007, -0.002, 0.003];
let sharpe = sharpe_ratio(&returns, 252.0, 0.0);

// Drawdown and time-under-water are computed on a *timestamped equity curve*,
// not on the return series: the function needs the timestamps to measure how
// long each high-water mark went un-recovered.
let t0 = NaiveDateTime::parse_from_str("2024-01-02 00:00:00", "%Y-%m-%d %H:%M:%S")?;
let mut equity = 1.0;
let curve: Vec<(NaiveDateTime, f64)> = returns
    .iter()
    .enumerate()
    .map(|(i, r)| {
        equity *= 1.0 + r;
        (t0 + Duration::days(i as i64), equity)
    })
    .collect();

// dollars = false reports each drawdown as a fraction of its high-water mark.
let (drawdowns, time_under_water) = drawdown_and_time_under_water(&curve, false);
println!("sharpe={sharpe:.3} drawdowns={drawdowns:?} tuw={time_under_water:?}");
```

## API Reference

### Python API

- `backtest_stats.sharpe_ratio`
- `backtest_stats.information_ratio`
- `backtest_stats.probabilistic_sharpe_ratio`
- `backtest_stats.deflated_sharpe_ratio`
- `backtest_stats.minimum_track_record_length`
- `backtest_stats.timing_of_flattening_and_flips`
- `backtest_stats.average_holding_period`
- `backtest_stats.bets_concentration`
- `backtest_stats.all_bets_concentration`
- `backtest_stats.drawdown_and_time_under_water`

### Rust API

- `sharpe_ratio`
- `deflated_sharpe_ratio`
- `probabilistic_sharpe_ratio`
- `drawdown_and_time_under_water`
- `average_holding_period`

## Implementation Notes

- Use annualization constants consistent with your bar frequency.
- Deflated Sharpe is useful when strategy mining many variants.
