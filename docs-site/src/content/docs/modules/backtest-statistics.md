---
title: "backtest_statistics"
description: "Performance diagnostics for strategy returns and position trajectories."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "backtest_statistics"
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
use openquant::backtest_statistics::{sharpe_ratio, drawdown_and_time_under_water};

let returns = vec![0.01, -0.005, 0.007, -0.002, 0.003];
let sr = sharpe_ratio(&returns, 252.0, 0.0);
let (dd, tuw) = drawdown_and_time_under_water(&returns);
println!("{sr} {dd:?} {tuw:?}");
```

## API Reference

### Rust API

- `sharpe_ratio`
- `deflated_sharpe_ratio`
- `probabilistic_sharpe_ratio`
- `drawdown_and_time_under_water`
- `average_holding_period`

## Implementation Notes

- Use annualization constants consistent with your bar frequency.
- Deflated Sharpe is useful when strategy mining many variants.
