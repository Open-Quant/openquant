---
title: "strategy_risk"
description: "AFML Chapter 15 strategy-viability diagnostics based on precision, payout asymmetry, and bet frequency."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "strategy_risk"
risk_notes:
  - "Inputs under manager control ({pi_minus, pi_plus, n}) should be analyzed separately from uncertain market precision p."
  - "Use this module for strategy-level viability and probability-of-failure diagnostics; use `risk_metrics` for portfolio-tail and drawdown risk."
rust_api:
  - "sharpe_symmetric"
  - "implied_precision_symmetric"
  - "implied_frequency_symmetric"
  - "sharpe_asymmetric"
  - "implied_precision_asymmetric"
  - "implied_frequency_asymmetric"
  - "estimate_strategy_failure_probability"
  - "StrategyRiskConfig"
  - "StrategyRiskReport"
sidebar:
  badge: Module
---

## Subject

**Portfolio Construction and Risk**

## Why This Module Exists

Strategy risk is the probability that a process fails to achieve a Sharpe objective over time; it is distinct from holdings/portfolio variance risk and should be monitored separately.

## Mathematical Foundations

### Symmetric Sharpe

$$\theta=\frac{2p-1}{2\sqrt{p(1-p)}}\sqrt{n}$$

### Asymmetric Sharpe

$$\theta=\frac{(\pi_+-\pi_-)p+\pi_-}{(\pi_+-\pi_-)\sqrt{p(1-p)}}\sqrt{n}$$

### Strategy Failure Probability

$$P_{fail}=\Pr[p\le p^*],\quad p^*=\text{impliedPrecision}(\theta^*,\pi_+,\pi_-,n)$$

## Usage Examples

### Rust

#### Estimate strategy-failure probability from realized bets

```rust
use openquant::strategy_risk::{estimate_strategy_failure_probability, StrategyRiskConfig};

let outcomes = vec![0.005, -0.01, 0.005, 0.005, -0.01, 0.005, 0.005, -0.01];
let report = estimate_strategy_failure_probability(
  &outcomes,
  StrategyRiskConfig {
    years_elapsed: 2.0,
    target_sharpe: 2.0,
    investor_horizon_years: 2.0,
    bootstrap_iterations: 10_000,
    seed: 7,
    kde_bandwidth: None,
  },
)?;

println!("p*: {:.4}", report.implied_precision_threshold);
println!("failure (KDE): {:.2}%", 100.0 * report.kde_failure_probability);
```

## API Reference

### Rust API

- `sharpe_symmetric`
- `implied_precision_symmetric`
- `implied_frequency_symmetric`
- `sharpe_asymmetric`
- `implied_precision_asymmetric`
- `implied_frequency_asymmetric`
- `estimate_strategy_failure_probability`
- `StrategyRiskConfig`
- `StrategyRiskReport`

## Implementation Notes

- Inputs under manager control ({pi_minus, pi_plus, n}) should be analyzed separately from uncertain market precision p.
- Use this module for strategy-level viability and probability-of-failure diagnostics; use `risk_metrics` for portfolio-tail and drawdown risk.
