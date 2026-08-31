---
title: "strategy_risk"
description: "AFML Chapter 15 strategy-viability diagnostics based on precision, payout asymmetry, and bet frequency."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-31'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "strategy_risk"
api_surface: "both"
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

## Concept Overview

AFML Chapter 15 asks a question portfolio risk does not: given the precision, payout asymmetry and bet frequency this strategy actually achieved, what is the probability that the *process* fails to reach its Sharpe target? The symmetric and asymmetric helpers invert the Sharpe relation for whichever variable you are solving for — implied precision, implied frequency — and `estimate_strategy_failure_probability` bootstraps the realised bet outcomes, fits a KDE to the resulting precision distribution, and reports the mass falling below the precision the target Sharpe requires.

## When to Use

Use it at strategy-approval time and then as a standing monitor: the implied precision threshold p* is a concrete kill criterion, and a strategy whose realised precision drifts toward it is failing before its PnL says so. Analyse the manager-controlled inputs — the payouts and the bet count — separately from market-determined precision, because the first are design choices and the second is not. This is strategy viability; use `risk_metrics` for holdings and tail risk.

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

### Python API

- `strategy_risk.sharpe_symmetric`
- `strategy_risk.implied_precision_symmetric`
- `strategy_risk.implied_frequency_symmetric`
- `strategy_risk.sharpe_asymmetric`
- `strategy_risk.implied_precision_asymmetric`
- `strategy_risk.implied_frequency_asymmetric`
- `strategy_risk.estimate_strategy_failure_probability`

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

## Risk Notes and Caveats

- Inputs under manager control ({pi_minus, pi_plus, n}) should be analyzed separately from uncertain market precision p.
- Use this module for strategy-level viability and probability-of-failure diagnostics; use `risk_metrics` for portfolio-tail and drawdown risk.

## Related Modules

- [`risk-metrics`](/modules/risk-metrics/)
- [`backtest-statistics`](/modules/backtest-statistics/)
- [`bet-sizing`](/modules/bet-sizing/)
- [`backtesting-engine`](/modules/backtesting-engine/)
