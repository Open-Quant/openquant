---
title: "synthetic_backtesting"
description: "Synthetic-data OTR backtesting with O-U calibration, PT/SL mesh search, and stability diagnostics."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "synthetic_backtesting"
risk_notes:
  - "Near-random-walk estimates (|phi| close to 1) often produce flat Sharpe heatmaps where any selected rule is unstable out-of-sample."
  - "Calibrating to process parameters and evaluating many synthetic paths reduces single-path lucky-fit risk compared to brute-force historical optimization."
rust_api:
  - "calibrate_ou_params"
  - "generate_ou_paths"
  - "evaluate_rule_on_paths"
  - "search_optimal_trading_rule"
  - "detect_no_stable_optimum"
  - "run_synthetic_otr_workflow"
sidebar:
  badge: Module
---

## Subject

**Sampling, Validation and ML Diagnostics**

## Why This Module Exists

AFML Chapter 13 shows that selecting PT/SL rules on a single historical path is prone to overfitting; synthetic path ensembles let us evaluate rule robustness under calibrated process dynamics.

## Key Public APIs

- `calibrate_ou_params`
- `generate_ou_paths`
- `evaluate_rule_on_paths`
- `search_optimal_trading_rule`
- `detect_no_stable_optimum`
- `run_synthetic_otr_workflow`

## Mathematical Definitions

### Discrete O-U (AR(1))

$$P_t=\alpha+\phi P_{t-1}+\sigma\epsilon_t,\quad \epsilon_t\sim\mathcal N(0,1)$$

### Equilibrium Level

$$\bar P=\frac{\alpha}{1-\phi}$$

### OTR Objective over Rule Mesh

$$R^*=\arg\max_{R\in\Omega}\frac{\mathbb E[\pi\mid R]}{\sigma[\pi\mid R]}$$

## Implementation Examples

### End-to-end synthetic OTR workflow

```rust
use openquant::synthetic_backtesting::{run_synthetic_otr_workflow, StabilityCriteria, SyntheticBacktestConfig};

let cfg = SyntheticBacktestConfig {
  initial_price: historical_prices[historical_prices.len() - 1],
  n_paths: 10_000,
  horizon: 128,
  seed: 42,
  profit_taking_grid: vec![0.5, 1.0, 1.5, 2.0, 3.0],
  stop_loss_grid: vec![0.5, 1.0, 1.5, 2.0, 3.0],
  max_holding_steps: 64,
  annualization_factor: 1.0,
  stability_criteria: StabilityCriteria::default(),
};

let out = run_synthetic_otr_workflow(&historical_prices, &cfg)?;
if out.diagnostics.no_stable_optimum {
  println!("Skip OTR optimization: {}", out.diagnostics.reason);
} else {
  println!("Best PT/SL: {:?}", out.best_rule);
}
```

## Implementation Notes

- Near-random-walk estimates (|phi| close to 1) often produce flat Sharpe heatmaps where any selected rule is unstable out-of-sample.
- Calibrating to process parameters and evaluating many synthetic paths reduces single-path lucky-fit risk compared to brute-force historical optimization.
