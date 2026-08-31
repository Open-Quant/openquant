---
title: "synthetic_backtesting"
description: "Synthetic-data OTR backtesting with O-U calibration, PT/SL mesh search, and stability diagnostics."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-31'
audience:
  - quant-dev
  - platform-engineering
module: "synthetic_backtesting"
api_surface: "both"
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

## Concept Overview

AFML Chapter 13's answer to profit-taking and stop-loss overfitting. Rather than searching the PT/SL mesh on the single historical path you have — where the winning cell is mostly luck — it calibrates an Ornstein-Uhlenbeck process to that path, generates thousands of synthetic paths from the fitted parameters, and evaluates the whole mesh across all of them. `detect_no_stable_optimum` then asks whether the resulting Sharpe surface has a peak worth trusting at all.

## When to Use

Use it before committing to any exit rule. Its most valuable output is often the negative one: when the fitted persistence is close to 1 the price is near a random walk, the Sharpe surface is flat, and `no_stable_optimum` says so — meaning no PT/SL pair is defensible and the honest move is to skip the optimisation rather than take the argmax of noise. It complements `backtesting_engine` rather than replacing it, since that validates on the real path.

## Mathematical Foundations

### Discrete O-U (AR(1))

$$P_t=\alpha+\phi P_{t-1}+\sigma\epsilon_t,\quad \epsilon_t\sim\mathcal N(0,1)$$

### Equilibrium Level

$$\bar P=\frac{\alpha}{1-\phi}$$

### OTR Objective over Rule Mesh

$$R^*=\arg\max_{R\in\Omega}\frac{\mathbb E[\pi\mid R]}{\sigma[\pi\mid R]}$$

## Usage Examples

### Rust

#### End-to-end synthetic OTR workflow

```rust
use openquant::synthetic_backtesting::{
    run_synthetic_otr_workflow, StabilityCriteria, SyntheticBacktestConfig,
};

// A realised price history is fitted to obtain the O-U parameters the synthetic
// paths are drawn from.
let historical_prices: Vec<f64> =
    (0..500).map(|i| 100.0 + (i as f64 * 0.05).sin() * 3.0).collect();

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

## API Reference

### Python API

- `synthetic_bt.calibrate_ou_params`
- `synthetic_bt.generate_ou_paths`
- `synthetic_bt.evaluate_rule_on_paths`
- `synthetic_bt.detect_no_stable_optimum`
- `synthetic_bt.run_synthetic_otr_workflow`
- `synthetic_bt.search_optimal_trading_rule`

### Rust API

- `calibrate_ou_params`
- `generate_ou_paths`
- `evaluate_rule_on_paths`
- `search_optimal_trading_rule`
- `detect_no_stable_optimum`
- `run_synthetic_otr_workflow`

## Risk Notes and Caveats

- Near-random-walk estimates (|phi| close to 1) often produce flat Sharpe heatmaps where any selected rule is unstable out-of-sample.
- Calibrating to process parameters and evaluating many synthetic paths reduces single-path lucky-fit risk compared to brute-force historical optimization.

## Related Modules

- [`backtesting-engine`](/modules/backtesting-engine/)
- [`labeling`](/modules/labeling/)
- [`backtest-statistics`](/modules/backtest-statistics/)
- [`bet-sizing`](/modules/bet-sizing/)
- [`strategy-risk`](/modules/strategy-risk/)
