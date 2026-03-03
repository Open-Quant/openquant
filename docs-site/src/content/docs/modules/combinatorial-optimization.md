---
title: "combinatorial_optimization"
description: "AFML Chapter 21 integer-encoded optimization and trajectory state-space tooling with exact baselines and solver adapters."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "combinatorial_optimization"
risk_notes:
  - "Exact enumeration scales exponentially in decision dimension/horizon; treat it as a correctness baseline and regression oracle."
  - "Use adapter interfaces to compare heuristic/external solvers against exact solutions on small calibration instances before production deployment."
rust_api:
  - "DecisionSchema"
  - "IntegerVariable"
  - "IntegerObjective"
  - "solve_exact"
  - "SolverAdapter"
  - "solve_with_adapter"
  - "compare_exact_and_adapter"
  - "TradingTrajectorySchema"
  - "enumerate_trading_paths"
  - "evaluate_trading_path"
  - "solve_trading_trajectory_exact"
sidebar:
  badge: Module
---

## Subject

**Scaling, HPC and Infrastructure**

## Why This Module Exists

Many trading/search problems are discrete and path-dependent; this module keeps integer structure explicit and provides exact small-instance baselines before scaling to heuristics.

## Mathematical Foundations

### Finite Integer Program

$$x^*=\arg\max_{x\in\mathcal X\subset\mathbb Z^d} f(x),\quad |\mathcal X|<\infty$$

### Path-Dependent Objective

$$J(\tau)=\sum_{t=1}^{T}\left(q_t r_t-\lambda q_t^2-c_t|\Delta q_t|-\kappa\,\mathbf 1_{\Delta q_t\ne0}\right)-\eta(q_T-q^*)^2$$

### Adapter Gap vs Exact

$$\Delta_{alg}=\begin{cases}f(x^*)-f(\hat x) & \text{maximize}\\f(\hat x)-f(x^*) & \text{minimize}\end{cases}$$

## Usage Examples

### Rust

#### Exact trajectory search with fixed ticket costs

```rust
use openquant::combinatorial_optimization::{
  TradeBounds, TradingTrajectoryObjectiveConfig, TradingTrajectoryPath, TradingTrajectorySchema,
  enumerate_trading_paths, evaluate_trading_path,
};

let schema = TradingTrajectorySchema {
  initial_inventory: 0,
  inventory_min: -2,
  inventory_max: 2,
  step_trade_bounds: vec![
    TradeBounds { min_trade: -1, max_trade: 1 },
    TradeBounds { min_trade: -1, max_trade: 1 },
    TradeBounds { min_trade: -1, max_trade: 1 },
  ],
  terminal_inventory: Some(0),
  max_paths: 50_000,
};
let cfg = TradingTrajectoryObjectiveConfig {
  expected_returns: vec![0.01, -0.015, 0.012],
  risk_aversion: 0.001,
  impact_coefficients: vec![0.0005, 0.0005, 0.0005],
  fixed_ticket_cost: 0.002,
  terminal_inventory_target: 0,
  terminal_inventory_penalty: 0.05,
};

let best = enumerate_trading_paths(&schema)?
  .into_iter()
  .map(|path| {
    let score = evaluate_trading_path(&path, &cfg)?;
    Ok::<(TradingTrajectoryPath, f64), openquant::combinatorial_optimization::CombinatorialOptimizationError>((path, score))
  })
  .collect::<Result<Vec<_>, _>>()?
  .into_iter()
  .max_by(|a, b| a.1.total_cmp(&b.1))
  .expect("at least one feasible path");

println!("best objective: {:.6}", best.1);
println!("trades: {:?}", best.0.trades);
```

## API Reference

### Rust API

- `DecisionSchema`
- `IntegerVariable`
- `IntegerObjective`
- `solve_exact`
- `SolverAdapter`
- `solve_with_adapter`
- `compare_exact_and_adapter`
- `TradingTrajectorySchema`
- `enumerate_trading_paths`
- `evaluate_trading_path`
- `solve_trading_trajectory_exact`

## Implementation Notes

- Exact enumeration scales exponentially in decision dimension/horizon; treat it as a correctness baseline and regression oracle.
- Use adapter interfaces to compare heuristic/external solvers against exact solutions on small calibration instances before production deployment.
