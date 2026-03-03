---
title: "portfolio_optimization"
description: "Mean-variance and constrained allocation methods with ergonomic APIs."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "portfolio_optimization"
risk_notes:
  - "Optimizer output is only as good as mean/covariance assumptions; stress-test inputs and rebalance frequency."
  - "Constraint design (asset caps, sector caps, long/short bounds) is usually more important than small objective tweaks."
  - "Track turnover, realized slippage, and drift between target and filled weights in production."
rust_api:
  - "allocate_inverse_variance"
  - "allocate_min_vol"
  - "allocate_max_sharpe"
  - "allocate_efficient_risk"
  - "AllocationOptions"
sidebar:
  badge: Module
---

## Subject

**Portfolio Construction and Risk**

## Why This Module Exists

Provides production-ready portfolio construction primitives with explicit options and constraints.

## Mathematical Foundations

### Constrained Mean-Variance Program

$$\begin{aligned}\min_{w}\;&\frac{1}{2}w^T\Sigma w-\lambda\mu^T w\\\text{s.t. }&\mathbf 1^T w=1,\quad l_i\le w_i\le u_i\end{aligned}$$

### Minimum Variance / Maximum Sharpe / Efficient Return

$$\begin{aligned}w_{MV}&=\arg\min_w\;w^T\Sigma w\\w_{MSR}&=\arg\max_w\;\frac{w^T(\mu-r_f\mathbf 1)}{\sqrt{w^T\Sigma w}}\\w_{ER}(r^*)&=\arg\min_w\;w^T\Sigma w\;\text{s.t. }w^T\mu\ge r^*\end{aligned}$$

### Exponential Mean Estimator

$$\mu_t=\frac{\sum_{k=0}^{T-1}(1-\alpha)^k r_{t-k}}{\sum_{k=0}^{T-1}(1-\alpha)^k},\qquad \alpha=\frac{2}{\text{span}+1}$$

## Usage Examples

### Rust

#### End-to-end: Compute and Compare Core Allocators

```rust
use nalgebra::DMatrix;
use openquant::portfolio_optimization::{
    allocate_inverse_variance,
    allocate_min_vol,
    allocate_max_sharpe,
    allocate_efficient_risk,
};

// rows=time, cols=assets
let prices: DMatrix<f64> = /* load matrix */ DMatrix::zeros(252, 6);

let ivp = allocate_inverse_variance(&prices)?;
let mv = allocate_min_vol(&prices, None, None)?;
let msr = allocate_max_sharpe(&prices, 0.01, None, None)?;
let er = allocate_efficient_risk(&prices, 0.12, None, None)?;

assert_eq!(ivp.weights.len(), prices.ncols());
assert!((mv.weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);
assert!((msr.weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);
assert!((er.weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);
```

#### End-to-end: Constrained Allocation with Exponential Returns and Resampling

```rust
use std::collections::HashMap;
use openquant::portfolio_optimization::{
    allocate_max_sharpe_with,
    AllocationOptions,
    ReturnsMethod,
};

let mut bounds = HashMap::new();
// Cap concentration in first asset; enforce long-only defaults elsewhere
bounds.insert(0usize, (0.0, 0.20));

let opts = AllocationOptions {
    risk_free_rate: 0.02,
    returns_method: ReturnsMethod::Exponential { span: 60 },
    resample_by: Some("W"),
    bounds: Some(bounds),
    tuple_bounds: Some((0.0, 0.40)),
    ..Default::default()
};

let constrained = allocate_max_sharpe_with(&prices, &opts)?;
assert!(constrained.weights.iter().all(|w| *w >= -1e-10));
```

## API Reference

### Rust API

- `allocate_inverse_variance`
- `allocate_min_vol`
- `allocate_max_sharpe`
- `allocate_efficient_risk`
- `AllocationOptions`

## Implementation Notes

- Optimizer output is only as good as mean/covariance assumptions; stress-test inputs and rebalance frequency.
- Constraint design (asset caps, sector caps, long/short bounds) is usually more important than small objective tweaks.
- Track turnover, realized slippage, and drift between target and filled weights in production.
