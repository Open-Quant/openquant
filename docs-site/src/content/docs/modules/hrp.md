---
title: "hrp"
description: "Hierarchical Risk Parity allocation with recursive bisection."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-31'
audience:
  - quant-dev
  - platform-engineering
module: "hrp"
api_surface: "both"
rust_api:
  - "HierarchicalRiskParity"
  - "HrpDendrogram"
sidebar:
  badge: Module
---

## Concept Overview

Hierarchical Risk Parity replaces matrix inversion with a tree. It clusters assets on a correlation distance, reorders the covariance matrix so that similar assets sit adjacent (quasi-diagonalisation), then recursively bisects that ordering, splitting capital between the two halves in inverse proportion to their cluster variance. Nothing is inverted, so the numerical instability that makes Markowitz weights swing violently under a noisy covariance estimate simply does not arise.

## When to Use

Use it when the asset count is large relative to the sample, when the covariance estimate is noisy, or whenever mean-variance weights are unstable between rebalances — which out of sample is most of the time. It needs no expected returns, which is both its robustness and its limit: if you have return views you trust, `cla` or `portfolio_optimization` will use them and HRP will not. Keep the asset ordering you pass in aligned with the dendrogram order you read back.

## Mathematical Foundations

### IVP Weight

$$w_i\propto\frac{1}{\sigma_i^2}$$

### Bisection Split

$$\alpha=1-\frac{\sigma_{left}^2}{\sigma_{left}^2+\sigma_{right}^2}$$

## Usage Examples

### Rust

#### Allocate with HRP

```rust
use nalgebra::DMatrix;
use openquant::hrp::HierarchicalRiskParity;

let asset_names: Vec<String> =
    ["SPY", "TLT", "GLD", "HYG"].iter().map(|s| s.to_string()).collect();
// rows = observations, cols = assets, in the same order as `asset_names`.
let prices = DMatrix::from_fn(250, 4, |i, j| 100.0 + (i as f64) * 0.05 + (j as f64) * 3.0);

let mut hrp = HierarchicalRiskParity::new();

// allocate() mutates the struct and returns Result<(), HrpError>; the weights are
// read back from `hrp.weights`. Exactly one of prices / returns / covariance must
// be supplied.
hrp.allocate(
    &asset_names,
    Some(&prices), // asset_prices
    None,          // asset_returns
    None,          // covariance_matrix
    None,          // resample_by
    false,         // use_shrinkage — Ledoit-Wolf shrinkage on the covariance
)?;

println!("weights: {:?}", hrp.weights);
println!("seriation order: {:?}", hrp.ordered_indices);
```

## API Reference

### Python API

- `hrp.allocate_hrp`

### Rust API

- `HierarchicalRiskParity`
- `HrpDendrogram`

## Risk Notes and Caveats

- HRP is often more robust under unstable covariance estimates.
- Ensure input asset order tracks produced dendrogram order.

## Related Modules

- [`hcaa`](/modules/hcaa/)
- [`codependence`](/modules/codependence/)
- [`onc`](/modules/onc/)
- [`portfolio-optimization`](/modules/portfolio-optimization/)
- [`cla`](/modules/cla/)
