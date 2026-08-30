---
title: "hrp"
description: "Hierarchical Risk Parity allocation with recursive bisection."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-30'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "hrp"
api_surface: "both"
risk_notes:
  - "HRP is often more robust under unstable covariance estimates."
  - "Ensure input asset order tracks produced dendrogram order."
rust_api:
  - "HierarchicalRiskParity"
  - "HrpDendrogram"
sidebar:
  badge: Module
---

## Subject

**Portfolio Construction and Risk**

## Why This Module Exists

Produces stable allocations without matrix inversion required by classic Markowitz.

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

## Implementation Notes

- HRP is often more robust under unstable covariance estimates.
- Ensure input asset order tracks produced dendrogram order.
