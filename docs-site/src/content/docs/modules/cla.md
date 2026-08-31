---
title: "cla"
description: "Critical Line Algorithm implementation for constrained mean-variance optimization."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-31'
audience:
  - quant-dev
  - platform-engineering
module: "cla"
api_surface: "both"
rust_api:
  - "CLA"
  - "covariance"
  - "ReturnsEstimation"
sidebar:
  badge: Module
---

## Concept Overview

Markowitz's Critical Line Algorithm in the Bailey-Lopez de Prado formulation: the exact solution to the constrained mean-variance problem with inequality bounds on every weight. Rather than calling a general quadratic solver it walks the efficient frontier from the maximum-return corner, computing each turning point where an asset enters or leaves the free set. That yields the whole frontier rather than one point on it, and it terminates — which quadratic solvers on near-singular covariance matrices frequently do not.

## When to Use

Use it when you need the full efficient frontier, when weight bounds are binding, or when a general optimiser is returning unstable or non-converging weights on an ill-conditioned covariance. If you only want one portfolio and the covariance is well behaved, `portfolio_optimization` is the shorter path. If the covariance itself is the problem, prefer `hrp`, which never inverts it. CLA still needs expected returns, so it inherits their estimation error.

## Mathematical Foundations

### MVO Objective

$$\min_w\;\frac{1}{2}w^T\Sigma w-\lambda\mu^T w$$

### Budget Constraint

$$\mathbf{1}^T w=1$$

## Usage Examples

### Rust

#### Prepare covariance for CLA

```rust
use nalgebra::DMatrix;
use openquant::cla::covariance;

let returns = DMatrix::from_row_slice(3, 2, &[0.01, 0.02, -0.01, 0.01, 0.015, 0.03]);
let sigma = covariance(&returns);
```

## API Reference

### Python API

- `cla.allocate_cla`

### Rust API

- `CLA`
- `covariance`
- `ReturnsEstimation`

## Risk Notes and Caveats

- CLA behavior depends on weight bounds and return estimates.
- Use robust covariance estimators when sample size is small.

## Related Modules

- [`portfolio-optimization`](/modules/portfolio-optimization/)
- [`hrp`](/modules/hrp/)
- [`hcaa`](/modules/hcaa/)
- [`risk-metrics`](/modules/risk-metrics/)
