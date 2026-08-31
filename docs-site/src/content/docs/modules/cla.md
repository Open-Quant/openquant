---
title: "cla"
description: "Critical Line Algorithm implementation for constrained mean-variance optimization."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-30'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "cla"
api_surface: "both"
risk_notes:
  - "CLA behavior depends on weight bounds and return estimates."
  - "Use robust covariance estimators when sample size is small."
rust_api:
  - "CLA"
  - "covariance"
  - "ReturnsEstimation"
sidebar:
  badge: Module
---

## Subject

**Portfolio Construction and Risk**

## Why This Module Exists

CLA solves constrained Markowitz problems efficiently with active-set style line updates.

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

## Implementation Notes

- CLA behavior depends on weight bounds and return estimates.
- Use robust covariance estimators when sample size is small.
