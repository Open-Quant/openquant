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
use openquant::hrp::HierarchicalRiskParity;

let mut hrp = HierarchicalRiskParity::new();
let weights = hrp.allocate(&prices)?;
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
