---
title: "hrp"
description: "Hierarchical Risk Parity allocation with recursive bisection."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "hrp"
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

## Key Public APIs

- `HierarchicalRiskParity`
- `HrpDendrogram`

## Mathematical Definitions

### IVP Weight

$$w_i\propto\frac{1}{\sigma_i^2}$$

### Bisection Split

$$\alpha=1-\frac{\sigma_{left}^2}{\sigma_{left}^2+\sigma_{right}^2}$$

## Implementation Examples

### Allocate with HRP

```rust
use openquant::hrp::HierarchicalRiskParity;

let mut hrp = HierarchicalRiskParity::new();
let weights = hrp.allocate(&prices)?;
```

## Implementation Notes

- HRP is often more robust under unstable covariance estimates.
- Ensure input asset order tracks produced dendrogram order.
