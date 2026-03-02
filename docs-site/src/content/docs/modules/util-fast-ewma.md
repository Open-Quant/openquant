---
title: "util::fast_ewma"
description: "Fast EWMA primitive shared across feature and volatility routines."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "util::fast_ewma"
risk_notes:
  - "Window length controls responsiveness vs smoothness."
  - "Prefer this helper over ad-hoc loops for consistency."
rust_api:
  - "ewma"
sidebar:
  badge: Module
---

## Subject

**Market Microstructure, Dependence and Regime Detection**

## Why This Module Exists

Provides performant smoothing for repeated rolling computations.

## Key Public APIs

- `ewma`

## Mathematical Definitions

### EWMA

$$m_t=\alpha x_t + (1-\alpha)m_{t-1}$$

### Smoothing

$$\alpha=\frac{2}{w+1}$$

## Implementation Examples

### Compute EWMA vector

```rust
use openquant::util::fast_ewma::ewma;

let x = vec![1.0, 2.0, 3.0, 4.0];
let y = ewma(&x, 3);
```

## Implementation Notes

- Window length controls responsiveness vs smoothness.
- Prefer this helper over ad-hoc loops for consistency.
