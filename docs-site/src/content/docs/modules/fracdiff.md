---
title: "fracdiff"
description: "Fractional differentiation to improve stationarity while retaining memory."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "fracdiff"
risk_notes:
  - "Tune d using stationarity tests and information retention."
  - "Threshold governs truncation error vs compute cost."
rust_api:
  - "get_weights"
  - "get_weights_ffd"
  - "frac_diff"
  - "frac_diff_ffd"
sidebar:
  badge: Module
---

## Subject

**Market Microstructure, Dependence and Regime Detection**

## Why This Module Exists

Balances stationarity and predictive memory better than integer differencing.

## Key Public APIs

- `get_weights`
- `get_weights_ffd`
- `frac_diff`
- `frac_diff_ffd`

## Mathematical Definitions

### FFD Weights

$$w_k = -w_{k-1}\frac{d-k+1}{k}$$

### Fractional Difference

$$y_t=\sum_{k=0}^{\infty}w_k x_{t-k}$$

## Implementation Examples

### Compute fixed-width fracdiff

```rust
use openquant::fracdiff::frac_diff_ffd;

let series = vec![100.0, 100.2, 100.1, 100.4, 100.6];
let out = frac_diff_ffd(&series, 0.4, 1e-4);
```

## Implementation Notes

- Tune d using stationarity tests and information retention.
- Threshold governs truncation error vs compute cost.
