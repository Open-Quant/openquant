---
title: "fracdiff"
description: "Fractional differentiation to improve stationarity while retaining memory."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "fracdiff"
api_surface: "both"
afml_chapters:
  - 5
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

## Concept Overview

Financial time series like prices are non-stationary — their statistical properties drift over time. Standard integer differencing (d=1, i.e., returns) makes the series stationary but destroys long-range memory that carries predictive signal.

Fractional differentiation (AFML Chapter 5) generalizes differencing to real-valued orders 0 < d < 1. A fractional difference applies an infinite series of weights to past observations, where the weights decay polynomially. At d=0 you have the raw price (full memory, non-stationary). At d=1 you have returns (stationary, no memory). The goal is to find the minimum d that passes stationarity tests (e.g., ADF) while preserving as much memory as possible.

The **fixed-width window (FFD)** variant truncates the weight series once weights fall below a threshold, making computation practical for long series. This is the recommended approach for production use.

## When to Use

Apply fractional differentiation to price or spread series *before* feature engineering. It replaces raw returns as the base transformation when you need stationarity without discarding mean-reversion or trend memory.

**Prerequisites**: A price series (close prices or mid-prices). Optionally, an ADF test loop to find the optimal d.

**Alternatives**: Standard returns (d=1) if stationarity is sufficient and memory isn't needed. Log prices if your downstream model handles non-stationarity.

## Mathematical Foundations

### FFD Weights

$$w_k = -w_{k-1}\frac{d-k+1}{k}$$

### Fractional Difference

$$y_t=\sum_{k=0}^{\infty}w_k x_{t-k}$$

## Key Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `d` | `f64` | Fractional differencing order; 0 = raw prices, 1 = returns | — |
| `threshold` | `f64` | Minimum absolute weight for FFD truncation; smaller = longer memory window, more compute | 1e-4 |

## Usage Examples

### Python

#### Fractionally differentiate a price series

```python
from openquant._core import fracdiff

prices = [100.0, 100.2, 100.1, 100.4, 100.6, 100.3, 100.8]

# Fixed-window fractional differentiation (d=0.4, threshold=1e-4)
stationary = fracdiff.frac_diff_ffd(prices, 0.4, 1e-4)

# Inspect the FFD weights to understand memory retention
weights = fracdiff.get_weights_ffd(0.4, 1e-4, len(prices))
```

### Rust

#### Compute fixed-width fracdiff

```rust
use openquant::fracdiff::frac_diff_ffd;

let series = vec![100.0, 100.2, 100.1, 100.4, 100.6];
let out = frac_diff_ffd(&series, 0.4, 1e-4);
```

## Common Pitfalls

- Using d=1 by default (standard returns) when the series has exploitable long-memory — run a d-search with ADF first.
- Setting threshold too large, which truncates weights aggressively and makes FFD behave like integer differencing.
- Applying fracdiff to already-differenced data — check whether your input is prices or returns.
- Forgetting that the first few observations are NaN/unreliable due to insufficient weight history — trim them before feeding into ML.

## API Reference

### Python API

- `fracdiff.frac_diff_ffd`
- `fracdiff.frac_diff`
- `fracdiff.get_weights_ffd`
- `fracdiff.get_weights`

### Rust API

- `get_weights`
- `get_weights_ffd`
- `frac_diff`
- `frac_diff_ffd`

## Implementation Notes

- Tune d using stationarity tests and information retention.
- Threshold governs truncation error vs compute cost.

## Related Modules

- [`data-structures`](/modules/data-structures/)
- [`filters`](/modules/filters/)
