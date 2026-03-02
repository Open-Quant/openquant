---
title: "util::volatility"
description: "Volatility estimators used across labeling and risk workflows."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "util::volatility"
risk_notes:
  - "Choose estimator based on available fields and microstructure noise."
  - "Daily-vol lookback should be matched to event horizon."
rust_api:
  - "get_daily_vol"
  - "get_parksinson_vol"
  - "get_garman_class_vol"
  - "get_yang_zhang_vol"
sidebar:
  badge: Module
---

## Subject

**Market Microstructure, Dependence and Regime Detection**

## Why This Module Exists

Volatility is a foundational scaling target for barriers, sizing, and risk controls.

## Key Public APIs

- `get_daily_vol`
- `get_parksinson_vol`
- `get_garman_class_vol`
- `get_yang_zhang_vol`

## Mathematical Definitions

### Parkinson

$$\sigma_P^2=\frac{1}{4\ln 2}\frac{1}{n}\sum (\ln(H_t/L_t))^2$$

### Yang-Zhang

$$\sigma_{YZ}^2=\sigma_o^2+k\sigma_c^2+(1-k)\sigma_{rs}^2$$

## Implementation Examples

### Compute daily and range-based volatility

```rust
use openquant::util::volatility::{get_daily_vol, get_parksinson_vol};

let dv = get_daily_vol(&close, 100);
let pv = get_parksinson_vol(&high, &low, 20);
```

## Implementation Notes

- Choose estimator based on available fields and microstructure noise.
- Daily-vol lookback should be matched to event horizon.
