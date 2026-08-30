---
title: "util::volatility"
description: "Volatility estimators used across labeling and risk workflows."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-30'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "util::volatility"
api_surface: "both"
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

## Mathematical Foundations

### Parkinson

$$\sigma_P^2=\frac{1}{4\ln 2}\frac{1}{n}\sum (\ln(H_t/L_t))^2$$

### Yang-Zhang

$$\sigma_{YZ}^2=\sigma_o^2+k\sigma_c^2+(1-k)\sigma_{rs}^2$$

## Usage Examples

### Rust

#### Compute daily and range-based volatility

```rust
use chrono::{Duration, NaiveDateTime};
use openquant::util::volatility::{get_daily_vol, get_parksinson_vol};

let t0 = NaiveDateTime::parse_from_str("2024-01-02 00:00:00", "%Y-%m-%d %H:%M:%S")?;
let close: Vec<(NaiveDateTime, f64)> = (0..300)
    .map(|i| (t0 + Duration::days(i), 100.0 + (i as f64 * 0.07).sin() * 2.0))
    .collect();
let high: Vec<f64> = close.iter().map(|(_, p)| p + 0.4).collect();
let low: Vec<f64> = close.iter().map(|(_, p)| p - 0.4).collect();

// Close-to-close EWMA vol on a timestamped series; `lookback` is the EWMA span.
let daily = get_daily_vol(&close, 100);
// Parkinson uses the high/low range, so it needs no timestamps — `window` bars.
let parkinson = get_parksinson_vol(&high, &low, 20);

println!("daily vol tail = {:?}", daily.last());
println!("parkinson vol tail = {:?}", parkinson.last());
```

## API Reference

### Python API

- `volatility.get_daily_vol`
- `volatility.get_parksinson_vol`
- `volatility.get_garman_class_vol`
- `volatility.get_yang_zhang_vol`

### Rust API

- `get_daily_vol`
- `get_parksinson_vol`
- `get_garman_class_vol`
- `get_yang_zhang_vol`

## Implementation Notes

- Choose estimator based on available fields and microstructure noise.
- Daily-vol lookback should be matched to event horizon.
