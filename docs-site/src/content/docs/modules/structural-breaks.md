---
title: "structural_breaks"
description: "Regime change and bubble diagnostics (Chow, CUSUM variants, SADF)."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-30'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "structural_breaks"
api_surface: "both"
risk_notes:
  - "SADF can be computationally expensive on long windows."
  - "Use dedicated slow/nightly test paths for heavy scenarios."
rust_api:
  - "get_chow_type_stat"
  - "get_chu_stinchcombe_white_statistics"
  - "get_sadf"
  - "SadfLags"
sidebar:
  badge: Module
---

## Subject

**Market Microstructure, Dependence and Regime Detection**

## Why This Module Exists

Regime instability can invalidate model assumptions; break detection is a core risk control.

## Mathematical Foundations

### ADF Regression

$$\Delta y_t=\alpha+\beta y_{t-1}+\sum_{i=1}^{k}\phi_i\Delta y_{t-i}+\epsilon_t$$

### SADF

$$SADF=\sup_{r_2\in[r_0,1]} ADF_0^{r_2}$$

## Usage Examples

### Rust

#### Compute SADF statistic

```rust
use openquant::structural_breaks::{get_sadf, SadfLags};

// SADF is defined on log prices.
let log_prices: Vec<f64> =
    (0..160).map(|i| (100.0 + i as f64 * 0.1 + ((i / 40) as f64) * 5.0).ln()).collect();

// (series, model, add_const, min_length, lags). `model` selects the regression
// specification — "linear", "quadratic", "sm_poly_1", "sm_poly_2", "sm_exp",
// "sm_power" — and `min_length` is the shortest window a statistic is computed on.
let sadf = get_sadf(&log_prices, "linear", true, 20, SadfLags::Fixed(1))?;

let peak = sadf.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
println!("{} SADF values, peak = {peak:.4}", sadf.len());
```

## API Reference

### Python API

- `structural_breaks.get_chow_type_stat`
- `structural_breaks.get_chu_stinchcombe_white_statistics`
- `structural_breaks.get_sadf`

### Rust API

- `get_chow_type_stat`
- `get_chu_stinchcombe_white_statistics`
- `get_sadf`
- `SadfLags`

## Implementation Notes

- SADF can be computationally expensive on long windows.
- Use dedicated slow/nightly test paths for heavy scenarios.
