---
title: "structural_breaks"
description: "Regime change and bubble diagnostics (Chow, CUSUM variants, SADF)."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "structural_breaks"
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

## Key Public APIs

- `get_chow_type_stat`
- `get_chu_stinchcombe_white_statistics`
- `get_sadf`
- `SadfLags`

## Mathematical Definitions

### ADF Regression

$$\Delta y_t=\alpha+\beta y_{t-1}+\sum_{i=1}^{k}\phi_i\Delta y_{t-i}+\epsilon_t$$

### SADF

$$SADF=\sup_{r_2\in[r_0,1]} ADF_0^{r_2}$$

## Implementation Examples

### Compute SADF statistic

```rust
use openquant::structural_breaks::{get_sadf, SadfLags};

let y = vec![100.0, 100.2, 100.4, 100.1, 99.8, 100.0];
let sadf = get_sadf(&y, 3, SadfLags::Fixed(1))?;
```

## Implementation Notes

- SADF can be computationally expensive on long windows.
- Use dedicated slow/nightly test paths for heavy scenarios.
