---
title: "util::fast_ewma"
description: "Fast EWMA primitive shared across feature and volatility routines."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-30'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "util::fast_ewma"
api_surface: "both"
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

## Mathematical Foundations

### EWMA

$$m_t=\alpha x_t + (1-\alpha)m_{t-1}$$

### Smoothing

$$\alpha=\frac{2}{w+1}$$

## Usage Examples

### Rust

#### Compute EWMA vector

```rust
use openquant::util::fast_ewma::ewma;

let x = vec![1.0, 2.0, 3.0, 4.0];
let y = ewma(&x, 3);
```

## API Reference

### Python API

- `fast_ewma.ewma`

### Rust API

- `ewma`

## Implementation Notes

- Window length controls responsiveness vs smoothness.
- Prefer this helper over ad-hoc loops for consistency.
