---
title: "util::fast_ewma"
description: "Fast EWMA primitive shared across feature and volatility routines."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-31'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "util::fast_ewma"
api_surface: "both"
rust_api:
  - "ewma"
sidebar:
  badge: Module
---

## Concept Overview

One function: a single-pass exponentially weighted moving average with span-style decay, alpha = 2/(window+1), corrected by the accumulated weight so that early values are not dragged toward the seed. It mirrors `mlfinlab.util.fast_ewma` exactly, which is the point — it is what makes daily volatility and every EWMA-derived feature numerically comparable between this library and a pandas reference implementation.

## When to Use

Use it instead of writing a rolling loop, so that everything downstream — `util::volatility`'s daily vol, the microstructure feature panel, dynamic threshold series for `filters` — shares one decay convention. Remember that `window` is a span rather than a hard lookback: the weight on a point w bars back is (1-alpha)^w, not zero, so the estimate remembers further than the number suggests. Size the span longer than the horizon you are trying to smooth over.

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

## Risk Notes and Caveats

- Window length controls responsiveness vs smoothness.
- Prefer this helper over ad-hoc loops for consistency.

## Related Modules

- [`util-volatility`](/modules/util-volatility/)
- [`filters`](/modules/filters/)
- [`microstructural-features`](/modules/microstructural-features/)
- [`labeling`](/modules/labeling/)
