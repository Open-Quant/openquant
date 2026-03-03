---
title: "codependence"
description: "Dependence metrics beyond linear correlation for feature and asset relationships."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "codependence"
risk_notes:
  - "Use with clustering and feature pruning workflows."
  - "Bin selection materially impacts MI estimates."
rust_api:
  - "distance_correlation"
  - "get_mutual_info"
  - "variation_of_information_score"
  - "angular_distance"
sidebar:
  badge: Module
---

## Subject

**Market Microstructure, Dependence and Regime Detection**

## Why This Module Exists

Financial relationships are often non-linear and regime-dependent; correlation alone is insufficient.

## Mathematical Foundations

### Mutual Information

$$I(X;Y)=\sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$$

### Variation of Information

$$VI(X,Y)=H(X)+H(Y)-2I(X;Y)$$

## Usage Examples

### Rust

#### Distance correlation between series

```rust
use openquant::codependence::distance_correlation;

let x = vec![1.0, 2.0, 3.0, 4.0];
let y = vec![1.1, 1.9, 3.2, 3.8];
let dcor = distance_correlation(&x, &y)?;
```

## API Reference

### Rust API

- `distance_correlation`
- `get_mutual_info`
- `variation_of_information_score`
- `angular_distance`

## Implementation Notes

- Use with clustering and feature pruning workflows.
- Bin selection materially impacts MI estimates.
