---
title: "onc"
description: "Optimal Number of Clusters utilities for clustering stability and allocation workflows."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "onc"
risk_notes:
  - "Run with repeated seeds/restarts for robust k selection."
  - "Use correlation cleaning before clustering unstable universes."
rust_api:
  - "get_onc_clusters"
  - "check_improve_clusters"
  - "OncResult"
sidebar:
  badge: Module
---

## Subject

**Portfolio Construction and Risk**

## Why This Module Exists

Cluster count selection is a key source of model risk in hierarchical portfolio methods.

## Mathematical Foundations

### Cluster Score

$$J(k)=\text{intra}(k)-\text{inter}(k)$$

### Selection

$$k^*=\arg\min_k J(k)$$

## Usage Examples

### Rust

#### Infer cluster structure

```rust
use openquant::onc::get_onc_clusters;

let out = get_onc_clusters(&corr, 20)?;
println!("{}", out.clusters.len());
```

## API Reference

### Rust API

- `get_onc_clusters`
- `check_improve_clusters`
- `OncResult`

## Implementation Notes

- Run with repeated seeds/restarts for robust k selection.
- Use correlation cleaning before clustering unstable universes.
