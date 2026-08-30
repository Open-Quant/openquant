---
title: "onc"
description: "Optimal Number of Clusters utilities for clustering stability and allocation workflows."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-30'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "onc"
api_surface: "both"
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

### Python API

- `onc.get_onc_clusters`

### Rust API

- `get_onc_clusters`
- `check_improve_clusters`
- `OncResult`

## Implementation Notes

- Run with repeated seeds/restarts for robust k selection.
- Use correlation cleaning before clustering unstable universes.
