---
title: "onc"
description: "Optimal Number of Clusters utilities for clustering stability and allocation workflows."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-31'
audience:
  - quant-dev
  - platform-engineering
module: "onc"
api_surface: "both"
rust_api:
  - "get_onc_clusters"
  - "check_improve_clusters"
  - "OncResult"
sidebar:
  badge: Module
---

## Concept Overview

Optimal Number of Clusters: runs k-means over a correlation matrix for a range of k, scores each partition by the mean-to-standard-deviation ratio of its silhouette scores, then re-clusters only the clusters that scored badly and keeps the result if it improves. Base k-means is unstable in both k and initialisation, so ONC restarts it `repeat` times and keeps the best — the point is a defensible cluster count, not a fast one.

## When to Use

Use it before any hierarchical allocation to decide how many clusters the universe actually supports, instead of hard-coding a number; its answer feeds `hcaa`'s `optimal_num_clusters` directly. Use it also to test whether a claimed grouping — sectors, factors, strategy families — survives contact with the data. Clean the correlation matrix first: on an unstable universe ONC will happily find structure in noise and report a confident k for it.

## Mathematical Foundations

### Cluster Score

$$J(k)=\text{intra}(k)-\text{inter}(k)$$

### Selection

$$k^*=\arg\min_k J(k)$$

## Usage Examples

### Rust

#### Infer cluster structure

```rust
use nalgebra::DMatrix;
use openquant::onc::get_onc_clusters;

// ONC consumes a *correlation* matrix, not raw prices — build one from your
// codependence measure of choice first.
let corr = DMatrix::from_row_slice(
    4,
    4,
    &[
        1.00, 0.85, 0.10, 0.05, //
        0.85, 1.00, 0.12, 0.08, //
        0.10, 0.12, 1.00, 0.78, //
        0.05, 0.08, 0.78, 1.00,
    ],
);

// `repeat` is the number of k-means restarts used to stabilise the partition.
let out = get_onc_clusters(&corr, 20)?;
println!("{} clusters", out.clusters.len());
println!("silhouette scores: {:?}", out.silhouette_scores);
```

## API Reference

### Python API

- `onc.get_onc_clusters`

### Rust API

- `get_onc_clusters`
- `check_improve_clusters`
- `OncResult`

## Risk Notes and Caveats

- Run with repeated seeds/restarts for robust k selection.
- Use correlation cleaning before clustering unstable universes.

## Related Modules

- [`hcaa`](/modules/hcaa/)
- [`hrp`](/modules/hrp/)
- [`codependence`](/modules/codependence/)
- [`portfolio-optimization`](/modules/portfolio-optimization/)
