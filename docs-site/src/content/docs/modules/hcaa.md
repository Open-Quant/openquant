---
title: "hcaa"
description: "Hierarchical Clustering Asset Allocation variant with cluster-level constraints."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-30'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "hcaa"
api_surface: "both"
risk_notes:
  - "Cluster linkage choices influence allocations."
  - "Use with robust codependence distances when possible."
rust_api:
  - "HierarchicalClusteringAssetAllocation"
  - "HcaaError"
sidebar:
  badge: Module
---

## Subject

**Portfolio Construction and Risk**

## Why This Module Exists

Allocates capital by hierarchy to reduce concentration and covariance-estimation fragility.

## Mathematical Foundations

### Cluster Variance

$$\sigma_C^2=w_C^T\Sigma_C w_C$$

### Recursive Split

$$w_{left},w_{right}\propto\frac{1}{\sigma_{left}^2},\frac{1}{\sigma_{right}^2}$$

## Usage Examples

### Rust

#### Fit HCAA allocator

```rust
use openquant::hcaa::HierarchicalClusteringAssetAllocation;

let mut hcaa = HierarchicalClusteringAssetAllocation::new();
let w = hcaa.allocate(&prices)?;
```

## API Reference

### Python API

- `hcaa.allocate_hcaa`

### Rust API

- `HierarchicalClusteringAssetAllocation`
- `HcaaError`

## Implementation Notes

- Cluster linkage choices influence allocations.
- Use with robust codependence distances when possible.
