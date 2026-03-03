---
title: "hcaa"
description: "Hierarchical Clustering Asset Allocation variant with cluster-level constraints."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "hcaa"
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

### Rust API

- `HierarchicalClusteringAssetAllocation`
- `HcaaError`

## Implementation Notes

- Cluster linkage choices influence allocations.
- Use with robust codependence distances when possible.
