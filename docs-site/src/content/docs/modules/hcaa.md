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
rust_api:
  - "HierarchicalClusteringAssetAllocation"
  - "HcaaError"
sidebar:
  badge: Module
---

## Concept Overview

Hierarchical Clustering Asset Allocation generalises HRP's recursive bisection to risk measures other than variance. Seriation and the cluster tree are built the same way, but the split at each node weights the two sides by the chosen `allocation_metric` — cluster variance, standard deviation, Sharpe ratio, expected shortfall or conditional drawdown — so the same hierarchy can express a tail-risk budget rather than only a variance budget. Like HRP it never inverts the covariance matrix.

## When to Use

Use it in place of `hrp` when your risk budget is not variance: expected shortfall or conditional drawdown for a drawdown-controlled mandate, Sharpe when you have return views you are willing to defend. Use `hrp` when you do not, since the variance split needs no expected-return estimate at all. The clustering is only as good as the distance fed to it, so build that with `codependence` rather than raw correlation, and sanity-check the cluster count with `onc`.

## Mathematical Foundations

### Cluster Risk

$$\sigma_C^2=w_C^{\top}\Sigma_C w_C$$

where $\Sigma_C$ is the covariance sub-matrix of cluster $C$ and $w_C$ its inverse-variance weights, normalised to sum to one within the cluster.

### Recursive Bisection Split

$$\alpha=1-\frac{m_{\text{left}}}{m_{\text{left}}+m_{\text{right}}},\qquad w_{\text{left}}\mathrel{*}=\alpha,\quad w_{\text{right}}\mathrel{*}=1-\alpha$$

where $m_C$ is the risk of cluster $C$ under the chosen `allocation_metric`: cluster variance ($\sigma_C^2$), standard deviation ($\sigma_C$), expected shortfall, or conditional drawdown. Lower risk on one side means a larger $\alpha$ for that side. This generalises the HRP split, which is the `minimum_variance` case. Two branches invert the sign: `sharpe_ratio` allocates $\alpha=\mathrm{SR}_{\text{left}}/(\mathrm{SR}_{\text{left}}+\mathrm{SR}_{\text{right}})$ because higher is better there, and `equal_weighting` skips the split entirely.

## Usage Examples

### Rust

#### Fit HCAA allocator

```rust
use nalgebra::DMatrix;
use openquant::hcaa::HierarchicalClusteringAssetAllocation;

let asset_names: Vec<String> =
    ["SPY", "TLT", "GLD", "HYG"].iter().map(|s| s.to_string()).collect();
// rows = observations, cols = assets, in the same order as `asset_names`.
let prices = DMatrix::from_fn(250, 4, |i, j| 100.0 + (i as f64) * 0.05 + (j as f64) * 3.0);

// The constructor argument selects how expected returns are estimated
// ("mean" or "exponential"); it is not optional.
let mut hcaa = HierarchicalClusteringAssetAllocation::new("mean");

// allocate() fills the struct in place and returns Result<(), HcaaError>.
// It does not return the weights — read them from `hcaa.weights` afterwards.
hcaa.allocate(
    &asset_names,
    Some(&prices),      // asset_prices
    None,               // asset_returns
    None,               // covariance_matrix
    None,               // expected_asset_returns
    "minimum_variance", // allocation_metric
    0.05,               // confidence_level, used by the tail-risk metrics
    None,               // optimal_num_clusters — inferred when None
    None,               // resample_by
)?;

println!("weights: {:?}", hcaa.weights);
println!("seriation order: {:?}", hcaa.ordered_indices);
```

## API Reference

### Python API

- `hcaa.allocate_hcaa`

### Rust API

- `HierarchicalClusteringAssetAllocation`
- `HcaaError`

## Risk Notes and Caveats

- Cluster linkage choices influence allocations.
- Use with robust codependence distances when possible.

## Related Modules

- [`hrp`](/modules/hrp/)
- [`onc`](/modules/onc/)
- [`codependence`](/modules/codependence/)
- [`portfolio-optimization`](/modules/portfolio-optimization/)
- [`cla`](/modules/cla/)
