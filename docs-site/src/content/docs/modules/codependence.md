---
title: "codependence"
description: "Dependence metrics beyond linear correlation for feature and asset relationships."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-31'
audience:
  - quant-dev
  - platform-engineering
module: "codependence"
api_surface: "both"
rust_api:
  - "distance_correlation"
  - "get_mutual_info"
  - "variation_of_information_score"
  - "angular_distance"
sidebar:
  badge: Module
---

## Concept Overview

Dependence measures that survive non-linearity, which Pearson correlation does not. Distance correlation is zero only under genuine independence. Mutual information and variation of information are information-theoretic and need a binning choice, which `get_optimal_number_of_bins` supplies. The angular distances turn a correlation into a proper metric — sqrt(2(1-rho)) and its absolute and squared variants — which is what hierarchical clustering needs in order to be well posed at all.

## When to Use

Use it upstream of any clustering or feature-pruning step: `hrp`, `hcaa` and `onc` all consume a distance matrix, and feeding them raw correlation silently assumes the relationship is linear. Use distance correlation when you suspect a non-monotone relationship, and variation of information when you want a true metric on discrete variables. Bin selection materially changes mutual-information estimates, so fix it explicitly and record it alongside the result.

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

### Python API

- `codependence.angular_distance`
- `codependence.absolute_angular_distance`
- `codependence.squared_angular_distance`
- `codependence.distance_correlation`
- `codependence.get_optimal_number_of_bins`
- `codependence.get_mutual_info`
- `codependence.variation_of_information_score`

### Rust API

- `distance_correlation`
- `get_mutual_info`
- `variation_of_information_score`
- `angular_distance`

## Risk Notes and Caveats

- Use with clustering and feature pruning workflows.
- Bin selection materially impacts MI estimates.

## Related Modules

- [`hrp`](/modules/hrp/)
- [`hcaa`](/modules/hcaa/)
- [`onc`](/modules/onc/)
- [`feature-importance`](/modules/feature-importance/)
- [`microstructural-features`](/modules/microstructural-features/)
