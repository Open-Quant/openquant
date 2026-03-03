---
title: "feature_diagnostics"
description: "Feature importance diagnostics: MDI, MDA, SFI, PCA orthogonalization, and substitution-effect analysis."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "feature_diagnostics"
api_surface: "python-only"
afml_chapters:
  - 8
risk_notes:
  - "MDI is biased toward high-cardinality features; cross-check with MDA."
  - "MDA uses purged k-fold CV internally to prevent leakage in importance estimates."
  - "SFI trains single-feature models — useful for detecting features that are only useful in combination."
  - "substitution_effect_report combines MDA + correlation + PCA in one call."
rust_api:
  - "mdi_importance"
  - "mda_importance"
  - "sfi_importance"
  - "orthogonalize_features_pca"
  - "substitution_effect_report"
sidebar:
  badge: Module
---

## Concept Overview

Feature importance is not a single number — AFML Chapter 8 argues you need multiple methods because each has different failure modes. **MDI** (Mean Decrease Impurity) measures how much each feature contributes to splits in an ensemble, but it's biased toward features with more unique values. **MDA** (Mean Decrease Accuracy) measures the score drop when a feature is permuted, which is unbiased but noisy. **SFI** (Single Feature Importance) trains one model per feature, revealing which features carry signal alone vs. only in combination.

The critical insight is **substitution effects**: when two features are correlated, MDI and MDA split importance between them arbitrarily. A feature that appears unimportant might be essential — its importance was just absorbed by its correlated partner. The `substitution_effect_report` detects this by comparing individual MDA scores against grouped-permutation scores, and by re-running MDA on PCA-orthogonalized features where substitution effects vanish.

All importance methods use purged k-fold cross-validation internally, preventing information leakage from overlapping labels.

## When to Use

Run feature diagnostics after training an initial model and before finalizing the feature set. Use the results to prune unstable features, detect redundancy, and validate that your model relies on economically meaningful signals.

**Prerequisites**: Feature matrix X, label vector y, and optionally event end indices for purged CV.

**Alternatives**: Rust-side `feature_importance` module for MDI/MDA on Rust models; this Python module adds SFI, PCA orthogonalization, and substitution-effect analysis.

## Mathematical Foundations

### MDI (Mean Decrease Impurity)

$$I_j^{MDI}=\frac{1}{B}\sum_{b=1}^B \frac{|\beta_j^{(b)}|}{\sum_k|\beta_k^{(b)}|}$$

### MDA (Mean Decrease Accuracy)

$$I_j^{MDA}=\frac{S_{base}-S_{perm(j)}}{1-S_{perm(j)}}$$

## Key Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `n_estimators` | `int` | Number of bootstrap rounds for MDI | 32 |
| `n_splits` | `int` | Number of purged k-fold splits for MDA/SFI | 5 |
| `pct_embargo` | `float` | Embargo fraction for purged CV | 0.01 |
| `scoring` | `str` | Scoring metric: 'neg_log_loss', 'accuracy', or 'f1' | 'neg_log_loss' |
| `corr_threshold` | `float` | Minimum |correlation| to flag a substitution-risk pair | 0.9 |
| `variance_threshold` | `float` | PCA cumulative variance to retain for orthogonalization | 0.95 |

## Usage Examples

### Python

#### Run all three importance methods and compare

```python
from openquant.feature_diagnostics import (
    mdi_importance, mda_importance, sfi_importance
)

X = [[0.1, 0.5, 0.3], [0.2, 0.4, 0.1], ...]  # n_samples × n_features
y = [1.0, 0.0, 1.0, ...]  # binary labels
names = ["momentum", "volatility", "spread"]

mdi = mdi_importance(X, y, feature_names=names, n_estimators=32)
mda = mda_importance(X, y, feature_names=names, n_splits=5, pct_embargo=0.01)
sfi = sfi_importance(X, y, feature_names=names, n_splits=5)

# Each returns: {"table": pl.DataFrame, "viz_payload": {...}, ...}
print(mdi["table"])  # feature | mean | std | stderr
print(mda["table"])
```

#### Detect substitution effects between correlated features

```python
from openquant.feature_diagnostics import substitution_effect_report

report = substitution_effect_report(
    X, y,
    feature_names=names,
    corr_threshold=0.7,   # flag pairs with |corr| > 0.7
    orthogonalize=True,   # also run MDA on PCA-orthogonalized features
)

# Correlated pairs with dilution risk
print(report["pairs"])
# feature_a | feature_b | corr | dilution_ratio | flag_substitution_risk

# Before/after orthogonalization comparison
print(report["orthogonalized"]["max_abs_corr_before"])   # e.g., 0.92
print(report["orthogonalized"]["max_abs_corr_after"])     # e.g., 0.03
```

## Common Pitfalls

- Relying on a single importance method — always cross-check MDI, MDA, and SFI for consistent rankings.
- Ignoring substitution effects: if two features are correlated, both may appear unimportant individually but one is essential.
- Not using event_end_indices with overlapping labels — without purging, importance estimates are biased by leakage.

## API Reference

### Python API

- `feature_diagnostics.mdi_importance`
- `feature_diagnostics.mda_importance`
- `feature_diagnostics.sfi_importance`
- `feature_diagnostics.orthogonalize_features_pca`
- `feature_diagnostics.substitution_effect_report`

### Key Functions

- `mdi_importance`
- `mda_importance`
- `sfi_importance`
- `orthogonalize_features_pca`
- `substitution_effect_report`

## Implementation Notes

- MDI is biased toward high-cardinality features; cross-check with MDA.
- MDA uses purged k-fold CV internally to prevent leakage in importance estimates.
- SFI trains single-feature models — useful for detecting features that are only useful in combination.
- substitution_effect_report combines MDA + correlation + PCA in one call.

## Related Modules

- [`feature-importance`](/modules/feature-importance/)
- [`cross-validation`](/modules/cross-validation/)
