---
name: feature-discovery
description: Build candidate features, run feature screening and diagnostics, and prepare accepted feature sets for downstream flywheel experiments.
---

# Feature Discovery

Use this skill when iterating on feature engineering and deciding which candidates are safe/useful for experiments.

## What this skill does

1. Builds/loads feature candidate matrices.
2. Applies quality and collinearity screening.
3. Runs importance diagnostics (MDI/MDA/SFI) where needed.
4. Produces a selected feature list and rejection rationale.

## Core Python APIs

- `openquant.feature_diagnostics.feature_screen_report(...)`
- `openquant.feature_diagnostics.mdi_importance(...)`
- `openquant.feature_diagnostics.mda_importance(...)`
- `openquant.feature_diagnostics.sfi_importance(...)`
- `openquant.feature_diagnostics.substitution_effect_report(...)`

## Minimal workflow snippet

```python
import openquant
import polars as pl

screen = openquant.feature_diagnostics.feature_screen_report(
    X=feature_df,
    min_coverage=0.95,
    max_corr=0.92,
)
selected = screen["selected_features"]
reasons = screen["rejection_reasons"]
```

## Notebook target

- `notebooks/python/07_feature_engineering_discovery_loop.ipynb`

## Validation

```bash
uv run --python .venv/bin/python pytest python/tests/test_feature_diagnostics_module.py python/tests/test_research_grid_and_screening.py -q
```
