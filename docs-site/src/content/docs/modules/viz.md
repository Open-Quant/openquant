---
title: "viz"
description: "Visualization payload builders for feature importance, drawdown, regime, frontier, and cluster charts."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "viz"
api_surface: "python-only"
risk_notes:
  - "Payloads are plain dicts — render with plotly, matplotlib, or pass to a frontend."
  - "prepare_feature_importance_payload sorts by importance descending and supports top_n filtering."
  - "prepare_feature_importance_comparison_payload creates side-by-side grouped bar payloads for before/after analysis."
rust_api:
  - "prepare_feature_importance_payload"
  - "prepare_drawdown_payload"
  - "prepare_regime_payload"
  - "prepare_frontier_payload"
  - "prepare_cluster_payload"
sidebar:
  badge: Module
---

## Concept Overview

The viz module produces structured chart payloads — plain Python dicts with chart type, axis data, and optional error bars or color channels. These payloads are plotting-library-agnostic: you can render them with Plotly, matplotlib, or pass them to a web frontend.

This decouples analysis from visualization: the feature_diagnostics module computes importance scores and calls viz internally to produce payloads, which you can render however you prefer. The pattern keeps the core modules free of plotting dependencies.

## When to Use

Use viz payloads when you want structured chart data from research outputs. Most diagnostic modules (feature_diagnostics, pipeline) already call viz internally and include payloads in their return dicts.

**Alternatives**: Build charts directly from DataFrames if you prefer a specific plotting library's API.

## Usage Examples

### Python

#### Build visualization payloads for research output

```python
from openquant.viz import (
    prepare_feature_importance_payload,
    prepare_drawdown_payload,
)

# Feature importance bar chart payload
payload = prepare_feature_importance_payload(
    feature_names=["momentum", "vol", "spread"],
    importance=[0.45, 0.35, 0.20],
    std=[0.05, 0.03, 0.02],
    top_n=10,
)
# {"chart": "bar", "x": [...], "y": [...], "error_y": [...]}

# Drawdown chart payload from equity curve
dd_payload = prepare_drawdown_payload(
    timestamps=["2024-01-02", "2024-01-03", "2024-01-04"],
    equity_curve=[1.0, 1.02, 0.98],
)
# {"chart": "line", "x": [...], "equity": [...], "drawdown": [...]}
```

## API Reference

### Python API

- `viz.prepare_feature_importance_payload`
- `viz.prepare_feature_importance_comparison_payload`
- `viz.prepare_drawdown_payload`
- `viz.prepare_regime_payload`
- `viz.prepare_frontier_payload`
- `viz.prepare_cluster_payload`

### Key Functions

- `prepare_feature_importance_payload`
- `prepare_drawdown_payload`
- `prepare_regime_payload`
- `prepare_frontier_payload`
- `prepare_cluster_payload`

## Implementation Notes

- Payloads are plain dicts — render with plotly, matplotlib, or pass to a frontend.
- prepare_feature_importance_payload sorts by importance descending and supports top_n filtering.
- prepare_feature_importance_comparison_payload creates side-by-side grouped bar payloads for before/after analysis.

## Related Modules

- [`feature-diagnostics`](/modules/feature-diagnostics/)
- [`pipeline`](/modules/pipeline/)
