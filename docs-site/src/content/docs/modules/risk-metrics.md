---
title: "risk_metrics"
description: "Portfolio and return-distribution risk measures for downside control."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-30'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "risk_metrics"
api_surface: "both"
risk_notes:
  - "Non-parametric estimates need enough tail observations."
  - "Use matrix variants for multi-asset return panels."
rust_api:
  - "RiskMetrics::calculate_value_at_risk"
  - "RiskMetrics::calculate_expected_shortfall"
  - "RiskMetrics::calculate_conditional_drawdown_risk"
  - "RiskMetrics::calculate_variance"
sidebar:
  badge: Module
---

## Subject

**Portfolio Construction and Risk**

## Why This Module Exists

Risk budgets and guardrails require coherent downside metrics beyond variance.

## Mathematical Foundations

### VaR

$$VaR_\alpha = -Q_\alpha(R)$$

### Expected Shortfall

$$ES_\alpha = -E[R \mid R \le Q_\alpha(R)]$$

## Usage Examples

### Rust

#### Compute VaR and ES

```rust
use openquant::risk_metrics::RiskMetrics;

let r = vec![-0.02, 0.01, -0.005, 0.003, 0.004];
let var95 = RiskMetrics::calculate_value_at_risk(&r, 0.05)?;
let es95 = RiskMetrics::calculate_expected_shortfall(&r, 0.05)?;
```

## API Reference

### Python API

- `risk.calculate_value_at_risk`
- `risk.calculate_expected_shortfall`
- `risk.calculate_conditional_drawdown_risk`
- `risk.calculate_variance`
- `risk.calculate_value_at_risk_from_matrix`
- `risk.calculate_expected_shortfall_from_matrix`
- `risk.calculate_conditional_drawdown_risk_from_matrix`

### Rust API

- `RiskMetrics::calculate_value_at_risk`
- `RiskMetrics::calculate_expected_shortfall`
- `RiskMetrics::calculate_conditional_drawdown_risk`
- `RiskMetrics::calculate_variance`

## Implementation Notes

- Non-parametric estimates need enough tail observations.
- Use matrix variants for multi-asset return panels.
