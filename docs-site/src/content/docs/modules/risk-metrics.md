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

let returns = vec![-0.02, 0.01, -0.005, 0.003, 0.004];

// These are &self methods on a unit struct, not associated functions: they need
// a receiver. `confidence_level` is the tail probability (0.05 = 95% VaR).
let metrics = RiskMetrics;
let var_95 = metrics.calculate_value_at_risk(&returns, 0.05)?;
let es_95 = metrics.calculate_expected_shortfall(&returns, 0.05)?;

println!("VaR(95%) = {var_95:.4}, ES(95%) = {es_95:.4}");
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
