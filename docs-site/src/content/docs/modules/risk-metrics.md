---
title: "risk_metrics"
description: "Portfolio and return-distribution risk measures for downside control."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "risk_metrics"
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

### Rust API

- `RiskMetrics::calculate_value_at_risk`
- `RiskMetrics::calculate_expected_shortfall`
- `RiskMetrics::calculate_conditional_drawdown_risk`
- `RiskMetrics::calculate_variance`

## Implementation Notes

- Non-parametric estimates need enough tail observations.
- Use matrix variants for multi-asset return panels.
