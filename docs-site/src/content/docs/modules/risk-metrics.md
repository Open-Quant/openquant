---
title: "risk_metrics"
description: "Portfolio and return-distribution risk measures for downside control."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-31'
audience:
  - quant-dev
  - platform-engineering
module: "risk_metrics"
api_surface: "both"
rust_api:
  - "RiskMetrics::calculate_value_at_risk"
  - "RiskMetrics::calculate_expected_shortfall"
  - "RiskMetrics::calculate_conditional_drawdown_risk"
  - "RiskMetrics::calculate_variance"
sidebar:
  badge: Module
---

## Concept Overview

Downside risk measures over a return series or a return panel: value at risk (the quantile at the given confidence level), expected shortfall (the mean loss beyond it), conditional drawdown at risk, and portfolio variance from a covariance matrix and a weight vector. Expected shortfall and CDaR are subadditive where VaR is not, which is why a risk budget built on VaR alone can be gamed by splitting one position across two sleeves.

## When to Use

Use it for portfolio-level guardrails and risk budgets, and as the input when `hcaa` should allocate on tail risk rather than on variance. Prefer expected shortfall to VaR whenever the number will be summed across books. These are non-parametric estimates, so they need enough tail observations to mean anything: at 95% confidence a 200-observation sample rests on ten points. All of them are `&self` methods on a unit struct, and the `_from_matrix` variants take return panels.

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

## Risk Notes and Caveats

- Non-parametric estimates need enough tail observations.
- Use matrix variants for multi-asset return panels.

## Related Modules

- [`hcaa`](/modules/hcaa/)
- [`portfolio-optimization`](/modules/portfolio-optimization/)
- [`backtest-statistics`](/modules/backtest-statistics/)
- [`strategy-risk`](/modules/strategy-risk/)
