---
title: Methodology and Leakage Controls
description: Required anti-leakage and methodology controls for OpenQuant research and evaluation.
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
risk_notes:
  - Naive random CV with overlapping labels is not acceptable.
  - Purging and embargo parameters must be versioned with results.
sidebar:
  order: 1
---

## Non-Negotiable Controls

1. **Time-aware splitting** for event-labeled data.
2. **Purging and embargo** when label windows overlap.
3. **Trial-count disclosure** for strategy selection diagnostics.
4. **Explicit assumptions** for costs, liquidity, and universe definition.

## Typical Failure Modes

- Random CV inflates apparent performance.
- Hidden data snooping from repeated tuning without trial tracking.
- Reporting only point metrics without distributional context.

## OpenQuant Modules Supporting These Controls

- `cross_validation`
- `backtesting_engine`
- `feature_importance`
- `strategy_risk`
