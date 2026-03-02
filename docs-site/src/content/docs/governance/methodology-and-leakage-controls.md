---
title: Methodology and Leakage Controls
description: Baseline controls for leakage-safe model research and evaluation.
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
risk_notes:
  - Do not use naive random CV for overlapping labels.
  - Record purging and embargo parameters with each backtest run.
sidebar:
  order: 1
---

- Use purged and embargoed validation for event-labeled datasets.
- Keep backtest safeguards and trial metadata with every report.
- Separate strategy-failure diagnostics from portfolio-volatility diagnostics.
