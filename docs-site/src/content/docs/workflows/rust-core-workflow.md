---
title: Rust Core Workflow
description: Core Rust workflow from event bars to risk diagnostics.
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
afml_chapter:
  - CHAPTER 2
  - CHAPTER 3
  - CHAPTER 7
  - CHAPTER 14
sidebar:
  order: 1
---

## Workflow stages

1. Build event-driven bars (`data_structures`, `filters`).
2. Label outcomes (`labeling`, `bet_sizing`).
3. Validate leakage-safe (`cross_validation`, `backtesting_engine`).
4. Evaluate strategy and risk (`backtest_statistics`, `risk_metrics`, `strategy_risk`).

Module deep links:
- [Module Pages](/module/)
- [Backtesting Engine](/module/backtesting-engine/)
- [Risk Metrics](/module/risk-metrics/)
