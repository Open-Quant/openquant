---
title: API Surfaces
description: High-level Rust and Python API surface map for core workflows.
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 1
---

## Rust modules (selected core)

- `data_structures`
- `filters`
- `labeling`
- `cross_validation`
- `backtesting_engine`
- `feature_importance`
- `backtest_statistics`
- `risk_metrics`
- `strategy_risk`

## Python namespaces (selected core)

- `data`: `load_ohlcv`, `clean_ohlcv`, `align_calendar`, `data_quality_report`
- `bars`: `build_time_bars`, `build_tick_bars`, `build_volume_bars`, `build_dollar_bars`
- `feature_diagnostics`: `mdi_importance`, `mda_importance`, `sfi_importance`
- `pipeline`: `run_mid_frequency_pipeline`, `summarize_pipeline`

Full module docs: [Module Pages](/module/)
