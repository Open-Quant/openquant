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

## Rust modules (core)

- [`data_structures`](/modules/data-structures/) — bar construction (dollar, volume, tick, imbalance, run, time)
- [`filters`](/modules/filters/) — CUSUM and z-score event filters
- [`labeling`](/modules/labeling/) — triple-barrier and meta-labeling
- [`fracdiff`](/modules/fracdiff/) — fractional differentiation (FFD)
- [`sampling`](/modules/sampling/) — indicator matrix and sequential bootstrap
- [`sample_weights`](/modules/sample-weights/) — uniqueness and time-decay weighting
- [`cross_validation`](/modules/cross-validation/) — purged k-fold with embargo
- [`feature_importance`](/modules/feature-importance/) — MDI, MDA, single-feature importance
- [`backtesting_engine`](/modules/backtesting-engine/) — walk-forward, purged CV, CPCV
- [`backtest_statistics`](/modules/backtest-statistics/) — Sharpe, drawdown, holding period
- [`risk_metrics`](/modules/risk-metrics/) — VaR, Expected Shortfall, CDaR
- [`strategy_risk`](/modules/strategy-risk/) — strategy failure probability
- [`bet_sizing`](/modules/bet-sizing/) — probability-to-position conversion
- [`hrp`](/modules/hrp/), [`hcaa`](/modules/hcaa/), [`cla`](/modules/cla/), [`portfolio_optimization`](/modules/portfolio-optimization/) — portfolio construction
- [`structural_breaks`](/modules/structural-breaks/) — SADF, Chow, CUSUM variants
- [`microstructural_features`](/modules/microstructural-features/) — impact, spread, VPIN, entropy
- [`hpc_parallel`](/modules/hpc-parallel/), [`streaming_hpc`](/modules/streaming-hpc/), [`combinatorial_optimization`](/modules/combinatorial-optimization/) — HPC utilities

## Python namespaces

- [`data`](/modules/data/) — `load_ohlcv`, `clean_ohlcv`, `align_calendar`, `data_quality_report`
- [`bars`](/modules/data-structures/) — `build_time_bars`, `build_tick_bars`, `build_volume_bars`, `build_dollar_bars`
- [`feature_diagnostics`](/modules/feature-diagnostics/) — `mdi_importance`, `mda_importance`, `sfi_importance`, `orthogonalize_features_pca`, `substitution_effect_report`
- [`pipeline`](/modules/pipeline/) — `run_mid_frequency_pipeline`, `run_mid_frequency_pipeline_frames`, `summarize_pipeline`
- [`research`](/modules/research/) — `make_synthetic_futures_dataset`, `run_flywheel_iteration`
- [`adapters`](/modules/adapters/) — `to_polars_signal_frame`, `to_polars_event_frame`, `to_polars_backtest_frame`, `SignalStreamBuffer`
- [`viz`](/modules/viz/) — `prepare_feature_importance_payload`, `prepare_drawdown_payload`, `prepare_regime_payload`, `prepare_frontier_payload`
- `_core.labeling` — `triple_barrier_labels`, `meta_labels` (Rust bindings)
- `_core.filters` — `cusum_filter_indices`, `cusum_filter_timestamps` (Rust bindings)
- `_core.fracdiff` — `frac_diff_ffd`, `get_weights_ffd` (Rust bindings)
- `_core.sampling` — `seq_bootstrap`, `get_ind_matrix` (Rust bindings)
- `_core.sample_weights` — `get_weights_by_return`, `get_weights_by_time_decay` (Rust bindings)

Full module docs: [All Modules](/modules/)
