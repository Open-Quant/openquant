---
title: "Module Reference Index"
description: "Full OpenQuant module documentation index with AFML-aligned summaries."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 1
---

This index contains one page per OpenQuant module with purpose, APIs, formulas, examples, and implementation notes.

### Data Ingestion and Quality

- [`adapters`](/modules/adapters/) — Polars DataFrame adapters for signals, events, weights, backtest curves, and streaming buffers.
- [`data`](/modules/data/) — OHLCV loading, cleaning, calendar alignment, and data quality reporting.

### Event-Driven Data and Labeling

- [`data_structures`](/modules/data-structures/) — Constructs standard/time/run/imbalance bars from trade streams.
- [`filters`](/modules/filters/) — CUSUM and z-score event filters for event-driven sampling.
- [`labeling`](/modules/labeling/) — Triple-barrier event labeling and metadata generation.
- [`sample_weights`](/modules/sample-weights/) — Sample weighting utilities for overlapping event structure.

### Market Microstructure, Dependence and Regime Detection

- [`codependence`](/modules/codependence/) — Dependence metrics beyond linear correlation for feature and asset relationships.
- [`fracdiff`](/modules/fracdiff/) — Fractional differentiation to improve stationarity while retaining memory.
- [`microstructural_features`](/modules/microstructural-features/) — Price-impact, spread, entropy, and flow toxicity estimators.
- [`structural_breaks`](/modules/structural-breaks/) — Regime change and bubble diagnostics (Chow, CUSUM variants, SADF).
- [`util::fast_ewma`](/modules/util-fast-ewma/) — Fast EWMA primitive shared across feature and volatility routines.
- [`util::volatility`](/modules/util-volatility/) — Volatility estimators used across labeling and risk workflows.

### Portfolio Construction and Risk

- [`backtest_statistics`](/modules/backtest-statistics/) — Performance diagnostics for strategy returns and position trajectories.
- [`cla`](/modules/cla/) — Critical Line Algorithm implementation for constrained mean-variance optimization.
- [`hcaa`](/modules/hcaa/) — Hierarchical Clustering Asset Allocation variant with cluster-level constraints.
- [`hrp`](/modules/hrp/) — Hierarchical Risk Parity allocation with recursive bisection.
- [`onc`](/modules/onc/) — Optimal Number of Clusters utilities for clustering stability and allocation workflows.
- [`portfolio_optimization`](/modules/portfolio-optimization/) — Mean-variance and constrained allocation methods with ergonomic APIs.
- [`risk_metrics`](/modules/risk-metrics/) — Portfolio and return-distribution risk measures for downside control.
- [`strategy_risk`](/modules/strategy-risk/) — AFML Chapter 15 strategy-viability diagnostics based on precision, payout asymmetry, and bet frequency.

### Position Sizing and Trade Construction

- [`bet_sizing`](/modules/bet-sizing/) — Transforms model confidence and constraints into executable position sizes.
- [`etf_trick`](/modules/etf-trick/) — Synthetic ETF and futures roll utilities for realistic PnL path construction.

### Research Workflows

- [`pipeline`](/modules/pipeline/) — End-to-end AFML research pipeline: events → signals → portfolio → risk → backtest with leakage checks.
- [`research`](/modules/research/) — Synthetic dataset generation and flywheel research iteration with cost modeling and promotion gates.
- [`viz`](/modules/viz/) — Visualization payload builders for feature importance, drawdown, regime, frontier, and cluster charts.

### Sampling, Validation and ML Diagnostics

- [`backtesting_engine`](/modules/backtesting-engine/) — Backtesting core with walk-forward, purged CV, and combinatorial purged CV (CPCV) workflows.
- [`cross_validation`](/modules/cross-validation/) — Purged cross-validation utilities designed for label overlap and leakage control.
- [`ef3m`](/modules/ef3m/) — Moment-based mixture fitting utilities for two-normal components.
- [`ensemble_methods`](/modules/ensemble-methods/) — Bias/variance diagnostics and practical bagging-vs-boosting ensemble utilities.
- [`feature_diagnostics`](/modules/feature-diagnostics/) — Feature importance diagnostics: MDI, MDA, SFI, PCA orthogonalization, and substitution-effect analysis.
- [`feature_importance`](/modules/feature-importance/) — Feature ranking methods: MDI, MDA, and single-feature importance with PCA diagnostics.
- [`fingerprint`](/modules/fingerprint/) — Model fingerprinting for linear, non-linear, and pairwise feature effects.
- [`hyperparameter_tuning`](/modules/hyperparameter-tuning/) — Leakage-aware grid/randomized hyper-parameter search with purged CV and weighted scoring.
- [`sampling`](/modules/sampling/) — Indicator matrix and sequential bootstrap tooling.
- [`sb_bagging`](/modules/sb-bagging/) — Sequentially bootstrapped bagging classifiers/regressors.
- [`synthetic_backtesting`](/modules/synthetic-backtesting/) — Synthetic-data OTR backtesting with O-U calibration, PT/SL mesh search, and stability diagnostics.

### Scaling, HPC and Infrastructure

- [`combinatorial_optimization`](/modules/combinatorial-optimization/) — AFML Chapter 21 integer-encoded optimization and trajectory state-space tooling with exact baselines and solver adapters.
- [`hpc_parallel`](/modules/hpc-parallel/) — AFML Chapter 20 atom/molecule execution utilities with serial/threaded modes and partition diagnostics.
- [`streaming_hpc`](/modules/streaming-hpc/) — AFML Chapter 22 streaming analytics utilities for low-latency early-warning metrics with bounded-memory incremental state.
