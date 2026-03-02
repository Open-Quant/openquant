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

- [`backtest_statistics`](/modules/backtest-statistics/) - Performance diagnostics for strategy returns and position trajectories.
- [`backtesting_engine`](/modules/backtesting-engine/) - Backtesting core with walk-forward, purged CV, and combinatorial purged CV (CPCV) workflows.
- [`bet_sizing`](/modules/bet-sizing/) - Transforms model confidence and constraints into executable position sizes.
- [`cla`](/modules/cla/) - Critical Line Algorithm implementation for constrained mean-variance optimization.
- [`codependence`](/modules/codependence/) - Dependence metrics beyond linear correlation for feature and asset relationships.
- [`combinatorial_optimization`](/modules/combinatorial-optimization/) - AFML Chapter 21 integer-encoded optimization and trajectory state-space tooling with exact baselines and solver adapters.
- [`cross_validation`](/modules/cross-validation/) - Purged cross-validation utilities designed for label overlap and leakage control.
- [`data_structures`](/modules/data-structures/) - Constructs standard/time/run/imbalance bars from trade streams.
- [`ef3m`](/modules/ef3m/) - Moment-based mixture fitting utilities for two-normal components.
- [`ensemble_methods`](/modules/ensemble-methods/) - Bias/variance diagnostics and practical bagging-vs-boosting ensemble utilities.
- [`etf_trick`](/modules/etf-trick/) - Synthetic ETF and futures roll utilities for realistic PnL path construction.
- [`feature_importance`](/modules/feature-importance/) - Feature ranking methods: MDI, MDA, and single-feature importance with PCA diagnostics.
- [`filters`](/modules/filters/) - CUSUM and z-score event filters for event-driven sampling.
- [`fingerprint`](/modules/fingerprint/) - Model fingerprinting for linear, non-linear, and pairwise feature effects.
- [`fracdiff`](/modules/fracdiff/) - Fractional differentiation to improve stationarity while retaining memory.
- [`hcaa`](/modules/hcaa/) - Hierarchical Clustering Asset Allocation variant with cluster-level constraints.
- [`hpc_parallel`](/modules/hpc-parallel/) - AFML Chapter 20 atom/molecule execution utilities with serial/threaded modes and partition diagnostics.
- [`hrp`](/modules/hrp/) - Hierarchical Risk Parity allocation with recursive bisection.
- [`hyperparameter_tuning`](/modules/hyperparameter-tuning/) - Leakage-aware grid/randomized hyper-parameter search with purged CV and weighted scoring.
- [`labeling`](/modules/labeling/) - Triple-barrier event labeling and metadata generation.
- [`microstructural_features`](/modules/microstructural-features/) - Price-impact, spread, entropy, and flow toxicity estimators.
- [`onc`](/modules/onc/) - Optimal Number of Clusters utilities for clustering stability and allocation workflows.
- [`portfolio_optimization`](/modules/portfolio-optimization/) - Mean-variance and constrained allocation methods with ergonomic APIs.
- [`risk_metrics`](/modules/risk-metrics/) - Portfolio and return-distribution risk measures for downside control.
- [`sample_weights`](/modules/sample-weights/) - Sample weighting utilities for overlapping event structure.
- [`sampling`](/modules/sampling/) - Indicator matrix and sequential bootstrap tooling.
- [`sb_bagging`](/modules/sb-bagging/) - Sequentially bootstrapped bagging classifiers/regressors.
- [`strategy_risk`](/modules/strategy-risk/) - AFML Chapter 15 strategy-viability diagnostics based on precision, payout asymmetry, and bet frequency.
- [`streaming_hpc`](/modules/streaming-hpc/) - AFML Chapter 22 streaming analytics utilities for low-latency early-warning metrics with bounded-memory incremental state.
- [`structural_breaks`](/modules/structural-breaks/) - Regime change and bubble diagnostics (Chow, CUSUM variants, SADF).
- [`synthetic_backtesting`](/modules/synthetic-backtesting/) - Synthetic-data OTR backtesting with O-U calibration, PT/SL mesh search, and stability diagnostics.
- [`util::fast_ewma`](/modules/util-fast-ewma/) - Fast EWMA primitive shared across feature and volatility routines.
- [`util::volatility`](/modules/util-volatility/) - Volatility estimators used across labeling and risk workflows.
