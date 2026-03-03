---
title: By AFML Chapter
description: Map AFML concepts to concrete OpenQuant modules and workflows.
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 2
---

## Chapter-to-Module Mapping

### Chapter 2: Financial Data Structures

Information-driven bars replace fixed-time sampling with activity-based aggregation, producing returns that are closer to IID normal. CUSUM and z-score filters extract structurally meaningful events from the bar stream.

- [`data_structures`](/modules/data-structures/) — bar construction (dollar, volume, tick, imbalance, run bars). **Entry point for raw trade data.**
- [`filters`](/modules/filters/) — CUSUM and z-score event filters
- [`etf_trick`](/modules/etf-trick/) — synthetic ETF and futures roll utilities

### Chapter 3: Labeling

Triple-barrier labeling converts filtered events into ML labels with controlled profit-taking, stop-loss, and holding-period barriers. Meta-labeling separates direction from sizing.

- [`labeling`](/modules/labeling/) — triple-barrier and meta-labeling. **Entry point for label generation.**
- [`bet_sizing`](/modules/bet-sizing/) — probability-to-position-size conversion

### Chapter 4: Sample Weights and Uniqueness

Overlapping labels violate IID assumptions. Sequential bootstrap and uniqueness weighting correct for this by measuring and accounting for label overlap.

- [`sampling`](/modules/sampling/) — indicator matrix and sequential bootstrap. **Entry point for overlap-aware sampling.**
- [`sample_weights`](/modules/sample-weights/) — uniqueness and time-decay weighting
- [`sb_bagging`](/modules/sb-bagging/) — sequentially bootstrapped bagging ensembles

### Chapter 5: Fractional Differentiation

Fractional differencing finds the minimum transformation order that achieves stationarity while preserving predictive long-memory in price series.

- [`fracdiff`](/modules/fracdiff/) — FFD and standard fractional differencing. **Entry point.**

### Chapter 6: Ensemble Methods

Bias-variance decomposition and bagging diagnostics determine whether bagging or boosting improves ensemble quality under financial label structure.

- [`ensemble_methods`](/modules/ensemble-methods/) — bias/variance diagnostics, aggregation, bagging-vs-boosting recommendation

### Chapter 7: Cross Validation in Finance

Standard k-fold CV leaks information through overlapping labels. Purged k-fold with embargo removes this leakage source.

- [`cross_validation`](/modules/cross-validation/) — purged k-fold CV with embargo. **Entry point.**
- [`backtesting_engine`](/modules/backtesting-engine/) — walk-forward and CPCV validation

### Chapter 8: Feature Importance

Multiple importance methods (MDI, MDA, SFI) are needed to detect substitution effects and unstable features before deploying models.

- [`feature_importance`](/modules/feature-importance/) — MDI, MDA, single-feature importance (Rust)
- [`fingerprint`](/modules/fingerprint/) — model fingerprinting for partial and pairwise effects

### Chapter 9: Hyper-parameter Tuning

Tuning must use purged CV to avoid leakage-inflated scores. Randomized search is preferred for large parameter spaces.

- [`hyperparameter_tuning`](/modules/hyperparameter-tuning/) — grid/randomized search with purged CV scoring

### Chapters 10-12: Position Sizing and Robust Backtesting

Backtesting is a scenario sanity check, not a performance estimator. CPCV provides path distributions instead of point estimates.

- [`bet_sizing`](/modules/bet-sizing/) — dynamic and reserve sizing for execution
- [`backtesting_engine`](/modules/backtesting-engine/) — walk-forward, purged CV, and CPCV. **Entry point for validation.**

### Chapter 13: Synthetic Backtesting

Selecting trading rules on a single historical path overfits. Synthetic path ensembles from calibrated O-U processes test rule robustness.

- [`synthetic_backtesting`](/modules/synthetic-backtesting/) — O-U calibration, PT/SL mesh search, stability diagnostics

### Chapters 14-15: Diagnostics and Strategy Risk

Strategy risk (probability of failing a Sharpe target) is distinct from portfolio risk (VaR, ES, drawdown).

- [`backtest_statistics`](/modules/backtest-statistics/) — Sharpe, deflated Sharpe, drawdown, holding period
- [`risk_metrics`](/modules/risk-metrics/) — VaR, Expected Shortfall, Conditional Drawdown Risk
- [`strategy_risk`](/modules/strategy-risk/) — strategy failure probability diagnostics

### Chapter 16: Portfolio Construction

Hierarchical methods (HRP, HCAA) avoid covariance inversion fragility. CLA solves constrained mean-variance problems exactly.

- [`hrp`](/modules/hrp/) — Hierarchical Risk Parity
- [`hcaa`](/modules/hcaa/) — Hierarchical Clustering Asset Allocation
- [`onc`](/modules/onc/) — Optimal Number of Clusters
- [`portfolio_optimization`](/modules/portfolio-optimization/) — mean-variance allocators (min-vol, max-Sharpe, efficient risk)
- [`cla`](/modules/cla/) — Critical Line Algorithm

### Chapters 17-19: Microstructure, Dependence, and Regime Detection

Microstructure features capture liquidity and order-flow dynamics invisible in OHLC bars. Structural break detection flags regime changes that invalidate model assumptions.

- [`structural_breaks`](/modules/structural-breaks/) — Chow, CUSUM variants, SADF bubble detection
- [`microstructural_features`](/modules/microstructural-features/) — Kyle/Amihud impact, VPIN, entropy
- [`codependence`](/modules/codependence/) — distance correlation, mutual information, variation of information

### Chapters 20-22: HPC and Advanced

Atom/molecule parallelism scales independent computations. Streaming analytics maintain bounded-memory indicators for real-time early warning.

- [`hpc_parallel`](/modules/hpc-parallel/) — partition and dispatch utilities
- [`combinatorial_optimization`](/modules/combinatorial-optimization/) — integer optimization with exact baselines
- [`streaming_hpc`](/modules/streaming-hpc/) — VPIN/HHI streaming with bounded-memory state

See the full per-module detail in the [Module Reference Index](/modules/).
