---
title: "Module Reference Index"
description: "Full OpenQuant module documentation index with AFML-aligned summaries."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-31'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 1
---

This is the canonical index of every OpenQuant module: one page each, with
purpose, APIs, formulas, examples, and implementation notes. It lists the same
39 modules twice over — by subject, and by the language surface they are
reachable through — because those are the two questions readers arrive with.
For the AFML chapter each module implements, see
[By AFML Chapter](/module-reference/by-afml-chapter/).

## By subject

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

## By language surface

### Rust core

- [`backtest_statistics`](/modules/backtest-statistics/) — Performance diagnostics for strategy returns and position trajectories.
- [`backtesting_engine`](/modules/backtesting-engine/) — Backtesting core with walk-forward, purged CV, and combinatorial purged CV (CPCV) workflows.
- [`bet_sizing`](/modules/bet-sizing/) — Transforms model confidence and constraints into executable position sizes.
- [`cla`](/modules/cla/) — Critical Line Algorithm implementation for constrained mean-variance optimization.
- [`codependence`](/modules/codependence/) — Dependence metrics beyond linear correlation for feature and asset relationships.
- [`combinatorial_optimization`](/modules/combinatorial-optimization/) — AFML Chapter 21 integer-encoded optimization and trajectory state-space tooling with exact baselines and solver adapters.
- [`cross_validation`](/modules/cross-validation/) — Purged cross-validation utilities designed for label overlap and leakage control.
- [`data`](/modules/data/) — OHLCV loading, cleaning, calendar alignment, and data quality reporting.
- [`data_structures`](/modules/data-structures/) — Constructs standard/time/run/imbalance bars from trade streams.
- [`ef3m`](/modules/ef3m/) — Moment-based mixture fitting utilities for two-normal components.
- [`ensemble_methods`](/modules/ensemble-methods/) — Bias/variance diagnostics and practical bagging-vs-boosting ensemble utilities.
- [`etf_trick`](/modules/etf-trick/) — Synthetic ETF and futures roll utilities for realistic PnL path construction.
- [`feature_importance`](/modules/feature-importance/) — Feature ranking methods: MDI, MDA, and single-feature importance with PCA diagnostics.
- [`filters`](/modules/filters/) — CUSUM and z-score event filters for event-driven sampling.
- [`fingerprint`](/modules/fingerprint/) — Model fingerprinting for linear, non-linear, and pairwise feature effects.
- [`fracdiff`](/modules/fracdiff/) — Fractional differentiation to improve stationarity while retaining memory.
- [`hcaa`](/modules/hcaa/) — Hierarchical Clustering Asset Allocation variant with cluster-level constraints.
- [`hpc_parallel`](/modules/hpc-parallel/) — AFML Chapter 20 atom/molecule execution utilities with serial/threaded modes and partition diagnostics.
- [`hrp`](/modules/hrp/) — Hierarchical Risk Parity allocation with recursive bisection.
- [`hyperparameter_tuning`](/modules/hyperparameter-tuning/) — Leakage-aware grid/randomized hyper-parameter search with purged CV and weighted scoring.
- [`labeling`](/modules/labeling/) — Triple-barrier event labeling and metadata generation.
- [`microstructural_features`](/modules/microstructural-features/) — Price-impact, spread, entropy, and flow toxicity estimators.
- [`onc`](/modules/onc/) — Optimal Number of Clusters utilities for clustering stability and allocation workflows.
- [`pipeline`](/modules/pipeline/) — End-to-end AFML research pipeline: events → signals → portfolio → risk → backtest with leakage checks.
- [`portfolio_optimization`](/modules/portfolio-optimization/) — Mean-variance and constrained allocation methods with ergonomic APIs.
- [`risk_metrics`](/modules/risk-metrics/) — Portfolio and return-distribution risk measures for downside control.
- [`sample_weights`](/modules/sample-weights/) — Sample weighting utilities for overlapping event structure.
- [`sampling`](/modules/sampling/) — Indicator matrix and sequential bootstrap tooling.
- [`sb_bagging`](/modules/sb-bagging/) — Sequentially bootstrapped bagging classifiers/regressors.
- [`strategy_risk`](/modules/strategy-risk/) — AFML Chapter 15 strategy-viability diagnostics based on precision, payout asymmetry, and bet frequency.
- [`streaming_hpc`](/modules/streaming-hpc/) — AFML Chapter 22 streaming analytics utilities for low-latency early-warning metrics with bounded-memory incremental state.
- [`structural_breaks`](/modules/structural-breaks/) — Regime change and bubble diagnostics (Chow, CUSUM variants, SADF).
- [`synthetic_backtesting`](/modules/synthetic-backtesting/) — Synthetic-data OTR backtesting with O-U calibration, PT/SL mesh search, and stability diagnostics.
- [`util::fast_ewma`](/modules/util-fast-ewma/) — Fast EWMA primitive shared across feature and volatility routines.
- [`util::volatility`](/modules/util-volatility/) — Volatility estimators used across labeling and risk workflows.

### Python namespaces

- [`adapters`](/modules/adapters/) — `to_polars_signal_frame`, `to_polars_event_frame`, `to_polars_backtest_frame`, `to_polars_weights_frame`, `to_polars_indicator_matrix`, `to_polars_frontier_frame`, `SignalStreamBuffer`, `to_pandas`
- [`backtest_stats`](/modules/backtest-statistics/) — `sharpe_ratio`, `information_ratio`, `probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`, `minimum_track_record_length`, `timing_of_flattening_and_flips`, `average_holding_period`, `bets_concentration`, `all_bets_concentration`, `drawdown_and_time_under_water`
- [`bars`](/modules/data-structures/) — `build_time_bars`, `build_tick_bars`, `build_volume_bars`, `build_dollar_bars`, `build_run_bars`, `build_imbalance_bars`
- [`bet_sizing`](/modules/bet-sizing/) — `get_signal`, `discrete_signal`, `bet_size`, `bet_size_sigmoid`, `bet_size_power`, `inv_price`, `inv_price_sigmoid`, `inv_price_power`, `get_w`, `get_w_sigmoid`, `get_w_power`, `get_target_pos`, `get_target_pos_sigmoid`, `get_target_pos_power`, `limit_price`, `limit_price_sigmoid`, `limit_price_power`, `avg_active_signals`, `bet_size_dynamic`, `cdf_mixture`, `single_bet_size_mixed`, `get_concurrent_sides`, `bet_size_budget`, `bet_size_probability`, `mp_avg_active_signals`, `bet_size_reserve`, `bet_size_reserve_with_fit`, `bet_size_reserve_full`
- [`cla`](/modules/cla/) — `allocate_cla`
- [`codependence`](/modules/codependence/) — `angular_distance`, `absolute_angular_distance`, `squared_angular_distance`, `distance_correlation`, `get_optimal_number_of_bins`, `get_mutual_info`, `variation_of_information_score`
- [`data`](/modules/data/) — `load_ohlcv`, `clean_ohlcv`, `align_calendar`, `data_quality_report`, `clean_ohlcv_df`, `quality_report_df`, `align_calendar_df`
- [`ef3m`](/modules/ef3m/) — `centered_moment`, `raw_moment`, `most_likely_parameters`, `fit_m2n`
- [`ensemble`](/modules/ensemble-methods/) — `bias_variance_noise`, `bootstrap_sample_indices`, `sequential_bootstrap_sample_indices`, `aggregate_regression_mean`, `aggregate_classification_vote`, `aggregate_classification_probability_mean`, `average_pairwise_prediction_correlation`, `bagging_ensemble_variance`, `recommend_bagging_vs_boosting`
- [`fast_ewma`](/modules/util-fast-ewma/) — `ewma`
- [`feature_diagnostics`](/modules/feature-diagnostics/) — `mdi_importance`, `mda_importance`, `sfi_importance`, `orthogonalize_features_pca`, `substitution_effect_report`
- [`filters`](/modules/filters/) — `cusum_filter_indices`, `cusum_filter_timestamps`, `z_score_filter_indices`, `z_score_filter_timestamps`
- [`fracdiff`](/modules/fracdiff/) — `get_weights`, `get_weights_ffd`, `frac_diff`, `frac_diff_ffd`
- [`hcaa`](/modules/hcaa/) — `allocate_hcaa`
- [`hrp`](/modules/hrp/) — `allocate_hrp`
- [`labeling`](/modules/labeling/) — `triple_barrier_labels`, `triple_barrier_events`, `meta_labels`, `add_vertical_barrier`, `get_events`, `get_bins`, `drop_labels`
- [`microstructural`](/modules/microstructural-features/) — `get_roll_measure`, `get_roll_impact`, `get_corwin_schultz_estimator`, `get_bekker_parkinson_vol`, `get_bar_based_kyle_lambda`, `get_bar_based_amihud_lambda`, `get_bar_based_hasbrouck_lambda`, `get_trades_based_kyle_lambda`, `get_trades_based_amihud_lambda`, `get_trades_based_hasbrouck_lambda`, `vwap`, `get_avg_tick_size`, `get_vpin`, `get_bvc_buy_volume`, `encode_tick_rule_array`, `quantile_mapping`, `sigma_mapping`, `encode_array`, `get_shannon_entropy`, `get_lempel_ziv_entropy`, `get_plug_in_entropy`, `get_konto_entropy`
- [`onc`](/modules/onc/) — `get_onc_clusters`
- [`pipeline`](/modules/pipeline/) — `run_mid_frequency_pipeline`, `run_mid_frequency_pipeline_frames`, `summarize_pipeline`
- [`portfolio`](/modules/portfolio-optimization/) — `allocate_inverse_variance`, `allocate_min_vol`, `allocate_max_sharpe`, `allocate_efficient_risk`, `allocate_with_solution`, `allocate_from_inputs`
- [`research`](/modules/research/) — `make_synthetic_futures_dataset`, `run_flywheel_iteration`, `ResearchDataset`
- [`risk`](/modules/risk-metrics/) — `calculate_value_at_risk`, `calculate_expected_shortfall`, `calculate_conditional_drawdown_risk`, `calculate_variance`, `calculate_value_at_risk_from_matrix`, `calculate_expected_shortfall_from_matrix`, `calculate_conditional_drawdown_risk_from_matrix`
- [`sample_weights`](/modules/sample-weights/) — `get_weights_by_return`, `get_weights_by_time_decay`
- [`sampling`](/modules/sampling/) — `get_ind_matrix`, `seq_bootstrap`, `get_ind_mat_average_uniqueness`, `get_ind_mat_label_uniqueness`, `bootstrap_loop_run`, `get_av_uniqueness_from_triple_barrier`, `num_concurrent_events`
- [`sb_bagging`](/modules/sb-bagging/) — `fit_predict_sb_classifier`, `fit_predict_sb_regressor`
- [`strategy_risk`](/modules/strategy-risk/) — `sharpe_symmetric`, `implied_precision_symmetric`, `implied_frequency_symmetric`, `sharpe_asymmetric`, `implied_precision_asymmetric`, `implied_frequency_asymmetric`, `estimate_strategy_failure_probability`
- [`streaming_hpc`](/modules/streaming-hpc/) — `run_streaming_pipeline`, `generate_synthetic_flash_crash_stream`
- [`structural_breaks`](/modules/structural-breaks/) — `get_chow_type_stat`, `get_chu_stinchcombe_white_statistics`, `get_sadf`
- [`synthetic_bt`](/modules/synthetic-backtesting/) — `calibrate_ou_params`, `generate_ou_paths`, `evaluate_rule_on_paths`, `detect_no_stable_optimum`, `run_synthetic_otr_workflow`, `search_optimal_trading_rule`
- [`viz`](/modules/viz/) — `prepare_feature_importance_payload`, `prepare_feature_importance_comparison_payload`, `prepare_drawdown_payload`, `prepare_regime_payload`, `prepare_frontier_payload`, `prepare_cluster_payload`
- [`volatility`](/modules/util-volatility/) — `get_daily_vol`, `get_parkinson_vol`, `get_garman_class_vol`, `get_yang_zhang_vol`
