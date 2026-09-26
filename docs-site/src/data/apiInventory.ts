// Generated file. Do not edit manually.
export const apiInventory = {
  "generatedAt": "generated-by-scripts/generate_api_inventory.py",
  "python": {
    "adapters": [
      "append",
      "clear",
      "frame",
      "to_pandas",
      "to_polars_backtest_frame",
      "to_polars_event_frame",
      "to_polars_frontier_frame",
      "to_polars_indicator_matrix",
      "to_polars_signal_frame",
      "to_polars_weights_frame"
    ],
    "backtesting_engine": [
      "assemble_cpcv_paths",
      "cpcv_path_count",
      "run_cpcv"
    ],
    "bars": [
      "bar_diagnostics",
      "build_dollar_bars",
      "build_tick_bars",
      "build_time_bars",
      "build_volume_bars"
    ],
    "cross_validation": [
      "count_train_test_overlaps",
      "cpcv_paths",
      "cpcv_splits",
      "naive_kfold_splits",
      "purged_kfold_splits",
      "split_with_diagnostics"
    ],
    "data": [
      "align_calendar",
      "clean_ohlcv",
      "data_quality_report",
      "dataset_hash",
      "default_cache_dir",
      "fetch",
      "fetch_symbol",
      "load_ohlcv",
      "quality_failures",
      "record_dataset_hash",
      "symbols"
    ],
    "evaluation": [
      "config_hash",
      "deflated_sharpe_ratio",
      "expected_max_sharpe",
      "meta_label_metrics",
      "minimum_track_record_length",
      "n_trials",
      "path",
      "probabilistic_sharpe_ratio",
      "record",
      "reload",
      "return_moments",
      "sharpe_std",
      "strategy_failure_probability",
      "trials"
    ],
    "feature_diagnostics": [
      "feature_screen_report",
      "mda_importance",
      "mdi_importance",
      "orthogonalize_features_pca",
      "sfi_importance",
      "substitution_effect_report"
    ],
    "feature_importance": [
      "mda_from_probabilities",
      "mean_decrease_accuracy",
      "mean_decrease_impurity",
      "sfi_from_probabilities",
      "single_feature_importance"
    ],
    "hyperparameter_tuning": [
      "classification_score",
      "expand_param_grid",
      "purged_search",
      "sample_param_sets"
    ],
    "pipeline": [
      "run_mid_frequency_pipeline",
      "run_mid_frequency_pipeline_frames",
      "summarize_pipeline"
    ],
    "research": [
      "make_synthetic_futures_dataset",
      "research_run_manifest",
      "run_flywheel_grid",
      "run_flywheel_iteration"
    ],
    "viz": [
      "prepare_cluster_payload",
      "prepare_drawdown_payload",
      "prepare_feature_importance_comparison_payload",
      "prepare_feature_importance_payload",
      "prepare_frontier_payload",
      "prepare_regime_payload"
    ]
  },
  "rust": {
    "openquant::backtest_statistics": [
      "all_bets_concentration",
      "average_holding_period",
      "bets_concentration",
      "deflated_sharpe_ratio",
      "drawdown_and_time_under_water",
      "information_ratio",
      "minimum_track_record_length",
      "probabilistic_sharpe_ratio",
      "sharpe_ratio",
      "timing_of_flattening_and_flips"
    ],
    "openquant::backtesting_engine": [
      "cpcv_path_count",
      "validate"
    ],
    "openquant::bet_sizing": [
      "avg_active_signals",
      "bet_size",
      "bet_size_budget",
      "bet_size_dynamic",
      "bet_size_power",
      "bet_size_probability",
      "bet_size_reserve",
      "bet_size_reserve_full",
      "bet_size_reserve_with_fit",
      "bet_size_sigmoid",
      "cdf_mixture",
      "confirm_and_cast_to_df",
      "discrete_signal",
      "get_concurrent_sides",
      "get_signal",
      "get_target_pos",
      "get_target_pos_power",
      "get_target_pos_sigmoid",
      "get_w",
      "get_w_power",
      "get_w_sigmoid",
      "inv_price",
      "inv_price_power",
      "inv_price_sigmoid",
      "limit_price",
      "limit_price_power",
      "limit_price_sigmoid",
      "mp_avg_active_signals",
      "single_bet_size_mixed"
    ],
    "openquant::cla": [
      "_initialise",
      "_purge_excess",
      "_purge_num_err",
      "allocate",
      "calculate_exponential_historical_returns",
      "calculate_mean_historical_returns",
      "calculate_returns",
      "covariance",
      "new"
    ],
    "openquant::codependence": [
      "absolute_angular_distance",
      "angular_distance",
      "distance_correlation",
      "get_mutual_info",
      "get_optimal_number_of_bins",
      "squared_angular_distance",
      "variation_of_information_score"
    ],
    "openquant::combinatorial_optimization": [
      "compare_exact_and_adapter",
      "decision_space_size",
      "enumerate_trading_paths",
      "evaluate_trading_path",
      "horizon",
      "solve_exact",
      "solve_trading_trajectory_exact",
      "solve_with_adapter",
      "validate"
    ],
    "openquant::cross_validation": [
      "count_train_test_overlaps",
      "cpcv_paths",
      "cpcv_splits",
      "ml_get_train_times",
      "naive_kfold_splits",
      "new",
      "split",
      "split_with_diagnostics"
    ],
    "openquant::data_processing": [
      "align_calendar_columns",
      "align_calendar_df",
      "align_calendar_rows",
      "clean_ohlcv_columns",
      "clean_ohlcv_df",
      "clean_ohlcv_rows",
      "quality_report",
      "quality_report_columns",
      "quality_report_df"
    ],
    "openquant::data_structures": [
      "imbalance_bars",
      "run_bars",
      "standard_bars",
      "time_bars"
    ],
    "openquant::dynamic_allocation": [
      "all_weights",
      "dynamic_optimal_portfolio",
      "new",
      "partition_count",
      "pigeonhole_partitions",
      "trajectory_sharpe_ratio",
      "transaction_costs",
      "weight_count"
    ],
    "openquant::ef3m": [
      "centered_moment",
      "fit",
      "get_moments",
      "iter_4",
      "iter_5",
      "most_likely_parameters",
      "mp_fit",
      "new",
      "raw_moment",
      "single_fit_loop",
      "with_defaults"
    ],
    "openquant::ensemble_methods": [
      "aggregate_classification_probability_mean",
      "aggregate_classification_vote",
      "aggregate_regression_mean",
      "average_pairwise_prediction_correlation",
      "bagging_ensemble_variance",
      "bias_variance_noise",
      "bootstrap_sample_indices",
      "recommend_bagging_vs_boosting",
      "sequential_bootstrap_sample_indices"
    ],
    "openquant::etf_trick": [
      "from_csv",
      "from_tables",
      "get_etf_series",
      "get_futures_roll_series",
      "reset"
    ],
    "openquant::feature_importance": [
      "feature_pca_analysis",
      "get_orthogonal_features",
      "mean_decrease_impurity",
      "plot_feature_importance"
    ],
    "openquant::filters": [
      "cusum_filter_indices",
      "cusum_filter_timestamps",
      "z_score_filter_indices",
      "z_score_filter_timestamps"
    ],
    "openquant::fingerprint": [
      "get_effects",
      "new",
      "plot_effects"
    ],
    "openquant::fracdiff": [
      "frac_diff",
      "frac_diff_ffd",
      "get_weights",
      "get_weights_ffd"
    ],
    "openquant::hcaa": [
      "allocate",
      "new",
      "with_distance"
    ],
    "openquant::hpc_parallel": [
      "is_empty",
      "is_finished",
      "len",
      "partition_atoms",
      "wait"
    ],
    "openquant::hrp": [
      "allocate",
      "new",
      "plot_clusters",
      "with_distance"
    ],
    "openquant::hyperparameter_tuning": [
      "as_bool",
      "as_f64",
      "as_i64",
      "classification_score",
      "expand_param_grid",
      "sample_param_sets"
    ],
    "openquant::labeling": [
      "add_vertical_barrier",
      "drop_labels",
      "get_bins",
      "get_events",
      "meta_labels",
      "triple_barrier_events",
      "triple_barrier_labels"
    ],
    "openquant::microstructural_features": [
      "encode_array",
      "encode_tick_rule_array",
      "get_avg_tick_size",
      "get_bar_based_amihud_lambda",
      "get_bar_based_hasbrouck_lambda",
      "get_bar_based_kyle_lambda",
      "get_bekker_parkinson_vol",
      "get_bvc_buy_volume",
      "get_corwin_schultz_estimator",
      "get_features_from_csv",
      "get_konto_entropy",
      "get_lempel_ziv_entropy",
      "get_plug_in_entropy",
      "get_roll_impact",
      "get_roll_measure",
      "get_shannon_entropy",
      "get_trades_based_amihud_lambda",
      "get_trades_based_hasbrouck_lambda",
      "get_trades_based_kyle_lambda",
      "get_vpin",
      "new_from_csv",
      "quantile_mapping",
      "sigma_mapping",
      "vwap"
    ],
    "openquant::onc": [
      "get_onc_clusters"
    ],
    "openquant::pipeline": [
      "run_mid_frequency_pipeline"
    ],
    "openquant::portfolio_optimization": [
      "allocate_efficient_risk",
      "allocate_efficient_risk_with",
      "allocate_from_inputs",
      "allocate_inverse_variance",
      "allocate_inverse_variance_with",
      "allocate_max_sharpe",
      "allocate_max_sharpe_with",
      "allocate_min_vol",
      "allocate_min_vol_with",
      "allocate_with_solution",
      "compute_expected_and_covariance",
      "returns_method_from_str"
    ],
    "openquant::risk_metrics": [
      "calculate_conditional_drawdown_risk",
      "calculate_conditional_drawdown_risk_from_matrix",
      "calculate_expected_shortfall",
      "calculate_expected_shortfall_from_matrix",
      "calculate_value_at_risk",
      "calculate_value_at_risk_from_matrix",
      "calculate_variance"
    ],
    "openquant::sample_weights": [
      "get_weights_by_return",
      "get_weights_by_time_decay"
    ],
    "openquant::sampling": [
      "bootstrap_loop_run",
      "get_av_uniqueness_from_triple_barrier",
      "get_ind_mat_average_uniqueness",
      "get_ind_mat_label_uniqueness",
      "get_ind_matrix",
      "num_concurrent_events",
      "seq_bootstrap"
    ],
    "openquant::sb_bagging": [
      "fit",
      "new",
      "predict",
      "predict_proba"
    ],
    "openquant::strategy_risk": [
      "estimate_strategy_failure_probability",
      "implied_frequency_asymmetric",
      "implied_frequency_symmetric",
      "implied_precision_asymmetric",
      "implied_precision_symmetric",
      "sharpe_asymmetric",
      "sharpe_symmetric"
    ],
    "openquant::streaming_hpc": [
      "completed_buckets",
      "current",
      "current_cdf",
      "generate_synthetic_flash_crash_stream",
      "new",
      "on_event",
      "run_streaming_pipeline",
      "run_streaming_pipeline_parallel",
      "total_volume",
      "update",
      "window_len"
    ],
    "openquant::structural_breaks": [
      "_get_betas",
      "_get_values_diff",
      "get_chow_type_stat",
      "get_chu_stinchcombe_white_statistics",
      "get_sadf"
    ],
    "openquant::synthetic_backtesting": [
      "calibrate_ou_params",
      "detect_no_stable_optimum",
      "evaluate_rule_on_paths",
      "generate_ou_paths",
      "run_synthetic_otr_workflow",
      "search_optimal_trading_rule"
    ],
    "openquant::util::fast_ewma": [
      "ewma"
    ],
    "openquant::util::volatility": [
      "get_daily_vol",
      "get_garman_class_vol",
      "get_parkinson_vol",
      "get_yang_zhang_vol"
    ],
    "pyopenquant::backtest_stats": [
      "register"
    ],
    "pyopenquant::backtesting_engine": [
      "register"
    ],
    "pyopenquant::bars": [
      "register"
    ],
    "pyopenquant::bet_sizing": [
      "register"
    ],
    "pyopenquant::cla": [
      "register"
    ],
    "pyopenquant::codependence": [
      "register"
    ],
    "pyopenquant::cross_validation": [
      "register"
    ],
    "pyopenquant::data": [
      "register"
    ],
    "pyopenquant::dynamic_allocation": [
      "register"
    ],
    "pyopenquant::ef3m": [
      "register"
    ],
    "pyopenquant::ensemble": [
      "register"
    ],
    "pyopenquant::fast_ewma": [
      "register"
    ],
    "pyopenquant::feature_importance": [
      "register"
    ],
    "pyopenquant::filters": [
      "register"
    ],
    "pyopenquant::fracdiff": [
      "register"
    ],
    "pyopenquant::hcaa": [
      "register"
    ],
    "pyopenquant::helpers": [
      "bars_to_rows",
      "build_labeling_events",
      "build_ohlcv_columns",
      "build_trades",
      "format_naive_datetime",
      "format_naive_datetimes",
      "matrix_from_rows",
      "pair_timestamps_values",
      "parse_datetime_str",
      "parse_naive_datetime",
      "parse_naive_datetimes",
      "parse_one_naive_datetime",
      "parse_vertical_barriers",
      "report_to_pydict"
    ],
    "pyopenquant::hrp": [
      "register"
    ],
    "pyopenquant::hyperparameter_tuning": [
      "register"
    ],
    "pyopenquant::labeling": [
      "register"
    ],
    "pyopenquant::microstructural": [
      "register"
    ],
    "pyopenquant::onc": [
      "register"
    ],
    "pyopenquant::pipeline": [
      "register"
    ],
    "pyopenquant::portfolio": [
      "register"
    ],
    "pyopenquant::risk": [
      "register"
    ],
    "pyopenquant::sample_weights": [
      "register"
    ],
    "pyopenquant::sampling": [
      "register"
    ],
    "pyopenquant::sb_bagging": [
      "register"
    ],
    "pyopenquant::strategy_risk": [
      "register"
    ],
    "pyopenquant::streaming_hpc": [
      "register"
    ],
    "pyopenquant::structural_breaks": [
      "register"
    ],
    "pyopenquant::synthetic_bt": [
      "register"
    ],
    "pyopenquant::volatility": [
      "register"
    ]
  },
  "rustItems": {
    "openquant::backtest_statistics": {
      "all_bets_concentration": "fn",
      "average_holding_period": "fn",
      "bets_concentration": "fn",
      "deflated_sharpe_ratio": "fn",
      "drawdown_and_time_under_water": "fn",
      "information_ratio": "fn",
      "minimum_track_record_length": "fn",
      "probabilistic_sharpe_ratio": "fn",
      "sharpe_ratio": "fn",
      "timing_of_flattening_and_flips": "fn"
    },
    "openquant::backtesting_engine": {
      "AntiLeakageDiagnostics": "struct",
      "BacktestData": "struct",
      "BacktestData::validate": "method",
      "BacktestDiagnostics": "struct",
      "BacktestError": "enum",
      "BacktestMode": "enum",
      "BacktestRunConfig": "struct",
      "BacktestSafeguards": "struct",
      "BacktestSafeguards::validate": "method",
      "CpcvConfig": "struct",
      "CpcvPathAssignment": "struct",
      "CpcvPathPerformance": "struct",
      "CpcvResult": "struct",
      "CrossValidationConfig": "struct",
      "CrossValidationResult": "struct",
      "FoldPerformance": "struct",
      "SplitDefinition": "struct",
      "WalkForwardConfig": "struct",
      "WalkForwardResult": "struct",
      "cpcv_path_count": "fn",
      "run_cpcv": "fn",
      "run_cross_validation": "fn",
      "run_walk_forward": "fn"
    },
    "openquant::bet_sizing": {
      "BetSizingError": "enum",
      "MixtureParams": "type",
      "ReserveBetSizeRow": "type",
      "avg_active_signals": "fn",
      "bet_size": "fn",
      "bet_size_budget": "fn",
      "bet_size_dynamic": "fn",
      "bet_size_power": "fn",
      "bet_size_probability": "fn",
      "bet_size_reserve": "fn",
      "bet_size_reserve_full": "fn",
      "bet_size_reserve_with_fit": "fn",
      "bet_size_sigmoid": "fn",
      "cdf_mixture": "fn",
      "confirm_and_cast_to_df": "fn",
      "discrete_signal": "fn",
      "get_concurrent_sides": "fn",
      "get_signal": "fn",
      "get_target_pos": "fn",
      "get_target_pos_power": "fn",
      "get_target_pos_sigmoid": "fn",
      "get_w": "fn",
      "get_w_power": "fn",
      "get_w_sigmoid": "fn",
      "inv_price": "fn",
      "inv_price_power": "fn",
      "inv_price_sigmoid": "fn",
      "limit_price": "fn",
      "limit_price_power": "fn",
      "limit_price_sigmoid": "fn",
      "mp_avg_active_signals": "fn",
      "single_bet_size_mixed": "fn"
    },
    "openquant::cla": {
      "AssetPrices": "struct",
      "AssetPrices::new": "method",
      "AssetPricesInput": "enum",
      "CLA": "struct",
      "CLA::_initialise": "method",
      "CLA::_purge_excess": "method",
      "CLA::_purge_num_err": "method",
      "CLA::allocate": "method",
      "CLA::new": "method",
      "ClaError": "enum",
      "ReturnsEstimation": "struct",
      "ReturnsEstimation::calculate_exponential_historical_returns": "method",
      "ReturnsEstimation::calculate_mean_historical_returns": "method",
      "ReturnsEstimation::calculate_returns": "method",
      "WeightBounds": "enum",
      "covariance": "fn"
    },
    "openquant::codependence": {
      "CodependenceError": "enum",
      "CodependenceResult": "type",
      "absolute_angular_distance": "fn",
      "angular_distance": "fn",
      "distance_correlation": "fn",
      "get_mutual_info": "fn",
      "get_optimal_number_of_bins": "fn",
      "squared_angular_distance": "fn",
      "variation_of_information_score": "fn"
    },
    "openquant::combinatorial_optimization": {
      "AdapterComparison": "struct",
      "CombinatorialOptimizationError": "enum",
      "DecisionSchema": "struct",
      "DecisionSchema::decision_space_size": "method",
      "DecisionSchema::validate": "method",
      "IntegerObjective": "trait",
      "IntegerVariable": "struct",
      "ObjectiveSense": "enum",
      "OptimizationResult": "struct",
      "SolverAdapter": "trait",
      "TradeBounds": "struct",
      "TradingTrajectoryObjective": "trait",
      "TradingTrajectoryObjectiveConfig": "struct",
      "TradingTrajectoryPath": "struct",
      "TradingTrajectoryPath::horizon": "method",
      "TradingTrajectorySchema": "struct",
      "TradingTrajectorySchema::horizon": "method",
      "TradingTrajectorySchema::validate": "method",
      "TrajectoryOptimizationResult": "struct",
      "compare_exact_and_adapter": "fn",
      "enumerate_trading_paths": "fn",
      "evaluate_trading_path": "fn",
      "solve_exact": "fn",
      "solve_trading_trajectory_exact": "fn",
      "solve_with_adapter": "fn"
    },
    "openquant::cross_validation": {
      "CpcvPath": "struct",
      "CpcvSplit": "struct",
      "CrossValidationError": "enum",
      "PurgedKFold": "struct",
      "PurgedKFold::cpcv_paths": "method",
      "PurgedKFold::cpcv_splits": "method",
      "PurgedKFold::new": "method",
      "PurgedKFold::split": "method",
      "PurgedKFold::split_with_diagnostics": "method",
      "PurgedSplit": "struct",
      "PurgedSplitDiagnostics": "struct",
      "Scoring": "enum",
      "SimpleClassifier": "trait",
      "TrainTestSplit": "type",
      "count_train_test_overlaps": "fn",
      "ml_cross_val_score": "fn",
      "ml_get_train_times": "fn",
      "naive_kfold_splits": "fn"
    },
    "openquant::data_processing": {
      "AlignedOhlcvColumns": "struct",
      "AlignedOhlcvRow": "struct",
      "CalendarAlignmentReport": "struct",
      "DataProcessingError": "enum",
      "DataQualityReport": "struct",
      "OhlcvColumns": "struct",
      "OhlcvRow": "struct",
      "align_calendar_columns": "fn",
      "align_calendar_df": "fn",
      "align_calendar_rows": "fn",
      "clean_ohlcv_columns": "fn",
      "clean_ohlcv_df": "fn",
      "clean_ohlcv_rows": "fn",
      "quality_report": "fn",
      "quality_report_columns": "fn",
      "quality_report_df": "fn"
    },
    "openquant::data_structures": {
      "ImbalanceBarType": "enum",
      "StandardBar": "struct",
      "StandardBarType": "enum",
      "Trade": "struct",
      "imbalance_bars": "fn",
      "run_bars": "fn",
      "standard_bars": "fn",
      "time_bars": "fn"
    },
    "openquant::dynamic_allocation": {
      "DEFAULT_MAX_TRAJECTORIES": "constant",
      "DynamicAllocation": "struct",
      "DynamicAllocationConfig": "struct",
      "DynamicAllocationConfig::new": "method",
      "DynamicAllocationError": "enum",
      "HorizonForecast": "struct",
      "all_weights": "fn",
      "dynamic_optimal_portfolio": "fn",
      "partition_count": "fn",
      "pigeonhole_partitions": "fn",
      "trajectory_sharpe_ratio": "fn",
      "transaction_costs": "fn",
      "weight_count": "fn"
    },
    "openquant::ef3m": {
      "FitResultRow": "struct",
      "M2N": "struct",
      "M2N::fit": "method",
      "M2N::get_moments": "method",
      "M2N::iter_4": "method",
      "M2N::iter_5": "method",
      "M2N::mp_fit": "method",
      "M2N::new": "method",
      "M2N::single_fit_loop": "method",
      "M2N::with_defaults": "method",
      "centered_moment": "fn",
      "most_likely_parameters": "fn",
      "raw_moment": "fn"
    },
    "openquant::ensemble_methods": {
      "BaggingBoostingDecision": "struct",
      "BiasVarianceNoise": "struct",
      "EnsembleError": "enum",
      "EnsembleMethod": "enum",
      "aggregate_classification_probability_mean": "fn",
      "aggregate_classification_vote": "fn",
      "aggregate_regression_mean": "fn",
      "average_pairwise_prediction_correlation": "fn",
      "bagging_ensemble_variance": "fn",
      "bias_variance_noise": "fn",
      "bootstrap_sample_indices": "fn",
      "recommend_bagging_vs_boosting": "fn",
      "sequential_bootstrap_sample_indices": "fn"
    },
    "openquant::etf_trick": {
      "EtfTrick": "struct",
      "EtfTrick::from_csv": "method",
      "EtfTrick::from_tables": "method",
      "EtfTrick::get_etf_series": "method",
      "EtfTrick::reset": "method",
      "EtfTrickError": "enum",
      "FuturesRollRow": "struct",
      "Table": "struct",
      "Table::from_csv": "method",
      "get_futures_roll_series": "fn"
    },
    "openquant::feature_importance": {
      "FeatureImportanceError": "enum",
      "ImportanceStats": "struct",
      "PcaCorrelation": "struct",
      "feature_pca_analysis": "fn",
      "get_orthogonal_features": "fn",
      "mean_decrease_accuracy": "fn",
      "mean_decrease_impurity": "fn",
      "plot_feature_importance": "fn",
      "single_feature_importance": "fn"
    },
    "openquant::filters": {
      "FilterError": "enum",
      "Threshold": "enum",
      "cusum_filter_indices": "fn",
      "cusum_filter_timestamps": "fn",
      "z_score_filter_indices": "fn",
      "z_score_filter_timestamps": "fn"
    },
    "openquant::fingerprint": {
      "ClassificationModelFingerprint": "struct",
      "ClassificationModelFingerprint::fit": "method",
      "ClassificationModelFingerprint::get_effects": "method",
      "ClassificationModelFingerprint::new": "method",
      "ClassificationModelFingerprint::plot_effects": "method",
      "ClassificationPredictor": "trait",
      "Effect": "struct",
      "FingerprintError": "enum",
      "PairwiseEffect": "struct",
      "RegressionModelFingerprint": "struct",
      "RegressionModelFingerprint::fit": "method",
      "RegressionModelFingerprint::get_effects": "method",
      "RegressionModelFingerprint::new": "method",
      "RegressionModelFingerprint::plot_effects": "method",
      "RegressionPredictor": "trait"
    },
    "openquant::fracdiff": {
      "frac_diff": "fn",
      "frac_diff_ffd": "fn",
      "get_weights": "fn",
      "get_weights_ffd": "fn"
    },
    "openquant::hcaa": {
      "HcaaDistance": "enum",
      "HcaaError": "enum",
      "HierarchicalClusteringAssetAllocation": "struct",
      "HierarchicalClusteringAssetAllocation::allocate": "method",
      "HierarchicalClusteringAssetAllocation::new": "method",
      "HierarchicalClusteringAssetAllocation::with_distance": "method"
    },
    "openquant::hpc_parallel": {
      "AsyncParallelHandle": "struct",
      "AsyncParallelHandle::is_finished": "method",
      "AsyncParallelHandle::wait": "method",
      "ExecutionMode": "enum",
      "HpcParallelConfig": "struct",
      "HpcParallelError": "enum",
      "HpcParallelMetrics": "struct",
      "MoleculePartition": "struct",
      "MoleculePartition::is_empty": "method",
      "MoleculePartition::len": "method",
      "ParallelRunReport": "struct",
      "PartitionStrategy": "enum",
      "ProgressSnapshot": "struct",
      "dispatch_async": "fn",
      "partition_atoms": "fn",
      "run_parallel": "fn"
    },
    "openquant::hrp": {
      "HierarchicalRiskParity": "struct",
      "HierarchicalRiskParity::allocate": "method",
      "HierarchicalRiskParity::new": "method",
      "HierarchicalRiskParity::plot_clusters": "method",
      "HierarchicalRiskParity::with_distance": "method",
      "HrpDendrogram": "struct",
      "HrpDistance": "enum",
      "HrpError": "enum"
    },
    "openquant::hyperparameter_tuning": {
      "HyperParamValue": "enum",
      "HyperParamValue::as_bool": "method",
      "HyperParamValue::as_f64": "method",
      "HyperParamValue::as_i64": "method",
      "ParamSet": "type",
      "RandomParamDistribution": "enum",
      "SearchData": "struct",
      "SearchResult": "struct",
      "SearchScoring": "enum",
      "SearchTrial": "struct",
      "TuningError": "enum",
      "classification_score": "fn",
      "expand_param_grid": "fn",
      "grid_search": "fn",
      "randomized_search": "fn",
      "sample_log_uniform": "fn",
      "sample_param_sets": "fn"
    },
    "openquant::labeling": {
      "Event": "struct",
      "LabeledEvent": "struct",
      "TripleBarrierConfig": "struct",
      "add_vertical_barrier": "fn",
      "drop_labels": "fn",
      "get_bins": "fn",
      "get_events": "fn",
      "meta_labels": "fn",
      "triple_barrier_events": "fn",
      "triple_barrier_labels": "fn"
    },
    "openquant::microstructural_features": {
      "MicrostructuralError": "enum",
      "MicrostructuralFeaturesGenerator": "struct",
      "MicrostructuralFeaturesGenerator::get_features_from_csv": "method",
      "MicrostructuralFeaturesGenerator::new_from_csv": "method",
      "encode_array": "fn",
      "encode_tick_rule_array": "fn",
      "get_avg_tick_size": "fn",
      "get_bar_based_amihud_lambda": "fn",
      "get_bar_based_hasbrouck_lambda": "fn",
      "get_bar_based_kyle_lambda": "fn",
      "get_bekker_parkinson_vol": "fn",
      "get_bvc_buy_volume": "fn",
      "get_corwin_schultz_estimator": "fn",
      "get_konto_entropy": "fn",
      "get_lempel_ziv_entropy": "fn",
      "get_plug_in_entropy": "fn",
      "get_roll_impact": "fn",
      "get_roll_measure": "fn",
      "get_shannon_entropy": "fn",
      "get_trades_based_amihud_lambda": "fn",
      "get_trades_based_hasbrouck_lambda": "fn",
      "get_trades_based_kyle_lambda": "fn",
      "get_vpin": "fn",
      "quantile_mapping": "fn",
      "sigma_mapping": "fn",
      "vwap": "fn"
    },
    "openquant::onc": {
      "OncError": "enum",
      "OncResult": "struct",
      "check_improve_clusters": "fn",
      "get_onc_clusters": "fn"
    },
    "openquant::pipeline": {
      "BacktestStage": "struct",
      "EventSelectionStage": "struct",
      "LeakageChecks": "struct",
      "PipelineError": "enum",
      "PortfolioStage": "struct",
      "ResearchPipelineConfig": "struct",
      "ResearchPipelineInput": "struct",
      "ResearchPipelineOutput": "struct",
      "RiskStage": "struct",
      "SignalStage": "struct",
      "run_mid_frequency_pipeline": "fn"
    },
    "openquant::portfolio_optimization": {
      "AllocError": "enum",
      "AllocationOptions": "struct",
      "MeanVariance": "struct",
      "ReturnsMethod": "enum",
      "allocate_efficient_risk": "fn",
      "allocate_efficient_risk_with": "fn",
      "allocate_from_inputs": "fn",
      "allocate_inverse_variance": "fn",
      "allocate_inverse_variance_with": "fn",
      "allocate_max_sharpe": "fn",
      "allocate_max_sharpe_with": "fn",
      "allocate_min_vol": "fn",
      "allocate_min_vol_with": "fn",
      "allocate_with_solution": "fn",
      "compute_expected_and_covariance": "fn",
      "returns_method_from_str": "fn"
    },
    "openquant::risk_metrics": {
      "RiskMetrics": "struct",
      "RiskMetrics::calculate_conditional_drawdown_risk": "method",
      "RiskMetrics::calculate_conditional_drawdown_risk_from_matrix": "method",
      "RiskMetrics::calculate_expected_shortfall": "method",
      "RiskMetrics::calculate_expected_shortfall_from_matrix": "method",
      "RiskMetrics::calculate_value_at_risk": "method",
      "RiskMetrics::calculate_value_at_risk_from_matrix": "method",
      "RiskMetrics::calculate_variance": "method",
      "RiskMetricsError": "enum"
    },
    "openquant::sample_weights": {
      "SampleWeightsError": "enum",
      "get_weights_by_return": "fn",
      "get_weights_by_time_decay": "fn"
    },
    "openquant::sampling": {
      "bootstrap_loop_run": "fn",
      "get_av_uniqueness_from_triple_barrier": "fn",
      "get_ind_mat_average_uniqueness": "fn",
      "get_ind_mat_label_uniqueness": "fn",
      "get_ind_matrix": "fn",
      "num_concurrent_events": "fn",
      "seq_bootstrap": "fn",
      "seq_bootstrap_with_rng": "fn"
    },
    "openquant::sb_bagging": {
      "MaxFeatures": "enum",
      "MaxSamples": "enum",
      "SbBaggingError": "enum",
      "SequentiallyBootstrappedBaggingClassifier": "struct",
      "SequentiallyBootstrappedBaggingClassifier::fit": "method",
      "SequentiallyBootstrappedBaggingClassifier::new": "method",
      "SequentiallyBootstrappedBaggingClassifier::predict": "method",
      "SequentiallyBootstrappedBaggingClassifier::predict_proba": "method",
      "SequentiallyBootstrappedBaggingRegressor": "struct",
      "SequentiallyBootstrappedBaggingRegressor::fit": "method",
      "SequentiallyBootstrappedBaggingRegressor::new": "method",
      "SequentiallyBootstrappedBaggingRegressor::predict": "method"
    },
    "openquant::strategy_risk": {
      "AsymmetricPayout": "struct",
      "StrategyRiskConfig": "struct",
      "StrategyRiskError": "enum",
      "StrategyRiskReport": "struct",
      "estimate_strategy_failure_probability": "fn",
      "implied_frequency_asymmetric": "fn",
      "implied_frequency_symmetric": "fn",
      "implied_precision_asymmetric": "fn",
      "implied_precision_symmetric": "fn",
      "sharpe_asymmetric": "fn",
      "sharpe_symmetric": "fn"
    },
    "openquant::streaming_hpc": {
      "AlertThresholds": "struct",
      "EarlyWarningSnapshot": "struct",
      "HhiConfig": "struct",
      "HhiState": "struct",
      "HhiState::current": "method",
      "HhiState::new": "method",
      "HhiState::update": "method",
      "HhiState::window_len": "method",
      "ParallelStreamingReport": "struct",
      "StreamEvent": "struct",
      "StreamEvent::total_volume": "method",
      "StreamSummary": "struct",
      "StreamingEarlyWarningEngine": "struct",
      "StreamingEarlyWarningEngine::new": "method",
      "StreamingEarlyWarningEngine::on_event": "method",
      "StreamingHpcError": "enum",
      "StreamingPipelineConfig": "struct",
      "StreamingRunMetrics": "struct",
      "StreamingRunReport": "struct",
      "SyntheticStreamConfig": "struct",
      "VpinConfig": "struct",
      "VpinState": "struct",
      "VpinState::completed_buckets": "method",
      "VpinState::current": "method",
      "VpinState::current_cdf": "method",
      "VpinState::new": "method",
      "VpinState::update": "method",
      "generate_synthetic_flash_crash_stream": "fn",
      "run_streaming_pipeline": "fn",
      "run_streaming_pipeline_parallel": "fn"
    },
    "openquant::structural_breaks": {
      "ChuStinchcombeWhiteResult": "struct",
      "SadfLags": "enum",
      "StructuralBreakError": "enum",
      "StructuralBreakResult": "type",
      "_get_betas": "fn",
      "_get_values_diff": "fn",
      "get_chow_type_stat": "fn",
      "get_chu_stinchcombe_white_statistics": "fn",
      "get_sadf": "fn"
    },
    "openquant::synthetic_backtesting": {
      "OtrSearchResult": "struct",
      "OuProcessParams": "struct",
      "RuleSurfacePoint": "struct",
      "StabilityCriteria": "struct",
      "StabilityDiagnostics": "struct",
      "SyntheticBacktestConfig": "struct",
      "SyntheticBacktestError": "enum",
      "TradingRule": "struct",
      "calibrate_ou_params": "fn",
      "detect_no_stable_optimum": "fn",
      "evaluate_rule_on_paths": "fn",
      "generate_ou_paths": "fn",
      "run_synthetic_otr_workflow": "fn",
      "search_optimal_trading_rule": "fn"
    },
    "openquant::util::fast_ewma": {
      "ewma": "fn"
    },
    "openquant::util::input_error": {
      "InputError": "enum"
    },
    "openquant::util::volatility": {
      "get_daily_vol": "fn",
      "get_garman_class_vol": "fn",
      "get_parkinson_vol": "fn",
      "get_yang_zhang_vol": "fn"
    }
  }
} as const;
