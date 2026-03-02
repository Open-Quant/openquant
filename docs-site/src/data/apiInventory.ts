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
    "bars": [
      "bar_diagnostics",
      "build_dollar_bars",
      "build_tick_bars",
      "build_time_bars",
      "build_volume_bars"
    ],
    "data": [
      "align_calendar",
      "clean_ohlcv",
      "data_quality_report",
      "load_ohlcv"
    ],
    "feature_diagnostics": [
      "mda_importance",
      "mdi_importance",
      "orthogonalize_features_pca",
      "sfi_importance",
      "substitution_effect_report"
    ],
    "pipeline": [
      "run_mid_frequency_pipeline",
      "run_mid_frequency_pipeline_frames",
      "summarize_pipeline"
    ],
    "research": [
      "make_synthetic_futures_dataset",
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
    "backtest_statistics": [
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
    "backtesting_engine": [
      "cpcv_path_count",
      "validate"
    ],
    "bet_sizing": [
      "avg_active_signals",
      "bet_size",
      "bet_size_budget",
      "bet_size_checked",
      "bet_size_dynamic",
      "bet_size_dynamic_checked",
      "bet_size_power",
      "bet_size_power_checked",
      "bet_size_probability",
      "bet_size_reserve",
      "bet_size_reserve_full",
      "bet_size_reserve_with_fit",
      "bet_size_sigmoid",
      "cdf_mixture",
      "confirm_and_cast_to_df",
      "confirm_and_cast_to_df_checked",
      "discrete_signal",
      "get_concurrent_sides",
      "get_signal",
      "get_target_pos",
      "get_target_pos_checked",
      "get_target_pos_power",
      "get_target_pos_sigmoid",
      "get_w",
      "get_w_checked",
      "get_w_power",
      "get_w_power_checked",
      "get_w_sigmoid",
      "inv_price",
      "inv_price_checked",
      "inv_price_power",
      "inv_price_sigmoid",
      "limit_price",
      "limit_price_checked",
      "limit_price_power",
      "limit_price_sigmoid",
      "mp_avg_active_signals",
      "single_bet_size_mixed"
    ],
    "cla": [
      "_compute_lambda",
      "_compute_w",
      "_free_bound_weight",
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
    "codependence": [
      "absolute_angular_distance",
      "angular_distance",
      "distance_correlation",
      "get_mutual_info",
      "get_optimal_number_of_bins",
      "squared_angular_distance",
      "variation_of_information_score"
    ],
    "combinatorial_optimization": [
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
    "cross_validation": [
      "ml_get_train_times",
      "new",
      "split"
    ],
    "data_processing": [
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
    "data_structures": [
      "imbalance_bars",
      "run_bars",
      "standard_bars",
      "time_bars"
    ],
    "ef3m": [
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
    "ensemble_methods": [
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
    "etf_trick": [
      "from_csv",
      "from_tables",
      "get_etf_series",
      "get_futures_roll_series",
      "reset"
    ],
    "feature_importance": [
      "feature_pca_analysis",
      "get_orthogonal_features",
      "mean_decrease_impurity",
      "plot_feature_importance"
    ],
    "filters": [
      "cusum_filter_indices",
      "cusum_filter_indices_checked",
      "cusum_filter_timestamps",
      "cusum_filter_timestamps_checked",
      "z_score_filter_indices",
      "z_score_filter_timestamps",
      "z_score_filter_timestamps_checked"
    ],
    "fingerprint": [
      "get_effects",
      "new",
      "plot_effects"
    ],
    "fracdiff": [
      "frac_diff",
      "frac_diff_ffd",
      "get_weights",
      "get_weights_ffd"
    ],
    "hcaa": [
      "allocate",
      "new"
    ],
    "hpc_parallel": [
      "is_empty",
      "is_finished",
      "len",
      "partition_atoms",
      "wait"
    ],
    "hrp": [
      "allocate",
      "new",
      "plot_clusters"
    ],
    "hyperparameter_tuning": [
      "as_bool",
      "as_f64",
      "as_i64",
      "classification_score",
      "expand_param_grid"
    ],
    "labeling": [
      "add_vertical_barrier",
      "drop_labels",
      "get_bins",
      "get_events",
      "meta_labels",
      "triple_barrier_events",
      "triple_barrier_labels"
    ],
    "microstructural_features": [
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
    "onc": [
      "get_onc_clusters"
    ],
    "pipeline": [
      "run_mid_frequency_pipeline"
    ],
    "portfolio_optimization": [
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
    "risk_metrics": [
      "calculate_conditional_drawdown_risk",
      "calculate_conditional_drawdown_risk_from_matrix",
      "calculate_expected_shortfall",
      "calculate_expected_shortfall_from_matrix",
      "calculate_value_at_risk",
      "calculate_value_at_risk_from_matrix",
      "calculate_variance"
    ],
    "sample_weights": [
      "get_weights_by_return",
      "get_weights_by_time_decay"
    ],
    "sampling": [
      "bootstrap_loop_run",
      "get_av_uniqueness_from_triple_barrier",
      "get_ind_mat_average_uniqueness",
      "get_ind_mat_label_uniqueness",
      "get_ind_matrix",
      "num_concurrent_events",
      "seq_bootstrap"
    ],
    "sb_bagging": [
      "fit",
      "new",
      "predict"
    ],
    "strategy_risk": [
      "estimate_strategy_failure_probability",
      "implied_frequency_asymmetric",
      "implied_frequency_symmetric",
      "implied_precision_asymmetric",
      "implied_precision_symmetric",
      "sharpe_asymmetric",
      "sharpe_symmetric"
    ],
    "streaming_hpc": [
      "completed_buckets",
      "current",
      "generate_synthetic_flash_crash_stream",
      "new",
      "on_event",
      "run_streaming_pipeline",
      "run_streaming_pipeline_parallel",
      "total_volume",
      "update",
      "window_len"
    ],
    "structural_breaks": [
      "_get_betas",
      "_get_values_diff",
      "get_chow_type_stat",
      "get_chu_stinchcombe_white_statistics",
      "get_sadf"
    ],
    "synthetic_backtesting": [
      "calibrate_ou_params",
      "detect_no_stable_optimum",
      "evaluate_rule_on_paths",
      "generate_ou_paths",
      "run_synthetic_otr_workflow",
      "search_optimal_trading_rule"
    ]
  }
} as const;
