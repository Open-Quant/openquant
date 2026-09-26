export type Formula = {
  label: string;
  latex: string;
  /**
   * Definitions for every symbol the equation introduces. Rendered directly
   * under the equation. Populate it whenever a formula names something the
   * reader has not already met on the page — an undefined symbol is the
   * difference between a foundation and a decoration.
   */
  where?: string;
};

export type ExampleBlock = {
  title: string;
  language: "rust" | "bash" | "python";
  code: string;
  /**
   * `docCheck: "skip"` marks a python example as illustrative rather than
   * runnable — it has `...` placeholders, reads a data file the repo does not
   * ship, or continues a variable defined in the prose. The generator turns it
   * into a ```python doc-check=skip fence, which check:python-examples parses
   * but does not execute. Every skip is listed on every run, so it is a public
   * exemption, not a quiet one. Anything that CAN run must not carry it.
   */
  docCheck?: "skip";
};

export type ParameterDoc = {
  name: string;
  type: string;
  description: string;
  default?: string;
};

export type ModuleDoc = {
  /**
   * The module's page is a hand-written .md file. The generator leaves it alone, and this
   * entry keeps only what the module index needs: slug, module, subject, summary, surface.
   * Every content field below is then unused. Issue #53 ends when every entry looks like this.
   */
  handwritten?: true;
  slug: string;
  module: string;
  subject: string;
  summary: string;
  /**
   * @deprecated Superseded by `conceptOverview`, which says the same thing
   * with substance behind it. No longer rendered on any page; retained only
   * so the 39 existing entries still type-check, and safe to delete once
   * something has been done with the one-liners.
   */
  whyItExists?: string;
  keyApis?: string[];
  formulas?: Formula[];
  examples?: ExampleBlock[];
  notes?: string[];
  /**
   * The three fields below are REQUIRED, and the generator asserts them.
   * They used to be optional, and 27 of the 39 modules simply omitted them:
   * the generator silently fell through to a `## Subject` heading skeleton,
   * so a missing overview shipped as a 130-word page instead of failing the
   * build. A new module with none of these is now a loud generator error.
   */
  conceptOverview?: string;
  whenToUse?: string;
  relatedModules?: string[];
  keyParameters?: ParameterDoc[];
  commonPitfalls?: string[];
  afmlChapters?: number[];
  pythonApis?: string[];
  apiSurface?: "rust-only" | "python-only" | "both";
};

export const moduleDocs: ModuleDoc[] = [
  {
    slug: "backtest-statistics",
    module: "backtest_statistics",
    subject: "Portfolio Construction and Risk",
    summary: "Probabilistic and deflated Sharpe ratios, minimum track record, drawdown and concentration.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["backtest_stats.sharpe_ratio", "backtest_stats.information_ratio", "backtest_stats.probabilistic_sharpe_ratio", "backtest_stats.deflated_sharpe_ratio", "backtest_stats.minimum_track_record_length", "backtest_stats.timing_of_flattening_and_flips", "backtest_stats.average_holding_period", "backtest_stats.bets_concentration", "backtest_stats.all_bets_concentration", "backtest_stats.drawdown_and_time_under_water"],
  },
  {
    slug: "backtesting-engine",
    module: "backtesting_engine",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Walk-forward, purged CV and combinatorial purged CV splits, with CPCV's out-of-sample paths.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["backtesting_engine.cpcv_path_count", "backtesting_engine.run_cpcv", "backtesting_engine.assemble_cpcv_paths"],
  },
  {
    slug: "bet-sizing",
    module: "bet_sizing",
    subject: "Position Sizing and Trade Construction",
    summary: "From a probability or a price forecast to a position size, averaged over live bets and discretised.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["bet_sizing.get_signal", "bet_sizing.discrete_signal", "bet_sizing.bet_size", "bet_sizing.bet_size_sigmoid", "bet_sizing.bet_size_power", "bet_sizing.inv_price", "bet_sizing.inv_price_sigmoid", "bet_sizing.inv_price_power", "bet_sizing.get_w", "bet_sizing.get_w_sigmoid", "bet_sizing.get_w_power", "bet_sizing.get_target_pos", "bet_sizing.get_target_pos_sigmoid", "bet_sizing.get_target_pos_power", "bet_sizing.limit_price", "bet_sizing.limit_price_sigmoid", "bet_sizing.limit_price_power", "bet_sizing.avg_active_signals", "bet_sizing.bet_size_dynamic", "bet_sizing.cdf_mixture", "bet_sizing.single_bet_size_mixed", "bet_sizing.get_concurrent_sides", "bet_sizing.bet_size_budget", "bet_sizing.bet_size_probability", "bet_sizing.mp_avg_active_signals", "bet_sizing.bet_size_reserve", "bet_sizing.bet_size_reserve_with_fit", "bet_sizing.bet_size_reserve_full"],
  },
  {
    slug: "cla",
    module: "cla",
    subject: "Portfolio Construction and Risk",
    summary: "The Critical Line Algorithm: the exact long-only efficient frontier as a sequence of turning points.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["cla.allocate_cla"],
  },
  {
    slug: "codependence",
    module: "codependence",
    subject: "Market Microstructure, Dependence and Regime Detection",
    summary: "Correlation distances, distance correlation, mutual information and variation of information.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["codependence.angular_distance", "codependence.absolute_angular_distance", "codependence.squared_angular_distance", "codependence.distance_correlation", "codependence.get_optimal_number_of_bins", "codependence.get_mutual_info", "codependence.variation_of_information_score"],
  },
  {
    slug: "cross-validation",
    module: "cross_validation",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Purged k-fold cross-validation with an embargo, for overlapping labels.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["cross_validation.purged_kfold_splits", "cross_validation.split_with_diagnostics", "cross_validation.cpcv_splits", "cross_validation.cpcv_paths", "cross_validation.naive_kfold_splits", "cross_validation.count_train_test_overlaps"],
  },
  {
    slug: "data-structures",
    module: "data_structures",
    subject: "Event-Driven Data and Labeling",
    summary: "Time, tick, volume, dollar, run and imbalance bars built from a stream of trades.",
    handwritten: true,
    afmlChapters: [2],
    apiSurface: "both",
    pythonApis: ["bars.build_time_bars", "bars.build_tick_bars", "bars.build_volume_bars", "bars.build_dollar_bars", "bars.bar_diagnostics"],
  },
  {
    slug: "hyperparameter-tuning",
    module: "hyperparameter_tuning",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Grid and randomised search on purged k-fold splits, scored with sample weights.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["hyperparameter_tuning.expand_param_grid", "hyperparameter_tuning.sample_param_sets", "hyperparameter_tuning.classification_score", "hyperparameter_tuning.purged_search"],
  },
  {
    slug: "ef3m",
    module: "ef3m",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "EF3M: a mixture of two Gaussians fitted by matching its first four or five moments exactly.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["ef3m.centered_moment", "ef3m.raw_moment", "ef3m.most_likely_parameters", "ef3m.fit_m2n"],
  },

  {
    slug: "ensemble-methods",
    module: "ensemble_methods",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Diagnostics for bagged ensembles: variance reduction given estimator correlation.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["ensemble.bias_variance_noise", "ensemble.bootstrap_sample_indices", "ensemble.sequential_bootstrap_sample_indices", "ensemble.aggregate_regression_mean", "ensemble.aggregate_classification_vote", "ensemble.aggregate_classification_probability_mean", "ensemble.average_pairwise_prediction_correlation", "ensemble.bagging_ensemble_variance", "ensemble.recommend_bagging_vs_boosting"],
  },
  {
    slug: "etf-trick",
    module: "etf_trick",
    subject: "Position Sizing and Trade Construction",
    summary: "A rebalanced futures basket, or one rolled contract, as a single continuous value series.",
    handwritten: true,
    afmlChapters: [2],
    apiSurface: "rust-only",
  },
  {
    slug: "feature-importance",
    module: "feature_importance",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "MDI, MDA and SFI feature importance, and a PCA cross-check, on purged folds.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["feature_importance.mean_decrease_impurity", "feature_importance.mean_decrease_accuracy", "feature_importance.single_feature_importance", "feature_importance.mda_from_probabilities", "feature_importance.sfi_from_probabilities"],
  },
  {
    slug: "filters",
    module: "filters",
    subject: "Event-Driven Data and Labeling",
    summary: "The symmetric CUSUM filter and a rolling z-score filter for event-based sampling.",
    handwritten: true,
    afmlChapters: [2],
    apiSurface: "both",
    pythonApis: ["filters.cusum_filter_indices", "filters.cusum_filter_timestamps", "filters.z_score_filter_indices", "filters.z_score_filter_timestamps"],
  },
  {
    slug: "fingerprint",
    module: "fingerprint",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Linear, non-linear and pairwise-interaction effects of each feature in a fitted model.",
    handwritten: true,
    apiSurface: "rust-only",
  },
  {
    slug: "fracdiff",
    module: "fracdiff",
    subject: "Market Microstructure, Dependence and Regime Detection",
    summary: "Fractional differentiation: stationarity with as much memory as possible.",
    handwritten: true,
    afmlChapters: [5],
    apiSurface: "both",
    pythonApis: ["fracdiff.get_weights", "fracdiff.get_weights_ffd", "fracdiff.frac_diff", "fracdiff.frac_diff_ffd"],
  },
  {
    slug: "hcaa",
    module: "hcaa",
    subject: "Portfolio Construction and Risk",
    summary: "Hierarchical allocation down the cluster tree, cut at a chosen number of clusters, with a choice of risk measure.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["hcaa.allocate_hcaa"],
  },
  {
    slug: "hrp",
    module: "hrp",
    subject: "Portfolio Construction and Risk",
    summary: "Hierarchical Risk Parity: weights from a clustering of the correlation matrix, with no inversion.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["hrp.allocate_hrp"],
  },
  {
    slug: "labeling",
    module: "labeling",
    subject: "Event-Driven Data and Labeling",
    summary: "Triple-barrier labels and meta-labels, with barriers in units of volatility at the event.",
    handwritten: true,
    afmlChapters: [3],
    apiSurface: "both",
    pythonApis: ["labeling.triple_barrier_labels", "labeling.triple_barrier_events", "labeling.meta_labels", "labeling.add_vertical_barrier", "labeling.get_events", "labeling.get_bins", "labeling.drop_labels"],
  },
  {
    slug: "microstructural-features",
    module: "microstructural_features",
    subject: "Market Microstructure, Dependence and Regime Detection",
    summary: "Spread, price-impact, order-flow and entropy features from bars or trades.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["microstructural.get_roll_measure", "microstructural.get_roll_impact", "microstructural.get_corwin_schultz_estimator", "microstructural.get_bekker_parkinson_vol", "microstructural.get_bar_based_kyle_lambda", "microstructural.get_bar_based_amihud_lambda", "microstructural.get_bar_based_hasbrouck_lambda", "microstructural.get_trades_based_kyle_lambda", "microstructural.get_trades_based_amihud_lambda", "microstructural.get_trades_based_hasbrouck_lambda", "microstructural.vwap", "microstructural.get_avg_tick_size", "microstructural.get_vpin", "microstructural.get_bvc_buy_volume", "microstructural.encode_tick_rule_array", "microstructural.quantile_mapping", "microstructural.sigma_mapping", "microstructural.encode_array", "microstructural.get_shannon_entropy", "microstructural.get_lempel_ziv_entropy", "microstructural.get_plug_in_entropy", "microstructural.get_konto_entropy"],
  },
  {
    slug: "onc",
    module: "onc",
    subject: "Portfolio Construction and Risk",
    summary: "Optimal Number of Clusters: k-means over a correlation matrix, with the count chosen by silhouette quality.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["onc.get_onc_clusters"],
  },
  {
    slug: "portfolio-optimization",
    module: "portfolio_optimization",
    subject: "Portfolio Construction and Risk",
    summary: "Mean-variance allocation with weight bounds: inverse variance, minimum volatility, maximum Sharpe, target return.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["portfolio.allocate_inverse_variance", "portfolio.allocate_min_vol", "portfolio.allocate_max_sharpe", "portfolio.allocate_efficient_risk", "portfolio.allocate_with_solution", "portfolio.allocate_from_inputs"],
  },
  {
    slug: "risk-metrics",
    module: "risk_metrics",
    subject: "Portfolio Construction and Risk",
    summary: "Historical value at risk, expected shortfall, conditional drawdown at risk and portfolio variance.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["risk.calculate_value_at_risk", "risk.calculate_expected_shortfall", "risk.calculate_conditional_drawdown_risk", "risk.calculate_variance", "risk.calculate_value_at_risk_from_matrix", "risk.calculate_expected_shortfall_from_matrix", "risk.calculate_conditional_drawdown_risk_from_matrix"],
  },
  {
    slug: "strategy-risk",
    module: "strategy_risk",
    subject: "Portfolio Construction and Risk",
    summary: "The precision and bet frequency a target Sharpe ratio requires, and the probability of falling short.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["strategy_risk.sharpe_symmetric", "strategy_risk.implied_precision_symmetric", "strategy_risk.implied_frequency_symmetric", "strategy_risk.sharpe_asymmetric", "strategy_risk.implied_precision_asymmetric", "strategy_risk.implied_frequency_asymmetric", "strategy_risk.estimate_strategy_failure_probability"],
  },
  {
    slug: "hpc-parallel",
    module: "hpc_parallel",
    subject: "Scaling, HPC and Infrastructure",
    summary: "AFML's atoms and molecules: equal-work partitions of a job, run in serial or on threads, outputs in order.",
    handwritten: true,
    apiSurface: "rust-only",
  },

  {
    slug: "combinatorial-optimization",
    module: "combinatorial_optimization",
    subject: "Scaling, HPC and Infrastructure",
    summary: "Exhaustive search over small integer problems and single-instrument trading paths, as an exact baseline.",
    handwritten: true,
    apiSurface: "rust-only",
  },

  {
    slug: "streaming-hpc",
    module: "streaming_hpc",
    subject: "Scaling, HPC and Infrastructure",
    summary: "VPIN and a venue-concentration HHI updated event by event in constant memory, with a joint alert.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["streaming_hpc.run_streaming_pipeline", "streaming_hpc.generate_synthetic_flash_crash_stream"],
  },

  {
    slug: "sample-weights",
    module: "sample_weights",
    subject: "Event-Driven Data and Labeling",
    summary: "Training weights for overlapping labels: return attribution and time decay.",
    handwritten: true,
    afmlChapters: [4],
    apiSurface: "both",
    pythonApis: ["sample_weights.get_weights_by_return", "sample_weights.get_weights_by_time_decay"],
  },
  {
    slug: "sampling",
    module: "sampling",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Label concurrency, average uniqueness and the sequential bootstrap.",
    handwritten: true,
    afmlChapters: [4],
    apiSurface: "both",
    pythonApis: ["sampling.get_ind_matrix", "sampling.seq_bootstrap", "sampling.get_ind_mat_average_uniqueness", "sampling.get_ind_mat_label_uniqueness", "sampling.bootstrap_loop_run", "sampling.get_av_uniqueness_from_triple_barrier", "sampling.num_concurrent_events"],
  },
  {
    slug: "sb-bagging",
    module: "sb_bagging",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "A bagging ensemble that draws each estimator's sample with the sequential bootstrap, around a one-feature base learner.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["sb_bagging.fit_predict_sb_classifier", "sb_bagging.fit_predict_sb_regressor", "sb_bagging.SequentiallyBootstrappedBaggingClassifier"],
  },
  {
    slug: "synthetic-backtesting",
    module: "synthetic_backtesting",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Profit-taking and stop-loss levels chosen on simulated paths of a fitted mean-reverting process.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["synthetic_bt.calibrate_ou_params", "synthetic_bt.generate_ou_paths", "synthetic_bt.evaluate_rule_on_paths", "synthetic_bt.detect_no_stable_optimum", "synthetic_bt.run_synthetic_otr_workflow", "synthetic_bt.search_optimal_trading_rule"],
  },
  {
    slug: "structural-breaks",
    module: "structural_breaks",
    subject: "Market Microstructure, Dependence and Regime Detection",
    summary: "SADF for explosive behaviour, a Chow-type Dickey-Fuller test, and the CSW CUSUM test.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["structural_breaks.get_chow_type_stat", "structural_breaks.get_chu_stinchcombe_white_statistics", "structural_breaks.get_sadf"],
  },
  {
    slug: "util-fast-ewma",
    conceptOverview:
      "One function: a single-pass exponentially weighted moving average with span-style decay, alpha = 2/(window+1), corrected by the accumulated weight so that early values are not dragged toward the seed. It mirrors `mlfinlab.util.fast_ewma` exactly, which is the point — it is what makes daily volatility and every EWMA-derived feature numerically comparable between this library and a pandas reference implementation.",
    whenToUse:
      "Use it instead of writing a rolling loop, so that everything downstream — `util::volatility`'s daily vol, the microstructure feature panel, dynamic threshold series for `filters` — shares one decay convention. Remember that `window` is a span rather than a hard lookback: the weight on a point w bars back is (1-alpha)^w, not zero, so the estimate remembers further than the number suggests. Size the span longer than the horizon you are trying to smooth over.",
    relatedModules: ["util-volatility", "filters", "microstructural-features", "labeling"],
    module: "util::fast_ewma",
    subject: "Market Microstructure, Dependence and Regime Detection",
    summary: "Fast EWMA primitive shared across feature and volatility routines.",
    whyItExists: "Provides performant smoothing for repeated rolling computations.",
    keyApis: ["ewma"],
    formulas: [
      { label: "EWMA", latex: "m_t=\\alpha x_t + (1-\\alpha)m_{t-1}" },
      { label: "Smoothing", latex: "\\alpha=\\frac{2}{w+1}" },
    ],
    examples: [
      {
        title: "Compute EWMA vector",
        language: "rust",
        code: `use openquant::util::fast_ewma::ewma;\n\nlet x = vec![1.0, 2.0, 3.0, 4.0];\nlet y = ewma(&x, 3)?;`,
      },
    ],
    notes: ["Window length controls responsiveness vs smoothness.", "Prefer this helper over ad-hoc loops for consistency."],
    apiSurface: "both",
    pythonApis: ["fast_ewma.ewma"],
  },
  {
    slug: "util-volatility",
    conceptOverview:
      "Four volatility estimators with different data requirements and different blind spots. `get_daily_vol` is a close-to-close EWMA over a timestamped series — the estimator AFML uses to scale triple-barrier widths. Parkinson uses the high-low range and extracts far more information per observation, but ignores overnight gaps and assumes no drift. Garman-Klass adds the open and close. Yang-Zhang combines an overnight, an open-to-close and a Rogers-Satchell term under a variance-minimising weight, and is the only one of the four that handles both opening gaps and intraday drift.",
    whenToUse:
      "Use `get_daily_vol` whenever volatility is a scaling target for barriers or position sizes, and match its lookback to the event horizon — a 100-bar volatility scaling a 3-bar barrier is measuring the wrong thing. Use the range-based estimators when you have OHLC and want more precision from the same number of bars, preferring Yang-Zhang for instruments that gap. All range estimators degrade when quoted spreads are wide, because the recorded high and low then reflect microstructure noise rather than price.",
    relatedModules: ["labeling", "filters", "util-fast-ewma", "bet-sizing", "microstructural-features"],
    module: "util::volatility",
    subject: "Market Microstructure, Dependence and Regime Detection",
    summary: "Volatility estimators used across labeling and risk workflows.",
    whyItExists: "Volatility is a foundational scaling target for barriers, sizing, and risk controls.",
    keyApis: ["get_daily_vol", "get_parkinson_vol", "get_garman_class_vol", "get_yang_zhang_vol"],
    formulas: [
      {
        label: "Parkinson",
        latex: "\\sigma_P^2=\\frac{1}{4\\ln 2}\\cdot\\frac{1}{n}\\sum_{t}\\left(\\ln\\frac{H_t}{L_t}\\right)^2",
        where: "$H_t,L_t$ are the bar high and low and $n$ the `window` length. It uses the range rather than the close, so it is far more efficient than close-to-close on the same sample — but it ignores overnight gaps and assumes no drift.",
      },
      {
        label: "Yang-Zhang",
        latex: "\\begin{aligned}\\sigma_{YZ}^2&=\\sigma_o^2+k\\,\\sigma_c^2+(1-k)\\,\\sigma_{rs}^2,\\qquad k=\\frac{0.34}{1.34+\\frac{n+1}{n-1}}\\\\\\sigma_o^2&=\\frac{1}{n-1}\\sum_{t}\\left(o_t-\\bar o\\right)^2,\\qquad o_t=\\ln\\frac{O_t}{C_{t-1}}\\\\\\sigma_c^2&=\\frac{1}{n-1}\\sum_{t}\\left(c_t-\\bar c\\right)^2,\\qquad c_t=\\ln\\frac{C_t}{O_t}\\\\\\sigma_{rs}^2&=\\frac{1}{n}\\sum_{t}\\left(\\ln\\frac{H_t}{C_t}\\ln\\frac{H_t}{O_t}+\\ln\\frac{L_t}{C_t}\\ln\\frac{L_t}{O_t}\\right)\\end{aligned}",
        where: "the sums run over the last $n$ bars, $n$ is the `window` length, and $\\bar o,\\bar c$ are the means over those bars (Yang & Zhang 2000). $\\sigma_o^2$ is the variance of the overnight (previous close to open) return, $\\sigma_c^2$ the variance of the open-to-close return, and $\\sigma_{rs}^2$ the Rogers-Satchell (1991) estimator. Because $\\sigma_o^2$ and $\\sigma_c^2$ are demeaned and $\\sigma_{rs}^2$ is unaffected by drift, a steady trend adds nothing to the estimate. $k$ is not a free parameter — it is the weight that minimises the estimator's variance, which is what makes Yang-Zhang the only one of these four that handles both overnight gaps and intraday drift. For a 20-bar window $k\\approx0.14$, so the overnight and Rogers-Satchell terms carry most of the estimate. `get_yang_zhang_vol` returns $\\sigma_{YZ}$ per bar (not annualised); the first $n$ values are NaN because $o_t$ needs the previous close. mlfinlab's version, which this function mirrored until #165, used $\\ln(C_t/O_{t-1})$ for the open-to-close term and did not demean.",
      },
    ],
    examples: [
      {
        title: "Compute daily and range-based volatility",
        language: "rust",
        code: `use chrono::{Duration, NaiveDateTime};\nuse openquant::util::volatility::{get_daily_vol, get_parkinson_vol};\n\nlet t0 = NaiveDateTime::parse_from_str("2024-01-02 00:00:00", "%Y-%m-%d %H:%M:%S")?;\nlet close: Vec<(NaiveDateTime, f64)> = (0..300)\n    .map(|i| (t0 + Duration::days(i), 100.0 + (i as f64 * 0.07).sin() * 2.0))\n    .collect();\nlet high: Vec<f64> = close.iter().map(|(_, p)| p + 0.4).collect();\nlet low: Vec<f64> = close.iter().map(|(_, p)| p - 0.4).collect();\n\n// Close-to-close EWMA vol on a timestamped series; \`lookback\` is the EWMA span.\nlet daily = get_daily_vol(&close, 100);\n// Parkinson uses the high/low range, so it needs no timestamps — \`window\` bars.\nlet parkinson = get_parkinson_vol(&high, &low, 20)?;\n\nprintln!("daily vol tail = {:?}", daily.last());\nprintln!("parkinson vol tail = {:?}", parkinson.last());`,
      },
    ],
    notes: ["Choose estimator based on available fields and microstructure noise.", "Daily-vol lookback should be matched to event horizon."],
    apiSurface: "both",
    pythonApis: ["volatility.get_daily_vol", "volatility.get_parkinson_vol", "volatility.get_garman_class_vol", "volatility.get_yang_zhang_vol"],
  },
  // ── Python-only modules ──────────────────────────────────────────────
  {
    slug: "data",
    module: "data",
    subject: "Data Ingestion and Quality",
    summary: "Fetching daily OHLCV through a cache, content hashes for run manifests, and OHLCV loading, cleaning, calendar alignment and quality reporting.",
    whyItExists: "Provides a consistent entrypoint for market data ingestion with automatic column normalization, deduplication, and quality diagnostics.",
    keyApis: ["load_ohlcv", "clean_ohlcv", "align_calendar", "data_quality_report"],
    formulas: [],
    examples: [
      {
        title: "Fetch through the cache and record the dataset hash",
        language: "python",
        code: `import tempfile

from openquant.data import data_quality_report, fetch, quality_failures, record_dataset_hash

cache = tempfile.mkdtemp()  # omit cache_dir to use ~/.cache/openquant/data

# The default source is the bundled SYNTHETIC sample (SYN_A ... SYN_E), so this runs offline.
df, meta = fetch(["SYN_A", "SYN_B"], "2023-01-01", "2023-03-31", cache_dir=cache, return_meta=True)
print(df.columns)  # ['ts', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
print(df.height, meta["cache"])  # 130 {'SYN_A': 'miss', 'SYN_B': 'miss'}
assert quality_failures(data_quality_report(df)) == []

# The same request again is read from disk; offline=True guarantees the source is not called.
again, meta2 = fetch(["SYN_A", "SYN_B"], "2023-01-01", "2023-03-31", cache_dir=cache, return_meta=True, offline=True)
print(meta2["cache"])  # {'SYN_A': 'hit', 'SYN_B': 'hit'}
assert again.equals(df) and meta2["dataset_hash"] == meta["dataset_hash"]

# Put the content hash, source, terms and request into the run manifest.
manifest = record_dataset_hash({"run_name": "demo"}, df, **meta)
print(manifest["dataset_hash"].startswith("sha256:"))  # True`,
      },
      {
        title: "Plug in your own data vendor (fetch-only, your own key)",
        language: "python",
        docCheck: "skip",
        code: `import os

import polars as pl
from openquant.data import CallableSource, fetch


def my_vendor(symbol, start, end):
    key = os.environ["MY_VENDOR_API_KEY"]  # your own key, from the environment; never commit it
    rows = my_vendor_client.daily_bars(symbol, start, end, api_key=key)  # your vendor's client
    return pl.DataFrame(rows)  # columns such as date/open/high/low/close/volume[/adj_close]


source = CallableSource(my_vendor, name="my-vendor", terms="https://my-vendor.example/terms")
df, meta = fetch(["SPY", "TLT"], "2020-01-01", "2024-12-31", source=source, return_meta=True)`,
      },
      {
        title: "Load, clean, and inspect OHLCV data",
        language: "python",
        docCheck: "skip",
        code: `from openquant.data import load_ohlcv, data_quality_report, align_calendar

# Load from CSV/Parquet with auto column normalization
df, report = load_ohlcv("prices.csv", symbol="AAPL", return_report=True)
print(report)
# {'row_count': 5040, 'symbol_count': 1, 'duplicate_key_count': 0, ...}

# Align to regular calendar (fills gaps with nulls + is_missing_bar flag)
aligned = align_calendar(df, interval="1d")

# Quality report on any DataFrame
quality = data_quality_report(df)`,
      },
    ],
    notes: [
      "The bundled sample is SYNTHETIC, not market data. Its symbols (SYN_A ... SYN_E) are not real tickers. DATA_SOURCES.md at the repository root records the terms of every source and why no real sample is committed yet.",
      "Your vendor's terms govern data you fetch with your own adapter. Several vendors forbid redistribution, and some forbid persistent storage on free plans. The cache is for you only: never commit it.",
      "dataset_hash (oq-dataset-sha256-v1) ignores row and column order and Parquet layout. It changes with any value, column name, column type or row, so equal hashes mean the same data.",
      "Column aliases are resolved automatically (e.g., 'timestamp' → 'ts', 'ticker' → 'symbol').",
      "clean_ohlcv deduplicates by (symbol, ts) and sorts chronologically.",
      "align_calendar marks missing bars with is_missing_bar=True for downstream imputation logic.",
      "`gap_interval_count` in the quality report depends on the bar spacing, which the report infers and returns as `inferred_interval_us`: the most common spacing between consecutive bars of one symbol, pooled over symbols (the smallest on a tie). For daily data (a spacing within an hour of one day) a gap is a skipped weekday, so weekends are not gaps; exchange holidays are, since no holiday calendar is applied. For any other spacing a gap is a spacing longer than the inferred one, so intraday data counts every overnight and weekend break as a gap. Before #168 any spacing over one day was a gap whatever the frequency: weekends counted on daily data, and nothing counted on intraday data.",
    ],
    conceptOverview: `Before any AFML workflow begins, raw market data must be fetched, loaded into a consistent schema, cleaned of duplicates and formatting issues, and aligned to a regular time grid. This module handles that ingestion layer.

\`fetch(symbols, start, end, source=..., cache_dir=...)\` gets daily OHLCV bars from a pluggable source. The source is any object with a \`name\` and a \`fetch_symbol(symbol, start, end)\` method (the \`DataSource\` protocol). Three sources ship with the module. \`LocalSampleSource\` reads the bundled synthetic sample and is the default. \`LocalFileSource\` reads your own CSV or Parquet file. \`CallableSource\` wraps your own function, which calls your vendor with your key. Each (source, symbol, date range) request is cached as a Parquet file and a JSON sidecar under \`<cache_dir>/<source>[@<version>]/<symbol>/<start>_<end>.parquet\`. A repeated request is served from disk without calling the source, so it works offline, and \`offline=True\` makes that a guarantee. The returned frame is the canonical \`clean_ohlcv\` frame, and it has passed \`data_quality_report\`.

\`dataset_hash(df)\` is a deterministic SHA-256 of a table's contents. \`record_dataset_hash(manifest, df, **meta)\` writes that hash, with the source, terms and request, into a run manifest, so every result can name the exact data behind it. \`experiments/run_pipeline.py\` records it in every \`run_manifest.json\`.

The loaders accept CSV or Parquet files with flexible column naming (e.g., "timestamp", "datetime", "date" all map to "ts"; "ticker" or "asset" map to "symbol") and produce a standardized Polars DataFrame with canonical OHLCV columns. Deduplication handles duplicate (symbol, timestamp) keys, and calendar alignment generates a regular grid with explicit gap markers.

The data quality report provides diagnostics — row counts, symbol counts, duplicate counts, gap intervals, and null counts — that should be inspected before feeding data into bars, labeling, or any downstream module.`,
    whenToUse: `Use \`fetch\` when a study needs daily bars for a list of symbols and must be reproducible. Record the dataset hash in the run manifest. Use the loaders directly when you already hold pre-aggregated OHLCV data (daily bars, minute bars from a vendor). If you have raw tick/trade data instead, use the \`data_structures\` module to construct bars first.

**Prerequisites**: For \`fetch\`, a source: the bundled synthetic sample, your own file, or your own vendor function and key. For the loaders, a CSV or Parquet file, or an existing Polars DataFrame with OHLCV-like columns.

**Alternatives**: Direct Polars/pandas loading if you handle column normalization and cleaning yourself.`,
    keyParameters: [
      { name: "symbols", type: "str | Iterable[str]", description: "Symbols to fetch; duplicates are dropped", default: "—" },
      { name: "start, end", type: "date | datetime | str", description: "Inclusive date range (ISO strings accepted)", default: "—" },
      { name: "source", type: "DataSource | None", description: "Where bars come from; None uses the synthetic LocalSampleSource", default: "None" },
      { name: "cache_dir", type: "str | Path | None", description: "Cache root; None uses $OPENQUANT_DATA_CACHE, $XDG_CACHE_HOME/openquant/data or ~/.cache/openquant/data", default: "None" },
      { name: "refresh / offline", type: "bool", description: "Refetch even if cached / never call the source (raise CacheMissError)", default: "False" },
      { name: "return_meta", type: "bool", description: "Also return provenance (source, terms, cache status, dataset_hash) for the run manifest", default: "False" },
      { name: "path", type: "str | Path", description: "File path to CSV or Parquet OHLCV data (load_ohlcv)", default: "—" },
      { name: "symbol", type: "str | None", description: "Symbol name if not present as a column in the data", default: "None" },
      { name: "interval", type: "str", description: "Calendar alignment interval (e.g., '1d', '1h', '5m')", default: "'1d'" },
      { name: "dedupe_keep", type: "str", description: "Which duplicate to keep: 'first' or 'last'", default: "'last'" },
    ],
    commonPitfalls: [
      "Treating the bundled sample as market data. It is synthetic, so any result computed on it says nothing about real markets.",
      "Committing a cache directory or fetched data to a repository. Almost every vendor's terms forbid that; see DATA_SOURCES.md.",
      "Expecting a cached range to serve a sub-range. The cache key is the exact (source, symbol, start, end), so a different range is a new fetch.",
      "Forgetting to check the quality report for gaps — missing bars silently create NaN features downstream.",
      "Using align_calendar with an interval shorter than the data's actual frequency — this creates many synthetic missing-bar rows.",
      "Using align_calendar with bars that are not on the grid. Each symbol's grid starts at its first bar and steps by `interval`, so a bar stamped off that grid (a daily bar at a different time of day, an irregular intraday print) is not in the output. `align_calendar(df, interval=..., return_report=True)` returns `(aligned, report)` with those bars in `report[\"off_grid_bars\"]` and their number in `report[\"off_grid_bar_count\"]`; without `return_report` a `UserWarning` says how many were dropped. The `_core.data.align_calendar` and `align_calendar_df` bindings take the same `return_report` flag.",
    ],
    relatedModules: ["data-structures", "research"],
    apiSurface: "both",
    pythonApis: ["data.fetch", "data.dataset_hash", "data.record_dataset_hash", "data.quality_failures", "data.default_cache_dir", "data.DataSource", "data.LocalSampleSource", "data.LocalFileSource", "data.CallableSource", "data.CacheMissError", "data.load_ohlcv", "data.clean_ohlcv", "data.align_calendar", "data.data_quality_report", "data.clean_ohlcv_df", "data.quality_report_df", "data.align_calendar_df"],
  },
  {
    slug: "evaluation",
    module: "evaluation",
    subject: "Research Workflows",
    summary: "PSR, deflated Sharpe and minimum track record from a returns series, with a trial registry that persists the count DSR deflates by.",
    handwritten: true,
    afmlChapters: [3, 14, 15],
    apiSurface: "python-only",
    pythonApis: ["evaluation.return_moments", "evaluation.probabilistic_sharpe_ratio", "evaluation.deflated_sharpe_ratio", "evaluation.expected_max_sharpe", "evaluation.minimum_track_record_length", "evaluation.meta_label_metrics", "evaluation.strategy_failure_probability", "evaluation.config_hash", "evaluation.TrialRegistry"],
  },
  {
    slug: "feature-diagnostics",
    module: "feature_diagnostics",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Python importance reports on purged folds, with a substitution-effect report and a feature screen.",
    handwritten: true,
    afmlChapters: [8],
    apiSurface: "python-only",
    pythonApis: ["feature_diagnostics.mdi_importance", "feature_diagnostics.mda_importance", "feature_diagnostics.sfi_importance", "feature_diagnostics.orthogonalize_features_pca", "feature_diagnostics.substitution_effect_report"],
  },
  {
    slug: "pipeline",
    module: "pipeline",
    subject: "Research Workflows",
    summary: "End-to-end AFML research pipeline: events → signals → portfolio → risk → backtest, with ordering checks.",
    whyItExists: "Chains the core AFML steps (event filtering, bet sizing, allocation, risk, backtest) into a single reproducible research call.",
    keyApis: ["run_mid_frequency_pipeline", "ResearchPipelineConfig", "ResearchPipelineInput", "ResearchPipelineOutput", "LeakageChecks"],
    formulas: [],
    examples: [
      {
        title: "Run a complete research pipeline",
        language: "python",
        docCheck: "skip",
        code: `from openquant.pipeline import run_mid_frequency_pipeline_frames, summarize_pipeline

out = run_mid_frequency_pipeline_frames(
    timestamps=timestamps,
    close=close,
    model_probabilities=probabilities,
    asset_prices=asset_prices,
    model_sides=sides,
    asset_names=["CL", "NG", "RB", "GC"],
    cusum_threshold=0.001,
)

# Polars DataFrames for each stage
signals_df = out["frames"]["signals"]
backtest_df = out["frames"]["backtest"]
weights_df = out["frames"]["weights"]

# One-row summary with key metrics
summary = summarize_pipeline(out)
print(summary)
# portfolio_sharpe | realized_sharpe | value_at_risk | timestamps_increasing | ...`,
      },
    ],
    notes: [
      "Mismatched input lengths are an error. leakage_checks reports two computed ordering checks, timestamps_increasing and event_indices_sorted; inputs_aligned (always true) and has_forward_look_bias (always false) are deprecated constants.",
      "run_mid_frequency_pipeline_frames and summarize_pipeline are Python-only helpers over the Rust run_mid_frequency_pipeline.",
      "run_mid_frequency_pipeline_frames adds Polars DataFrames to the raw dict output.",
      "summarize_pipeline extracts key metrics into a single-row DataFrame for notebook display.",
    ],
    conceptOverview: `The pipeline module orchestrates the full AFML research workflow in a single function call. It chains: CUSUM event detection → bet sizing from the model's probabilities → a max-Sharpe portfolio allocation → risk metrics → a single-asset backtest. No labeling or model fitting happens here: the model probabilities and sides are inputs, one per bar, and the signal is traded with a one-bar lag. The output also reports whether the timestamps and the event positions are in increasing order. The pipeline does not detect look-ahead in the probabilities you pass it; that is your responsibility.

This is designed for rapid research iteration — change a parameter, re-run the pipeline, and compare the summary table. The \`_frames\` variant enriches output with Polars DataFrames for each stage, making notebook exploration ergonomic.`,
    whenToUse: `Use this when you want to run a complete AFML workflow without manually chaining individual modules. It's the fastest path from "I have prices and a model" to "I have a backtested strategy with risk metrics."

**Prerequisites**: Timestamps, close prices, model probability forecasts, and multi-asset price matrix.

**Alternatives**: Call individual modules (filters, labeling, bet_sizing, etc.) for more control over each stage.`,
    keyParameters: [
      { name: "cusum_threshold", type: "float", description: "CUSUM event filter threshold", default: "0.001" },
      { name: "num_classes", type: "int", description: "Number of label classes for bet sizing", default: "2" },
      { name: "step_size", type: "float", description: "Bet size discretization step", default: "0.1" },
      { name: "risk_free_rate", type: "float", description: "Risk-free rate for Sharpe calculations", default: "0.0" },
      { name: "confidence_level", type: "float", description: "Confidence level for VaR/ES", default: "0.05" },
    ],
    commonPitfalls: [
      "Reading has_forward_look_bias as a test: it is a deprecated constant (always false). The pipeline cannot see look-ahead inside model_probabilities; fit them on data available at each bar's close.",
      "Unordered timestamps do not stop the run; check leakage_checks.timestamps_increasing.",
      "Using the raw dict output when DataFrames are more convenient — prefer run_mid_frequency_pipeline_frames.",
    ],
    relatedModules: ["filters", "labeling", "bet-sizing", "backtest-statistics", "risk-metrics"],
    apiSurface: "both",
    pythonApis: ["pipeline.run_mid_frequency_pipeline", "pipeline.run_mid_frequency_pipeline_frames", "pipeline.summarize_pipeline"],
  },
  {
    slug: "research",
    module: "research",
    subject: "Research Workflows",
    summary: "Synthetic dataset generation and flywheel research iteration with cost modeling and promotion gates.",
    whyItExists: "Provides a reproducible research loop: generate data → run pipeline → estimate costs → check promotion criteria.",
    keyApis: ["make_synthetic_futures_dataset", "run_flywheel_iteration", "ResearchDataset"],
    formulas: [],
    examples: [
      {
        title: "Synthetic research loop with cost-aware promotion",
        language: "python",
        code: `from openquant.research import make_synthetic_futures_dataset, run_flywheel_iteration

# Generate deterministic synthetic multi-asset futures data
dataset = make_synthetic_futures_dataset(n_bars=192, seed=7)

# Run full pipeline + cost model + promotion checks
result = run_flywheel_iteration(dataset, config={
    "cusum_threshold": 0.001,
    "commission_bps": 1.5,
    "spread_bps": 2.0,
    "min_net_sharpe": 0.30,
})

# Cost breakdown
print(result["costs"])
# {'turnover': 12.3, 'net_sharpe': 0.42, 'estimated_total_cost': 0.018, ...}

# Promotion gate results
print(result["promotion"])
# {'passed_net_sharpe': True, 'promote_candidate': True, ...}

# Full summary DataFrame
print(result["summary"])`,
      },
    ],
    notes: [
      "make_synthetic_futures_dataset is deterministic given seed — use for regression tests and reproducible notebooks.",
      "run_flywheel_iteration includes turnover estimation, transaction cost modeling, and net-of-cost Sharpe.",
      "Promotion gates check realized Sharpe, net Sharpe, and leakage guards before flagging a strategy as deployment-ready.",
    ],
    conceptOverview: `The research module implements the "research flywheel" pattern: a tight loop of hypothesis → synthetic test → cost estimation → promotion gate. It wraps the pipeline module with additional cost modeling (commissions, spread, slippage proportional to realized volatility) and strategy-readiness checks.

\`make_synthetic_futures_dataset\` generates a deterministic multi-asset futures dataset with realistic properties (seasonal patterns, correlated assets, noisy model forecasts). This lets you develop and test research workflows without real market data, and provides a stable baseline for regression testing.

\`run_flywheel_iteration\` runs the full pipeline, computes turnover and estimated transaction costs, calculates net-of-cost Sharpe, and evaluates promotion criteria. The result tells you whether a strategy variant passes minimum viability thresholds.`,
    whenToUse: `Use this for rapid strategy research iteration, especially during development when you don't have (or don't want to use) real market data. Also useful for CI regression tests and notebook tutorials.

**Prerequisites**: None for synthetic data. For real data, construct a ResearchDataset from your own prices and model forecasts.`,
    keyParameters: [
      { name: "n_bars", type: "int", description: "Number of bars in synthetic dataset", default: "192" },
      { name: "seed", type: "int", description: "Random seed for reproducibility", default: "7" },
      { name: "commission_bps", type: "float", description: "Commission in basis points per turn", default: "1.5" },
      { name: "spread_bps", type: "float", description: "Spread cost in basis points", default: "2.0" },
      { name: "min_net_sharpe", type: "float", description: "Minimum net-of-cost Sharpe for promotion", default: "0.30" },
    ],
    commonPitfalls: [
      "Over-optimizing on synthetic data — the data generator has known dynamics; validate on real data before deployment.",
      "Ignoring cost estimates — gross Sharpe is misleading for high-turnover strategies.",
    ],
    relatedModules: ["pipeline"],
    apiSurface: "python-only",
    pythonApis: ["research.make_synthetic_futures_dataset", "research.run_flywheel_iteration", "research.ResearchDataset"],
  },
  {
    slug: "adapters",
    module: "adapters",
    subject: "Data Ingestion and Quality",
    summary: "Polars DataFrame adapters for signals, events, weights, backtest curves, and streaming buffers.",
    whyItExists: "Bridges raw dict/list outputs from the Rust core into typed Polars DataFrames for ergonomic notebook and pipeline use.",
    keyApis: ["to_polars_signal_frame", "to_polars_event_frame", "to_polars_backtest_frame", "to_polars_weights_frame", "SignalStreamBuffer"],
    formulas: [],
    examples: [
      {
        title: "Convert pipeline outputs to typed DataFrames",
        language: "python",
        code: `from openquant.adapters import (
    to_polars_signal_frame,
    to_polars_weights_frame,
    SignalStreamBuffer,
)

# Signal frame from raw timestamps + values
signals = to_polars_signal_frame(
    timestamps=["2024-01-02T09:30:00", "2024-01-02T09:31:00"],
    signal=[0.5, -0.3],
    side=[1.0, -1.0],
    symbol="CL",
)

# Streaming buffer for incremental signal updates
buf = SignalStreamBuffer()
buf.append(timestamps=["2024-01-02T09:32:00"], signal=[0.1])
buf.append(timestamps=["2024-01-02T09:33:00"], signal=[-0.2])
all_signals = buf.frame()  # concat into single DataFrame`,
      },
    ],
    notes: [
      "All adapter functions validate input length alignment before constructing frames.",
      "SignalStreamBuffer supports incremental append for streaming research notebooks.",
      "to_pandas() is available for downstream tools that require pandas; requires pandas to be installed.",
    ],
    conceptOverview: `The Rust core returns results as plain dicts and lists. The adapters module converts these into typed Polars DataFrames with proper datetime parsing, column naming, and validation. This is the standard way to move data between the Rust computation engine and Python analysis/visualization code.

\`SignalStreamBuffer\` provides an incremental append interface for streaming workflows where signals arrive in chunks — common in live research notebooks or paper-trading loops.`,
    whenToUse: `Use adapters whenever you receive output from the Rust core or pipeline module and need DataFrames for analysis, visualization, or further processing. The pipeline module's \`_frames\` variant calls these adapters internally.

**Alternatives**: Manual Polars DataFrame construction from dicts, but you lose validation and timestamp parsing.`,
    relatedModules: ["pipeline", "data"],
    apiSurface: "python-only",
    pythonApis: ["adapters.to_polars_signal_frame", "adapters.to_polars_event_frame", "adapters.to_polars_backtest_frame", "adapters.to_polars_weights_frame", "adapters.to_polars_indicator_matrix", "adapters.to_polars_frontier_frame", "adapters.SignalStreamBuffer", "adapters.to_pandas"],
  },
  {
    slug: "viz",
    module: "viz",
    subject: "Research Workflows",
    summary: "Visualization payload builders for feature importance, drawdown, regime, frontier, and cluster charts.",
    whyItExists: "Produces structured chart payloads (bar, line, scatter, tree) that can be rendered by any frontend without coupling to a specific plotting library.",
    keyApis: ["prepare_feature_importance_payload", "prepare_drawdown_payload", "prepare_regime_payload", "prepare_frontier_payload", "prepare_cluster_payload"],
    formulas: [],
    examples: [
      {
        title: "Build visualization payloads for research output",
        language: "python",
        code: `from openquant.viz import (
    prepare_feature_importance_payload,
    prepare_drawdown_payload,
)

# Feature importance bar chart payload
payload = prepare_feature_importance_payload(
    feature_names=["momentum", "vol", "spread"],
    importance=[0.45, 0.35, 0.20],
    std=[0.05, 0.03, 0.02],
    top_n=10,
)
# {"chart": "bar", "x": [...], "y": [...], "error_y": [...]}

# Drawdown chart payload from equity curve
dd_payload = prepare_drawdown_payload(
    timestamps=["2024-01-02", "2024-01-03", "2024-01-04"],
    equity_curve=[1.0, 1.02, 0.98],
)
# {"chart": "line", "x": [...], "equity": [...], "drawdown": [...]}`,
      },
    ],
    notes: [
      "Payloads are plain dicts — render with plotly, matplotlib, or pass to a frontend.",
      "prepare_feature_importance_payload sorts by importance descending and supports top_n filtering.",
      "prepare_feature_importance_comparison_payload creates side-by-side grouped bar payloads for before/after analysis.",
    ],
    conceptOverview: `The viz module produces structured chart payloads — plain Python dicts with chart type, axis data, and optional error bars or color channels. These payloads are plotting-library-agnostic: you can render them with Plotly, matplotlib, or pass them to a web frontend.

This decouples analysis from visualization: the feature_diagnostics module computes importance scores and calls viz internally to produce payloads, which you can render however you prefer. The pattern keeps the core modules free of plotting dependencies.`,
    whenToUse: `Use viz payloads when you want structured chart data from research outputs. Most diagnostic modules (feature_diagnostics, pipeline) already call viz internally and include payloads in their return dicts.

**Alternatives**: Build charts directly from DataFrames if you prefer a specific plotting library's API.`,
    relatedModules: ["feature-diagnostics", "pipeline"],
    apiSurface: "python-only",
    pythonApis: ["viz.prepare_feature_importance_payload", "viz.prepare_feature_importance_comparison_payload", "viz.prepare_drawdown_payload", "viz.prepare_regime_payload", "viz.prepare_frontier_payload", "viz.prepare_cluster_payload"],
  },
];
