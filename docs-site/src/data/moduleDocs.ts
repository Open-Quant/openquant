export type Formula = {
  label: string;
  latex: string;
};

export type ExampleBlock = {
  title: string;
  language: "rust" | "bash" | "python";
  code: string;
};

export type ParameterDoc = {
  name: string;
  type: string;
  description: string;
  default?: string;
};

export type ModuleDoc = {
  slug: string;
  module: string;
  subject: string;
  summary: string;
  whyItExists: string;
  keyApis: string[];
  formulas: Formula[];
  examples: ExampleBlock[];
  notes: string[];
  conceptOverview?: string;
  whenToUse?: string;
  keyParameters?: ParameterDoc[];
  commonPitfalls?: string[];
  relatedModules?: string[];
  afmlChapters?: number[];
  pythonApis?: string[];
  apiSurface?: "rust-only" | "python-only" | "both";
};

export const moduleDocs: ModuleDoc[] = [
  {
    slug: "backtest-statistics",
    module: "backtest_statistics",
    subject: "Portfolio Construction and Risk",
    summary: "Performance diagnostics for strategy returns and position trajectories.",
    whyItExists: "Turns raw PnL/returns into risk-adjusted diagnostics used in model selection and production monitoring.",
    keyApis: [
      "sharpe_ratio",
      "deflated_sharpe_ratio",
      "probabilistic_sharpe_ratio",
      "drawdown_and_time_under_water",
      "average_holding_period",
    ],
    formulas: [
      { label: "Sharpe", latex: "S=\\frac{\\mu-r_f}{\\sigma}" },
      { label: "Information Ratio", latex: "IR=\\frac{\\mu-r_b}{\\sigma_{(r-r_b)}}" },
    ],
    examples: [
      {
        title: "Compute Sharpe and drawdown",
        language: "rust",
        code: `use openquant::backtest_statistics::{sharpe_ratio, drawdown_and_time_under_water};\n\nlet returns = vec![0.01, -0.005, 0.007, -0.002, 0.003];\nlet sr = sharpe_ratio(&returns, 252.0, 0.0);\nlet (dd, tuw) = drawdown_and_time_under_water(&returns);\nprintln!("{sr} {dd:?} {tuw:?}");`,
      },
    ],
    notes: [
      "Use annualization constants consistent with your bar frequency.",
      "Deflated Sharpe is useful when strategy mining many variants.",
    ],
    apiSurface: "both",
    pythonApis: ["backtest_stats.sharpe_ratio", "backtest_stats.information_ratio", "backtest_stats.probabilistic_sharpe_ratio", "backtest_stats.deflated_sharpe_ratio", "backtest_stats.minimum_track_record_length", "backtest_stats.timing_of_flattening_and_flips", "backtest_stats.average_holding_period", "backtest_stats.bets_concentration", "backtest_stats.all_bets_concentration", "backtest_stats.drawdown_and_time_under_water"],
  },
  {
    slug: "backtesting-engine",
    module: "backtesting_engine",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Backtesting core with walk-forward, purged CV, and combinatorial purged CV (CPCV) workflows.",
    whyItExists:
      "AFML Chapters 11-12 require scenario-based validation with explicit anti-leakage controls, split provenance, and path-wise uncertainty rather than single-score reporting.",
    keyApis: [
      "run_walk_forward",
      "run_cross_validation",
      "run_cpcv",
      "cpcv_path_count",
      "BacktestRunConfig",
      "BacktestSafeguards",
      "WalkForwardConfig",
      "CrossValidationConfig",
      "CpcvConfig",
    ],
    formulas: [
      {
        label: "CPCV Path Count",
        latex: "\\phi[N,k]=\\binom{N}{k}\\frac{k}{N}=\\binom{N-1}{k-1}",
      },
      {
        label: "Purge + Embargo Train Set",
        latex:
          "\\mathcal T_{train}^{*}=\\mathcal T_{train}\\setminus\\{i: \\exists j\\in\\mathcal T_{test},\\;I_i\\cap I_j\\neq\\varnothing\\}\\setminus\\mathcal E(\\mathcal T_{test},p)",
      },
      {
        label: "Per-Path Sharpe",
        latex: "S_{path}=\\frac{\\bar r_{path}}{\\sigma_{path}}\\sqrt{T_{path}}",
      },
    ],
    examples: [
      {
        title: "Run CPCV and inspect Sharpe distribution",
        language: "rust",
        code: `use openquant::backtesting_engine::{\n  run_cpcv, BacktestData, BacktestRunConfig, BacktestSafeguards, CpcvConfig,\n};\n\nlet result = run_cpcv(\n  &data,\n  &BacktestRunConfig {\n    mode_provenance: \"research_v3_with_costs\".to_string(),\n    trials_count: 24,\n    safeguards: BacktestSafeguards {\n      survivorship_bias_control: \"point-in-time universe\".to_string(),\n      look_ahead_control: \"lagged features\".to_string(),\n      data_mining_control: \"frozen split protocol\".to_string(),\n      cost_assumption: \"spread + slippage\".to_string(),\n      multiple_testing_control: \"trial count logged\".to_string(),\n    },\n  },\n  &CpcvConfig { n_groups: 8, test_groups: 2, pct_embargo: 0.01 },\n  |split| Ok(split.test_indices.iter().map(|i| pnl[*i]).collect()),\n)?;\n\nprintln!(\"phi = {}\", result.path_count);\nprintln!(\"path sharpe count = {}\", result.path_distribution.len());`,
      },
    ],
    notes: [
      "Chapter 11: a backtest is a scenario sanity check; keep safeguards and assumptions attached to every run.",
      "Chapter 12: compare WF/CV/CPCV results by mode rather than averaging them into one statistic.",
      "CPCV output is a path distribution, enabling robust Sharpe diagnostics (e.g., quantiles) instead of point estimates.",
    ],
  },
  {
    slug: "bet-sizing",
    module: "bet_sizing",
    subject: "Position Sizing and Trade Construction",
    summary: "Transforms model confidence and constraints into executable position sizes.",
    whyItExists: "A model signal is not tradable until converted into bounded, discrete, and risk-aware position sizes.",
    keyApis: [
      "bet_size_probability",
      "bet_size_dynamic",
      "bet_size_budget",
      "bet_size_reserve",
      "bet_size_reserve_full",
      "get_target_pos",
      "limit_price",
    ],
    formulas: [
      {
        label: "From Classification Probability to Signed Bet",
        latex:
          "\\begin{aligned}z_t&=\\frac{p_t-1/K}{\\sqrt{p_t(1-p_t)}}\\\\m_t&=\\operatorname{side}_t\\left(2\\Phi(z_t)-1\\right)\\\\\\tilde m_t&=\\operatorname{clip}_{[-1,1]}\\!\\left(\\Delta\\,\\mathrm{round}\\!\\left(\\frac{m_t}{\\Delta}\\right)\\right)\\end{aligned}",
      },
      {
        label: "Dynamic Position Target and Limit Price",
        latex:
          "\\begin{aligned}w&=\\frac{x^2(1-m^2)}{m^2}\\quad (x=f-m_p)\\\\m(x)&=\\frac{x}{\\sqrt{w+x^2}}\\\\\\text{target}&=\\operatorname{maxPos}\\cdot m(f-m_p)\\\\\\text{limitPrice}&=\\frac{1}{|q^*-q|}\\sum_{j=q}^{q^*}\\operatorname{invPrice}(j)\\end{aligned}",
      },
      {
        label: "Budget and Reserve Concurrency Sizing",
        latex:
          "\\begin{aligned}b_t^{budget}&=\\frac{L_t}{\\max_s L_s}-\\frac{S_t}{\\max_s S_s}\\\\c_t&=L_t-S_t\\\\b_t^{reserve}&=\\frac{F(c_t)-F(0)}{1-F(0)}\\;\\mathbf 1_{c_t\\ge0}+\\frac{F(c_t)-F(0)}{F(0)}\\;\\mathbf 1_{c_t<0}\\end{aligned}",
      },
    ],
    examples: [
      {
        title: "End-to-end: Probability Forecasts -> Discrete Executable Bet Sizes",
        language: "rust",
        code: `use chrono::{Duration, NaiveDateTime};\nuse openquant::bet_sizing::bet_size_probability;\n\n// 1) Build event stream: (start, end, class probability, trade side)\nlet t0 = NaiveDateTime::parse_from_str(\"2024-01-01 09:30:00\", \"%Y-%m-%d %H:%M:%S\")?;\nlet events = vec![\n    (t0, t0 + Duration::minutes(20), 0.56,  1.0),\n    (t0 + Duration::minutes(5), t0 + Duration::minutes(35), 0.62,  1.0),\n    (t0 + Duration::minutes(10), t0 + Duration::minutes(30), 0.48, -1.0),\n    (t0 + Duration::minutes(15), t0 + Duration::minutes(45), 0.67,  1.0),\n];\n\n// 2) Convert probabilities -> signed signal -> discretized size (step=0.1)\nlet sizes = bet_size_probability(&events, 2, 0.1, true);\n\n// 3) sizes are directly executable as timestamped target exposure in [-1, 1]\nassert!(!sizes.is_empty());`,
      },
      {
        title: "End-to-end: Dynamic + Reserve Sizing for Execution and Inventory Control",
        language: "rust",
        code: `use chrono::{Duration, NaiveDateTime};\nuse openquant::bet_sizing::{bet_size_dynamic, bet_size_reserve_full};\n\n// Dynamic sizing inputs (position, max position, market price, forecast price)\nlet pos = vec![0.0, 1.0, 1.0, 2.0, 1.0];\nlet max_pos = vec![10.0; 5];\nlet market = vec![100.0, 100.1, 100.0, 100.2, 100.15];\nlet forecast = vec![100.3, 100.4, 100.2, 100.5, 100.45];\n\nlet dynamic = bet_size_dynamic(&pos, &max_pos, &market, &forecast);\n// tuple: (bet_size, target_position, limit_price)\n\n// Reserve sizing from overlapping long/short events\nlet t0 = NaiveDateTime::parse_from_str(\"2024-01-01 09:30:00\", \"%Y-%m-%d %H:%M:%S\")?;\nlet t1 = vec![\n  (t0, t0 + Duration::minutes(30)),\n  (t0 + Duration::minutes(10), t0 + Duration::minutes(40)),\n  (t0 + Duration::minutes(20), t0 + Duration::minutes(50)),\n];\nlet side = vec![1.0, -1.0, 1.0];\nlet (reserve, fit) = bet_size_reserve_full(&t1, &side, 8, 1e-6, 200, true);\n\nassert_eq!(dynamic.len(), 5);\nassert!(fit.is_some());\nassert!(!reserve.is_empty());`,
      },
    ],
    notes: [
      "Keep sizing logic coupled to latency and fill assumptions; limit price from dynamic sizing is a decision boundary, not a guaranteed fill.",
      "Use reserve sizing when overlapping books or strategy stacking can create hidden gross exposure.",
      "Calibrate step_size to real execution granularity (lots/contracts), not arbitrary decimals.",
    ],
    apiSurface: "both",
    pythonApis: ["bet_sizing.get_signal", "bet_sizing.discrete_signal", "bet_sizing.bet_size", "bet_sizing.bet_size_sigmoid", "bet_sizing.bet_size_power", "bet_sizing.inv_price", "bet_sizing.inv_price_sigmoid", "bet_sizing.inv_price_power", "bet_sizing.get_w", "bet_sizing.get_w_sigmoid", "bet_sizing.get_w_power", "bet_sizing.get_target_pos", "bet_sizing.get_target_pos_sigmoid", "bet_sizing.get_target_pos_power", "bet_sizing.limit_price", "bet_sizing.limit_price_sigmoid", "bet_sizing.limit_price_power", "bet_sizing.avg_active_signals", "bet_sizing.bet_size_dynamic", "bet_sizing.cdf_mixture", "bet_sizing.single_bet_size_mixed", "bet_sizing.get_concurrent_sides", "bet_sizing.bet_size_budget", "bet_sizing.bet_size_probability", "bet_sizing.mp_avg_active_signals", "bet_sizing.bet_size_reserve", "bet_sizing.bet_size_reserve_with_fit", "bet_sizing.bet_size_reserve_full"],
  },
  {
    slug: "cla",
    module: "cla",
    subject: "Portfolio Construction and Risk",
    summary: "Critical Line Algorithm implementation for constrained mean-variance optimization.",
    whyItExists: "CLA solves constrained Markowitz problems efficiently with active-set style line updates.",
    keyApis: ["CLA", "covariance", "ReturnsEstimation"],
    formulas: [
      { label: "MVO Objective", latex: "\\min_w\\;\\frac{1}{2}w^T\\Sigma w-\\lambda\\mu^T w" },
      { label: "Budget Constraint", latex: "\\mathbf{1}^T w=1" },
    ],
    examples: [
      {
        title: "Prepare covariance for CLA",
        language: "rust",
        code: `use nalgebra::DMatrix;\nuse openquant::cla::covariance;\n\nlet returns = DMatrix::from_row_slice(3, 2, &[0.01, 0.02, -0.01, 0.01, 0.015, 0.03]);\nlet sigma = covariance(&returns);`,
      },
    ],
    notes: ["CLA behavior depends on weight bounds and return estimates.", "Use robust covariance estimators when sample size is small."],
    apiSurface: "both",
    pythonApis: ["cla.allocate_cla"],
  },
  {
    slug: "codependence",
    module: "codependence",
    subject: "Market Microstructure, Dependence and Regime Detection",
    summary: "Dependence metrics beyond linear correlation for feature and asset relationships.",
    whyItExists: "Financial relationships are often non-linear and regime-dependent; correlation alone is insufficient.",
    keyApis: ["distance_correlation", "get_mutual_info", "variation_of_information_score", "angular_distance"],
    formulas: [
      { label: "Mutual Information", latex: "I(X;Y)=\\sum_{x,y}p(x,y)\\log\\frac{p(x,y)}{p(x)p(y)}" },
      { label: "Variation of Information", latex: "VI(X,Y)=H(X)+H(Y)-2I(X;Y)" },
    ],
    examples: [
      {
        title: "Distance correlation between series",
        language: "rust",
        code: `use openquant::codependence::distance_correlation;\n\nlet x = vec![1.0, 2.0, 3.0, 4.0];\nlet y = vec![1.1, 1.9, 3.2, 3.8];\nlet dcor = distance_correlation(&x, &y)?;`,
      },
    ],
    notes: ["Use with clustering and feature pruning workflows.", "Bin selection materially impacts MI estimates."],
    apiSurface: "both",
    pythonApis: ["codependence.angular_distance", "codependence.absolute_angular_distance", "codependence.squared_angular_distance", "codependence.distance_correlation", "codependence.get_optimal_number_of_bins", "codependence.get_mutual_info", "codependence.variation_of_information_score"],
  },
  {
    slug: "cross-validation",
    module: "cross_validation",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Purged cross-validation utilities designed for label overlap and leakage control.",
    whyItExists: "Time-dependent labels violate IID assumptions; purging/embargoing reduces leakage bias.",
    keyApis: ["ml_cross_val_score", "ml_get_train_times", "PurgedKFold", "Scoring"],
    formulas: [
      { label: "Purged Train Set", latex: "\\mathcal{T}_{train}=\\mathcal{T}\\setminus(\\mathcal{T}_{test}\\oplus e)" },
      { label: "Embargo", latex: "e=\\lfloor p\\cdot T\\rfloor" },
    ],
    examples: [
      {
        title: "Configure PurgedKFold",
        language: "rust",
        code: `use openquant::cross_validation::PurgedKFold;\n\nlet cv = PurgedKFold::new(5, 0.01);`,
      },
    ],
    notes: ["Always align event end-times when purging.", "Report variance across folds, not only mean score."],
    apiSurface: "rust-only",
  },
  {
    slug: "data-structures",
    module: "data_structures",
    subject: "Event-Driven Data and Labeling",
    summary: "Constructs standard/time/run/imbalance bars from trade streams.",
    whyItExists: "Event-based bars reduce heteroskedasticity and improve stationarity versus fixed-time sampling.",
    keyApis: ["standard_bars", "time_bars", "run_bars", "imbalance_bars", "Trade", "StandardBar", "StandardBarType", "ImbalanceBarType"],
    formulas: [
      { label: "Dollar Bar Trigger", latex: "\\sum_{i=t_0}^{t} p_i v_i \\ge \\theta" },
      { label: "Imbalance Trigger", latex: "\\left|\\sum b_i\\right| \\ge E[|\\sum b_i|]" },
    ],
    examples: [
      {
        title: "Build dollar bars from a Polars DataFrame",
        language: "python",
        code: `from openquant.bars import build_dollar_bars, bar_diagnostics
import polars as pl

# Input: Polars DataFrame with ts, symbol, open, high, low, close, volume columns
df = pl.read_parquet("trades.parquet")

# Dollar bars: each bar aggregates ~$5M of notional
bars = build_dollar_bars(df, dollar_value_per_bar=5_000_000.0)
# Returns: Polars DataFrame with ts, symbol, open, high, low, close, volume, adj_close, start_ts, n_obs, dollar_value

# Check bar quality: low autocorrelation = good
diag = bar_diagnostics(bars)
print(diag)  # {"n_bars": 482.0, "lag1_return_autocorr": -0.02, ...}`,
      },
      {
        title: "Build tick and volume bars",
        language: "python",
        code: `from openquant.bars import build_tick_bars, build_volume_bars, build_time_bars

tick_bars = build_tick_bars(df, ticks_per_bar=50)
vol_bars = build_volume_bars(df, volume_per_bar=100_000.0)
time_bars = build_time_bars(df, interval="5m")`,
      },
      {
        title: "Build bars from Rust",
        language: "rust",
        code: `use chrono::Duration;\nuse openquant::data_structures::{\n    standard_bars, time_bars, run_bars, imbalance_bars,\n    Trade, StandardBarType, ImbalanceBarType,\n};\n\nlet trades: Vec<Trade> = vec![];\n\n// Fixed-time bars\nlet t_bars = time_bars(&trades, Duration::minutes(5));\n\n// Dollar bars via standard_bars\nlet d_bars = standard_bars(&trades, 50_000.0, StandardBarType::Dollar);\n\n// Run bars (Rust-only)\nlet r_bars = run_bars(&trades, 100);\n\n// Tick imbalance bars (Rust-only)\nlet ib = imbalance_bars(&trades, 500.0, ImbalanceBarType::Tick);`,
      },
    ],
    notes: [
      "Threshold selection controls bar frequency and noise level.",
      "Keep OHLCV semantics consistent across downstream features.",
      "Run bars and imbalance bars are available via bars.build_run_bars and bars.build_imbalance_bars.",
      "`bar_diagnostics` is Python-only; use it to verify low return autocorrelation after bar construction.",
    ],
    conceptOverview: `Traditional financial data uses fixed-time bars (1-minute, daily), but these sample uniformly regardless of market activity. During quiet periods you get noise; during volatile periods you under-sample important information.

Information-driven bars (AFML Chapter 2) sample based on market activity instead of clock time. **Dollar bars** trigger a new bar when cumulative traded dollar volume reaches a threshold, producing roughly equal-information observations. **Volume bars** trigger on cumulative share volume. **Tick bars** trigger on trade count.

**Imbalance bars** go further: they detect when the net signed trade flow (buy minus sell) exceeds its expected magnitude, capturing points where informed trading pressure shifts. **Run bars** detect runs of same-signed trades exceeding expectations.

The key insight is that information-driven bars produce returns that are closer to IID normal, which makes downstream ML models (labeling, feature importance, cross-validation) better behaved. All AFML workflows assume information-driven bars as input.`,
    whenToUse: `This is the first module in any AFML pipeline. Raw tick or trade data goes in; structured OHLCV bars come out. Everything downstream — labeling, features, sampling — consumes these bars.

**Prerequisites**: Raw trade or tick data with timestamps, prices, and volumes.

**Alternatives**: Standard time bars if your data is already aggregated. For pre-aggregated OHLCV data, use the \`data\` module's \`load_ohlcv\` and \`clean_ohlcv\` functions instead.`,
    keyParameters: [
      { name: "dollar_value_per_bar", type: "float", description: "Dollar notional threshold for dollar bars (Python)", default: "5_000_000.0" },
      { name: "volume_per_bar", type: "float", description: "Cumulative volume threshold for volume bars (Python)", default: "100_000.0" },
      { name: "ticks_per_bar", type: "int", description: "Trade count threshold for tick bars (Python)", default: "50" },
      { name: "interval", type: "str", description: "Time interval for time bars, e.g. '1d', '5m', '1h' (Python)", default: "'1d'" },
      { name: "threshold", type: "f64", description: "Bar trigger threshold for standard_bars, run_bars, imbalance_bars (Rust)", default: "—" },
      { name: "bar_type", type: "StandardBarType", description: "Tick, Volume, or Dollar — selects accumulation metric (Rust)", default: "—" },
    ],
    commonPitfalls: [
      "Using time bars when your data has highly variable activity — dollar or volume bars will produce more stationary returns.",
      "Setting the threshold too low, creating extremely noisy high-frequency bars, or too high, losing intraday resolution.",
      "Forgetting to assign trade direction (buy/sell sign) before constructing imbalance or run bars — these require signed volume.",
      "Mixing bar types across train and inference: if you train on dollar bars, your live pipeline must also use dollar bars with the same threshold.",
      "Run bars and imbalance bars are available in Python via bars.build_run_bars and bars.build_imbalance_bars.",
    ],
    relatedModules: ["filters", "labeling", "fracdiff"],
    afmlChapters: [2],
    apiSurface: "both",
    pythonApis: ["bars.build_time_bars", "bars.build_tick_bars", "bars.build_volume_bars", "bars.build_dollar_bars", "bars.build_run_bars", "bars.build_imbalance_bars"],
  },
  {
    slug: "hyperparameter-tuning",
    module: "hyperparameter_tuning",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Leakage-aware grid/randomized hyper-parameter search with purged CV and weighted scoring.",
    whyItExists:
      "AFML Chapter 9 recommends tuning under PurgedKFold, using randomized search for large spaces, and scoring with metrics aligned to trading objectives.",
    keyApis: [
      "grid_search",
      "randomized_search",
      "expand_param_grid",
      "sample_log_uniform",
      "classification_score",
      "SearchScoring",
      "RandomParamDistribution",
    ],
    formulas: [
      {
        label: "Purged CV Objective",
        latex: "\\hat\\theta=\\arg\\max_{\\theta\\in\\Theta}\\frac{1}{K}\\sum_{k=1}^{K}\\mathrm{Score}(f_\\theta,\\mathcal T_k^{train},\\mathcal T_k^{test})",
      },
      {
        label: "Log-Uniform Draw",
        latex: "\\log x\\sim U(\\log a,\\log b),\\; a>0,\\;x\\in(a,b)",
      },
      {
        label: "Weighted Neg Log Loss",
        latex: "-\\frac{1}{\\sum_i w_i}\\sum_i w_i\\left[y_i\\log p_i + (1-y_i)\\log(1-p_i)\\right]",
      },
    ],
    examples: [
      {
        title: "Randomized search with PurgedKFold semantics",
        language: "rust",
        code: `use std::collections::BTreeMap;\nuse openquant::hyperparameter_tuning::{\n  randomized_search, RandomParamDistribution, SearchData, SearchScoring,\n};\n\nlet mut space = BTreeMap::new();\nspace.insert(\"C\".to_string(), RandomParamDistribution::LogUniform { low: 1e-2, high: 1e2 });\nspace.insert(\"gamma\".to_string(), RandomParamDistribution::LogUniform { low: 1e-3, high: 1e1 });\n\nlet result = randomized_search(\n  build_model,\n  &space,\n  25,\n  42,\n  SearchData { x: &x, y: &y, sample_weight: Some(&w), samples_info_sets: &info_sets },\n  5,\n  0.01,\n  SearchScoring::NegLogLoss,\n)?;\nprintln!(\"best score = {}\", result.best_score);`,
      },
    ],
    notes: [
      "Use Accuracy only when each prediction has similar economic value (equal bet sizing).",
      "Prefer weighted NegLogLoss when probabilities drive position sizing or outcomes have different economic magnitude.",
      "BalancedAccuracy is useful for severe class imbalance, especially in meta-labeling where recall of positives matters.",
    ],
    apiSurface: "rust-only",
  },
  {
    slug: "ef3m",
    module: "ef3m",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Moment-based mixture fitting utilities for two-normal components.",
    whyItExists: "Provides robust parameter estimation for bimodal return mixtures when full MLE is heavy.",
    keyApis: ["M2N", "centered_moment", "raw_moment", "most_likely_parameters"],
    formulas: [
      { label: "Raw Moment", latex: "m_k=E[X^k]" },
      { label: "Mixture Mean", latex: "\\mu=p\\mu_1+(1-p)\\mu_2" },
    ],
    examples: [
      {
        title: "Estimate moments",
        language: "rust",
        code: `use openquant::ef3m::centered_moment;\n\nlet moments = vec![0.0, 1.0, 0.1, 3.0];\nlet m3 = centered_moment(&moments, 3);`,
      },
    ],
    notes: ["Use as initialization for more expensive optimizers.", "Sensitive to higher-moment estimation noise."],
    apiSurface: "both",
    pythonApis: ["ef3m.centered_moment", "ef3m.raw_moment", "ef3m.most_likely_parameters", "ef3m.fit_m2n"],
  },
  {
    slug: "ensemble-methods",
    module: "ensemble_methods",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Bias/variance diagnostics and practical bagging-vs-boosting ensemble utilities.",
    whyItExists:
      "AFML Chapter 6 emphasizes that ensemble gains depend on error decomposition and forecast dependence, not just estimator count.",
    keyApis: [
      "bias_variance_noise",
      "bootstrap_sample_indices",
      "sequential_bootstrap_sample_indices",
      "aggregate_classification_vote",
      "aggregate_classification_probability_mean",
      "average_pairwise_prediction_correlation",
      "bagging_ensemble_variance",
      "recommend_bagging_vs_boosting",
    ],
    formulas: [
      {
        label: "Error Decomposition",
        latex: "\\operatorname{MSE}=\\operatorname{Bias}^2+\\operatorname{Var}+\\operatorname{Noise}",
      },
      {
        label: "Bagging Variance Under Average Correlation",
        latex: "\\sigma^2_{bag}=\\sigma^2\\left(\\rho+\\frac{1-\\rho}{N}\\right)",
      },
      {
        label: "Majority Vote and Mean Probability",
        latex:
          "\\hat y=\\mathbf 1\\left(\\frac{1}{N}\\sum_{m=1}^N \\hat p_m \\ge \\tau\\right),\\quad \\hat p=\\frac{1}{N}\\sum_{m=1}^N \\hat p_m",
      },
    ],
    examples: [
      {
        title: "Assess Ensemble Variance and Recommendation",
        language: "rust",
        code: `use openquant::ensemble_methods::{\n  average_pairwise_prediction_correlation,\n  bagging_ensemble_variance,\n  recommend_bagging_vs_boosting,\n};\n\nlet preds = vec![\n  vec![0.51, 0.49, 0.52, 0.50],\n  vec![0.50, 0.48, 0.53, 0.49],\n  vec![0.52, 0.50, 0.51, 0.50],\n];\n\nlet rho = average_pairwise_prediction_correlation(&preds)?;\nlet bag_var = bagging_ensemble_variance(1.0, rho, 20)?;\nlet decision = recommend_bagging_vs_boosting(0.54, rho, 0.75, 1.0, 20)?;\n\nprintln!(\"rho={rho:.3}, var={bag_var:.3}, rec={:?}\", decision.recommended);`,
      },
      {
        title: "Aggregate Bagged Classifier Outputs",
        language: "rust",
        code: `use openquant::ensemble_methods::{\n  aggregate_classification_vote,\n  aggregate_classification_probability_mean,\n};\n\nlet vote = aggregate_classification_vote(&[\n  vec![1, 0, 1],\n  vec![1, 1, 0],\n  vec![0, 1, 1],\n])?;\n\nlet (mean_prob, labels) = aggregate_classification_probability_mean(&[\n  vec![0.9, 0.2, 0.6],\n  vec![0.8, 0.3, 0.5],\n  vec![0.7, 0.4, 0.4],\n], 0.5)?;\n\nassert_eq!(vote, vec![1, 1, 1]);\nassert_eq!(labels, vec![1, 0, 1]);\nassert_eq!(mean_prob.len(), 3);`,
      },
    ],
    notes: [
      "If base learners are highly correlated, bagging variance reduction is minimal even with many estimators.",
      "Sequential-bootstrap-style sampling is preferable under heavy label overlap and non-IID observations.",
      "Boosting is usually preferable for weak learners (bias reduction); bagging is usually preferable for unstable learners (variance reduction).",
    ],
    apiSurface: "both",
    pythonApis: ["ensemble.bias_variance_noise", "ensemble.bootstrap_sample_indices", "ensemble.sequential_bootstrap_sample_indices", "ensemble.aggregate_regression_mean", "ensemble.aggregate_classification_vote", "ensemble.aggregate_classification_probability_mean", "ensemble.average_pairwise_prediction_correlation", "ensemble.bagging_ensemble_variance", "ensemble.recommend_bagging_vs_boosting"],
  },
  {
    slug: "etf-trick",
    module: "etf_trick",
    subject: "Position Sizing and Trade Construction",
    summary: "Synthetic ETF and futures roll utilities for realistic PnL path construction.",
    whyItExists: "Backtests must include financing, carry, and contract-roll mechanics to avoid optimistic bias.",
    keyApis: ["EtfTrick", "EtfTrick::from_tables", "EtfTrick::from_csv", "EtfTrick::get_etf_series", "get_futures_roll_series", "FuturesRollRow", "Table"],
    formulas: [
      { label: "ETF NAV Update", latex: "NAV_t=NAV_{t-1}(1+r_t-c_t)" },
      { label: "Roll Return", latex: "r^{roll}_t=\\frac{F^{near}_t-F^{far}_t}{F^{far}_t}" },
    ],
    examples: [
      {
        title: "Construct synthetic ETF series",
        language: "rust",
        code: `use openquant::etf_trick::{EtfTrick, Table};\n\n// Load open/close/allocation/cost tables from CSV\nlet etf = EtfTrick::from_csv(\n    "open.csv", "close.csv", "alloc.csv", "costs.csv", Some("rates.csv"),\n).unwrap();\n\n// Generate synthetic ETF NAV series\nlet series = etf.get_etf_series(252).unwrap();\n// Returns Vec<(date_string, nav_value)>`,
      },
      {
        title: "Compute futures roll-adjusted series",
        language: "rust",
        code: `use openquant::etf_trick::{get_futures_roll_series, FuturesRollRow};\n\nlet rows: Vec<FuturesRollRow> = vec![/* ... */];\nlet adjusted = get_futures_roll_series(&rows, "backward", true).unwrap();`,
      },
    ],
    notes: [
      "Verify contract calendar assumptions.",
      "Costs and rates should come from the same clock as price data.",
      "This module is Rust-only — no Python bindings are currently exposed.",
    ],
    apiSurface: "rust-only",
  },
  {
    slug: "feature-importance",
    module: "feature_importance",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Feature ranking methods: MDI, MDA, and single-feature importance with PCA diagnostics.",
    whyItExists: "Improves model interpretability and helps remove unstable or redundant features.",
    keyApis: ["mean_decrease_impurity", "mean_decrease_accuracy", "single_feature_importance", "feature_pca_analysis"],
    formulas: [
      { label: "MDI", latex: "I_j=\\sum_{t\\in T_j} p(t)\\Delta i(t)" },
      { label: "MDA", latex: "I_j=Score(X)-Score(X_{perm(j)})" },
    ],
    examples: [
      {
        title: "Run MDA with classifier",
        language: "rust",
        code: `use openquant::feature_importance::mean_decrease_accuracy;\n\n// Plug in your classifier implementing SimpleClassifier\nlet importance = mean_decrease_accuracy(&clf, &x, &y, 5)?;`,
      },
    ],
    notes: ["Cross-validated MDA is preferred when leakage risk is high.", "Compare ranking stability across folds/time windows."],
    apiSurface: "rust-only",
  },
  {
    slug: "filters",
    module: "filters",
    subject: "Event-Driven Data and Labeling",
    summary: "CUSUM and z-score event filters for event-driven sampling.",
    whyItExists: "Extracts informative events from noisy high-frequency sequences.",
    keyApis: ["cusum_filter_indices", "cusum_filter_timestamps", "cusum_filter_indices_checked", "cusum_filter_timestamps_checked", "z_score_filter_indices", "z_score_filter_timestamps", "z_score_filter_timestamps_checked", "Threshold", "FilterError"],
    formulas: [
      { label: "CUSUM", latex: "S_t=\\max(0, S_{t-1}+r_t),\\; trigger\\;if\\;|S_t|>h" },
      { label: "Z-score", latex: "z_t=\\frac{x_t-\\mu_t}{\\sigma_t}" },
    ],
    examples: [
      {
        title: "CUSUM and z-score event detection",
        language: "python",
        code: `import openquant

close = [100.0, 100.1, 99.9, 100.2, 100.05, 100.3, 99.7, 100.1]
timestamps = [
    "2024-01-02T09:30:00", "2024-01-02T09:31:00",
    "2024-01-02T09:32:00", "2024-01-02T09:33:00",
    "2024-01-02T09:34:00", "2024-01-02T09:35:00",
    "2024-01-02T09:36:00", "2024-01-02T09:37:00",
]

# CUSUM filter: fires when cumulative deviation exceeds threshold
event_indices = openquant.filters.cusum_filter_indices(close, 0.02)

# With timestamps: returns event timestamps directly
event_ts = openquant.filters.cusum_filter_timestamps(close, timestamps, 0.02)

# Z-score filter: fires when z-score exceeds threshold
z_indices = openquant.filters.z_score_filter_indices(close, mean_window=20, std_window=20, threshold=2.0)
z_ts = openquant.filters.z_score_filter_timestamps(close, timestamps, mean_window=20, std_window=20, threshold=2.0)`,
      },
      {
        title: "CUSUM with static and dynamic thresholds",
        language: "rust",
        code: `use openquant::filters::{cusum_filter_indices, cusum_filter_indices_checked, Threshold};\n\nlet close = vec![100.0, 100.1, 99.9, 100.2];\n\n// Static threshold\nlet idx = cusum_filter_indices(&close, Threshold::Scalar(0.02));\n\n// Dynamic threshold (e.g. volatility-scaled per bar)\nlet dynamic_h = vec![0.02, 0.025, 0.018, 0.022];\nlet idx = cusum_filter_indices_checked(&close, Threshold::Dynamic(dynamic_h)).unwrap();`,
      },
    ],
    notes: [
      "Calibrate thresholds to target event frequency, not just sensitivity.",
      "Use identical filtering in train and live pipelines.",
      "Rust API supports dynamic (per-bar) thresholds via Threshold::Dynamic; Python bindings accept only a scalar threshold.",
      "Rust _checked variants return Result<..., FilterError> for input validation; Python raises exceptions.",
    ],
    conceptOverview: `Instead of sampling at fixed intervals, AFML Chapter 2 uses structural event filters to detect when something meaningful happens in the price process. This produces training examples that correspond to real market inflection points rather than arbitrary calendar dates.

The **CUSUM filter** tracks a cumulative sum of returns (or price changes). It resets to zero when the cumulative deviation exceeds a threshold h, and the reset point becomes an event. This captures points where the price has moved "enough" since the last event. The filter is directional: it tracks both positive and negative cumulative deviations separately.

The **z-score filter** standardizes the current value against a rolling mean and standard deviation, firing when the z-score exceeds a threshold. This is useful for mean-reverting signals where you want events when the price deviates significantly from its recent average.

Both filters replace the naive approach of labeling every bar, which creates highly correlated and redundant training examples.`,
    whenToUse: `Apply event filters immediately after bar construction and before labeling. They bridge raw bars to the labeling module: bars go in, event timestamps come out.

**Prerequisites**: A price series (close prices from bars), and optionally timestamps.

**Alternatives**: Fixed-interval sampling (simpler but creates redundant events), or custom event logic for strategy-specific triggers.`,
    keyParameters: [
      { name: "close", type: "list[float]", description: "Input price series (close prices)", default: "—" },
      { name: "threshold", type: "float", description: "CUSUM trigger level; controls event frequency (Python: scalar only)", default: "—" },
      { name: "threshold", type: "Threshold", description: "CUSUM trigger: Threshold::Scalar(f64) or Threshold::Dynamic(Vec<f64>) (Rust)", default: "—" },
      { name: "mean_window", type: "int", description: "Rolling mean lookback for z-score filter", default: "—" },
      { name: "std_window", type: "int", description: "Rolling std lookback for z-score filter", default: "—" },
      { name: "timestamps", type: "list[str]", description: "Optional timestamps; use _timestamps variants to get event times instead of indices", default: "—" },
    ],
    commonPitfalls: [
      "Setting the CUSUM threshold too tight in volatile regimes — you get too many events and labels become noisy. Scale h by recent volatility.",
      "Using different thresholds in training vs live inference — the event distribution shifts and the model sees a different regime.",
      "Applying CUSUM to non-stationary raw prices instead of returns or log-returns — the filter becomes meaningless as the price drifts.",
      "Python bindings only support scalar thresholds — use the Rust API directly if you need dynamic (per-bar) thresholds.",
    ],
    relatedModules: ["data-structures", "labeling", "sample-weights"],
    afmlChapters: [2],
    apiSurface: "both",
    pythonApis: ["filters.cusum_filter_indices", "filters.cusum_filter_timestamps", "filters.z_score_filter_indices", "filters.z_score_filter_timestamps"],
  },
  {
    slug: "fingerprint",
    module: "fingerprint",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Model fingerprinting for linear, non-linear, and pairwise feature effects.",
    whyItExists: "Quantifies behavior of fitted models beyond scalar accuracy metrics.",
    keyApis: ["RegressionModelFingerprint", "ClassificationModelFingerprint", "Effect", "PairwiseEffect"],
    formulas: [
      { label: "Partial Effect", latex: "f_j(x_j)=E_{X_{-j}}[f(X)|X_j=x_j]" },
      { label: "Pairwise Interaction", latex: "I_{ij}=f(x_i,x_j)-f_i(x_i)-f_j(x_j)" },
    ],
    examples: [
      {
        title: "Create regression fingerprint",
        language: "rust",
        code: `use openquant::fingerprint::RegressionModelFingerprint;\n\nlet fp = RegressionModelFingerprint::new(&model, &x);\nlet effects = fp.linear_effects()?;`,
      },
    ],
    notes: ["Compare fingerprints across retrains for drift detection.", "Use pairwise effects to detect hidden interaction risk."],
    apiSurface: "rust-only",
  },
  {
    slug: "fracdiff",
    module: "fracdiff",
    subject: "Market Microstructure, Dependence and Regime Detection",
    summary: "Fractional differentiation to improve stationarity while retaining memory.",
    whyItExists: "Balances stationarity and predictive memory better than integer differencing.",
    keyApis: ["get_weights", "get_weights_ffd", "frac_diff", "frac_diff_ffd"],
    formulas: [
      { label: "FFD Weights", latex: "w_k = -w_{k-1}\\frac{d-k+1}{k}" },
      { label: "Fractional Difference", latex: "y_t=\\sum_{k=0}^{\\infty}w_k x_{t-k}" },
    ],
    examples: [
      {
        title: "Fractionally differentiate a price series",
        language: "python",
        code: `from openquant._core import fracdiff

prices = [100.0, 100.2, 100.1, 100.4, 100.6, 100.3, 100.8]

# Fixed-window fractional differentiation (d=0.4, threshold=1e-4)
stationary = fracdiff.frac_diff_ffd(prices, 0.4, 1e-4)

# Inspect the FFD weights to understand memory retention
weights = fracdiff.get_weights_ffd(0.4, 1e-4, len(prices))`,
      },
      {
        title: "Compute fixed-width fracdiff",
        language: "rust",
        code: `use openquant::fracdiff::frac_diff_ffd;\n\nlet series = vec![100.0, 100.2, 100.1, 100.4, 100.6];\nlet out = frac_diff_ffd(&series, 0.4, 1e-4);`,
      },
    ],
    notes: ["Tune d using stationarity tests and information retention.", "Threshold governs truncation error vs compute cost."],
    conceptOverview: `Financial time series like prices are non-stationary — their statistical properties drift over time. Standard integer differencing (d=1, i.e., returns) makes the series stationary but destroys long-range memory that carries predictive signal.

Fractional differentiation (AFML Chapter 5) generalizes differencing to real-valued orders 0 < d < 1. A fractional difference applies an infinite series of weights to past observations, where the weights decay polynomially. At d=0 you have the raw price (full memory, non-stationary). At d=1 you have returns (stationary, no memory). The goal is to find the minimum d that passes stationarity tests (e.g., ADF) while preserving as much memory as possible.

The **fixed-width window (FFD)** variant truncates the weight series once weights fall below a threshold, making computation practical for long series. This is the recommended approach for production use.`,
    whenToUse: `Apply fractional differentiation to price or spread series *before* feature engineering. It replaces raw returns as the base transformation when you need stationarity without discarding mean-reversion or trend memory.

**Prerequisites**: A price series (close prices or mid-prices). Optionally, an ADF test loop to find the optimal d.

**Alternatives**: Standard returns (d=1) if stationarity is sufficient and memory isn't needed. Log prices if your downstream model handles non-stationarity.`,
    keyParameters: [
      { name: "d", type: "f64", description: "Fractional differencing order; 0 = raw prices, 1 = returns", default: "—" },
      { name: "threshold", type: "f64", description: "Minimum absolute weight for FFD truncation; smaller = longer memory window, more compute", default: "1e-4" },
    ],
    commonPitfalls: [
      "Using d=1 by default (standard returns) when the series has exploitable long-memory — run a d-search with ADF first.",
      "Setting threshold too large, which truncates weights aggressively and makes FFD behave like integer differencing.",
      "Applying fracdiff to already-differenced data — check whether your input is prices or returns.",
      "Forgetting that the first few observations are NaN/unreliable due to insufficient weight history — trim them before feeding into ML.",
    ],
    relatedModules: ["data-structures", "filters"],
    afmlChapters: [5],
    apiSurface: "both",
    pythonApis: ["fracdiff.get_weights", "fracdiff.get_weights_ffd", "fracdiff.frac_diff", "fracdiff.frac_diff_ffd"],
  },
  {
    slug: "hcaa",
    module: "hcaa",
    subject: "Portfolio Construction and Risk",
    summary: "Hierarchical Clustering Asset Allocation variant with cluster-level constraints.",
    whyItExists: "Allocates capital by hierarchy to reduce concentration and covariance-estimation fragility.",
    keyApis: ["HierarchicalClusteringAssetAllocation", "HcaaError"],
    formulas: [
      { label: "Cluster Variance", latex: "\\sigma_C^2=w_C^T\\Sigma_C w_C" },
      { label: "Recursive Split", latex: "w_{left},w_{right}\\propto\\frac{1}{\\sigma_{left}^2},\\frac{1}{\\sigma_{right}^2}" },
    ],
    examples: [
      {
        title: "Fit HCAA allocator",
        language: "rust",
        code: `use openquant::hcaa::HierarchicalClusteringAssetAllocation;\n\nlet mut hcaa = HierarchicalClusteringAssetAllocation::new();\nlet w = hcaa.allocate(&prices)?;`,
      },
    ],
    notes: ["Cluster linkage choices influence allocations.", "Use with robust codependence distances when possible."],
    apiSurface: "both",
    pythonApis: ["hcaa.allocate_hcaa"],
  },
  {
    slug: "hrp",
    module: "hrp",
    subject: "Portfolio Construction and Risk",
    summary: "Hierarchical Risk Parity allocation with recursive bisection.",
    whyItExists: "Produces stable allocations without matrix inversion required by classic Markowitz.",
    keyApis: ["HierarchicalRiskParity", "HrpDendrogram"],
    formulas: [
      { label: "IVP Weight", latex: "w_i\\propto\\frac{1}{\\sigma_i^2}" },
      { label: "Bisection Split", latex: "\\alpha=1-\\frac{\\sigma_{left}^2}{\\sigma_{left}^2+\\sigma_{right}^2}" },
    ],
    examples: [
      {
        title: "Allocate with HRP",
        language: "rust",
        code: `use openquant::hrp::HierarchicalRiskParity;\n\nlet mut hrp = HierarchicalRiskParity::new();\nlet weights = hrp.allocate(&prices)?;`,
      },
    ],
    notes: ["HRP is often more robust under unstable covariance estimates.", "Ensure input asset order tracks produced dendrogram order."],
    apiSurface: "both",
    pythonApis: ["hrp.allocate_hrp"],
  },
  {
    slug: "labeling",
    module: "labeling",
    subject: "Event-Driven Data and Labeling",
    summary: "Triple-barrier event labeling and metadata generation.",
    whyItExists: "Converts event outcomes into ML labels with controlled horizon and risk barriers.",
    keyApis: ["add_vertical_barrier", "get_events", "get_bins", "drop_labels", "Event"],
    formulas: [
      {
        label: "Triple-Barrier Event Time",
        latex:
          "\\tau=\\min\\left(\\tau_{pt},\\tau_{sl},t_1\\right),\\quad\\tau_{pt}=\\inf\\{u>t:r_{t,u}\\ge pt\\cdot\\sigma_t\\},\\quad\\tau_{sl}=\\inf\\{u>t:r_{t,u}\\le-sl\\cdot\\sigma_t\\}",
      },
      {
        label: "Labeling Rule",
        latex:
          "y_t=\\begin{cases}1,&r_{t,\\tau}>0\\\\0,&r_{t,\\tau}=0\\\\-1,&r_{t,\\tau}<0\\end{cases},\\qquad\\text{meta label: }y_t^{meta}=\\mathbf 1\\{\\operatorname{side}_t\\cdot r_{t,\\tau}>0\\}",
      },
      {
        label: "Target Volatility Scaling",
        latex:
          "\\sigma_t=\\operatorname{EWMA}\\big(|r_t|\\big),\\qquad\\text{barrier widths }\\propto \\sigma_t",
      },
    ],
    examples: [
      {
        title: "Triple-barrier labels from price series",
        language: "python",
        code: `from openquant._core import labeling, filters

# 1) Detect events with CUSUM filter
timestamps = ["2024-01-01T09:30:00", "2024-01-01T09:31:00", ...]
close = [100.0, 100.1, 99.9, 100.2, 100.05, 100.3, ...]
event_ts = filters.cusum_filter_timestamps(close, timestamps, 0.02)

# 2) Estimate target volatility (use your own EWMA or rolling std)
target_ts = event_ts
target_vals = [0.02] * len(event_ts)  # simplified constant target

# 3) Compute triple-barrier labels
labels = labeling.triple_barrier_labels(
    close_timestamps=timestamps,
    close_prices=close,
    t_events=event_ts,
    target_timestamps=target_ts,
    target_values=target_vals,
    pt=1.0, sl=1.0, min_ret=0.005,
)
# Each label: (event_ts, return, target, label_int, touch_ts)`,
      },
      {
        title: "Meta-labeling: learn when to act on a primary signal",
        language: "python",
        code: `from openquant._core import labeling

# Primary model gives side predictions (+1 or -1) at each event
side_prediction = [1.0, -1.0, 1.0, 1.0, -1.0, ...]

meta_labels = labeling.meta_labels(
    close_timestamps=timestamps,
    close_prices=close,
    t_events=event_ts,
    target_timestamps=target_ts,
    target_values=target_vals,
    side_prediction=side_prediction,
    pt=1.0, sl=1.0, min_ret=0.005,
)
# Train a secondary classifier on meta_labels to filter false signals`,
      },
      {
        title: "End-to-end: Event Filter -> Vertical Barrier -> Triple Barrier Labels",
        language: "rust",
        code: `use chrono::NaiveDateTime;\nuse openquant::filters::{cusum_filter_timestamps, Threshold};\nuse openquant::labeling::{add_vertical_barrier, get_events, get_bins};\nuse openquant::util::volatility::get_daily_vol;\n\n// 1) price series and timestamps\nlet close: Vec<(NaiveDateTime, f64)> = /* load bars */ vec![];\nlet prices: Vec<f64> = close.iter().map(|(_, p)| *p).collect();\nlet ts: Vec<NaiveDateTime> = close.iter().map(|(t, _)| *t).collect();\n\n// 2) detect candidate events via CUSUM filter\nlet events = cusum_filter_timestamps(&prices, &ts, Threshold::Scalar(0.02));\n\n// 3) estimate target volatility and add max-holding horizon\nlet target = get_daily_vol(&close, 100);\nlet vbars = add_vertical_barrier(&events, &close, 1, 0, 0, 0);\n\n// 4) compute barrier touches and labels\nlet ev = get_events(&close, &events, (1.0, 1.0), &target, 0.005, 3, Some(&vbars), None);\nlet bins = get_bins(&ev, &close);\nassert!(!bins.is_empty());`,
      },
      {
        title: "Meta-Labeling Workflow with Side Signal",
        language: "rust",
        code: `use chrono::NaiveDateTime;\nuse openquant::labeling::{get_events, get_bins};\n\nlet close: Vec<(NaiveDateTime, f64)> = /* bars */ vec![];\nlet events: Vec<NaiveDateTime> = /* primary event timestamps */ vec![];\nlet target: Vec<(NaiveDateTime, f64)> = /* vol target */ vec![];\nlet vbars: Vec<(NaiveDateTime, NaiveDateTime)> = /* horizon */ vec![];\n\n// Primary model side forecast (+1 / -1)\nlet side: Vec<(NaiveDateTime, f64)> = events.iter().map(|t| (*t, 1.0)).collect();\n\nlet meta_events = get_events(\n    &close,\n    &events,\n    (1.0, 1.0),\n    &target,\n    0.005,\n    3,\n    Some(&vbars),\n    Some(&side),\n);\nlet meta_bins = get_bins(&meta_events, &close);\n// Use meta_bins to train a second-stage filter (take/skip decision)\nassert!(!meta_bins.is_empty());`,
      },
    ],
    notes: [
      "Label stability is dominated by event quality and volatility-target quality; calibrate these before tuning ML models.",
      "Always audit class balance and average holding time after labeling; both drive downstream model behavior.",
      "In meta-labeling, side alignment and timestamp joins are a frequent hidden bug source.",
    ],
    conceptOverview: `The triple-barrier method (AFML Chapter 3) replaces fixed-horizon labeling with a path-dependent approach. Instead of asking "did the price go up in 10 days?", it asks "which barrier did the price hit first — a profit-taking ceiling, a stop-loss floor, or a maximum holding horizon?"

This matters because fixed-horizon labels create artifacts: a trade that hits +5% then reverses to -1% at the horizon gets labeled as a loss. Triple-barrier labels capture the actual trade outcome under realistic exit rules.

**Meta-labeling** is a two-stage extension: a primary model predicts direction (side), while a secondary model learns *when to act* on that signal. The secondary model's label is binary (1 = the primary model was correct, 0 = it wasn't). This separation lets you combine a simple directional model with a sophisticated sizing/filtering model.

Barrier widths are scaled by a volatility target (typically EWMA of returns), making them adaptive across regimes. Events are sourced from structural filters like CUSUM rather than calendar time.`,
    whenToUse: `Use this module immediately after event detection (CUSUM/z-score filters) and volatility estimation. It sits at the start of the ML pipeline: raw price events go in, labeled training examples come out.

**Prerequisites**: A price series with timestamps, filtered event timestamps, and a volatility target series.

**Alternatives**: Fixed-horizon labeling (simpler but regime-blind), or trend-scanning labels for continuous-valued targets instead of classification.`,
    keyParameters: [
      { name: "pt", type: "f64", description: "Profit-taking barrier multiplier (× volatility target)", default: "1.0" },
      { name: "sl", type: "f64", description: "Stop-loss barrier multiplier (× volatility target)", default: "1.0" },
      { name: "min_ret", type: "f64", description: "Minimum return threshold; events with smaller absolute returns are labeled 0", default: "0.0" },
      { name: "vertical_barrier_times", type: "Option<Vec>", description: "Maximum holding period timestamps; events expire if neither profit nor stop barrier is hit", default: "None" },
      { name: "side_prediction", type: "Option<Vec<f64>>", description: "Primary model side forecasts (+1/−1) for meta-labeling mode", default: "None" },
    ],
    commonPitfalls: [
      "Setting symmetric barriers (pt=sl=1) when the strategy has asymmetric payoff — calibrate each barrier width independently.",
      "Using calendar-time vertical barriers with information-driven bars — the holding period should match bar frequency, not wall time.",
      "Ignoring class imbalance after labeling: if 80% of events hit the vertical barrier, the model learns to predict 'no movement' and the labels need recalibration.",
      "Forgetting that meta-labeling requires aligned timestamps between the primary model's side predictions and the event set — off-by-one joins silently corrupt labels.",
    ],
    relatedModules: ["filters", "sample-weights", "sampling", "bet-sizing"],
    afmlChapters: [3],
    pythonApis: ["labeling.triple_barrier_labels", "labeling.triple_barrier_events", "labeling.meta_labels", "labeling.add_vertical_barrier", "labeling.get_events", "labeling.get_bins", "labeling.drop_labels"],
    apiSurface: "both",
  },
  {
    slug: "microstructural-features",
    module: "microstructural_features",
    subject: "Market Microstructure, Dependence and Regime Detection",
    summary: "Price-impact, spread, entropy, and flow toxicity estimators.",
    whyItExists: "Microstructure features capture liquidity and order-flow dynamics not visible in OHLC bars alone.",
    keyApis: ["get_roll_measure", "get_corwin_schultz_estimator", "get_bar_based_kyle_lambda", "get_vpin", "MicrostructuralFeaturesGenerator"],
    formulas: [
      {
        label: "Kyle / Amihud / Hasbrouck Impact Families",
        latex:
          "\\Delta p_t=\\lambda_K q_t+\\epsilon_t,\\qquad r_t=\\lambda_A\\frac{1}{DV_t}+\\epsilon_t,\\qquad r_t=\\lambda_H\\frac{q_t}{\\sqrt{DV_t}}+\\epsilon_t",
      },
      {
        label: "Spread and Volatility Proxies",
        latex:
          "\\text{Roll spread}\\approx 2\\sqrt{-\\operatorname{cov}(\\Delta p_t,\\Delta p_{t-1})},\\qquad\\sigma_{CS}=f(H_t,L_t,H_{t-1},L_{t-1})",
      },
      {
        label: "Flow Toxicity and Entropy",
        latex:
          "\\mathrm{VPIN}_t=\\frac{1}{n}\\sum_{i=t-n+1}^{t}\\frac{|V_i^b-V_i^s|}{V_i},\\qquad H=-\\sum_j p_j\\log p_j",
      },
    ],
    examples: [
      {
        title: "End-to-end: Build Core Liquidity Feature Panel",
        language: "rust",
        code: `use openquant::microstructural_features::{\n    get_roll_measure,\n    get_corwin_schultz_estimator,\n    get_bar_based_kyle_lambda,\n    get_bar_based_amihud_lambda,\n    get_vpin,\n};\n\n// 1) Inputs from bar construction\nlet close = vec![100.0, 100.2, 100.1, 100.3, 100.25, 100.4];\nlet high = vec![100.1, 100.25, 100.2, 100.35, 100.3, 100.45];\nlet low = vec![99.9, 100.0, 99.95, 100.1, 100.05, 100.2];\nlet volume = vec![1000.0, 1200.0, 900.0, 1100.0, 1300.0, 1250.0];\nlet dollar_volume: Vec<f64> = close.iter().zip(volume.iter()).map(|(p, v)| p * v).collect();\nlet buy_volume = vec![600.0, 700.0, 480.0, 650.0, 800.0, 760.0];\n\n// 2) Liquidity and spread proxies\nlet roll = get_roll_measure(&close, 3);\nlet cs_spread = get_corwin_schultz_estimator(&high, &low, 3);\nlet kyle = get_bar_based_kyle_lambda(&close, &volume, 3);\nlet amihud = get_bar_based_amihud_lambda(&close, &dollar_volume, 3);\nlet vpin = get_vpin(&volume, &buy_volume, 3);\n\n// 3) Feature panel is ready for regime model / execution model\nassert_eq!(roll.len(), close.len());\nassert_eq!(vpin.len(), close.len());`,
      },
      {
        title: "From Encoded Tick Signs to Entropy Features",
        language: "rust",
        code: `use openquant::microstructural_features::{\n    encode_tick_rule_array,\n    get_shannon_entropy,\n    get_lempel_ziv_entropy,\n    get_plug_in_entropy,\n};\n\nlet tick_rule = vec![1, 1, -1, -1, 1, -1, 1, 1, 1, -1];\nlet msg = encode_tick_rule_array(&tick_rule)?;\n\nlet h_shannon = get_shannon_entropy(&msg);\nlet h_lz = get_lempel_ziv_entropy(&msg);\nlet h_plugin = get_plug_in_entropy(&msg, 2);\n\nassert!(h_shannon.is_finite());\nassert!(h_lz.is_finite());\nassert!(h_plugin.is_finite());`,
      },
    ],
    notes: [
      "Microstructure signals are highly regime-dependent; normalize and standardize within venue/time bucket before cross-asset comparison.",
      "Use shared bar definitions between training and live pipelines, otherwise feature drift is structural.",
      "Entropy features are sensitive to encoding; freeze symbol maps in production.",
    ],
    apiSurface: "both",
    pythonApis: ["microstructural.get_roll_measure", "microstructural.get_roll_impact", "microstructural.get_corwin_schultz_estimator", "microstructural.get_bekker_parkinson_vol", "microstructural.get_bar_based_kyle_lambda", "microstructural.get_bar_based_amihud_lambda", "microstructural.get_bar_based_hasbrouck_lambda", "microstructural.get_trades_based_kyle_lambda", "microstructural.get_trades_based_amihud_lambda", "microstructural.get_trades_based_hasbrouck_lambda", "microstructural.vwap", "microstructural.get_avg_tick_size", "microstructural.get_vpin", "microstructural.get_bvc_buy_volume", "microstructural.encode_tick_rule_array", "microstructural.quantile_mapping", "microstructural.sigma_mapping", "microstructural.encode_array", "microstructural.get_shannon_entropy", "microstructural.get_lempel_ziv_entropy", "microstructural.get_plug_in_entropy", "microstructural.get_konto_entropy"],
  },
  {
    slug: "onc",
    module: "onc",
    subject: "Portfolio Construction and Risk",
    summary: "Optimal Number of Clusters utilities for clustering stability and allocation workflows.",
    whyItExists: "Cluster count selection is a key source of model risk in hierarchical portfolio methods.",
    keyApis: ["get_onc_clusters", "check_improve_clusters", "OncResult"],
    formulas: [
      { label: "Cluster Score", latex: "J(k)=\\text{intra}(k)-\\text{inter}(k)" },
      { label: "Selection", latex: "k^*=\\arg\\min_k J(k)" },
    ],
    examples: [
      {
        title: "Infer cluster structure",
        language: "rust",
        code: `use openquant::onc::get_onc_clusters;\n\nlet out = get_onc_clusters(&corr, 20)?;\nprintln!("{}", out.clusters.len());`,
      },
    ],
    notes: ["Run with repeated seeds/restarts for robust k selection.", "Use correlation cleaning before clustering unstable universes."],
    apiSurface: "both",
    pythonApis: ["onc.get_onc_clusters"],
  },
  {
    slug: "portfolio-optimization",
    module: "portfolio_optimization",
    subject: "Portfolio Construction and Risk",
    summary: "Mean-variance and constrained allocation methods with ergonomic APIs.",
    whyItExists: "Provides production-ready portfolio construction primitives with explicit options and constraints.",
    keyApis: ["allocate_inverse_variance", "allocate_min_vol", "allocate_max_sharpe", "allocate_efficient_risk", "AllocationOptions"],
    formulas: [
      {
        label: "Constrained Mean-Variance Program",
        latex:
          "\\begin{aligned}\\min_{w}\\;&\\frac{1}{2}w^T\\Sigma w-\\lambda\\mu^T w\\\\\\text{s.t. }&\\mathbf 1^T w=1,\\quad l_i\\le w_i\\le u_i\\end{aligned}",
      },
      {
        label: "Minimum Variance / Maximum Sharpe / Efficient Return",
        latex:
          "\\begin{aligned}w_{MV}&=\\arg\\min_w\\;w^T\\Sigma w\\\\w_{MSR}&=\\arg\\max_w\\;\\frac{w^T(\\mu-r_f\\mathbf 1)}{\\sqrt{w^T\\Sigma w}}\\\\w_{ER}(r^*)&=\\arg\\min_w\\;w^T\\Sigma w\\;\\text{s.t. }w^T\\mu\\ge r^*\\end{aligned}",
      },
      {
        label: "Exponential Mean Estimator",
        latex:
          "\\mu_t=\\frac{\\sum_{k=0}^{T-1}(1-\\alpha)^k r_{t-k}}{\\sum_{k=0}^{T-1}(1-\\alpha)^k},\\qquad \\alpha=\\frac{2}{\\text{span}+1}",
      },
    ],
    examples: [
      {
        title: "End-to-end: Compute and Compare Core Allocators",
        language: "rust",
        code: `use nalgebra::DMatrix;\nuse openquant::portfolio_optimization::{\n    allocate_inverse_variance,\n    allocate_min_vol,\n    allocate_max_sharpe,\n    allocate_efficient_risk,\n};\n\n// rows=time, cols=assets\nlet prices: DMatrix<f64> = /* load matrix */ DMatrix::zeros(252, 6);\n\nlet ivp = allocate_inverse_variance(&prices)?;\nlet mv = allocate_min_vol(&prices, None, None)?;\nlet msr = allocate_max_sharpe(&prices, 0.01, None, None)?;\nlet er = allocate_efficient_risk(&prices, 0.12, None, None)?;\n\nassert_eq!(ivp.weights.len(), prices.ncols());\nassert!((mv.weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);\nassert!((msr.weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);\nassert!((er.weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);`,
      },
      {
        title: "End-to-end: Constrained Allocation with Exponential Returns and Resampling",
        language: "rust",
        code: `use std::collections::HashMap;\nuse openquant::portfolio_optimization::{\n    allocate_max_sharpe_with,\n    AllocationOptions,\n    ReturnsMethod,\n};\n\nlet mut bounds = HashMap::new();\n// Cap concentration in first asset; enforce long-only defaults elsewhere\nbounds.insert(0usize, (0.0, 0.20));\n\nlet opts = AllocationOptions {\n    risk_free_rate: 0.02,\n    returns_method: ReturnsMethod::Exponential { span: 60 },\n    resample_by: Some(\"W\"),\n    bounds: Some(bounds),\n    tuple_bounds: Some((0.0, 0.40)),\n    ..Default::default()\n};\n\nlet constrained = allocate_max_sharpe_with(&prices, &opts)?;\nassert!(constrained.weights.iter().all(|w| *w >= -1e-10));`,
      },
    ],
    notes: [
      "Optimizer output is only as good as mean/covariance assumptions; stress-test inputs and rebalance frequency.",
      "Constraint design (asset caps, sector caps, long/short bounds) is usually more important than small objective tweaks.",
      "Track turnover, realized slippage, and drift between target and filled weights in production.",
    ],
    apiSurface: "both",
    pythonApis: ["portfolio.allocate_inverse_variance", "portfolio.allocate_min_vol", "portfolio.allocate_max_sharpe", "portfolio.allocate_efficient_risk", "portfolio.allocate_with_solution", "portfolio.allocate_from_inputs"],
  },
  {
    slug: "risk-metrics",
    module: "risk_metrics",
    subject: "Portfolio Construction and Risk",
    summary: "Portfolio and return-distribution risk measures for downside control.",
    whyItExists: "Risk budgets and guardrails require coherent downside metrics beyond variance.",
    keyApis: ["RiskMetrics::calculate_value_at_risk", "RiskMetrics::calculate_expected_shortfall", "RiskMetrics::calculate_conditional_drawdown_risk", "RiskMetrics::calculate_variance"],
    formulas: [
      { label: "VaR", latex: "VaR_\\alpha = -Q_\\alpha(R)" },
      { label: "Expected Shortfall", latex: "ES_\\alpha = -E[R \\mid R \\le Q_\\alpha(R)]" },
    ],
    examples: [
      {
        title: "Compute VaR and ES",
        language: "rust",
        code: `use openquant::risk_metrics::RiskMetrics;\n\nlet r = vec![-0.02, 0.01, -0.005, 0.003, 0.004];\nlet var95 = RiskMetrics::calculate_value_at_risk(&r, 0.05)?;\nlet es95 = RiskMetrics::calculate_expected_shortfall(&r, 0.05)?;`,
      },
    ],
    notes: ["Non-parametric estimates need enough tail observations.", "Use matrix variants for multi-asset return panels."],
    apiSurface: "both",
    pythonApis: ["risk.calculate_value_at_risk", "risk.calculate_expected_shortfall", "risk.calculate_conditional_drawdown_risk", "risk.calculate_variance", "risk.calculate_value_at_risk_from_matrix", "risk.calculate_expected_shortfall_from_matrix", "risk.calculate_conditional_drawdown_risk_from_matrix"],
  },
  {
    slug: "strategy-risk",
    module: "strategy_risk",
    subject: "Portfolio Construction and Risk",
    summary: "AFML Chapter 15 strategy-viability diagnostics based on precision, payout asymmetry, and bet frequency.",
    whyItExists:
      "Strategy risk is the probability that a process fails to achieve a Sharpe objective over time; it is distinct from holdings/portfolio variance risk and should be monitored separately.",
    keyApis: [
      "sharpe_symmetric",
      "implied_precision_symmetric",
      "implied_frequency_symmetric",
      "sharpe_asymmetric",
      "implied_precision_asymmetric",
      "implied_frequency_asymmetric",
      "estimate_strategy_failure_probability",
      "StrategyRiskConfig",
      "StrategyRiskReport",
    ],
    formulas: [
      {
        label: "Symmetric Sharpe",
        latex: "\\theta=\\frac{2p-1}{2\\sqrt{p(1-p)}}\\sqrt{n}",
      },
      {
        label: "Asymmetric Sharpe",
        latex:
          "\\theta=\\frac{(\\pi_+-\\pi_-)p+\\pi_-}{(\\pi_+-\\pi_-)\\sqrt{p(1-p)}}\\sqrt{n}",
      },
      {
        label: "Strategy Failure Probability",
        latex: "P_{fail}=\\Pr[p\\le p^*],\\quad p^*=\\text{impliedPrecision}(\\theta^*,\\pi_+,\\pi_-,n)",
      },
    ],
    examples: [
      {
        title: "Estimate strategy-failure probability from realized bets",
        language: "rust",
        code: `use openquant::strategy_risk::{estimate_strategy_failure_probability, StrategyRiskConfig};\n\nlet outcomes = vec![0.005, -0.01, 0.005, 0.005, -0.01, 0.005, 0.005, -0.01];\nlet report = estimate_strategy_failure_probability(\n  &outcomes,\n  StrategyRiskConfig {\n    years_elapsed: 2.0,\n    target_sharpe: 2.0,\n    investor_horizon_years: 2.0,\n    bootstrap_iterations: 10_000,\n    seed: 7,\n    kde_bandwidth: None,\n  },\n)?;\n\nprintln!(\"p*: {:.4}\", report.implied_precision_threshold);\nprintln!(\"failure (KDE): {:.2}%\", 100.0 * report.kde_failure_probability);`,
      },
    ],
    notes: [
      "Inputs under manager control ({pi_minus, pi_plus, n}) should be analyzed separately from uncertain market precision p.",
      "Use this module for strategy-level viability and probability-of-failure diagnostics; use `risk_metrics` for portfolio-tail and drawdown risk.",
    ],
    apiSurface: "both",
    pythonApis: ["strategy_risk.sharpe_symmetric", "strategy_risk.implied_precision_symmetric", "strategy_risk.implied_frequency_symmetric", "strategy_risk.sharpe_asymmetric", "strategy_risk.implied_precision_asymmetric", "strategy_risk.implied_frequency_asymmetric", "strategy_risk.estimate_strategy_failure_probability"],
  },
  {
    slug: "hpc-parallel",
    module: "hpc_parallel",
    subject: "Scaling, HPC and Infrastructure",
    summary: "AFML Chapter 20 atom/molecule execution utilities with serial/threaded modes and partition diagnostics.",
    whyItExists:
      "Research pipelines bottleneck on repeated independent computations; this module exposes reproducible partitioning and dispatch controls to scale those workloads safely.",
    keyApis: [
      "partition_atoms",
      "run_parallel",
      "dispatch_async",
      "ExecutionMode",
      "PartitionStrategy",
      "HpcParallelConfig",
      "ParallelRunReport",
      "HpcParallelMetrics",
    ],
    formulas: [
      {
        label: "Linear Partition Boundary",
        latex: "b_i=\\left\\lfloor\\frac{iN}{M}\\right\\rfloor,\\;i=0,\\dots,M",
      },
      {
        label: "Nested Partition Boundary",
        latex: "b_i=\\left\\lfloor N\\sqrt{\\frac{i}{M}}\\right\\rfloor,\\;i=0,\\dots,M",
      },
      {
        label: "Throughput",
        latex: "\\text{throughput}=\\frac{\\text{atoms processed}}{\\text{runtime seconds}}",
      },
    ],
    examples: [
      {
        title: "Run atom->molecule callback in threaded mode",
        language: "rust",
        code: `use openquant::hpc_parallel::{run_parallel, ExecutionMode, HpcParallelConfig, PartitionStrategy};\n\nlet atoms: Vec<f64> = (0..10_000).map(|i| i as f64).collect();\nlet report = run_parallel(\n  &atoms,\n  HpcParallelConfig {\n    mode: ExecutionMode::Threaded { num_threads: 8 },\n    partition: PartitionStrategy::Nested,\n    mp_batches: 4,\n    progress_every: 4,\n  },\n  |chunk| Ok::<f64, &'static str>(chunk.iter().map(|x| x.sqrt()).sum()),\n)?;\n\nprintln!(\"molecules={} atoms/s={:.0}\", report.metrics.molecules_total, report.metrics.throughput_atoms_per_sec);`,
      },
    ],
    notes: [
      "Use `ExecutionMode::Serial` for deterministic debugging with identical callback semantics.",
      "If per-atom cost rises with atom index (e.g., expanding windows), nested partitioning can reduce tail stragglers versus linear chunking.",
    ],
    apiSurface: "rust-only",
  },
  {
    slug: "combinatorial-optimization",
    module: "combinatorial_optimization",
    subject: "Scaling, HPC and Infrastructure",
    summary:
      "AFML Chapter 21 integer-encoded optimization and trajectory state-space tooling with exact baselines and solver adapters.",
    whyItExists:
      "Many trading/search problems are discrete and path-dependent; this module keeps integer structure explicit and provides exact small-instance baselines before scaling to heuristics.",
    keyApis: [
      "DecisionSchema",
      "IntegerVariable",
      "IntegerObjective",
      "solve_exact",
      "SolverAdapter",
      "solve_with_adapter",
      "compare_exact_and_adapter",
      "TradingTrajectorySchema",
      "enumerate_trading_paths",
      "evaluate_trading_path",
      "solve_trading_trajectory_exact",
    ],
    formulas: [
      {
        label: "Finite Integer Program",
        latex: "x^*=\\arg\\max_{x\\in\\mathcal X\\subset\\mathbb Z^d} f(x),\\quad |\\mathcal X|<\\infty",
      },
      {
        label: "Path-Dependent Objective",
        latex:
          "J(\\tau)=\\sum_{t=1}^{T}\\left(q_t r_t-\\lambda q_t^2-c_t|\\Delta q_t|-\\kappa\\,\\mathbf 1_{\\Delta q_t\\ne0}\\right)-\\eta(q_T-q^*)^2",
      },
      {
        label: "Adapter Gap vs Exact",
        latex:
          "\\Delta_{alg}=\\begin{cases}f(x^*)-f(\\hat x) & \\text{maximize}\\\\f(\\hat x)-f(x^*) & \\text{minimize}\\end{cases}",
      },
    ],
    examples: [
      {
        title: "Exact trajectory search with fixed ticket costs",
        language: "rust",
        code: `use openquant::combinatorial_optimization::{\n  TradeBounds, TradingTrajectoryObjectiveConfig, TradingTrajectoryPath, TradingTrajectorySchema,\n  enumerate_trading_paths, evaluate_trading_path,\n};\n\nlet schema = TradingTrajectorySchema {\n  initial_inventory: 0,\n  inventory_min: -2,\n  inventory_max: 2,\n  step_trade_bounds: vec![\n    TradeBounds { min_trade: -1, max_trade: 1 },\n    TradeBounds { min_trade: -1, max_trade: 1 },\n    TradeBounds { min_trade: -1, max_trade: 1 },\n  ],\n  terminal_inventory: Some(0),\n  max_paths: 50_000,\n};\nlet cfg = TradingTrajectoryObjectiveConfig {\n  expected_returns: vec![0.01, -0.015, 0.012],\n  risk_aversion: 0.001,\n  impact_coefficients: vec![0.0005, 0.0005, 0.0005],\n  fixed_ticket_cost: 0.002,\n  terminal_inventory_target: 0,\n  terminal_inventory_penalty: 0.05,\n};\n\nlet best = enumerate_trading_paths(&schema)?\n  .into_iter()\n  .map(|path| {\n    let score = evaluate_trading_path(&path, &cfg)?;\n    Ok::<(TradingTrajectoryPath, f64), openquant::combinatorial_optimization::CombinatorialOptimizationError>((path, score))\n  })\n  .collect::<Result<Vec<_>, _>>()?\n  .into_iter()\n  .max_by(|a, b| a.1.total_cmp(&b.1))\n  .expect(\"at least one feasible path\");\n\nprintln!(\"best objective: {:.6}\", best.1);\nprintln!(\"trades: {:?}\", best.0.trades);`,
      },
    ],
    notes: [
      "Exact enumeration scales exponentially in decision dimension/horizon; treat it as a correctness baseline and regression oracle.",
      "Use adapter interfaces to compare heuristic/external solvers against exact solutions on small calibration instances before production deployment.",
    ],
    apiSurface: "rust-only",
  },
  {
    slug: "streaming-hpc",
    module: "streaming_hpc",
    subject: "Scaling, HPC and Infrastructure",
    summary:
      "AFML Chapter 22 streaming analytics utilities for low-latency early-warning metrics with bounded-memory incremental state.",
    whyItExists:
      "Streaming decisions are turnaround-time constrained; this module maintains VPIN/HHI-style indicators incrementally and supports multi-stream scaling across cores/chunk sizes.",
    keyApis: [
      "StreamEvent",
      "VpinState",
      "HhiState",
      "StreamingEarlyWarningEngine",
      "run_streaming_pipeline",
      "run_streaming_pipeline_parallel",
      "generate_synthetic_flash_crash_stream",
      "StreamingPipelineConfig",
      "StreamingRunMetrics",
    ],
    formulas: [
      {
        label: "VPIN (Rolling Buckets)",
        latex: "\\mathrm{VPIN}_t=\\frac{1}{N}\\sum_{i=t-N+1}^{t}\\frac{|V_i^B-V_i^S|}{V_i}",
      },
      {
        label: "Market Fragmentation HHI",
        latex: "\\mathrm{HHI}_t=\\sum_{v=1}^{K}\\left(\\frac{n_{v,t}}{\\sum_j n_{j,t}}\\right)^2",
      },
      {
        label: "Streaming Throughput",
        latex: "\\mathrm{throughput}=\\frac{\\#\\mathrm{events\\ processed}}{\\mathrm{runtime\\ seconds}}",
      },
    ],
    examples: [
      {
        title: "Incremental early-warning pipeline on streaming trades",
        language: "rust",
        code: `use openquant::hpc_parallel::{ExecutionMode, HpcParallelConfig, PartitionStrategy};\nuse openquant::streaming_hpc::{\n  run_streaming_pipeline_parallel, AlertThresholds, HhiConfig, StreamingPipelineConfig,\n  SyntheticStreamConfig, VpinConfig, generate_synthetic_flash_crash_stream,\n};\n\nlet streams: Vec<_> = (0..16)\n  .map(|k| generate_synthetic_flash_crash_stream(SyntheticStreamConfig {\n    events: 2_000,\n    crash_start_fraction: 0.7,\n    calm_venues: 8,\n    shock_venue: k % 2,\n  }))\n  .collect::<Result<Vec<_>, _>>()?;\n\nlet report = run_streaming_pipeline_parallel(\n  &streams,\n  StreamingPipelineConfig {\n    vpin: VpinConfig { bucket_volume: 1_000.0, support_buckets: 20 },\n    hhi: HhiConfig { lookback_events: 200 },\n    thresholds: AlertThresholds { vpin: 0.45, hhi: 0.30 },\n  },\n  HpcParallelConfig {\n    mode: ExecutionMode::Threaded { num_threads: 8 },\n    partition: PartitionStrategy::Linear,\n    mp_batches: 4,\n    progress_every: 8,\n  },\n)?;\n\nprintln!(\"streams={} molecules={} events/s={:.0}\",\n  report.stream_summaries.len(),\n  report.parallel_metrics.molecules_total,\n  report.parallel_metrics.throughput_atoms_per_sec\n);`,
      },
    ],
    notes: [
      "Chapter 22 stresses turnaround-time over pure throughput: bounded rolling windows avoid unbounded latency/memory growth.",
      "For low-latency alerts, keep stream partitioning stable and calibrate `mp_batches` against scheduling overhead and cache locality.",
      "Use synthetic flash-crash replays to validate that warning thresholds react early without excessive false positives.",
    ],
    apiSurface: "both",
    pythonApis: ["streaming_hpc.run_streaming_pipeline", "streaming_hpc.generate_synthetic_flash_crash_stream"],
  },
  {
    slug: "sample-weights",
    module: "sample_weights",
    subject: "Event-Driven Data and Labeling",
    summary: "Sample weighting utilities for overlapping event structure.",
    whyItExists: "Adjusts training influence to avoid overcounting dense overlapping labels.",
    keyApis: ["get_weights_by_return", "get_weights_by_time_decay"],
    formulas: [
      { label: "Uniqueness Weight", latex: "w_i=\\sum_t\\frac{I_{t,i}}{\\sum_j I_{t,j}}" },
      { label: "Time Decay", latex: "w_i=(\\frac{i}{T})^\\delta" },
    ],
    examples: [
      {
        title: "Compute sample weights for overlapping labels",
        language: "python",
        code: `from openquant._core import sample_weights

# Returns from labeled events (used for return-attribution weighting)
returns = [0.01, -0.005, 0.007, -0.002, 0.003, 0.01, -0.008]

# Weight by absolute return (higher-impact events get more weight)
w_return = sample_weights.get_weights_by_return(returns)

# Weight by time decay (more recent events weighted higher, delta=0.5)
w_decay = sample_weights.get_weights_by_time_decay(returns, 0.5)

# Use these weights in model training:
# model.fit(X, y, sample_weight=w_return)`,
      },
      {
        title: "Compute event weights",
        language: "rust",
        code: `use openquant::sample_weights::get_weights_by_time_decay;\n\nlet w = get_weights_by_time_decay(&returns, 0.5);`,
      },
    ],
    notes: ["Pair with sequential bootstrap for robust label sampling.", "Time-decay controls recency bias explicitly."],
    conceptOverview: `In AFML's event-driven framework (Chapter 4), labels are derived from overlapping price paths. When two events overlap in time, their labels share information — the price observations that determine event A's outcome also influence event B's outcome. Treating these labels as independent samples inflates effective sample size and biases model training.

**Uniqueness-based weighting** addresses this by computing how unique each sample is at each time step. If a bar contributes to 3 concurrent events, each event gets 1/3 credit for that bar. The total weight of a sample is the sum of its per-bar uniqueness scores. Samples that overlap with many others get down-weighted; isolated samples get full weight.

**Return-attribution weighting** weights samples by their absolute return, giving more training influence to economically significant events.

**Time-decay weighting** applies a power-law decay so recent observations contribute more than older ones, useful when the data-generating process evolves over time.

These weights should be passed as \`sample_weight\` to your classifier or loss function.`,
    whenToUse: `Apply sample weights after labeling and before model training. They correct for the non-IID structure caused by overlapping triple-barrier labels.

**Prerequisites**: Labeled events from the labeling module, with event start/end times.

**Alternatives**: Equal weights (ignores overlap, biases toward dense clusters), or sequential bootstrap (sampling-based approach instead of weighting).`,
    keyParameters: [
      { name: "delta", type: "f64", description: "Time-decay exponent; 0 = uniform, 1 = linear decay, >1 = aggressive recency bias", default: "1.0" },
    ],
    commonPitfalls: [
      "Training without any overlap correction — highly overlapping labels effectively duplicate data and overfit the dense-event regime.",
      "Using uniqueness weights without the indicator matrix from the sampling module — the weights require knowledge of which bars each event spans.",
      "Combining time-decay and uniqueness weights incorrectly — multiply them element-wise, don't add.",
    ],
    relatedModules: ["labeling", "sampling", "sb-bagging"],
    afmlChapters: [4],
    apiSurface: "both",
    pythonApis: ["sample_weights.get_weights_by_return", "sample_weights.get_weights_by_time_decay"],
  },
  {
    slug: "sampling",
    module: "sampling",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Indicator matrix and sequential bootstrap tooling.",
    whyItExists: "Produces less correlated training samples when labels overlap heavily in time.",
    keyApis: ["get_ind_matrix", "seq_bootstrap", "get_ind_mat_average_uniqueness", "num_concurrent_events"],
    formulas: [
      { label: "Average Uniqueness", latex: "u_i=\\frac{1}{|T_i|}\\sum_{t\\in T_i}\\frac{1}{c_t}" },
      { label: "Sequential Draw Prob", latex: "P(i)\\propto E[u_i \\mid \\mathcal{S}]" },
    ],
    examples: [
      {
        title: "Sequential bootstrap with overlap-aware sampling",
        language: "python",
        code: `from openquant._core import sampling

# Indicator matrix: rows=bars, cols=labels
# 1 means bar i is active for label j
ind_matrix = [
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [0, 1, 0],
    [1, 0, 0],
]

# Average uniqueness per label (diagnostic)
avg_u = sampling.get_ind_mat_average_uniqueness(ind_matrix)
# e.g., [0.72, 0.58, 0.44] — label 0 is most unique

# Sequential bootstrap: draw n samples favoring unique labels
drawn_indices = sampling.seq_bootstrap(ind_matrix, n_samples=3)
# Returns label indices selected with overlap-aware probabilities`,
      },
      {
        title: "Run sequential bootstrap",
        language: "rust",
        code: `use openquant::sampling::seq_bootstrap;\n\nlet ind = vec![vec![1,0,1], vec![0,1,1], vec![1,1,0]];\nlet idx = seq_bootstrap(&ind, Some(3), None);`,
      },
    ],
    notes: ["Indicator matrix quality drives bootstrap quality.", "Use average uniqueness as a diagnostics KPI."],
    conceptOverview: `Standard bootstrap assumes IID observations: draw N samples with replacement uniformly. But AFML labels overlap in time — event A might span bars 1-5 while event B spans bars 3-8. Drawing both A and B into the same bootstrap sample introduces information leakage between train/test, because they share bars 3-5.

The **sequential bootstrap** (AFML Chapter 4) fixes this by making draws overlap-aware. It builds an **indicator matrix** that maps which bars each label spans. At each draw step, it computes the average uniqueness of each remaining label *given what's already been drawn*, then samples proportionally to uniqueness. Labels that would create heavy overlap with already-drawn samples have low uniqueness and are unlikely to be selected.

The result is a bootstrap sample where the drawn labels are as independent as possible given the underlying overlap structure. This is critical for bagging classifiers trained on financial labels, where naive bootstrap produces ensembles with highly correlated base learners.

**Average uniqueness** is the key diagnostic: it tells you what fraction of each label's information is non-redundant. Low average uniqueness (< 0.5) means heavy overlap and sequential bootstrap becomes essential.`,
    whenToUse: `Use sequential bootstrap whenever you're bagging or bootstrapping with overlapping labels. It replaces standard \`np.random.choice\` in any ensemble or bootstrap workflow.

**Prerequisites**: An indicator matrix from event start/end times, and optionally the concurrent event count per bar.

**Alternatives**: Standard IID bootstrap (fast but leakage-prone), or sample weighting (correct expected value but doesn't reduce sample correlation).`,
    keyParameters: [
      { name: "ind_matrix", type: "Vec<Vec<i32>>", description: "Indicator matrix: rows=bars, cols=labels. Entry is 1 if bar i is active during label j", default: "—" },
      { name: "n_samples", type: "Option<usize>", description: "Number of bootstrap draws; defaults to number of labels", default: "None (= n_labels)" },
    ],
    commonPitfalls: [
      "Building the indicator matrix with wrong event boundaries — off-by-one errors silently break uniqueness calculations.",
      "Using sequential bootstrap with very short labels that don't overlap — it degenerates to standard bootstrap and just adds overhead.",
      "Forgetting to pass sequential bootstrap indices to the bagging estimator — the sampling module produces indices, your estimator must use them.",
    ],
    relatedModules: ["sample-weights", "sb-bagging", "labeling"],
    afmlChapters: [4],
    apiSurface: "both",
    pythonApis: ["sampling.get_ind_matrix", "sampling.seq_bootstrap", "sampling.get_ind_mat_average_uniqueness", "sampling.get_ind_mat_label_uniqueness", "sampling.bootstrap_loop_run", "sampling.get_av_uniqueness_from_triple_barrier", "sampling.num_concurrent_events"],
  },
  {
    slug: "sb-bagging",
    module: "sb_bagging",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Sequentially bootstrapped bagging classifiers/regressors.",
    whyItExists: "Combines ensemble variance reduction with overlap-aware sampling.",
    keyApis: ["SequentiallyBootstrappedBaggingClassifier", "SequentiallyBootstrappedBaggingRegressor", "MaxSamples", "MaxFeatures"],
    formulas: [
      { label: "Bagging Predictor", latex: "\\hat f(x)=\\frac{1}{B}\\sum_{b=1}^{B} f_b(x)" },
      { label: "Bootstrap Sampling", latex: "S_b\\sim P_{seq}(u)" },
    ],
    examples: [
      {
        title: "Instantiate SB bagging classifier",
        language: "rust",
        code: `use openquant::sb_bagging::SequentiallyBootstrappedBaggingClassifier;\n\nlet bag = SequentiallyBootstrappedBaggingClassifier::new(100);`,
      },
    ],
    notes: ["Sequential bootstrap improves diversity under event overlap.", "Tune max_samples/max_features with out-of-sample monitoring."],
    apiSurface: "both",
    pythonApis: ["sb_bagging.fit_predict_sb_classifier", "sb_bagging.fit_predict_sb_regressor"],
  },
  {
    slug: "synthetic-backtesting",
    module: "synthetic_backtesting",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Synthetic-data OTR backtesting with O-U calibration, PT/SL mesh search, and stability diagnostics.",
    whyItExists:
      "AFML Chapter 13 shows that selecting PT/SL rules on a single historical path is prone to overfitting; synthetic path ensembles let us evaluate rule robustness under calibrated process dynamics.",
    keyApis: [
      "calibrate_ou_params",
      "generate_ou_paths",
      "evaluate_rule_on_paths",
      "search_optimal_trading_rule",
      "detect_no_stable_optimum",
      "run_synthetic_otr_workflow",
    ],
    formulas: [
      {
        label: "Discrete O-U (AR(1))",
        latex: "P_t=\\alpha+\\phi P_{t-1}+\\sigma\\epsilon_t,\\quad \\epsilon_t\\sim\\mathcal N(0,1)",
      },
      {
        label: "Equilibrium Level",
        latex: "\\bar P=\\frac{\\alpha}{1-\\phi}",
      },
      {
        label: "OTR Objective over Rule Mesh",
        latex: "R^*=\\arg\\max_{R\\in\\Omega}\\frac{\\mathbb E[\\pi\\mid R]}{\\sigma[\\pi\\mid R]}",
      },
    ],
    examples: [
      {
        title: "End-to-end synthetic OTR workflow",
        language: "rust",
        code: `use openquant::synthetic_backtesting::{run_synthetic_otr_workflow, StabilityCriteria, SyntheticBacktestConfig};\n\nlet cfg = SyntheticBacktestConfig {\n  initial_price: historical_prices[historical_prices.len() - 1],\n  n_paths: 10_000,\n  horizon: 128,\n  seed: 42,\n  profit_taking_grid: vec![0.5, 1.0, 1.5, 2.0, 3.0],\n  stop_loss_grid: vec![0.5, 1.0, 1.5, 2.0, 3.0],\n  max_holding_steps: 64,\n  annualization_factor: 1.0,\n  stability_criteria: StabilityCriteria::default(),\n};\n\nlet out = run_synthetic_otr_workflow(&historical_prices, &cfg)?;\nif out.diagnostics.no_stable_optimum {\n  println!(\"Skip OTR optimization: {}\", out.diagnostics.reason);\n} else {\n  println!(\"Best PT/SL: {:?}\", out.best_rule);\n}`,
      },
    ],
    notes: [
      "Near-random-walk estimates (|phi| close to 1) often produce flat Sharpe heatmaps where any selected rule is unstable out-of-sample.",
      "Calibrating to process parameters and evaluating many synthetic paths reduces single-path lucky-fit risk compared to brute-force historical optimization.",
    ],
    apiSurface: "both",
    pythonApis: ["synthetic_bt.calibrate_ou_params", "synthetic_bt.generate_ou_paths", "synthetic_bt.evaluate_rule_on_paths", "synthetic_bt.detect_no_stable_optimum", "synthetic_bt.run_synthetic_otr_workflow", "synthetic_bt.search_optimal_trading_rule"],
  },
  {
    slug: "structural-breaks",
    module: "structural_breaks",
    subject: "Market Microstructure, Dependence and Regime Detection",
    summary: "Regime change and bubble diagnostics (Chow, CUSUM variants, SADF).",
    whyItExists: "Regime instability can invalidate model assumptions; break detection is a core risk control.",
    keyApis: ["get_chow_type_stat", "get_chu_stinchcombe_white_statistics", "get_sadf", "SadfLags"],
    formulas: [
      { label: "ADF Regression", latex: "\\Delta y_t=\\alpha+\\beta y_{t-1}+\\sum_{i=1}^{k}\\phi_i\\Delta y_{t-i}+\\epsilon_t" },
      { label: "SADF", latex: "SADF=\\sup_{r_2\\in[r_0,1]} ADF_0^{r_2}" },
    ],
    examples: [
      {
        title: "Compute SADF statistic",
        language: "rust",
        code: `use openquant::structural_breaks::{get_sadf, SadfLags};\n\nlet y = vec![100.0, 100.2, 100.4, 100.1, 99.8, 100.0];\nlet sadf = get_sadf(&y, 3, SadfLags::Fixed(1))?;`,
      },
    ],
    notes: ["SADF can be computationally expensive on long windows.", "Use dedicated slow/nightly test paths for heavy scenarios."],
    apiSurface: "both",
    pythonApis: ["structural_breaks.get_chow_type_stat", "structural_breaks.get_chu_stinchcombe_white_statistics", "structural_breaks.get_sadf"],
  },
  {
    slug: "util-fast-ewma",
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
        code: `use openquant::util::fast_ewma::ewma;\n\nlet x = vec![1.0, 2.0, 3.0, 4.0];\nlet y = ewma(&x, 3);`,
      },
    ],
    notes: ["Window length controls responsiveness vs smoothness.", "Prefer this helper over ad-hoc loops for consistency."],
    apiSurface: "both",
    pythonApis: ["fast_ewma.ewma"],
  },
  {
    slug: "util-volatility",
    module: "util::volatility",
    subject: "Market Microstructure, Dependence and Regime Detection",
    summary: "Volatility estimators used across labeling and risk workflows.",
    whyItExists: "Volatility is a foundational scaling target for barriers, sizing, and risk controls.",
    keyApis: ["get_daily_vol", "get_parksinson_vol", "get_garman_class_vol", "get_yang_zhang_vol"],
    formulas: [
      { label: "Parkinson", latex: "\\sigma_P^2=\\frac{1}{4\\ln 2}\\frac{1}{n}\\sum (\\ln(H_t/L_t))^2" },
      { label: "Yang-Zhang", latex: "\\sigma_{YZ}^2=\\sigma_o^2+k\\sigma_c^2+(1-k)\\sigma_{rs}^2" },
    ],
    examples: [
      {
        title: "Compute daily and range-based volatility",
        language: "rust",
        code: `use openquant::util::volatility::{get_daily_vol, get_parksinson_vol};\n\nlet dv = get_daily_vol(&close, 100);\nlet pv = get_parksinson_vol(&high, &low, 20);`,
      },
    ],
    notes: ["Choose estimator based on available fields and microstructure noise.", "Daily-vol lookback should be matched to event horizon."],
    apiSurface: "both",
    pythonApis: ["volatility.get_daily_vol", "volatility.get_parksinson_vol", "volatility.get_garman_class_vol", "volatility.get_yang_zhang_vol"],
  },
  // ── Python-only modules ──────────────────────────────────────────────
  {
    slug: "data",
    module: "data",
    subject: "Data Ingestion and Quality",
    summary: "OHLCV loading, cleaning, calendar alignment, and data quality reporting.",
    whyItExists: "Provides a consistent entrypoint for market data ingestion with automatic column normalization, deduplication, and quality diagnostics.",
    keyApis: ["load_ohlcv", "clean_ohlcv", "align_calendar", "data_quality_report"],
    formulas: [],
    examples: [
      {
        title: "Load, clean, and inspect OHLCV data",
        language: "python",
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
      "Column aliases are resolved automatically (e.g., 'timestamp' → 'ts', 'ticker' → 'symbol').",
      "clean_ohlcv deduplicates by (symbol, ts) and sorts chronologically.",
      "align_calendar marks missing bars with is_missing_bar=True for downstream imputation logic.",
    ],
    conceptOverview: `Before any AFML workflow begins, raw market data must be loaded into a consistent schema, cleaned of duplicates and formatting issues, and aligned to a regular time grid. This module handles that ingestion layer.

It accepts CSV or Parquet files with flexible column naming (e.g., "timestamp", "datetime", "date" all map to "ts"; "ticker" or "asset" map to "symbol") and produces a standardized Polars DataFrame with canonical OHLCV columns. Deduplication handles duplicate (symbol, timestamp) keys, and calendar alignment generates a regular grid with explicit gap markers.

The data quality report provides diagnostics — row counts, symbol counts, duplicate counts, gap intervals, and null counts — that should be inspected before feeding data into bars, labeling, or any downstream module.`,
    whenToUse: `Use this module as the first step when working with pre-aggregated OHLCV data (daily bars, minute bars from a vendor). If you have raw tick/trade data instead, use the \`data_structures\` module to construct bars first.

**Prerequisites**: A CSV or Parquet file, or an existing Polars DataFrame with OHLCV-like columns.

**Alternatives**: Direct Polars/pandas loading if you handle column normalization and cleaning yourself.`,
    keyParameters: [
      { name: "path", type: "str | Path", description: "File path to CSV or Parquet OHLCV data", default: "—" },
      { name: "symbol", type: "str | None", description: "Symbol name if not present as a column in the data", default: "None" },
      { name: "interval", type: "str", description: "Calendar alignment interval (e.g., '1d', '1h', '5m')", default: "'1d'" },
      { name: "dedupe_keep", type: "str", description: "Which duplicate to keep: 'first' or 'last'", default: "'last'" },
    ],
    commonPitfalls: [
      "Forgetting to check the quality report for gaps — missing bars silently create NaN features downstream.",
      "Using align_calendar with an interval shorter than the data's actual frequency — this creates many synthetic missing-bar rows.",
    ],
    relatedModules: ["data-structures"],
    apiSurface: "both",
    pythonApis: ["data.load_ohlcv", "data.clean_ohlcv", "data.align_calendar", "data.data_quality_report", "data.clean_ohlcv_df", "data.quality_report_df", "data.align_calendar_df"],
  },
  {
    slug: "feature-diagnostics",
    module: "feature_diagnostics",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Feature importance diagnostics: MDI, MDA, SFI, PCA orthogonalization, and substitution-effect analysis.",
    whyItExists: "AFML Chapter 8 requires multiple importance methods to detect substitution effects and unstable features before deploying models.",
    keyApis: ["mdi_importance", "mda_importance", "sfi_importance", "orthogonalize_features_pca", "substitution_effect_report"],
    formulas: [
      { label: "MDI (Mean Decrease Impurity)", latex: "I_j^{MDI}=\\frac{1}{B}\\sum_{b=1}^B \\frac{|\\beta_j^{(b)}|}{\\sum_k|\\beta_k^{(b)}|}" },
      { label: "MDA (Mean Decrease Accuracy)", latex: "I_j^{MDA}=\\frac{S_{base}-S_{perm(j)}}{1-S_{perm(j)}}" },
    ],
    examples: [
      {
        title: "Run all three importance methods and compare",
        language: "python",
        code: `from openquant.feature_diagnostics import (
    mdi_importance, mda_importance, sfi_importance
)

X = [[0.1, 0.5, 0.3], [0.2, 0.4, 0.1], ...]  # n_samples × n_features
y = [1.0, 0.0, 1.0, ...]  # binary labels
names = ["momentum", "volatility", "spread"]

mdi = mdi_importance(X, y, feature_names=names, n_estimators=32)
mda = mda_importance(X, y, feature_names=names, n_splits=5, pct_embargo=0.01)
sfi = sfi_importance(X, y, feature_names=names, n_splits=5)

# Each returns: {"table": pl.DataFrame, "viz_payload": {...}, ...}
print(mdi["table"])  # feature | mean | std | stderr
print(mda["table"])`,
      },
      {
        title: "Detect substitution effects between correlated features",
        language: "python",
        code: `from openquant.feature_diagnostics import substitution_effect_report

report = substitution_effect_report(
    X, y,
    feature_names=names,
    corr_threshold=0.7,   # flag pairs with |corr| > 0.7
    orthogonalize=True,   # also run MDA on PCA-orthogonalized features
)

# Correlated pairs with dilution risk
print(report["pairs"])
# feature_a | feature_b | corr | dilution_ratio | flag_substitution_risk

# Before/after orthogonalization comparison
print(report["orthogonalized"]["max_abs_corr_before"])   # e.g., 0.92
print(report["orthogonalized"]["max_abs_corr_after"])     # e.g., 0.03`,
      },
    ],
    notes: [
      "MDI is biased toward high-cardinality features; cross-check with MDA.",
      "MDA uses purged k-fold CV internally to prevent leakage in importance estimates.",
      "SFI trains single-feature models — useful for detecting features that are only useful in combination.",
      "substitution_effect_report combines MDA + correlation + PCA in one call.",
    ],
    conceptOverview: `Feature importance is not a single number — AFML Chapter 8 argues you need multiple methods because each has different failure modes. **MDI** (Mean Decrease Impurity) measures how much each feature contributes to splits in an ensemble, but it's biased toward features with more unique values. **MDA** (Mean Decrease Accuracy) measures the score drop when a feature is permuted, which is unbiased but noisy. **SFI** (Single Feature Importance) trains one model per feature, revealing which features carry signal alone vs. only in combination.

The critical insight is **substitution effects**: when two features are correlated, MDI and MDA split importance between them arbitrarily. A feature that appears unimportant might be essential — its importance was just absorbed by its correlated partner. The \`substitution_effect_report\` detects this by comparing individual MDA scores against grouped-permutation scores, and by re-running MDA on PCA-orthogonalized features where substitution effects vanish.

All importance methods use purged k-fold cross-validation internally, preventing information leakage from overlapping labels.`,
    whenToUse: `Run feature diagnostics after training an initial model and before finalizing the feature set. Use the results to prune unstable features, detect redundancy, and validate that your model relies on economically meaningful signals.

**Prerequisites**: Feature matrix X, label vector y, and optionally event end indices for purged CV.

**Alternatives**: Rust-side \`feature_importance\` module for MDI/MDA on Rust models; this Python module adds SFI, PCA orthogonalization, and substitution-effect analysis.`,
    keyParameters: [
      { name: "n_estimators", type: "int", description: "Number of bootstrap rounds for MDI", default: "32" },
      { name: "n_splits", type: "int", description: "Number of purged k-fold splits for MDA/SFI", default: "5" },
      { name: "pct_embargo", type: "float", description: "Embargo fraction for purged CV", default: "0.01" },
      { name: "scoring", type: "str", description: "Scoring metric: 'neg_log_loss', 'accuracy', or 'f1'", default: "'neg_log_loss'" },
      { name: "corr_threshold", type: "float", description: "Minimum |correlation| to flag a substitution-risk pair", default: "0.9" },
      { name: "variance_threshold", type: "float", description: "PCA cumulative variance to retain for orthogonalization", default: "0.95" },
    ],
    commonPitfalls: [
      "Relying on a single importance method — always cross-check MDI, MDA, and SFI for consistent rankings.",
      "Ignoring substitution effects: if two features are correlated, both may appear unimportant individually but one is essential.",
      "Not using event_end_indices with overlapping labels — without purging, importance estimates are biased by leakage.",
    ],
    relatedModules: ["feature-importance", "cross-validation"],
    afmlChapters: [8],
    apiSurface: "python-only",
    pythonApis: ["feature_diagnostics.mdi_importance", "feature_diagnostics.mda_importance", "feature_diagnostics.sfi_importance", "feature_diagnostics.orthogonalize_features_pca", "feature_diagnostics.substitution_effect_report"],
  },
  {
    slug: "pipeline",
    module: "pipeline",
    subject: "Research Workflows",
    summary: "End-to-end AFML research pipeline: events → signals → portfolio → risk → backtest with leakage checks.",
    whyItExists: "Chains the core AFML steps (filtering, labeling, sizing, allocation, risk) into a single reproducible research call with built-in leakage guards.",
    keyApis: ["run_mid_frequency_pipeline", "run_mid_frequency_pipeline_frames", "summarize_pipeline"],
    formulas: [],
    examples: [
      {
        title: "Run a complete research pipeline",
        language: "python",
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
# portfolio_sharpe | realized_sharpe | value_at_risk | has_forward_look_bias`,
      },
    ],
    notes: [
      "The pipeline enforces input alignment and event ordering as leakage guards.",
      "run_mid_frequency_pipeline_frames adds Polars DataFrames to the raw dict output.",
      "summarize_pipeline extracts key metrics into a single-row DataFrame for notebook display.",
    ],
    conceptOverview: `The pipeline module orchestrates the full AFML research workflow in a single function call. It chains: CUSUM event detection → triple-barrier labeling → bet sizing → portfolio allocation → risk metrics → backtest statistics. Each stage passes its output to the next, and built-in leakage checks verify that inputs are aligned, events are chronologically ordered, and no forward-looking bias is present.

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
      "Not checking leakage_checks in the output — the pipeline flags forward-look bias but doesn't stop execution.",
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
