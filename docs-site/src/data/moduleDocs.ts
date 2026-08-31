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
  /**
   * @deprecated Superseded by `conceptOverview`, which says the same thing
   * with substance behind it. No longer rendered on any page; retained only
   * so the 39 existing entries still type-check, and safe to delete once
   * something has been done with the one-liners.
   */
  whyItExists: string;
  keyApis: string[];
  formulas: Formula[];
  examples: ExampleBlock[];
  notes: string[];
  /**
   * The three fields below are REQUIRED, and the generator asserts them.
   * They used to be optional, and 27 of the 39 modules simply omitted them:
   * the generator silently fell through to a `## Subject` heading skeleton,
   * so a missing overview shipped as a 130-word page instead of failing the
   * build. A new module with none of these is now a loud generator error.
   */
  conceptOverview: string;
  whenToUse: string;
  relatedModules: string[];
  keyParameters?: ParameterDoc[];
  commonPitfalls?: string[];
  afmlChapters?: number[];
  pythonApis?: string[];
  apiSurface?: "rust-only" | "python-only" | "both";
};

export const moduleDocs: ModuleDoc[] = [
  {
    slug: "backtest-statistics",
    conceptOverview:
      "Turns a return or equity series into the handful of statistics a strategy is actually judged on: annualised Sharpe, information ratio, the drawdown and time-under-water profile, average holding period, bet concentration, and the multiple-testing corrections — probabilistic and deflated Sharpe — that say whether a Sharpe is real. Those corrections are why this module exists rather than a two-line Sharpe helper: AFML Chapter 14's point is that a Sharpe reported without the number of trials behind it is uninterpretable.",
    whenToUse:
      "Reach for it after a backtest run, at model-selection time, and again in production monitoring. Use `deflated_sharpe_ratio` whenever the strategy is the survivor of a search — a grid, a parameter sweep, a family of variants — and pass the trial count honestly; `sharpe_ratio` alone flatters every one of them. Note that `drawdown_and_time_under_water` consumes a timestamped equity curve, not a return vector, and that every annualisation constant must match your bar frequency.",
    relatedModules: ["backtesting-engine", "strategy-risk", "risk-metrics", "synthetic-backtesting"],
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
      {
        label: "Sharpe Ratio",
        latex: "\\mathrm{SR}=\\frac{\\mu-r_f}{\\sigma}\\sqrt{n}",
        where: "$\\mu$ and $\\sigma$ are the mean and standard deviation of the per-bar returns, $r_f$ the per-bar risk-free rate, and $n$ the number of bars per year (`entries_per_year`) — the annualisation constant must match your bar frequency.",
      },
      {
        label: "Information Ratio",
        latex: "\\mathrm{IR}=\\frac{\\mu-r_b}{\\sigma_{(r-r_b)}}",
        where: "$r_b$ is the benchmark return and $\\sigma_{(r-r_b)}$ the tracking error, i.e. the standard deviation of the *excess* return series.",
      },
      {
        label: "Probabilistic Sharpe Ratio",
        latex: "\\mathrm{PSR}(\\mathrm{SR}^*)=Z\\left[\\frac{(\\widehat{\\mathrm{SR}}-\\mathrm{SR}^*)\\sqrt{T-1}}{\\sqrt{1-\\hat\\gamma_3\\widehat{\\mathrm{SR}}+\\frac{\\hat\\gamma_4-1}{4}\\widehat{\\mathrm{SR}}^2}}\\right]",
        where: "$Z[\\cdot]$ is the standard normal CDF, $\\widehat{\\mathrm{SR}}$ the observed (non-annualised) Sharpe ratio, $\\mathrm{SR}^*$ the benchmark being tested against, $T$ the number of returns, and $\\hat\\gamma_3,\\hat\\gamma_4$ the sample skewness and kurtosis. Non-normal returns lower the confidence a given Sharpe deserves.",
      },
      {
        label: "Deflated Sharpe Ratio",
        latex: "\\mathrm{DSR}=\\mathrm{PSR}(\\mathrm{SR}_0),\\qquad \\mathrm{SR}_0=\\sqrt{V[\\{\\widehat{\\mathrm{SR}}_n\\}]}\\left((1-\\gamma)Z^{-1}\\!\\left[1-\\tfrac{1}{N}\\right]+\\gamma Z^{-1}\\!\\left[1-\\tfrac{e^{-1}}{N}\\right]\\right)",
        where: "$N$ is the number of strategy variants you tried, $V[\\{\\widehat{\\mathrm{SR}}_n\\}]$ the variance of their Sharpe ratios, $\\gamma\\approx0.5772$ the Euler-Mascheroni constant, and $Z^{-1}$ the normal quantile function. $\\mathrm{SR}_0$ is the Sharpe you would *expect* the best of $N$ independent worthless strategies to post, so DSR is the PSR measured against that bar instead of against zero. `deflated_sharpe_ratio` accepts either the raw $\\{\\widehat{\\mathrm{SR}}_n\\}$ or the $(\\text{sd}, N)$ pair via `estimates_param`.",
      },
    ],
    examples: [
      {
        title: "Compute Sharpe and drawdown",
        language: "rust",
        code: `use chrono::{Duration, NaiveDateTime};\nuse openquant::backtest_statistics::{drawdown_and_time_under_water, sharpe_ratio};\n\nlet returns = vec![0.01, -0.005, 0.007, -0.002, 0.003];\nlet sharpe = sharpe_ratio(&returns, 252.0, 0.0);\n\n// Drawdown and time-under-water are computed on a *timestamped equity curve*,\n// not on the return series: the function needs the timestamps to measure how\n// long each high-water mark went un-recovered.\nlet t0 = NaiveDateTime::parse_from_str("2024-01-02 00:00:00", "%Y-%m-%d %H:%M:%S")?;\nlet mut equity = 1.0;\nlet curve: Vec<(NaiveDateTime, f64)> = returns\n    .iter()\n    .enumerate()\n    .map(|(i, r)| {\n        equity *= 1.0 + r;\n        (t0 + Duration::days(i as i64), equity)\n    })\n    .collect();\n\n// dollars = false reports each drawdown as a fraction of its high-water mark.\nlet (drawdowns, time_under_water) = drawdown_and_time_under_water(&curve, false);\nprintln!("sharpe={sharpe:.3} drawdowns={drawdowns:?} tuw={time_under_water:?}");`,
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
    conceptOverview:
      "Three validation modes over one data contract: walk-forward, purged k-fold cross-validation, and combinatorial purged CV. CPCV is the one that justifies the extra cost — instead of a single backtest path it produces phi[N,k] = C(N-1, k-1) paths, so the output is a *distribution* of per-path Sharpe ratios you can take quantiles of rather than a point estimate you can fool yourself with. Every run carries a `BacktestSafeguards` record (survivorship, look-ahead, data-mining, cost and multiple-testing controls) so the assumptions travel attached to the number.",
    whenToUse:
      "Use walk-forward when the question is \"would this have worked as deployed\"; use purged CV when you need many folds out of limited data; use CPCV when you are about to make a go/no-go decision and need to know how much of the reported Sharpe is path luck. All three require `label_spans` — the label lifetimes — not just observation timestamps, because that is what purging acts on. Compare the three modes against each other rather than averaging them into one statistic.",
    relatedModules: ["cross-validation", "sample-weights", "backtest-statistics", "synthetic-backtesting", "hyperparameter-tuning"],
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
        code: `use chrono::{Duration, NaiveDateTime};\nuse openquant::backtesting_engine::{\n    run_cpcv, BacktestData, BacktestRunConfig, BacktestSafeguards, CpcvConfig,\n};\n\nlet t0 = NaiveDateTime::parse_from_str("2024-01-02 00:00:00", "%Y-%m-%d %H:%M:%S")?;\nlet pnl: Vec<f64> = (0..240).map(|i| ((i % 7) as f64 - 3.0) / 1000.0).collect();\n\n// Each observation carries the span its label was drawn over. That span — not the\n// observation's timestamp — is what purging and the embargo act on.\nlet data = BacktestData {\n    returns: pnl.clone(),\n    label_spans: (0..240)\n        .map(|i| (t0 + Duration::days(i), t0 + Duration::days(i + 2)))\n        .collect(),\n};\n\nlet result = run_cpcv(\n    &data,\n    &BacktestRunConfig {\n        mode_provenance: "research_v3_with_costs".to_string(),\n        trials_count: 24,\n        safeguards: BacktestSafeguards {\n            survivorship_bias_control: "point-in-time universe".to_string(),\n            look_ahead_control: "lagged features".to_string(),\n            data_mining_control: "frozen split protocol".to_string(),\n            cost_assumption: "spread + slippage".to_string(),\n            multiple_testing_control: "trial count logged".to_string(),\n        },\n    },\n    &CpcvConfig { n_groups: 8, test_groups: 2, pct_embargo: 0.01 },\n    |split| Ok(split.test_indices.iter().map(|i| pnl[*i]).collect()),\n)?;\n\nprintln!("phi = {}", result.path_count);\nprintln!("path sharpe count = {}", result.path_distribution.len());`,
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
    conceptOverview:
      "The layer between a model's confidence and an order. `bet_size_probability` maps class probabilities to a signed size in [-1, 1] through the t-statistic of the probability against the null of no edge, averages sizes across bets that are still active, and discretises to your execution granularity. `bet_size_dynamic` works from a price forecast instead: given the current and maximum position it returns the target position and the limit price at which that size is justified. `bet_size_reserve` sizes from a fitted mixture of long/short concurrency rather than from any model score.",
    whenToUse:
      "Between signal generation and execution, always — a raw model score is not a position. Use the probability path when a classifier emits calibrated probabilities, the dynamic path when you have a price forecast and want a limit-order boundary, and reserve sizing when overlapping books or stacked strategies can accumulate hidden gross exposure. Set `step_size` to real lot or contract granularity, not an arbitrary decimal, and treat the limit price as a decision boundary rather than a fill you will get.",
    relatedModules: ["labeling", "sample-weights", "strategy-risk", "portfolio-optimization"],
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
    conceptOverview:
      "Markowitz's Critical Line Algorithm in the Bailey-Lopez de Prado formulation: the exact solution to the constrained mean-variance problem with inequality bounds on every weight. Rather than calling a general quadratic solver it walks the efficient frontier from the maximum-return corner, computing each turning point where an asset enters or leaves the free set. That yields the whole frontier rather than one point on it, and it terminates — which quadratic solvers on near-singular covariance matrices frequently do not.",
    whenToUse:
      "Use it when you need the full efficient frontier, when weight bounds are binding, or when a general optimiser is returning unstable or non-converging weights on an ill-conditioned covariance. If you only want one portfolio and the covariance is well behaved, `portfolio_optimization` is the shorter path. If the covariance itself is the problem, prefer `hrp`, which never inverts it. CLA still needs expected returns, so it inherits their estimation error.",
    relatedModules: ["portfolio-optimization", "hrp", "hcaa", "risk-metrics"],
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
    conceptOverview:
      "Dependence measures that survive non-linearity, which Pearson correlation does not. Distance correlation is zero only under genuine independence. Mutual information and variation of information are information-theoretic and need a binning choice, which `get_optimal_number_of_bins` supplies. The angular distances turn a correlation into a proper metric — sqrt(2(1-rho)) and its absolute and squared variants — which is what hierarchical clustering needs in order to be well posed at all.",
    whenToUse:
      "Use it upstream of any clustering or feature-pruning step: `hrp`, `hcaa` and `onc` all consume a distance matrix, and feeding them raw correlation silently assumes the relationship is linear. Use distance correlation when you suspect a non-monotone relationship, and variation of information when you want a true metric on discrete variables. Bin selection materially changes mutual-information estimates, so fix it explicitly and record it alongside the result.",
    relatedModules: ["hrp", "hcaa", "onc", "feature-importance", "microstructural-features"],
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
    conceptOverview:
      "Standard k-fold leaks in finance because labels overlap: an observation's label is realised over a span of bars, and a training observation whose span touches a test observation's span has effectively seen the answer. `PurgedKFold` takes those spans as `samples_info_sets`, drops the overlapping training observations (purging), then drops a further `pct_embargo` fraction of observations immediately after each test fold to catch the serial correlation the spans do not literally share.",
    whenToUse:
      "Use it in place of plain k-fold for every model whose labels are event-based — which is every model built on `labeling`. `ml_cross_val_score` wraps it for scoring and `ml_get_train_times` exposes the purged training index if you are driving your own loop. Report fold-to-fold variance, not only the mean: a high mean with high variance across purged folds usually means the leakage moved rather than disappeared.",
    relatedModules: ["labeling", "sample-weights", "backtesting-engine", "hyperparameter-tuning", "feature-importance"],
    module: "cross_validation",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Purged cross-validation utilities designed for label overlap and leakage control.",
    whyItExists: "Time-dependent labels violate IID assumptions; purging/embargoing reduces leakage bias.",
    keyApis: ["ml_cross_val_score", "ml_get_train_times", "PurgedKFold", "Scoring"],
    formulas: [
      {
        label: "Purged Train Set",
        latex: "\\mathcal{T}_{\\text{train}}=\\mathcal{T}\\setminus\\{i:\\;\\exists j\\in\\mathcal{T}_{\\text{test}},\\;[t_{i,0},t_{i,1}]\\cap[t_{j,0},t_{j,1}]\\neq\\varnothing\\}\\setminus\\mathcal{E}",
        where: "$[t_{i,0},t_{i,1}]$ is observation $i$'s label span — the `samples_info_sets` entry `PurgedKFold::new` requires. *Purging* drops any training observation whose label lifetime overlaps a test label's; $\\mathcal{E}$ is the embargo set below. Overlap, not adjacency, is what leaks: two observations sampled a month apart still share information if their labels resolve on the same bar.",
      },
      {
        label: "Embargo",
        latex: "e=\\lfloor p\\cdot T\\rfloor,\\qquad \\mathcal{E}=\\{i:\\;\\max(\\mathcal{T}_{\\text{test}})<i\\le\\max(\\mathcal{T}_{\\text{test}})+e\\}",
        where: "$T$ is the total number of observations and $p$ the `pct_embargo` fraction (0.01 = 1%), so $e$ is an observation count. The embargo drops the $e$ observations immediately *after* each test fold, which catches serial correlation that purging alone misses because the label spans do not literally overlap.",
      },
    ],
    examples: [
      {
        title: "Configure PurgedKFold",
        language: "rust",
        code: `use chrono::{Duration, NaiveDateTime};\nuse openquant::cross_validation::PurgedKFold;\n\nlet t0 = NaiveDateTime::parse_from_str("2024-01-02 00:00:00", "%Y-%m-%d %H:%M:%S")?;\n\n// samples_info_sets is one (label_start, label_end) span per observation. It is\n// mandatory: without label lifetimes there is nothing to purge against.\nlet samples_info_sets: Vec<(NaiveDateTime, NaiveDateTime)> = (0..100)\n    .map(|i| (t0 + Duration::days(i), t0 + Duration::days(i + 3)))\n    .collect();\n\n// n_splits = 5 folds; pct_embargo = 0.01 drops a further 1% of the sample\n// immediately after each test fold. new() validates and returns a Result.\nlet cv = PurgedKFold::new(5, samples_info_sets, 0.01)?;\n\nlet splits = cv.split(100)?;\nprintln!("{} folds; fold 0 keeps {} training rows", splits.len(), splits[0].0.len());`,
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
    conceptOverview:
      "Grid and randomized search that run under `PurgedKFold` rather than plain k-fold, so the tuning loop cannot buy its score with leakage. `randomized_search` samples from `RandomParamDistribution`, including log-uniform — the right prior for scale parameters such as C and gamma — and AFML Chapter 9's argument is that beyond a couple of dimensions random sampling dominates grid search per unit of compute. The scoring choice exposed by `SearchScoring` is an economic decision, not a statistical one.",
    whenToUse:
      "Any time you tune a model whose labels overlap. Use `NegLogLoss` when probabilities drive position size, since it penalises confident wrong answers the way a bet does; use `Accuracy` only when every prediction carries similar economic weight; use `BalancedAccuracy` for the severe class imbalance typical of meta-labelling, where recall of the positive class is what matters. Pass `sample_weight` from `sample_weights` — tuning on unweighted overlapping observations rewards the wrong model.",
    relatedModules: ["cross-validation", "sample-weights", "sb-bagging", "ensemble-methods", "backtesting-engine"],
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
        code: `use chrono::{Duration, NaiveDateTime};\nuse openquant::cross_validation::SimpleClassifier;\nuse openquant::hyperparameter_tuning::{\n    randomized_search, ParamSet, RandomParamDistribution, SearchData, SearchScoring,\n};\nuse std::collections::BTreeMap;\n\n// The search builds a fresh model from each sampled parameter set.\nstruct Logistic {\n    c: f64,\n}\nimpl SimpleClassifier for Logistic {\n    fn fit(&mut self, _x: &[Vec<f64>], _y: &[f64], _sample_weight: Option<&[f64]>) {}\n    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {\n        x.iter().map(|row| 1.0 / (1.0 + (-self.c * row[0]).exp())).collect()\n    }\n}\nlet build_model =\n    |params: &ParamSet| Logistic { c: params["C"].as_f64().unwrap_or(1.0) };\n\nlet mut space = BTreeMap::new();\nspace.insert("C".to_string(), RandomParamDistribution::LogUniform { low: 1e-2, high: 1e2 });\nspace.insert("gamma".to_string(), RandomParamDistribution::LogUniform { low: 1e-3, high: 1e1 });\n\nlet t0 = NaiveDateTime::parse_from_str("2024-01-02 00:00:00", "%Y-%m-%d %H:%M:%S")?;\nlet x: Vec<Vec<f64>> = (0..60).map(|i| vec![(i as f64 - 30.0) / 30.0]).collect();\nlet y: Vec<f64> = (0..60).map(|i| if i >= 30 { 1.0 } else { 0.0 }).collect();\nlet w = vec![1.0f64; 60];\n// Label spans again — the search purges internally, so it needs them.\nlet info_sets: Vec<(NaiveDateTime, NaiveDateTime)> =\n    (0..60).map(|i| (t0 + Duration::days(i), t0 + Duration::days(i + 2))).collect();\n\nlet result = randomized_search(\n    build_model,\n    &space,\n    25,   // n_iter — parameter sets sampled\n    42,   // seed\n    SearchData { x: &x, y: &y, sample_weight: Some(&w), samples_info_sets: &info_sets },\n    5,    // n_splits\n    0.01, // pct_embargo\n    SearchScoring::NegLogLoss,\n)?;\nprintln!("best score = {} with {:?}", result.best_score, result.best_params);`,
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
    conceptOverview:
      "Exact Fit of the first 3, 4 or 5 Moments: fits a mixture of two Gaussians by matching sample moments instead of by maximum likelihood. `M2N` takes the observed moments and searches over the second mean and the mixing probability, solving the remaining parameters analytically at each candidate (`iter_4` and `iter_5` for the four- and five-moment variants); `most_likely_parameters` then picks the modal solution across that search. It is fast and derivative-free, which is what makes it usable as an initialiser.",
    whenToUse:
      "Use it when a return or bet-outcome distribution is visibly bimodal — two regimes, or a mixture of trades that ran and trades that were stopped — and you want the components without paying for EM. It is the standard way to obtain the mixture parameters `bet_size_reserve` needs. Because it works from higher moments it is sensitive to tail estimation noise, so on small samples treat its output as an initialisation for a heavier optimiser rather than a final answer.",
    relatedModules: ["bet-sizing", "backtest-statistics", "strategy-risk"],
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
    conceptOverview:
      "The diagnostics behind the bagging-versus-boosting choice rather than another ensemble implementation. `bias_variance_noise` decomposes the error; `average_pairwise_prediction_correlation` measures how correlated your base learners actually are; `bagging_ensemble_variance` turns that rho into the variance a bagged ensemble can reach, sigma^2(rho + (1-rho)/N). The consequence AFML Chapter 6 draws is the useful one: as N grows the ensemble variance floors at sigma^2·rho, so with highly correlated learners more estimators buy nothing at all.",
    whenToUse:
      "Use it before scaling an ensemble. If measured rho is 0.9, going from 20 to 200 estimators is wasted compute, and `recommend_bagging_vs_boosting` will say so from the numbers rather than from folklore. Reach for bagging when the base learner is unstable (variance-dominated) and boosting when it is weak (bias-dominated). Under heavy label overlap use `sequential_bootstrap_sample_indices` instead of the IID bootstrap, or the bags will be near-duplicates of each other.",
    relatedModules: ["sb-bagging", "sampling", "sample-weights", "cross-validation", "feature-importance"],
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
    conceptOverview:
      "The ETF trick turns a series of futures contracts — each with its own roll, financing cost and carry — into one continuous, reinvestable price series that a backtest can treat like a tradable instrument. `EtfTrick` consumes aligned open, close, allocation and cost tables plus optional financing rates and produces a NAV series; `get_futures_roll_series` applies backward or forward roll adjustment to a single contract chain. Both exist because naively concatenating contract prices manufactures a return at every roll.",
    whenToUse:
      "Use it whenever a backtest spans a contract roll, or whenever the traded object is a basket whose weights change over time. Suspiciously smooth PnL around roll dates is the symptom of skipping it. Costs and financing rates must come from the same clock as the price data, and the contract calendar assumptions are worth verifying against the exchange rather than inferring from the data. This module is Rust-only — no Python bindings are exposed.",
    relatedModules: ["data-structures", "backtesting-engine", "backtest-statistics", "bet-sizing"],
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
    conceptOverview:
      "The three AFML Chapter 8 importance methods on the Rust side, each with a different blind spot. MDI is in-sample and tree-specific: it sums each feature's impurity decrease across splits, cheap but defeated by substitution, since two interchangeable features split the credit and both then look weak. MDA permutes a feature in the *test* fold and measures the score drop, so it is model-agnostic and out-of-sample but still substitution-prone. Single-feature importance trains on one feature at a time, immune to substitution but blind to interactions. `feature_pca_analysis` cross-checks the ranking against an unsupervised one.",
    whenToUse:
      "Run at least two of the three: agreement between MDI and MDA is evidence, MDI alone is not. Prefer MDA when leakage risk is high, since it is the only one scored out of sample — and give it purged splits from `cross_validation`, not a fold count. Compare rankings across time windows before trusting them; a feature that is important in only one regime is a feature that will fail in the next.",
    relatedModules: ["feature-diagnostics", "cross-validation", "sample-weights", "codependence", "fingerprint"],
    module: "feature_importance",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Feature ranking methods: MDI, MDA, and single-feature importance with PCA diagnostics.",
    whyItExists: "Improves model interpretability and helps remove unstable or redundant features.",
    keyApis: ["mean_decrease_impurity", "mean_decrease_accuracy", "single_feature_importance", "feature_pca_analysis"],
    formulas: [
      {
        label: "MDI — Mean Decrease Impurity",
        latex: "I_j=\\frac{1}{B}\\sum_{b=1}^{B}\\;\\sum_{t\\in T_j^{(b)}} p(t)\\,\\Delta i(t)",
        where: "$T_j^{(b)}$ are the nodes of tree $b$ that split on feature $j$, $p(t)$ the fraction of samples reaching node $t$, and $\\Delta i(t)$ the impurity drop at that split. This is the tree-based definition: it is in-sample, computable only for tree ensembles, and `mean_decrease_impurity` takes the per-tree importance vectors a fitted forest already exposes. The Python `feature_diagnostics.mdi_importance` uses a different, linear-model estimator under the same acronym — see that page.",
      },
      {
        label: "MDA — Mean Decrease Accuracy",
        latex: "I_j=\\frac{1}{K}\\sum_{k=1}^{K}\\big(S_k-S_{k,\\text{perm}(j)}\\big)",
        where: "$S_k$ is the out-of-sample score on purged fold $k$ and $S_{k,\\text{perm}(j)}$ the same score after column $j$ is randomly permuted in the test set. Unlike MDI it is model-agnostic and out-of-sample, which is why `mean_decrease_accuracy` demands the CV splits rather than a fold count.",
      },
    ],
    examples: [
      {
        title: "Run MDA with classifier",
        language: "rust",
        code: `use openquant::cross_validation::{Scoring, SimpleClassifier};\nuse openquant::feature_importance::mean_decrease_accuracy;\n\n// MDA works with any model implementing SimpleClassifier; this stand-in keeps\n// the example self-contained.\nstruct MeanThreshold {\n    threshold: f64,\n}\nimpl SimpleClassifier for MeanThreshold {\n    fn fit(&mut self, x: &[Vec<f64>], _y: &[f64], _sample_weight: Option<&[f64]>) {\n        self.threshold = x.iter().map(|row| row[0]).sum::<f64>() / x.len() as f64;\n    }\n    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {\n        x.iter().map(|row| if row[0] > self.threshold { 0.9 } else { 0.1 }).collect()\n    }\n}\n\nlet x: Vec<Vec<f64>> = (0..40).map(|i| vec![i as f64, (i % 7) as f64]).collect();\nlet y: Vec<f64> = (0..40).map(|i| if i >= 20 { 1.0 } else { 0.0 }).collect();\nlet feature_names = vec!["trend".to_string(), "noise".to_string()];\n\n// MDA is measured out of sample, so it takes the *already-purged splits* — not a\n// fold count. Feed it the output of PurgedKFold::split so the score is leak-free.\nlet splits = vec![\n    ((0..20).collect::<Vec<usize>>(), (20..40).collect::<Vec<usize>>()),\n    ((20..40).collect::<Vec<usize>>(), (0..20).collect::<Vec<usize>>()),\n];\n\nlet mut model = MeanThreshold { threshold: 0.0 };\nlet importance = mean_decrease_accuracy(\n    &mut model,\n    &x,\n    &y,\n    &feature_names,\n    &splits,\n    None, // sample_weight — pass uniqueness weights from \`sample_weights\` in practice\n    Scoring::Accuracy,\n)?;\n\nprintln!("trend: mean={:.4} std={:.4}", importance["trend"].mean, importance["trend"].std);`,
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
      {
        label: "Symmetric CUSUM Filter",
        latex: "S_t^{+}=\\max\\!\\left(0,\\,S_{t-1}^{+}+r_t\\right),\\qquad S_t^{-}=\\min\\!\\left(0,\\,S_{t-1}^{-}+r_t\\right),\\qquad \\text{event at }t\\iff S_t^{+}>h_t\\;\\lor\\;S_t^{-}<-h_t",
        where: "$r_t=\\ln(p_t/p_{t-1})$ is the log return and $h_t$ the threshold — a constant for `Threshold::Scalar`, a per-bar series for `Threshold::Dynamic`. Both arms are needed: $S^{+}$ alone only ever detects upward runs. Whichever arm breaches is reset to $0$ and the bar is emitted as an event, so the filter measures *runs* away from the last event rather than a cumulative level.",
      },
      {
        label: "Z-score Filter",
        latex: "z_t=\\frac{x_t-\\mu_t}{\\sigma_t},\\qquad \\text{event at }t\\iff|z_t|>h",
        where: "$\\mu_t$ and $\\sigma_t$ are the rolling mean and standard deviation over the lookback window ending at $t$.",
      },
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
    conceptOverview:
      "Model fingerprinting decomposes a fitted model's behaviour into a linear effect, a non-linear effect and pairwise interaction effects per feature, by sweeping each feature across a grid and measuring how the prediction moves. The result describes *what the model learned* rather than how well it scored — two models with identical accuracy can have entirely different fingerprints, and only one of them may be relying on something that will still be there next quarter.",
    whenToUse:
      "Use it after fitting and before deploying, and again on every retrain: comparing fingerprints across retrains is a drift signal that accuracy metrics do not give you. Use the pairwise effects to find interaction risk, since a large pairwise term means the model's response to one feature depends on another, which makes its extrapolation fragile. It works with any model — implement `RegressionPredictor` or `ClassificationPredictor` and pass it to `fit`.",
    relatedModules: ["feature-importance", "feature-diagnostics", "ensemble-methods", "backtesting-engine"],
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
        code: `use openquant::fingerprint::{RegressionModelFingerprint, RegressionPredictor};\n\n// Fingerprinting is model-agnostic: anything that can predict will do.\nstruct LinearModel {\n    beta: Vec<f64>,\n}\nimpl RegressionPredictor for LinearModel {\n    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {\n        x.iter()\n            .map(|row| row.iter().zip(self.beta.iter()).map(|(v, b)| v * b).sum())\n            .collect()\n    }\n}\n\nlet model = LinearModel { beta: vec![1.5, -0.5] };\nlet x: Vec<Vec<f64>> =\n    (0..50).map(|i| vec![i as f64 / 50.0, ((i % 5) as f64) / 5.0]).collect();\n\n// new() takes no arguments; the model and data go to fit(), which needs &mut self.\n// num_values is the partial-dependence grid resolution.\nlet mut fingerprint = RegressionModelFingerprint::new();\nfingerprint.fit(&model, &x, 10, Some(&[(0, 1)]))?;\n\n// The accessor is get_effects(), returning (linear, non-linear, optional pairwise).\nlet (linear, non_linear, pairwise) = fingerprint.get_effects()?;\nprintln!("linear={:?}", linear.norm);\nprintln!("non_linear={:?}", non_linear.norm);\nprintln!("pairwise={:?}", pairwise.map(|p| p.norm.clone()));`,
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
    conceptOverview:
      "Hierarchical Clustering Asset Allocation generalises HRP's recursive bisection to risk measures other than variance. Seriation and the cluster tree are built the same way, but the split at each node weights the two sides by the chosen `allocation_metric` — cluster variance, standard deviation, Sharpe ratio, expected shortfall or conditional drawdown — so the same hierarchy can express a tail-risk budget rather than only a variance budget. Like HRP it never inverts the covariance matrix.",
    whenToUse:
      "Use it in place of `hrp` when your risk budget is not variance: expected shortfall or conditional drawdown for a drawdown-controlled mandate, Sharpe when you have return views you are willing to defend. Use `hrp` when you do not, since the variance split needs no expected-return estimate at all. The clustering is only as good as the distance fed to it, so build that with `codependence` rather than raw correlation, and sanity-check the cluster count with `onc`.",
    relatedModules: ["hrp", "onc", "codependence", "portfolio-optimization", "cla"],
    module: "hcaa",
    subject: "Portfolio Construction and Risk",
    summary: "Hierarchical Clustering Asset Allocation variant with cluster-level constraints.",
    whyItExists: "Allocates capital by hierarchy to reduce concentration and covariance-estimation fragility.",
    keyApis: ["HierarchicalClusteringAssetAllocation", "HcaaError"],
    formulas: [
      {
        label: "Cluster Risk",
        latex: "\\sigma_C^2=w_C^{\\top}\\Sigma_C w_C",
        where: "$\\Sigma_C$ is the covariance sub-matrix of cluster $C$ and $w_C$ its inverse-variance weights, normalised to sum to one within the cluster.",
      },
      {
        label: "Recursive Bisection Split",
        latex: "\\alpha=1-\\frac{m_{\\text{left}}}{m_{\\text{left}}+m_{\\text{right}}},\\qquad w_{\\text{left}}\\mathrel{*}=\\alpha,\\quad w_{\\text{right}}\\mathrel{*}=1-\\alpha",
        where: "$m_C$ is the risk of cluster $C$ under the chosen `allocation_metric`: cluster variance ($\\sigma_C^2$), standard deviation ($\\sigma_C$), expected shortfall, or conditional drawdown. Lower risk on one side means a larger $\\alpha$ for that side. This generalises the HRP split, which is the `minimum_variance` case. Two branches invert the sign: `sharpe_ratio` allocates $\\alpha=\\mathrm{SR}_{\\text{left}}/(\\mathrm{SR}_{\\text{left}}+\\mathrm{SR}_{\\text{right}})$ because higher is better there, and `equal_weighting` skips the split entirely.",
      },
    ],
    examples: [
      {
        title: "Fit HCAA allocator",
        language: "rust",
        code: `use nalgebra::DMatrix;\nuse openquant::hcaa::HierarchicalClusteringAssetAllocation;\n\nlet asset_names: Vec<String> =\n    ["SPY", "TLT", "GLD", "HYG"].iter().map(|s| s.to_string()).collect();\n// rows = observations, cols = assets, in the same order as \`asset_names\`.\nlet prices = DMatrix::from_fn(250, 4, |i, j| 100.0 + (i as f64) * 0.05 + (j as f64) * 3.0);\n\n// The constructor argument selects how expected returns are estimated\n// ("mean" or "exponential"); it is not optional.\nlet mut hcaa = HierarchicalClusteringAssetAllocation::new("mean");\n\n// allocate() fills the struct in place and returns Result<(), HcaaError>.\n// It does not return the weights — read them from \`hcaa.weights\` afterwards.\nhcaa.allocate(\n    &asset_names,\n    Some(&prices),      // asset_prices\n    None,               // asset_returns\n    None,               // covariance_matrix\n    None,               // expected_asset_returns\n    "minimum_variance", // allocation_metric\n    0.05,               // confidence_level, used by the tail-risk metrics\n    None,               // optimal_num_clusters — inferred when None\n    None,               // resample_by\n)?;\n\nprintln!("weights: {:?}", hcaa.weights);\nprintln!("seriation order: {:?}", hcaa.ordered_indices);`,
      },
    ],
    notes: ["Cluster linkage choices influence allocations.", "Use with robust codependence distances when possible."],
    apiSurface: "both",
    pythonApis: ["hcaa.allocate_hcaa"],
  },
  {
    slug: "hrp",
    conceptOverview:
      "Hierarchical Risk Parity replaces matrix inversion with a tree. It clusters assets on a correlation distance, reorders the covariance matrix so that similar assets sit adjacent (quasi-diagonalisation), then recursively bisects that ordering, splitting capital between the two halves in inverse proportion to their cluster variance. Nothing is inverted, so the numerical instability that makes Markowitz weights swing violently under a noisy covariance estimate simply does not arise.",
    whenToUse:
      "Use it when the asset count is large relative to the sample, when the covariance estimate is noisy, or whenever mean-variance weights are unstable between rebalances — which out of sample is most of the time. It needs no expected returns, which is both its robustness and its limit: if you have return views you trust, `cla` or `portfolio_optimization` will use them and HRP will not. Keep the asset ordering you pass in aligned with the dendrogram order you read back.",
    relatedModules: ["hcaa", "codependence", "onc", "portfolio-optimization", "cla"],
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
        code: `use nalgebra::DMatrix;\nuse openquant::hrp::HierarchicalRiskParity;\n\nlet asset_names: Vec<String> =\n    ["SPY", "TLT", "GLD", "HYG"].iter().map(|s| s.to_string()).collect();\n// rows = observations, cols = assets, in the same order as \`asset_names\`.\nlet prices = DMatrix::from_fn(250, 4, |i, j| 100.0 + (i as f64) * 0.05 + (j as f64) * 3.0);\n\nlet mut hrp = HierarchicalRiskParity::new();\n\n// allocate() mutates the struct and returns Result<(), HrpError>; the weights are\n// read back from \`hrp.weights\`. Exactly one of prices / returns / covariance must\n// be supplied.\nhrp.allocate(\n    &asset_names,\n    Some(&prices), // asset_prices\n    None,          // asset_returns\n    None,          // covariance_matrix\n    None,          // resample_by\n    false,         // use_shrinkage — Ledoit-Wolf shrinkage on the covariance\n)?;\n\nprintln!("weights: {:?}", hrp.weights);\nprintln!("seriation order: {:?}", hrp.ordered_indices);`,
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
    conceptOverview:
      "Features computed from bar-level order flow rather than from price alone, in three families: effective-spread proxies (Roll, Corwin-Schultz), price-impact coefficients (Kyle's lambda, Amihud, Hasbrouck) and flow-toxicity or entropy measures (VPIN, plus Shannon, Lempel-Ziv and plug-in entropy over encoded tick signs). Together they estimate what OHLC bars omit: how expensive the instrument is to trade, and how likely it is that the counterparty knows something you do not.",
    whenToUse:
      "Use them as features when the edge or its cost depends on liquidity — execution models, regime detection, and any signal that decays with trade size. VPIN in particular is an early-warning indicator for flow toxicity ahead of liquidity events. Normalise within venue and time bucket before comparing across assets, since these are strongly regime-dependent, and freeze the symbol encoding used for entropy features or the values will not be comparable between training and production.",
    relatedModules: ["data-structures", "streaming-hpc", "structural-breaks", "filters", "codependence"],
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
          "\\mathrm{VPIN}_t=\\frac{1}{V_t}\\cdot\\frac{1}{n}\\sum_{i=t-n+1}^{t}\\left|V_i^{B}-V_i^{S}\\right|,\\qquad H=-\\sum_j p_j\\log p_j",
        where: "$V_i^{B}$ and $V_i^{S}$ are buy- and sell-initiated volume in bar $i$ (`get_bvc_buy_volume` will estimate the split when it is not observed), $V_t$ the current bar's total volume, and $n$ the rolling `window`. The normaliser sits *outside* the sum because bars are not equal-volume: `get_vpin` averages the imbalance over the window and then scales by the latest bar. The equal-volume-bucket form used by [`streaming-hpc`](/modules/streaming-hpc/) divides each term by the same constant bucket size instead; the two agree when bars carry equal volume. $H$ is the entropy of the tick-sign message, with $p_j$ the empirical frequency of symbol $j$.",
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
    conceptOverview:
      "Optimal Number of Clusters: runs k-means over a correlation matrix for a range of k, scores each partition by the mean-to-standard-deviation ratio of its silhouette scores, then re-clusters only the clusters that scored badly and keeps the result if it improves. Base k-means is unstable in both k and initialisation, so ONC restarts it `repeat` times and keeps the best — the point is a defensible cluster count, not a fast one.",
    whenToUse:
      "Use it before any hierarchical allocation to decide how many clusters the universe actually supports, instead of hard-coding a number; its answer feeds `hcaa`'s `optimal_num_clusters` directly. Use it also to test whether a claimed grouping — sectors, factors, strategy families — survives contact with the data. Clean the correlation matrix first: on an unstable universe ONC will happily find structure in noise and report a confident k for it.",
    relatedModules: ["hcaa", "hrp", "codependence", "portfolio-optimization"],
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
        code: `use nalgebra::DMatrix;\nuse openquant::onc::get_onc_clusters;\n\n// ONC consumes a *correlation* matrix, not raw prices — build one from your\n// codependence measure of choice first.\nlet corr = DMatrix::from_row_slice(\n    4,\n    4,\n    &[\n        1.00, 0.85, 0.10, 0.05, //\n        0.85, 1.00, 0.12, 0.08, //\n        0.10, 0.12, 1.00, 0.78, //\n        0.05, 0.08, 0.78, 1.00,\n    ],\n);\n\n// \`repeat\` is the number of k-means restarts used to stabilise the partition.\nlet out = get_onc_clusters(&corr, 20)?;\nprintln!("{} clusters", out.clusters.len());\nprintln!("silhouette scores: {:?}", out.silhouette_scores);`,
      },
    ],
    notes: ["Run with repeated seeds/restarts for robust k selection.", "Use correlation cleaning before clustering unstable universes."],
    apiSurface: "both",
    pythonApis: ["onc.get_onc_clusters"],
  },
  {
    slug: "portfolio-optimization",
    conceptOverview:
      "Mean-variance allocation with the constraints production actually needs. Four objectives — inverse variance, minimum volatility, maximum Sharpe, and efficient risk (maximum return at a target volatility) — each with a `_with` variant taking `AllocationOptions`: per-asset bounds, a global tuple bound, the expected-returns estimator (historical mean or exponentially weighted) and price resampling. The options struct is really the module; the constraint set matters far more to out-of-sample behaviour than the choice of objective.",
    whenToUse:
      "Use it when you have expected returns you are willing to defend, and `hrp` or `hcaa` when you do not. Treat `allocate_inverse_variance` as the baseline to beat — it uses no return estimate at all and is hard to improve on out of sample. Cap concentration through `bounds` before tuning the objective, and monitor turnover and the drift between target and filled weights, which usually account for more of the backtest-to-live gap than the optimiser does.",
    relatedModules: ["hrp", "hcaa", "cla", "risk-metrics", "backtest-statistics"],
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
        code: `use nalgebra::DMatrix;\nuse openquant::portfolio_optimization::{\n    allocate_max_sharpe_with, AllocationOptions, ReturnsMethod,\n};\nuse std::collections::HashMap;\n\n// rows = time, cols = assets\nlet prices = DMatrix::from_fn(252, 6, |i, j| 100.0 + (i as f64) * 0.03 + (j as f64) * 2.0);\n\nlet mut bounds = HashMap::new();\n// Cap concentration in the first asset; the tuple bound applies to the rest.\nbounds.insert(0usize, (0.0, 0.20));\n\nlet opts = AllocationOptions {\n    risk_free_rate: 0.02,\n    returns_method: ReturnsMethod::Exponential { span: 60 },\n    resample_by: Some("W"),\n    bounds: Some(bounds),\n    tuple_bounds: Some((0.0, 0.40)),\n    ..Default::default()\n};\n\nlet constrained = allocate_max_sharpe_with(&prices, &opts)?;\nassert!(constrained.weights.iter().all(|w| *w >= -1e-10));`,
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
    conceptOverview:
      "Downside risk measures over a return series or a return panel: value at risk (the quantile at the given confidence level), expected shortfall (the mean loss beyond it), conditional drawdown at risk, and portfolio variance from a covariance matrix and a weight vector. Expected shortfall and CDaR are subadditive where VaR is not, which is why a risk budget built on VaR alone can be gamed by splitting one position across two sleeves.",
    whenToUse:
      "Use it for portfolio-level guardrails and risk budgets, and as the input when `hcaa` should allocate on tail risk rather than on variance. Prefer expected shortfall to VaR whenever the number will be summed across books. These are non-parametric estimates, so they need enough tail observations to mean anything: at 95% confidence a 200-observation sample rests on ten points. All of them are `&self` methods on a unit struct, and the `_from_matrix` variants take return panels.",
    relatedModules: ["hcaa", "portfolio-optimization", "backtest-statistics", "strategy-risk"],
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
        code: `use openquant::risk_metrics::RiskMetrics;\n\nlet returns = vec![-0.02, 0.01, -0.005, 0.003, 0.004];\n\n// These are &self methods on a unit struct, not associated functions: they need\n// a receiver. \`confidence_level\` is the tail probability (0.05 = 95% VaR).\nlet metrics = RiskMetrics;\nlet var_95 = metrics.calculate_value_at_risk(&returns, 0.05)?;\nlet es_95 = metrics.calculate_expected_shortfall(&returns, 0.05)?;\n\nprintln!("VaR(95%) = {var_95:.4}, ES(95%) = {es_95:.4}");`,
      },
    ],
    notes: ["Non-parametric estimates need enough tail observations.", "Use matrix variants for multi-asset return panels."],
    apiSurface: "both",
    pythonApis: ["risk.calculate_value_at_risk", "risk.calculate_expected_shortfall", "risk.calculate_conditional_drawdown_risk", "risk.calculate_variance", "risk.calculate_value_at_risk_from_matrix", "risk.calculate_expected_shortfall_from_matrix", "risk.calculate_conditional_drawdown_risk_from_matrix"],
  },
  {
    slug: "strategy-risk",
    conceptOverview:
      "AFML Chapter 15 asks a question portfolio risk does not: given the precision, payout asymmetry and bet frequency this strategy actually achieved, what is the probability that the *process* fails to reach its Sharpe target? The symmetric and asymmetric helpers invert the Sharpe relation for whichever variable you are solving for — implied precision, implied frequency — and `estimate_strategy_failure_probability` bootstraps the realised bet outcomes, fits a KDE to the resulting precision distribution, and reports the mass falling below the precision the target Sharpe requires.",
    whenToUse:
      "Use it at strategy-approval time and then as a standing monitor: the implied precision threshold p* is a concrete kill criterion, and a strategy whose realised precision drifts toward it is failing before its PnL says so. Analyse the manager-controlled inputs — the payouts and the bet count — separately from market-determined precision, because the first are design choices and the second is not. This is strategy viability; use `risk_metrics` for holdings and tail risk.",
    relatedModules: ["risk-metrics", "backtest-statistics", "bet-sizing", "backtesting-engine"],
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
    conceptOverview:
      "AFML Chapter 20's atom/molecule model: a job is a list of independent atoms, atoms are grouped into molecules, and molecules are dispatched to workers. What this adds over a plain thread pool is the partitioning choice — linear for uniform-cost atoms, nested for the triangular workloads that dominate this library, where atom k touches k earlier observations — together with a metrics report and a serial mode whose callback semantics are identical to the threaded one.",
    whenToUse:
      "Use it for any embarrassingly parallel research loop: per-asset feature computation, bootstrap replicas, parameter sweeps. Choose `PartitionStrategy::Nested` when per-atom cost grows with the atom index, otherwise the final molecule becomes the whole runtime; choose `Linear` when atoms cost the same. Debug with `ExecutionMode::Serial` first — the callback contract is unchanged, so a bug that reproduces there is not a concurrency bug and you have just halved the search space.",
    relatedModules: ["streaming-hpc", "combinatorial-optimization", "sampling", "backtesting-engine"],
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
        where: "$N$ is the number of atoms, $M$ the number of molecules (`mp_batches` x workers), and molecule $i$ covers atoms $[b_{i-1},b_i)$. Every molecule gets the same *count* of atoms, which is correct only when atoms cost the same.",
      },
      {
        label: "Nested Partition Boundary",
        latex: "b_i=\\left\\lfloor N\\sqrt{\\frac{i}{M}}\\right\\rfloor,\\;i=0,\\dots,M",
        where: "The same $N$ and $M$, for the triangular workloads that dominate this library — building an overlap or codependence matrix, where atom $k$ touches $k$ earlier observations, so its cost grows linearly with $k$. Later molecules therefore hold fewer atoms.",
      },
      {
        label: "Equal-Cost Condition",
        latex: "\\text{cost}(i)\\;\\propto\\;\\frac{b_i^2-b_{i-1}^2}{2}=\\frac{N^2}{2M}\\quad\\text{for every }i",
        where: "$b_i$ and $M$ are as above. This is why the square root is there: if atom $k$ costs $\\propto k$, a molecule spanning $[b_{i-1},b_i)$ costs $\\propto(b_i^2-b_{i-1}^2)/2$; substituting $b_i=N\\sqrt{i/M}$ makes that $N^2/(2M)$, the same for every molecule. Linear partitioning on the same workload leaves the last molecule roughly $2M-1$ times more expensive than the first, and the run is only as fast as that straggler.",
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
    conceptOverview:
      "AFML Chapter 21 tooling for discrete, path-dependent problems, built around keeping the integer structure explicit rather than relaxing it away. `DecisionSchema` describes an integer decision space and `solve_exact` enumerates it. `TradingTrajectorySchema` describes a trading path — per-step trade bounds, inventory limits, an optional terminal inventory — and `enumerate_trading_paths` produces every feasible trajectory, which `evaluate_trading_path` scores against expected returns, risk aversion, market impact and a fixed per-ticket cost.",
    whenToUse:
      "Use exact enumeration on small instances as a correctness oracle: `compare_exact_and_adapter` exists precisely so a heuristic or external solver can be validated against ground truth before it is trusted at scale. The decision space grows exponentially in horizon and dimension and `max_paths` will stop you — treat that as the signal to move to an adapter, not to raise the cap. The fixed ticket cost is what makes the problem genuinely combinatorial; without it a continuous relaxation would do.",
    relatedModules: ["hpc-parallel", "bet-sizing", "portfolio-optimization", "backtesting-engine"],
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
    conceptOverview:
      "AFML Chapter 22 is about turnaround time rather than throughput: an early-warning metric that arrives after the event is worthless however fast it was computed. This module keeps VPIN and venue-concentration HHI as incremental state with bounded memory — VPIN fills equal-volume buckets and retains a fixed-length window of completed ones, HHI retains a fixed event lookback — so per-event cost and memory stay constant however long the stream runs. `run_streaming_pipeline_parallel` fans many streams across workers through `hpc_parallel`.",
    whenToUse:
      "Use it for live or replayed order-flow monitoring where the alert has to fire during the event, not after it. The bundled `generate_synthetic_flash_crash_stream` exists to calibrate thresholds against a known-bad path first: a threshold pair that fires late on a synthetic crash will fire late on a real one. For batch feature computation over a completed history use `microstructural_features` instead, which is cheaper per bar and gives the same quantities.",
    relatedModules: ["hpc-parallel", "microstructural-features", "structural-breaks", "data-structures"],
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
        label: "VPIN (Rolling Volume Buckets)",
        latex: "\\mathrm{VPIN}_t=\\frac{1}{N}\\sum_{i=t-N+1}^{t}\\frac{\\left|V_i^{B}-V_i^{S}\\right|}{V},\\qquad V_i^{B}+V_i^{S}=V",
        where: "$V_i^{B}$ and $V_i^{S}$ are buy- and sell-initiated volume in bucket $i$, $V$ the fixed `bucket_volume` every bucket is filled to, and $N$ = `support_buckets` the rolling window. Because buckets are equal-volume by construction, the denominator is a constant — this is the canonical Easley-Lopez de Prado form. The bar-based `get_vpin` in [`microstructural-features`](/modules/microstructural-features/) estimates the same quantity over unequal bars and so must normalise differently.",
      },
      {
        label: "Market Fragmentation HHI",
        latex: "\\mathrm{HHI}_t=\\sum_{v=1}^{K}\\left(\\frac{n_{v,t}}{\\sum_j n_{j,t}}\\right)^2",
        where: "$n_{v,t}$ is the event count on venue $v$ over the trailing `lookback_events` window and $K$ the number of venues. $1/K$ means flow is spread evenly; $1$ means one venue carries everything. Concentration spikes are the fragmentation half of a flash-crash signature.",
      },
      {
        label: "Alert Condition",
        latex: "\\text{alert}_t\\iff \\mathrm{VPIN}_t\\ge\\tau_V\\;\\land\\;\\mathrm{HHI}_t\\ge\\tau_H,\\qquad \\text{risk}_t=\\frac{1}{2}\\left(\\frac{\\mathrm{VPIN}_t}{\\tau_V}+\\frac{\\mathrm{HHI}_t}{\\tau_H}\\right)",
        where: "$\\tau_V$ and $\\tau_H$ are `AlertThresholds { vpin, hhi }`. Both conditions must hold — toxic flow alone, or concentrated flow alone, is common; together they are not. $\\text{risk}_t$ is the threshold-normalised score reported alongside the boolean, and is undefined until both estimators have filled their windows.",
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
        code: `use chrono::{Duration, NaiveDateTime};\nuse openquant::sample_weights::get_weights_by_time_decay;\n\nlet t0 = NaiveDateTime::parse_from_str("2024-01-02 00:00:00", "%Y-%m-%d %H:%M:%S")?;\n\n// Weighting is driven by triple-barrier events (t_in, t_out, label) — the label\n// lifetimes — plus the close series they span. It is not a function of returns.\nlet triple_barrier_events: Vec<(NaiveDateTime, NaiveDateTime, f64)> = (0..20)\n    .map(|i| (t0 + Duration::days(i), t0 + Duration::days(i + 2), 1.0))\n    .collect();\nlet close: Vec<(NaiveDateTime, f64)> =\n    (0..25).map(|i| (t0 + Duration::days(i), 100.0 + i as f64 * 0.1)).collect();\n\n// decay = 0.5: the oldest observation keeps half the weight of the newest.\n// decay <= 0 erases the oldest observations entirely.\nlet weights = get_weights_by_time_decay(&triple_barrier_events, &close, 0.5)?;\nprintln!("{} weights; newest = {:.4}", weights.len(), weights.last().map(|w| w.1).unwrap_or(0.0));`,
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
    conceptOverview:
      "Bagging in which the resampling respects label overlap. The standard bootstrap assumes IID draws; with triple-barrier labels whose spans overlap, an IID bag is full of near-duplicates, the base learners end up correlated, and the variance reduction bagging promises never materialises. Sequential bootstrap instead draws each index with probability proportional to its average uniqueness *given what has already been drawn*, so each bag is as close to independent as the data permits.",
    whenToUse:
      "Use it in place of ordinary bagging whenever the labels come from `labeling` — that is, whenever observations overlap in time. Measure the benefit rather than assuming it: `ensemble_methods::average_pairwise_prediction_correlation` will tell you whether the base learners actually decorrelated, and if rho is still high the extra sampling cost bought nothing. Note that `new()` takes the random seed, not the ensemble size: `n_estimators` defaults to 10 and must be set explicitly.",
    relatedModules: ["sampling", "ensemble-methods", "sample-weights", "labeling", "cross-validation"],
    module: "sb_bagging",
    subject: "Sampling, Validation and ML Diagnostics",
    summary: "Sequentially bootstrapped bagging classifiers/regressors.",
    whyItExists: "Combines ensemble variance reduction with overlap-aware sampling.",
    keyApis: ["SequentiallyBootstrappedBaggingClassifier", "SequentiallyBootstrappedBaggingRegressor", "MaxSamples", "MaxFeatures"],
    formulas: [
      {
        label: "Bagging Predictor",
        latex: "\\hat f(x)=\\frac{1}{B}\\sum_{b=1}^{B} f_b(x)",
        where: "$B$ = `n_estimators` and $f_b$ is the base learner fitted to the $b$-th resample. Note that $B$ defaults to $10$, not to the constructor argument, which is the random seed.",
      },
      {
        label: "Sequential Bootstrap Draw",
        latex: "\\Pr\\!\\left[i\\mid\\varphi\\right]=\\frac{\\bar u_i(\\varphi)}{\\sum_j \\bar u_j(\\varphi)},\\qquad \\bar u_i(\\varphi)=\\frac{1}{|T_i|}\\sum_{t\\in T_i}\\frac{1}{1+c_t(\\varphi)}",
        where: "$\\varphi$ is the set of indices drawn so far, $T_i$ the bars spanned by observation $i$'s label, and $c_t(\\varphi)$ the number of already-drawn observations whose label also covers bar $t$. Drawing an observation that overlaps what is already in the bag drives $\\bar u_i$ down, so the next draw prefers something disjoint — this is what stops the standard IID bootstrap from silently resampling the same overlapping event $B$ times. Probabilities are recomputed after every draw. See [`sampling`](/modules/sampling/) for the uniqueness machinery.",
      },
    ],
    examples: [
      {
        title: "Instantiate SB bagging classifier",
        language: "rust",
        code: `use openquant::sb_bagging::SequentiallyBootstrappedBaggingClassifier;\n\n// The single constructor argument is \`random_state\` — NOT the ensemble size.\n// n_estimators defaults to 10 and has to be set explicitly.\nlet mut bag = SequentiallyBootstrappedBaggingClassifier::new(42);\nbag.n_estimators = 100;\nbag.oob_score = true;\n\nprintln!("{} estimators, seed {}", bag.n_estimators, bag.random_state);`,
      },
    ],
    notes: ["Sequential bootstrap improves diversity under event overlap.", "Tune max_samples/max_features with out-of-sample monitoring."],
    apiSurface: "both",
    pythonApis: ["sb_bagging.fit_predict_sb_classifier", "sb_bagging.fit_predict_sb_regressor"],
  },
  {
    slug: "synthetic-backtesting",
    conceptOverview:
      "AFML Chapter 13's answer to profit-taking and stop-loss overfitting. Rather than searching the PT/SL mesh on the single historical path you have — where the winning cell is mostly luck — it calibrates an Ornstein-Uhlenbeck process to that path, generates thousands of synthetic paths from the fitted parameters, and evaluates the whole mesh across all of them. `detect_no_stable_optimum` then asks whether the resulting Sharpe surface has a peak worth trusting at all.",
    whenToUse:
      "Use it before committing to any exit rule. Its most valuable output is often the negative one: when the fitted persistence is close to 1 the price is near a random walk, the Sharpe surface is flat, and `no_stable_optimum` says so — meaning no PT/SL pair is defensible and the honest move is to skip the optimisation rather than take the argmax of noise. It complements `backtesting_engine` rather than replacing it, since that validates on the real path.",
    relatedModules: ["backtesting-engine", "labeling", "backtest-statistics", "bet-sizing", "strategy-risk"],
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
        code: `use openquant::synthetic_backtesting::{\n    run_synthetic_otr_workflow, StabilityCriteria, SyntheticBacktestConfig,\n};\n\n// A realised price history is fitted to obtain the O-U parameters the synthetic\n// paths are drawn from.\nlet historical_prices: Vec<f64> =\n    (0..500).map(|i| 100.0 + (i as f64 * 0.05).sin() * 3.0).collect();\n\nlet cfg = SyntheticBacktestConfig {\n    initial_price: historical_prices[historical_prices.len() - 1],\n    n_paths: 10_000,\n    horizon: 128,\n    seed: 42,\n    profit_taking_grid: vec![0.5, 1.0, 1.5, 2.0, 3.0],\n    stop_loss_grid: vec![0.5, 1.0, 1.5, 2.0, 3.0],\n    max_holding_steps: 64,\n    annualization_factor: 1.0,\n    stability_criteria: StabilityCriteria::default(),\n};\n\nlet out = run_synthetic_otr_workflow(&historical_prices, &cfg)?;\nif out.diagnostics.no_stable_optimum {\n    println!("Skip OTR optimization: {}", out.diagnostics.reason);\n} else {\n    println!("Best PT/SL: {:?}", out.best_rule);\n}`,
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
    conceptOverview:
      "Three families of break test. Chow-type statistics test for a break at a known or scanned candidate date. Chu-Stinchcombe-White is a sequential monitoring statistic that can be run online as data arrives. SADF — the supremum of ADF statistics over expanding windows — tests for *explosive* rather than merely non-stationary behaviour, which is the econometric signature of a bubble: an autoregressive coefficient that exceeds 1 rather than approaching it from below.",
    whenToUse:
      "Use SADF as a regime guard on any model whose parameters are estimated: a break means the training distribution no longer describes the present, and refitting then becomes a decision rather than a formality. Use the sequential statistics for online monitoring between refits. SADF cost grows quadratically with series length, because every endpoint re-runs an expanding-window regression, so keep long-window scenarios on a nightly path rather than in an interactive loop.",
    relatedModules: ["filters", "microstructural-features", "fracdiff", "cross-validation"],
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
        code: `use openquant::structural_breaks::{get_sadf, SadfLags};\n\n// SADF is defined on log prices.\nlet log_prices: Vec<f64> =\n    (0..160).map(|i| (100.0 + i as f64 * 0.1 + ((i / 40) as f64) * 5.0).ln()).collect();\n\n// (series, model, add_const, min_length, lags). \`model\` selects the regression\n// specification — "linear", "quadratic", "sm_poly_1", "sm_poly_2", "sm_exp",\n// "sm_power" — and \`min_length\` is the shortest window a statistic is computed on.\nlet sadf = get_sadf(&log_prices, "linear", true, 20, SadfLags::Fixed(1))?;\n\nlet peak = sadf.iter().cloned().fold(f64::NEG_INFINITY, f64::max);\nprintln!("{} SADF values, peak = {peak:.4}", sadf.len());`,
      },
    ],
    notes: ["SADF can be computationally expensive on long windows.", "Use dedicated slow/nightly test paths for heavy scenarios."],
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
        code: `use openquant::util::fast_ewma::ewma;\n\nlet x = vec![1.0, 2.0, 3.0, 4.0];\nlet y = ewma(&x, 3);`,
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
        latex: "\\sigma_{YZ}^2=\\sigma_o^2+k\\,\\sigma_c^2+(1-k)\\,\\sigma_{rs}^2,\\qquad k=\\frac{0.34}{1.34+\\frac{n+1}{n-1}}",
        where: "$\\sigma_o^2$ is the overnight (close-to-open) variance, $\\sigma_c^2$ the open-to-close variance, and $\\sigma_{rs}^2$ the Rogers-Satchell estimator; $n$ is the `window` length. $k$ is not a free parameter — it is the weight that minimises the estimator's variance, which is what makes Yang-Zhang the only one of these four that handles both overnight gaps and intraday drift. For a 20-bar window $k\\approx0.14$, so the overnight and Rogers-Satchell terms carry most of the estimate.",
      },
    ],
    examples: [
      {
        title: "Compute daily and range-based volatility",
        language: "rust",
        code: `use chrono::{Duration, NaiveDateTime};\nuse openquant::util::volatility::{get_daily_vol, get_parkinson_vol};\n\nlet t0 = NaiveDateTime::parse_from_str("2024-01-02 00:00:00", "%Y-%m-%d %H:%M:%S")?;\nlet close: Vec<(NaiveDateTime, f64)> = (0..300)\n    .map(|i| (t0 + Duration::days(i), 100.0 + (i as f64 * 0.07).sin() * 2.0))\n    .collect();\nlet high: Vec<f64> = close.iter().map(|(_, p)| p + 0.4).collect();\nlet low: Vec<f64> = close.iter().map(|(_, p)| p - 0.4).collect();\n\n// Close-to-close EWMA vol on a timestamped series; \`lookback\` is the EWMA span.\nlet daily = get_daily_vol(&close, 100);\n// Parkinson uses the high/low range, so it needs no timestamps — \`window\` bars.\nlet parkinson = get_parkinson_vol(&high, &low, 20);\n\nprintln!("daily vol tail = {:?}", daily.last());\nprintln!("parkinson vol tail = {:?}", parkinson.last());`,
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
      {
        label: "In-Sample Importance (`mdi_importance`)",
        latex: "I_j=\\frac{1}{B}\\sum_{b=1}^{B}\\frac{\\left|\\beta_j^{(b)}\\right|}{\\sum_k\\left|\\beta_k^{(b)}\\right|}",
        where: "$\\beta^{(b)}$ are the coefficients of a linear probability model fitted to bootstrap replica $b$, and $B$ = `n_estimators`. **This is normalised coefficient magnitude, not impurity decrease.** The function is named `mdi_importance` because it fills MDI's role — a cheap in-sample ranking computed from the fitted model alone, with the same substitution-effect weakness — but there is no tree and no impurity term here. For the tree-based $I_j=\\frac{1}{B}\\sum_b\\sum_{t\\in T_j^{(b)}}p(t)\\Delta i(t)$, see the Rust [`feature-importance`](/modules/feature-importance/) module. The two are not interchangeable and will rank features differently: this one measures linear sensitivity, that one measures split usefulness. Features must be standardised for the magnitudes to be comparable.",
      },
      {
        label: "Out-of-Sample Importance (`mda_importance`)",
        latex: "I_j=\\frac{1}{K}\\sum_{k=1}^{K}\\frac{S_k-S_{k,\\text{perm}(j)}}{d(S_{k,\\text{perm}(j)})},\\qquad d(S)=\\begin{cases}-S & \\text{scoring}=\\texttt{neg\\_log\\_loss}\\\\ 1-S & \\text{scoring}=\\texttt{accuracy}\\end{cases}",
        where: "$S_k$ is the score on purged fold $k$ and $S_{k,\\text{perm}(j)}$ the score after permuting column $j$ in that fold's test set. The denominator differs by scoring rule, and the default is `neg_log_loss` — with negative scores $-S$, not $1-S$, is what puts folds on a comparable scale. Splits come from `_purged_kfold_splits`, so `event_end_indices` must be supplied for the purge to do anything.",
      },
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
