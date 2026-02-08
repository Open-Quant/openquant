export type Formula = {
  label: string;
  latex: string;
};

export type ExampleBlock = {
  title: string;
  language: "rust" | "bash";
  code: string;
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
      "get_target_pos",
      "limit_price",
    ],
    formulas: [
      { label: "Sigmoid Sizing", latex: "m(x)=\\frac{x}{\\sqrt{w+x^2}}" },
      { label: "Probability Bet", latex: "m=2\\Phi(z)-1" },
    ],
    examples: [
      {
        title: "Probability-based sizing",
        language: "rust",
        code: `use chrono::NaiveDateTime;\nuse openquant::bet_sizing::bet_size_probability;\n\nlet t0 = NaiveDateTime::parse_from_str("2024-01-01 09:30:00", "%Y-%m-%d %H:%M:%S")?;\nlet t1 = NaiveDateTime::parse_from_str("2024-01-01 16:00:00", "%Y-%m-%d %H:%M:%S")?;\nlet events = vec![(t0, t1, 0.56, 1.0), (t0, t1, 0.62, 1.0), (t0, t1, 0.48, -1.0)];\nlet sized = bet_size_probability(&events, 2, 0.1, false);`,
      },
    ],
    notes: [
      "Discretize to execution lot sizes using step_size.",
      "Reserve sizing is preferred when capital utilization is a hard constraint.",
    ],
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
  },
  {
    slug: "data-structures",
    module: "data_structures",
    subject: "Event-Driven Data and Labeling",
    summary: "Constructs standard/time/run/imbalance bars from trade streams.",
    whyItExists: "Event-based bars reduce heteroskedasticity and improve stationarity versus fixed-time sampling.",
    keyApis: ["standard_bars", "time_bars", "run_bars", "imbalance_bars", "Trade", "StandardBar"],
    formulas: [
      { label: "Dollar Bar Trigger", latex: "\\sum_{i=t_0}^{t} p_i v_i \\ge \\theta" },
      { label: "Imbalance Trigger", latex: "\\left|\\sum b_i\\right| \\ge E[|\\sum b_i|]" },
    ],
    examples: [
      {
        title: "Build time bars",
        language: "rust",
        code: `use chrono::Duration;\nuse openquant::data_structures::{time_bars, Trade};\n\nlet trades: Vec<Trade> = vec![];\nlet bars = time_bars(&trades, Duration::minutes(5));`,
      },
    ],
    notes: ["Threshold selection controls bar frequency and noise level.", "Keep OHLCV semantics consistent across downstream features."],
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
  },
  {
    slug: "etf-trick",
    module: "etf_trick",
    subject: "Position Sizing and Trade Construction",
    summary: "Synthetic ETF and futures roll utilities for realistic PnL path construction.",
    whyItExists: "Backtests must include financing, carry, and contract-roll mechanics to avoid optimistic bias.",
    keyApis: ["EtfTrick", "get_futures_roll_series", "FuturesRollRow"],
    formulas: [
      { label: "ETF NAV Update", latex: "NAV_t=NAV_{t-1}(1+r_t-c_t)" },
      { label: "Roll Return", latex: "r^{roll}_t=\\frac{F^{near}_t-F^{far}_t}{F^{far}_t}" },
    ],
    examples: [
      {
        title: "Compute futures roll series",
        language: "rust",
        code: `use openquant::etf_trick::get_futures_roll_series;\n\nlet roll = get_futures_roll_series(/* input tables */);`,
      },
    ],
    notes: ["Verify contract calendar assumptions.", "Costs and rates should come from the same clock as price data."],
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
  },
  {
    slug: "filters",
    module: "filters",
    subject: "Event-Driven Data and Labeling",
    summary: "CUSUM and z-score event filters for event-driven sampling.",
    whyItExists: "Extracts informative events from noisy high-frequency sequences.",
    keyApis: ["cusum_filter_indices", "cusum_filter_timestamps", "z_score_filter_indices", "Threshold"],
    formulas: [
      { label: "CUSUM", latex: "S_t=\\max(0, S_{t-1}+r_t),\\; trigger\\;if\\;|S_t|>h" },
      { label: "Z-score", latex: "z_t=\\frac{x_t-\\mu_t}{\\sigma_t}" },
    ],
    examples: [
      {
        title: "Run CUSUM over closes",
        language: "rust",
        code: `use openquant::filters::{cusum_filter_indices, Threshold};\n\nlet close = vec![100.0, 100.1, 99.9, 100.2];\nlet idx = cusum_filter_indices(&close, Threshold::Scalar(0.02));`,
      },
    ],
    notes: ["Calibrate thresholds to target event frequency, not just sensitivity.", "Use identical filtering in train and live pipelines."],
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
        title: "Compute fixed-width fracdiff",
        language: "rust",
        code: `use openquant::fracdiff::frac_diff_ffd;\n\nlet series = vec![100.0, 100.2, 100.1, 100.4, 100.6];\nlet out = frac_diff_ffd(&series, 0.4, 1e-4);`,
      },
    ],
    notes: ["Tune d using stationarity tests and information retention.", "Threshold governs truncation error vs compute cost."],
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
  },
  {
    slug: "labeling",
    module: "labeling",
    subject: "Event-Driven Data and Labeling",
    summary: "Triple-barrier event labeling and metadata generation.",
    whyItExists: "Converts event outcomes into ML labels with controlled horizon and risk barriers.",
    keyApis: ["add_vertical_barrier", "get_events", "get_bins", "drop_labels", "Event"],
    formulas: [
      { label: "Return", latex: "r_{t,t_1}=\\frac{P_{t_1}}{P_t}-1" },
      { label: "Barrier Hit", latex: "\\tau=\\min(\\tau_{pt},\\tau_{sl},t_1)" },
    ],
    examples: [
      {
        title: "Generate triple-barrier labels",
        language: "rust",
        code: `use openquant::labeling::{add_vertical_barrier, get_events, get_bins};\n\nlet barriers = add_vertical_barrier(&events, &close, 1, 0, 0, 0);\nlet ev = get_events(&close, &events, (1.0, 1.0), &target, 0.005, 3, Some(&barriers), None);\nlet bins = get_bins(&ev, &close);`,
      },
    ],
    notes: ["Target volatility quality strongly affects labels.", "Meta-labeling requires a side signal aligned on event timestamps."],
  },
  {
    slug: "microstructural-features",
    module: "microstructural_features",
    subject: "Market Microstructure, Dependence and Regime Detection",
    summary: "Price-impact, spread, entropy, and flow toxicity estimators.",
    whyItExists: "Microstructure features capture liquidity and order-flow dynamics not visible in OHLC bars alone.",
    keyApis: ["get_roll_measure", "get_corwin_schultz_estimator", "get_bar_based_kyle_lambda", "get_vpin", "MicrostructuralFeaturesGenerator"],
    formulas: [
      { label: "Kyle Lambda", latex: "\\Delta p_t=\\lambda q_t+\\epsilon_t" },
      { label: "Shannon Entropy", latex: "H=-\\sum_i p_i\\log p_i" },
    ],
    examples: [
      {
        title: "Estimate spread and impact",
        language: "rust",
        code: `use openquant::microstructural_features::{get_roll_measure, get_bar_based_kyle_lambda};\n\nlet close = vec![100.0, 100.2, 100.1, 100.3];\nlet volume = vec![1000.0, 1200.0, 900.0, 1100.0];\nlet spread = get_roll_measure(&close, 2);\nlet kyle = get_bar_based_kyle_lambda(&close, &volume, 2);`,
      },
    ],
    notes: ["Normalize by volume/liquidity regime when comparing across assets.", "Entropy features require stable encoding maps."],
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
  },
  {
    slug: "portfolio-optimization",
    module: "portfolio_optimization",
    subject: "Portfolio Construction and Risk",
    summary: "Mean-variance and constrained allocation methods with ergonomic APIs.",
    whyItExists: "Provides production-ready portfolio construction primitives with explicit options and constraints.",
    keyApis: ["allocate_inverse_variance", "allocate_min_vol", "allocate_max_sharpe", "allocate_efficient_risk", "AllocationOptions"],
    formulas: [
      { label: "Min Variance", latex: "\\min_w\\;w^T\\Sigma w\\;s.t.\\;\\mathbf{1}^T w=1" },
      { label: "Max Sharpe", latex: "\\max_w\\;\\frac{w^T(\\mu-r_f\\mathbf{1})}{\\sqrt{w^T\\Sigma w}}" },
    ],
    examples: [
      {
        title: "Compute inverse-variance and max-sharpe portfolios",
        language: "rust",
        code: `use openquant::portfolio_optimization::{allocate_inverse_variance, allocate_max_sharpe};\n\nlet ivp = allocate_inverse_variance(&prices)?;\nlet msr = allocate_max_sharpe(&prices, 0.0, None, None)?;`,
      },
    ],
    notes: ["Bounds and return-estimation method materially change optimal weights.", "Use out-of-sample evaluation with turnover/cost constraints."],
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
        title: "Compute event weights",
        language: "rust",
        code: `use openquant::sample_weights::get_weights_by_time_decay;\n\nlet w = get_weights_by_time_decay(&returns, 0.5);`,
      },
    ],
    notes: ["Pair with sequential bootstrap for robust label sampling.", "Time-decay controls recency bias explicitly."],
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
        title: "Run sequential bootstrap",
        language: "rust",
        code: `use openquant::sampling::seq_bootstrap;\n\nlet ind = vec![vec![1,0,1], vec![0,1,1], vec![1,1,0]];\nlet idx = seq_bootstrap(&ind, Some(3), None);`,
      },
    ],
    notes: ["Indicator matrix quality drives bootstrap quality.", "Use average uniqueness as a diagnostics KPI."],
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
  },
];
