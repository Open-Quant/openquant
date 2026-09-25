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
    apiSurface: "rust-only",
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
        code: `use openquant::ef3m::centered_moment;\n\nlet moments = vec![0.0, 1.0, 0.1, 3.0];\nlet m3 = centered_moment(&moments, 3)?;`,
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
    apiSurface: "rust-only",
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
    summary: "Hierarchical allocation with a choice of risk measure; currently HRP's bisection, not Raffinot's cut.",
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
    summary: "A bagging ensemble meant to draw samples with the sequential bootstrap; see its status note.",
    handwritten: true,
    apiSurface: "both",
    pythonApis: ["sb_bagging.fit_predict_sb_classifier", "sb_bagging.fit_predict_sb_regressor"],
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
        latex: "\\sigma_{YZ}^2=\\sigma_o^2+k\\,\\sigma_c^2+(1-k)\\,\\sigma_{rs}^2,\\qquad k=\\frac{0.34}{1.34+\\frac{n+1}{n-1}}",
        where: "$\\sigma_o^2$ is the overnight (close-to-open) variance, $\\sigma_c^2$ the open-to-close variance, and $\\sigma_{rs}^2$ the Rogers-Satchell estimator; $n$ is the `window` length. $k$ is not a free parameter — it is the weight that minimises the estimator's variance, which is what makes Yang-Zhang the only one of these four that handles both overnight gaps and intraday drift. For a 20-bar window $k\\approx0.14$, so the overnight and Rogers-Satchell terms carry most of the estimate.",
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
    summary: "OHLCV loading, cleaning, calendar alignment, and data quality reporting.",
    whyItExists: "Provides a consistent entrypoint for market data ingestion with automatic column normalization, deduplication, and quality diagnostics.",
    keyApis: ["load_ohlcv", "clean_ohlcv", "align_calendar", "data_quality_report"],
    formulas: [],
    examples: [
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
    summary: "End-to-end AFML research pipeline: events → signals → portfolio → risk → backtest with leakage checks.",
    whyItExists: "Chains the core AFML steps (filtering, labeling, sizing, allocation, risk) into a single reproducible research call with built-in leakage guards.",
    keyApis: ["run_mid_frequency_pipeline", "run_mid_frequency_pipeline_frames", "summarize_pipeline"],
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
