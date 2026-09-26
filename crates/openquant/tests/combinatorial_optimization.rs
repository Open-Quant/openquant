use openquant::combinatorial_optimization::{
    compare_exact_and_adapter, enumerate_trading_paths, evaluate_trading_path, solve_exact,
    solve_trading_trajectory_exact, solve_with_adapter, CombinatorialOptimizationError,
    DecisionSchema, IntegerObjective, IntegerVariable, ObjectiveSense, OptimizationResult,
    SolverAdapter, TradeBounds, TradingTrajectoryObjective, TradingTrajectoryObjectiveConfig,
    TradingTrajectoryPath, TradingTrajectorySchema,
};

struct NonConvexIntegerObjective;

impl IntegerObjective for NonConvexIntegerObjective {
    fn sense(&self) -> ObjectiveSense {
        ObjectiveSense::Maximize
    }

    fn evaluate(&self, decision: &[i64]) -> Result<f64, CombinatorialOptimizationError> {
        if decision.len() != 2 {
            return Err(CombinatorialOptimizationError::DecisionLengthMismatch {
                expected: 2,
                found: decision.len(),
            });
        }
        let x = decision[0] as f64;
        let y = decision[1] as f64;
        let smooth = -((x - 1.0).powi(2) + (y + 1.0).powi(2));
        let fixed =
            if decision[0] != 0 { 1.5 } else { 0.0 } + if decision[1] != 0 { 0.5 } else { 0.0 };
        let discrete_bonus = if decision[0] * decision[1] == -1 { 3.0 } else { 0.0 };
        Ok(smooth - fixed + discrete_bonus)
    }
}

struct ZeroVectorAdapter;

impl SolverAdapter for ZeroVectorAdapter {
    fn solve(
        &self,
        schema: &DecisionSchema,
        objective: &dyn IntegerObjective,
    ) -> Result<OptimizationResult, CombinatorialOptimizationError> {
        schema.validate()?;
        let decision = vec![0_i64; schema.variables.len()];
        let value = objective.evaluate(&decision)?;
        Ok(OptimizationResult {
            best_decision: decision,
            best_objective: value,
            evaluated_candidates: 1,
        })
    }
}

struct TrajectoryObjective {
    cfg: TradingTrajectoryObjectiveConfig,
}

impl TradingTrajectoryObjective for TrajectoryObjective {
    fn sense(&self) -> ObjectiveSense {
        ObjectiveSense::Maximize
    }

    fn evaluate(
        &self,
        path: &TradingTrajectoryPath,
    ) -> Result<f64, CombinatorialOptimizationError> {
        evaluate_trading_path(path, &self.cfg)
    }
}

#[test]
fn exact_solver_finds_best_non_convex_integer_solution() {
    let schema = DecisionSchema {
        variables: vec![
            IntegerVariable { lower: -2, upper: 2, step: 1 },
            IntegerVariable { lower: -2, upper: 2, step: 1 },
        ],
        max_enumeration: 100,
    };
    let objective = NonConvexIntegerObjective;
    let result = solve_exact(&schema, &objective).expect("exact solve should succeed");

    assert_eq!(result.best_decision, vec![1, -1]);
    assert_eq!(result.evaluated_candidates, 25);
    assert!((result.best_objective - 1.0).abs() < 1e-12);
}

#[test]
fn adapter_comparison_reports_gap_vs_exact() {
    let schema = DecisionSchema {
        variables: vec![
            IntegerVariable { lower: -2, upper: 2, step: 1 },
            IntegerVariable { lower: -2, upper: 2, step: 1 },
        ],
        max_enumeration: 100,
    };
    let objective = NonConvexIntegerObjective;
    let adapter = ZeroVectorAdapter;
    let comparison =
        compare_exact_and_adapter(&schema, &objective, &adapter).expect("comparison should run");

    assert_eq!(comparison.exact.best_decision, vec![1, -1]);
    assert_eq!(comparison.adapter.best_decision, vec![0, 0]);
    assert!(comparison.objective_gap_vs_exact > 0.0);
}

#[test]
fn trajectory_objective_captures_non_convex_ticket_cost() {
    let path = TradingTrajectoryPath { trades: vec![1, -1, 0], inventory_path: vec![0, 1, 0, 0] };
    let mut cfg = TradingTrajectoryObjectiveConfig {
        expected_returns: vec![0.01, -0.02, 0.015],
        risk_aversion: 0.001,
        impact_coefficients: vec![0.0005, 0.0005, 0.0005],
        fixed_ticket_cost: 0.002,
        terminal_inventory_target: 0,
        terminal_inventory_penalty: 0.1,
    };

    let with_ticket = evaluate_trading_path(&path, &cfg).expect("evaluation should succeed");
    cfg.fixed_ticket_cost = 0.0;
    let without_ticket = evaluate_trading_path(&path, &cfg).expect("evaluation should succeed");

    assert!(without_ticket > with_ticket);
    assert!(((without_ticket - with_ticket) - 0.004).abs() < 1e-12);
}

#[test]
fn trajectory_enumeration_and_exact_solver_match_manual_best() {
    let schema = TradingTrajectorySchema {
        initial_inventory: 0,
        inventory_min: -2,
        inventory_max: 2,
        step_trade_bounds: vec![
            TradeBounds { min_trade: -1, max_trade: 1 },
            TradeBounds { min_trade: -1, max_trade: 1 },
            TradeBounds { min_trade: -1, max_trade: 1 },
            TradeBounds { min_trade: -1, max_trade: 1 },
        ],
        terminal_inventory: Some(0),
        max_paths: 10_000,
    };
    let cfg = TradingTrajectoryObjectiveConfig {
        expected_returns: vec![0.015, -0.01, 0.02, -0.005],
        risk_aversion: 0.001,
        impact_coefficients: vec![0.0005, 0.001, 0.0005, 0.001],
        fixed_ticket_cost: 0.0015,
        terminal_inventory_target: 0,
        terminal_inventory_penalty: 0.05,
    };
    let objective = TrajectoryObjective { cfg: cfg.clone() };
    let paths = enumerate_trading_paths(&schema).expect("path enumeration should succeed");
    let manual_best = paths
        .iter()
        .map(|path| evaluate_trading_path(path, &cfg).expect("manual objective should evaluate"))
        .fold(f64::NEG_INFINITY, f64::max);

    let result =
        solve_trading_trajectory_exact(&schema, &objective).expect("exact solve should succeed");

    assert_eq!(result.evaluated_paths, paths.len());
    assert_eq!(result.best_path.inventory_path.first().copied(), Some(schema.initial_inventory));
    assert_eq!(result.best_path.inventory_path.last().copied(), schema.terminal_inventory);
    assert!((result.best_objective - manual_best).abs() < 1e-12);
}

#[test]
fn trajectory_path_limit_guard_triggers() {
    let schema = TradingTrajectorySchema {
        initial_inventory: 0,
        inventory_min: -2,
        inventory_max: 2,
        step_trade_bounds: vec![
            TradeBounds { min_trade: -1, max_trade: 1 },
            TradeBounds { min_trade: -1, max_trade: 1 },
            TradeBounds { min_trade: -1, max_trade: 1 },
        ],
        terminal_inventory: None,
        max_paths: 5,
    };
    let err = enumerate_trading_paths(&schema).expect_err("limit should be exceeded");
    assert!(matches!(err, CombinatorialOptimizationError::EnumerationLimitExceeded { limit: 5 }));
}

#[test]
fn trajectory_path_limit_admits_exactly_max_paths() {
    // Seven round trips of three +/-1 trades end flat. A cap of seven must admit them, as
    // `DecisionSchema::max_enumeration` admits a space of exactly its size. It used to fail
    // whenever a leaf pruned by the terminal constraint was visited after the seventh path.
    let schema = |max_paths| TradingTrajectorySchema {
        initial_inventory: 0,
        inventory_min: -2,
        inventory_max: 2,
        step_trade_bounds: vec![TradeBounds { min_trade: -1, max_trade: 1 }; 3],
        terminal_inventory: Some(0),
        max_paths,
    };
    assert_eq!(enumerate_trading_paths(&schema(7)).expect("seven paths fit").len(), 7);
    assert!(matches!(
        enumerate_trading_paths(&schema(6)),
        Err(CombinatorialOptimizationError::EnumerationLimitExceeded { limit: 6 })
    ));
}

/// #184 item 5: `final inventory - terminal_inventory_target` was an unchecked `i64`
/// subtraction: a panic in debug builds and a wrapped difference (here -1) in release.
/// #186 item 23: `evaluate_trading_path` checked only lengths. A negative impact coefficient
/// paid the trader to trade, and an inventory path that is not the running sum of the trades
/// was scored as given.
#[test]
fn evaluate_trading_path_rejects_bad_impact_and_inconsistent_inventory() {
    let path = TradingTrajectoryPath { trades: vec![2, -1], inventory_path: vec![0, 2, 1] };
    let cfg = TradingTrajectoryObjectiveConfig {
        expected_returns: vec![0.01, 0.02],
        risk_aversion: 0.001,
        impact_coefficients: vec![0.001, 0.002],
        fixed_ticket_cost: 0.005,
        terminal_inventory_target: 0,
        terminal_inventory_penalty: 0.01,
    };
    assert!(evaluate_trading_path(&path, &cfg).is_ok());

    for bad in [-0.001, f64::NAN, f64::INFINITY] {
        let cfg = TradingTrajectoryObjectiveConfig {
            impact_coefficients: vec![0.001, bad],
            ..cfg.clone()
        };
        assert_eq!(
            evaluate_trading_path(&path, &cfg),
            Err(CombinatorialOptimizationError::InvalidInput(
                "impact_coefficients must be finite and >= 0"
            )),
            "{bad}"
        );
    }

    // Buy 2 then sell 1 cannot end at 3.
    let wrong = TradingTrajectoryPath { trades: vec![2, -1], inventory_path: vec![0, 2, 3] };
    assert_eq!(
        evaluate_trading_path(&wrong, &cfg),
        Err(CombinatorialOptimizationError::InconsistentInventoryPath { step: 1 })
    );
    // A running sum that overflows i64 is inconsistent too, not a panic.
    let overflow =
        TradingTrajectoryPath { trades: vec![1], inventory_path: vec![i64::MAX, i64::MIN] };
    let one_step = TradingTrajectoryObjectiveConfig {
        expected_returns: vec![0.0],
        impact_coefficients: vec![0.0],
        ..cfg.clone()
    };
    assert_eq!(
        evaluate_trading_path(&overflow, &one_step),
        Err(CombinatorialOptimizationError::InconsistentInventoryPath { step: 0 })
    );
}

#[test]
fn terminal_inventory_difference_does_not_overflow() {
    let path = TradingTrajectoryPath { trades: vec![], inventory_path: vec![i64::MAX] };
    let cfg = TradingTrajectoryObjectiveConfig {
        expected_returns: vec![],
        risk_aversion: 0.0,
        impact_coefficients: vec![],
        fixed_ticket_cost: 0.0,
        terminal_inventory_target: i64::MIN,
        terminal_inventory_penalty: 1e-40,
    };
    let value = std::panic::catch_unwind(|| evaluate_trading_path(&path, &cfg))
        .expect("must not panic")
        .expect("finite objective");
    // The true difference is 2^64 - 1, so the penalty is 1e-40 * (2^64 - 1)^2.
    let expected = -1e-40 * 2f64.powi(64).powi(2);
    assert!((value / expected - 1.0).abs() < 1e-12, "{value} vs {expected}");
}

/// Returns a fixed answer, whatever the problem.
struct FixedAdapter {
    decision: Vec<i64>,
    objective: f64,
}

impl SolverAdapter for FixedAdapter {
    fn solve(
        &self,
        _schema: &DecisionSchema,
        _objective: &dyn IntegerObjective,
    ) -> Result<OptimizationResult, CombinatorialOptimizationError> {
        Ok(OptimizationResult {
            best_decision: self.decision.clone(),
            best_objective: self.objective,
            evaluated_candidates: 1,
        })
    }
}

/// #185 item 15: an adapter runs on a box far above `max_enumeration`; only `solve_exact`
/// enumerates.
#[test]
fn adapter_runs_on_a_box_too_large_to_enumerate() {
    // A box of about 4e12 points against a cap of 10.
    let wide = IntegerVariable { lower: -1_000_000, upper: 1_000_000, step: 1 };
    let schema = DecisionSchema { variables: vec![wide, wide], max_enumeration: 10 };
    assert!(matches!(
        solve_exact(&schema, &NonConvexIntegerObjective),
        Err(CombinatorialOptimizationError::EnumerationLimitExceeded { limit: 10 })
    ));
    let objective = NonConvexIntegerObjective.evaluate(&[1, -1]).unwrap();
    let adapter = FixedAdapter { decision: vec![1, -1], objective };
    let result = solve_with_adapter(&schema, &NonConvexIntegerObjective, &adapter).unwrap();
    assert_eq!(result.best_decision, [1, -1]);
    assert_eq!(result.best_objective, objective);
    // A malformed schema is still rejected.
    let bad = DecisionSchema {
        variables: vec![IntegerVariable { lower: 1, upper: 0, step: 1 }],
        max_enumeration: 1,
    };
    assert!(matches!(
        solve_with_adapter(&bad, &NonConvexIntegerObjective, &adapter),
        Err(CombinatorialOptimizationError::InvalidInput(_))
    ));
}

/// #185 item 15: the adapter's answer is checked, not trusted.
#[test]
fn adapter_result_must_be_in_the_box_and_scored_honestly() {
    let grid = IntegerVariable { lower: -2, upper: 2, step: 2 };
    let schema = DecisionSchema { variables: vec![grid, grid], max_enumeration: 100 };
    let run = |decision: Vec<i64>, objective: f64| {
        solve_with_adapter(
            &schema,
            &NonConvexIntegerObjective,
            &FixedAdapter { decision, objective },
        )
    };
    let at = |d: &[i64]| NonConvexIntegerObjective.evaluate(d).unwrap();
    for (decision, why) in [
        (vec![4, 0], "outside the bounds"),
        (vec![1, 0], "off the step-2 grid"),
        (vec![0], "wrong length"),
    ] {
        assert!(
            matches!(
                run(decision.clone(), 0.0),
                Err(CombinatorialOptimizationError::InvalidAdapterResult(_))
            ),
            "{why}"
        );
    }
    // Claiming a better value than the decision has: rejected, as is a NaN claim.
    let claimed = at(&[0, 0]) + 5.0;
    assert!(matches!(
        run(vec![0, 0], claimed),
        Err(CombinatorialOptimizationError::InvalidAdapterResult(_))
    ));
    assert!(matches!(
        run(vec![0, 0], f64::NAN),
        Err(CombinatorialOptimizationError::InvalidAdapterResult(_))
    ));
    // An honest answer passes, and the gap is not floored: it is the true shortfall.
    let comparison = compare_exact_and_adapter(
        &schema,
        &NonConvexIntegerObjective,
        &FixedAdapter { decision: vec![2, 2], objective: at(&[2, 2]) },
    )
    .unwrap();
    let exact = solve_exact(&schema, &NonConvexIntegerObjective).unwrap();
    assert_eq!(comparison.objective_gap_vs_exact, exact.best_objective - at(&[2, 2]));
    assert!(comparison.objective_gap_vs_exact > 0.0);
}
