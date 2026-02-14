use openquant::combinatorial_optimization::{
    compare_exact_and_adapter, enumerate_trading_paths, evaluate_trading_path, solve_exact,
    solve_trading_trajectory_exact, CombinatorialOptimizationError, DecisionSchema,
    IntegerObjective, IntegerVariable, ObjectiveSense, OptimizationResult, SolverAdapter,
    TradeBounds, TradingTrajectoryObjective, TradingTrajectoryObjectiveConfig,
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
