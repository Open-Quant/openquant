//! AFML Chapter 21: brute-force and combinatorial optimization adapters.
//!
//! This module provides integer decision schemas, exact finite-set solvers,
//! adapter traits for heuristic/external solvers, and trading-trajectory
//! state-space utilities with path-dependent objective evaluation.

use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveSense {
    Minimize,
    Maximize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CombinatorialOptimizationError {
    InvalidInput(&'static str),
    DecisionLengthMismatch { expected: usize, found: usize },
    ObjectiveNotFinite,
    EmptyDomain,
    EnumerationLimitExceeded { limit: usize },
    NoFeasibleSolution,
}

impl Display for CombinatorialOptimizationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::DecisionLengthMismatch { expected, found } => {
                write!(f, "decision length mismatch: expected {expected}, found {found}")
            }
            Self::ObjectiveNotFinite => write!(f, "objective evaluation returned non-finite value"),
            Self::EmptyDomain => write!(f, "decision domain is empty"),
            Self::EnumerationLimitExceeded { limit } => {
                write!(f, "enumeration limit exceeded: more than {limit} candidates")
            }
            Self::NoFeasibleSolution => write!(f, "no feasible solution found"),
        }
    }
}

impl std::error::Error for CombinatorialOptimizationError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegerVariable {
    pub lower: i64,
    pub upper: i64,
    pub step: i64,
}

impl IntegerVariable {
    fn validate(self) -> Result<(), CombinatorialOptimizationError> {
        if self.step <= 0 {
            return Err(CombinatorialOptimizationError::InvalidInput("step must be > 0"));
        }
        if self.lower > self.upper {
            return Err(CombinatorialOptimizationError::InvalidInput(
                "variable lower bound must be <= upper bound",
            ));
        }
        Ok(())
    }

    fn cardinality(self) -> Result<usize, CombinatorialOptimizationError> {
        self.validate()?;
        let span = i128::from(self.upper) - i128::from(self.lower);
        let step = i128::from(self.step);
        let count = span / step + 1;
        usize::try_from(count).map_err(|_| {
            CombinatorialOptimizationError::InvalidInput("variable cardinality does not fit usize")
        })
    }

    fn values(self) -> Result<Vec<i64>, CombinatorialOptimizationError> {
        self.validate()?;
        let mut out = Vec::with_capacity(self.cardinality()?);
        let mut current = self.lower;
        while current <= self.upper {
            out.push(current);
            current = match current.checked_add(self.step) {
                Some(next) => next,
                None => break,
            };
        }
        if out.is_empty() {
            return Err(CombinatorialOptimizationError::EmptyDomain);
        }
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionSchema {
    pub variables: Vec<IntegerVariable>,
    /// Hard cap for exact finite-set enumeration.
    pub max_enumeration: usize,
}

impl DecisionSchema {
    pub fn validate(&self) -> Result<(), CombinatorialOptimizationError> {
        if self.variables.is_empty() {
            return Err(CombinatorialOptimizationError::EmptyDomain);
        }
        if self.max_enumeration == 0 {
            return Err(CombinatorialOptimizationError::InvalidInput(
                "max_enumeration must be > 0",
            ));
        }
        for var in &self.variables {
            var.validate()?;
        }
        let size = self.decision_space_size()?;
        if size > self.max_enumeration {
            return Err(CombinatorialOptimizationError::EnumerationLimitExceeded {
                limit: self.max_enumeration,
            });
        }
        Ok(())
    }

    pub fn decision_space_size(&self) -> Result<usize, CombinatorialOptimizationError> {
        if self.variables.is_empty() {
            return Err(CombinatorialOptimizationError::EmptyDomain);
        }
        self.variables.iter().try_fold(1usize, |acc, var| {
            let n = var.cardinality()?;
            acc.checked_mul(n).ok_or(CombinatorialOptimizationError::InvalidInput(
                "decision space cardinality overflowed usize",
            ))
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationResult {
    pub best_decision: Vec<i64>,
    pub best_objective: f64,
    pub evaluated_candidates: usize,
}

pub trait IntegerObjective {
    fn sense(&self) -> ObjectiveSense;
    fn evaluate(&self, decision: &[i64]) -> Result<f64, CombinatorialOptimizationError>;
}

pub trait SolverAdapter {
    fn solve(
        &self,
        schema: &DecisionSchema,
        objective: &dyn IntegerObjective,
    ) -> Result<OptimizationResult, CombinatorialOptimizationError>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdapterComparison {
    pub exact: OptimizationResult,
    pub adapter: OptimizationResult,
    /// Non-negative gap in objective space relative to the exact optimum.
    pub objective_gap_vs_exact: f64,
}

pub fn solve_exact(
    schema: &DecisionSchema,
    objective: &dyn IntegerObjective,
) -> Result<OptimizationResult, CombinatorialOptimizationError> {
    schema.validate()?;
    let values = schema
        .variables
        .iter()
        .copied()
        .map(IntegerVariable::values)
        .collect::<Result<Vec<_>, _>>()?;
    if values.iter().any(Vec::is_empty) {
        return Err(CombinatorialOptimizationError::EmptyDomain);
    }

    let mut current = vec![0_i64; schema.variables.len()];
    let mut best_decision: Option<Vec<i64>> = None;
    let mut best_objective = 0.0;
    let mut evaluated = 0usize;

    enumerate_decisions(&values, 0, &mut current, &mut |decision| {
        let value = objective.evaluate(decision)?;
        if !value.is_finite() {
            return Err(CombinatorialOptimizationError::ObjectiveNotFinite);
        }
        if best_decision.is_none() || is_better(value, best_objective, objective.sense()) {
            best_decision = Some(decision.to_vec());
            best_objective = value;
        }
        evaluated = evaluated.saturating_add(1);
        Ok(())
    })?;

    let best_decision = best_decision.ok_or(CombinatorialOptimizationError::NoFeasibleSolution)?;
    Ok(OptimizationResult { best_decision, best_objective, evaluated_candidates: evaluated })
}

pub fn solve_with_adapter(
    schema: &DecisionSchema,
    objective: &dyn IntegerObjective,
    adapter: &dyn SolverAdapter,
) -> Result<OptimizationResult, CombinatorialOptimizationError> {
    schema.validate()?;
    adapter.solve(schema, objective)
}

pub fn compare_exact_and_adapter(
    schema: &DecisionSchema,
    objective: &dyn IntegerObjective,
    adapter: &dyn SolverAdapter,
) -> Result<AdapterComparison, CombinatorialOptimizationError> {
    let exact = solve_exact(schema, objective)?;
    let adapter_result = solve_with_adapter(schema, objective, adapter)?;
    let gap = match objective.sense() {
        ObjectiveSense::Maximize => (exact.best_objective - adapter_result.best_objective).max(0.0),
        ObjectiveSense::Minimize => (adapter_result.best_objective - exact.best_objective).max(0.0),
    };
    Ok(AdapterComparison { exact, adapter: adapter_result, objective_gap_vs_exact: gap })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TradeBounds {
    pub min_trade: i64,
    pub max_trade: i64,
}

impl TradeBounds {
    fn validate(self) -> Result<(), CombinatorialOptimizationError> {
        if self.min_trade > self.max_trade {
            return Err(CombinatorialOptimizationError::InvalidInput(
                "trade bounds must satisfy min_trade <= max_trade",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TradingTrajectorySchema {
    pub initial_inventory: i64,
    pub inventory_min: i64,
    pub inventory_max: i64,
    pub step_trade_bounds: Vec<TradeBounds>,
    pub terminal_inventory: Option<i64>,
    pub max_paths: usize,
}

impl TradingTrajectorySchema {
    pub fn validate(&self) -> Result<(), CombinatorialOptimizationError> {
        if self.inventory_min > self.inventory_max {
            return Err(CombinatorialOptimizationError::InvalidInput(
                "inventory_min must be <= inventory_max",
            ));
        }
        if self.initial_inventory < self.inventory_min
            || self.initial_inventory > self.inventory_max
        {
            return Err(CombinatorialOptimizationError::InvalidInput(
                "initial_inventory must be inside [inventory_min, inventory_max]",
            ));
        }
        if self.max_paths == 0 {
            return Err(CombinatorialOptimizationError::InvalidInput("max_paths must be > 0"));
        }
        if self.step_trade_bounds.is_empty() {
            return Err(CombinatorialOptimizationError::EmptyDomain);
        }
        for bounds in &self.step_trade_bounds {
            bounds.validate()?;
        }
        if let Some(term) = self.terminal_inventory {
            if term < self.inventory_min || term > self.inventory_max {
                return Err(CombinatorialOptimizationError::InvalidInput(
                    "terminal_inventory must be inside [inventory_min, inventory_max]",
                ));
            }
        }
        Ok(())
    }

    pub fn horizon(&self) -> usize {
        self.step_trade_bounds.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TradingTrajectoryPath {
    pub trades: Vec<i64>,
    /// Inventory path includes initial inventory at index 0.
    pub inventory_path: Vec<i64>,
}

impl TradingTrajectoryPath {
    pub fn horizon(&self) -> usize {
        self.trades.len()
    }
}

pub trait TradingTrajectoryObjective {
    fn sense(&self) -> ObjectiveSense;
    fn evaluate(&self, path: &TradingTrajectoryPath)
        -> Result<f64, CombinatorialOptimizationError>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryOptimizationResult {
    pub best_path: TradingTrajectoryPath,
    pub best_objective: f64,
    pub evaluated_paths: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TradingTrajectoryObjectiveConfig {
    pub expected_returns: Vec<f64>,
    /// Penalty on held inventory each step (path-dependent risk proxy).
    pub risk_aversion: f64,
    /// Per-step linear impact coefficient applied to |trade|.
    pub impact_coefficients: Vec<f64>,
    /// Fixed per-step ticket cost whenever trade != 0 (non-convex).
    pub fixed_ticket_cost: f64,
    pub terminal_inventory_target: i64,
    pub terminal_inventory_penalty: f64,
}

pub fn enumerate_trading_paths(
    schema: &TradingTrajectorySchema,
) -> Result<Vec<TradingTrajectoryPath>, CombinatorialOptimizationError> {
    schema.validate()?;
    let mut all_paths = Vec::new();
    let mut trades = Vec::with_capacity(schema.horizon());
    let mut inventory_path = Vec::with_capacity(schema.horizon() + 1);
    inventory_path.push(schema.initial_inventory);
    dfs_paths(schema, 0, &mut trades, &mut inventory_path, &mut all_paths)?;
    if all_paths.is_empty() {
        return Err(CombinatorialOptimizationError::NoFeasibleSolution);
    }
    Ok(all_paths)
}

pub fn solve_trading_trajectory_exact(
    schema: &TradingTrajectorySchema,
    objective: &dyn TradingTrajectoryObjective,
) -> Result<TrajectoryOptimizationResult, CombinatorialOptimizationError> {
    let paths = enumerate_trading_paths(schema)?;
    let mut best_idx = None::<usize>;
    let mut best_value = 0.0;
    for (idx, path) in paths.iter().enumerate() {
        let value = objective.evaluate(path)?;
        if !value.is_finite() {
            return Err(CombinatorialOptimizationError::ObjectiveNotFinite);
        }
        if best_idx.is_none() || is_better(value, best_value, objective.sense()) {
            best_idx = Some(idx);
            best_value = value;
        }
    }
    let idx = best_idx.ok_or(CombinatorialOptimizationError::NoFeasibleSolution)?;
    Ok(TrajectoryOptimizationResult {
        best_path: paths[idx].clone(),
        best_objective: best_value,
        evaluated_paths: paths.len(),
    })
}

pub fn evaluate_trading_path(
    path: &TradingTrajectoryPath,
    cfg: &TradingTrajectoryObjectiveConfig,
) -> Result<f64, CombinatorialOptimizationError> {
    if path.inventory_path.len() != path.trades.len() + 1 {
        return Err(CombinatorialOptimizationError::InvalidInput(
            "inventory_path must have exactly trades.len() + 1 entries",
        ));
    }
    if cfg.expected_returns.len() != path.horizon() {
        return Err(CombinatorialOptimizationError::DecisionLengthMismatch {
            expected: path.horizon(),
            found: cfg.expected_returns.len(),
        });
    }
    if cfg.impact_coefficients.len() != path.horizon() {
        return Err(CombinatorialOptimizationError::DecisionLengthMismatch {
            expected: path.horizon(),
            found: cfg.impact_coefficients.len(),
        });
    }
    if !cfg.risk_aversion.is_finite()
        || !cfg.fixed_ticket_cost.is_finite()
        || !cfg.terminal_inventory_penalty.is_finite()
        || cfg.risk_aversion < 0.0
        || cfg.fixed_ticket_cost < 0.0
        || cfg.terminal_inventory_penalty < 0.0
    {
        return Err(CombinatorialOptimizationError::InvalidInput(
            "risk/cost coefficients must be finite and >= 0",
        ));
    }

    let mut objective = 0.0;
    for step in 0..path.horizon() {
        let trade = path.trades[step] as f64;
        let inventory_after = path.inventory_path[step + 1] as f64;
        let step_return = cfg.expected_returns[step];
        let directional_pnl = inventory_after * step_return;
        let risk_penalty = cfg.risk_aversion * inventory_after.powi(2);
        let impact_cost = cfg.impact_coefficients[step] * trade.abs();
        let fixed_cost = if path.trades[step] == 0 { 0.0 } else { cfg.fixed_ticket_cost };
        objective += directional_pnl - risk_penalty - impact_cost - fixed_cost;
    }

    let terminal_diff = path.inventory_path[path.horizon()] - cfg.terminal_inventory_target;
    objective -= cfg.terminal_inventory_penalty * (terminal_diff as f64).powi(2);

    if !objective.is_finite() {
        return Err(CombinatorialOptimizationError::ObjectiveNotFinite);
    }
    Ok(objective)
}

fn dfs_paths(
    schema: &TradingTrajectorySchema,
    step: usize,
    trades: &mut Vec<i64>,
    inventory_path: &mut Vec<i64>,
    out: &mut Vec<TradingTrajectoryPath>,
) -> Result<(), CombinatorialOptimizationError> {
    if out.len() >= schema.max_paths {
        return Err(CombinatorialOptimizationError::EnumerationLimitExceeded {
            limit: schema.max_paths,
        });
    }
    if step == schema.horizon() {
        let inventory_final = *inventory_path.last().unwrap_or(&schema.initial_inventory);
        if schema.terminal_inventory.is_some_and(|required| required != inventory_final) {
            return Ok(());
        }
        out.push(TradingTrajectoryPath {
            trades: trades.clone(),
            inventory_path: inventory_path.clone(),
        });
        return Ok(());
    }

    let bounds = schema.step_trade_bounds[step];
    for trade in bounds.min_trade..=bounds.max_trade {
        let current = *inventory_path.last().unwrap_or(&schema.initial_inventory);
        let next = match current.checked_add(trade) {
            Some(v) => v,
            None => continue,
        };
        if next < schema.inventory_min || next > schema.inventory_max {
            continue;
        }
        trades.push(trade);
        inventory_path.push(next);
        dfs_paths(schema, step + 1, trades, inventory_path, out)?;
        inventory_path.pop();
        trades.pop();
    }
    Ok(())
}

fn enumerate_decisions(
    values: &[Vec<i64>],
    depth: usize,
    current: &mut [i64],
    visit: &mut dyn FnMut(&[i64]) -> Result<(), CombinatorialOptimizationError>,
) -> Result<(), CombinatorialOptimizationError> {
    if depth == values.len() {
        return visit(current);
    }
    for value in &values[depth] {
        current[depth] = *value;
        enumerate_decisions(values, depth + 1, current, visit)?;
    }
    Ok(())
}

fn is_better(candidate: f64, incumbent: f64, sense: ObjectiveSense) -> bool {
    match sense {
        ObjectiveSense::Maximize => candidate > incumbent,
        ObjectiveSense::Minimize => candidate < incumbent,
    }
}
