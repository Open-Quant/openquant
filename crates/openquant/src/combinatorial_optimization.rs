//! AFML Chapter 21: brute-force and combinatorial optimization adapters.
//!
//! This module provides integer decision schemas, exact finite-set solvers,
//! adapter traits for heuristic/external solvers, and trading-trajectory
//! state-space utilities with path-dependent objective evaluation.
//!
//! AFML Chapter 21 (§21.2 Combinatorial Optimization, §21.3 The Objective Function,
//! §21.5 An Integer Optimization Approach) argues that when the choices are discrete and
//! finite, listing and scoring every one gives the global optimum of any objective, smooth
//! or not. The chapter's own multi-asset problem (pigeonhole partitions and the Sharpe-ratio
//! trajectory search of Snippets 21.1–21.3) lives in [`crate::dynamic_allocation`]; this
//! module is the generic exact-enumeration baseline underneath it:
//!
//! - [`solve_exact`] visits every point of a box of integer grids ([`DecisionSchema`]) and
//!   scores it with an [`IntegerObjective`] you write, without storing the candidates.
//! - [`enumerate_trading_paths`] and [`solve_trading_trajectory_exact`] list and score every
//!   inventory path of a **single** instrument ([`TradingTrajectorySchema`]).
//!   [`evaluate_trading_path`] is a ready-made, non-convex objective (linear impact plus a
//!   fixed ticket cost per trade), in the spirit of Garleanu and Pedersen (2013) and
//!   Rosenberg et al. (2016); it is not the chapter's square-root cost.
//! - [`SolverAdapter`] and [`solve_with_adapter`] run a heuristic or external solver on a box
//!   of any size (`max_enumeration` does not apply) and check its answer;
//!   [`compare_exact_and_adapter`] scores one against the exact answer on instances small
//!   enough to enumerate.
//!
//! Conventions: decisions, trades and inventories are integers (units or lots). Candidates
//! are visited in lexicographic order starting from the lower bounds, and ties go to the
//! first candidate found (comparisons are strict). An objective cannot mark a candidate
//! infeasible: an error, or a non-finite score
//! ([`CombinatorialOptimizationError::ObjectiveNotFinite`]), aborts the whole search, so
//! constraints the schema cannot express must be large finite penalties. The search space is
//! exponential; hitting `max_enumeration` or `max_paths` is the cue for a heuristic, not for a
//! higher cap. Nothing here runs in parallel.
//!
//! ```
//! use openquant::combinatorial_optimization::{
//!     solve_exact, CombinatorialOptimizationError, DecisionSchema, IntegerObjective,
//!     IntegerVariable, ObjectiveSense,
//! };
//!
//! // Maximise -(x - 3)^2 - (y + 1)^2 with x on the even grid 0..=10 and y in -2..=2.
//! struct Bowl;
//! impl IntegerObjective for Bowl {
//!     fn sense(&self) -> ObjectiveSense {
//!         ObjectiveSense::Maximize
//!     }
//!     fn evaluate(&self, d: &[i64]) -> Result<f64, CombinatorialOptimizationError> {
//!         Ok(-((d[0] - 3).pow(2) + (d[1] + 1).pow(2)) as f64)
//!     }
//! }
//!
//! # fn main() -> Result<(), CombinatorialOptimizationError> {
//! let schema = DecisionSchema {
//!     variables: vec![
//!         IntegerVariable { lower: 0, upper: 10, step: 2 },
//!         IntegerVariable { lower: -2, upper: 2, step: 1 },
//!     ],
//!     max_enumeration: 100,
//! };
//! let best = solve_exact(&schema, &Bowl)?;
//! // x = 2 and x = 4 tie at -1; the first one visited wins.
//! assert_eq!(best.best_decision, [2, -1]);
//! assert_eq!(best.best_objective, -1.0);
//! assert_eq!(best.evaluated_candidates, 30); // 6 x 5 grid points
//! # Ok(())
//! # }
//! ```
#![deny(missing_docs)]

use std::fmt::{Display, Formatter};

/// Whether an objective is to be minimised or maximised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveSense {
    /// Smaller objective values are better.
    Minimize,
    /// Larger objective values are better.
    Maximize,
}

/// Errors returned by the combinatorial-optimization functions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CombinatorialOptimizationError {
    /// An input is out of its domain; the message names the violated condition.
    InvalidInput(&'static str),
    /// A per-step input of [`evaluate_trading_path`] does not match the path's horizon.
    DecisionLengthMismatch {
        /// The required length (the path's horizon).
        expected: usize,
        /// The length supplied.
        found: usize,
    },
    /// A path given to [`evaluate_trading_path`] is not self-consistent:
    /// `inventory_path[step + 1]` is not `inventory_path[step] + trades[step]` (or that sum
    /// overflows `i64`).
    InconsistentInventoryPath {
        /// The first trade index at which the inventory does not follow the trades.
        step: usize,
    },
    /// The objective returned a NaN or infinite value; the search is aborted.
    ObjectiveNotFinite,
    /// The schema has no variables, or no trading steps.
    EmptyDomain,
    /// The search space holds more candidates than the configured cap.
    EnumerationLimitExceeded {
        /// The cap that was exceeded (`max_enumeration` or `max_paths`).
        limit: usize,
    },
    /// No candidate satisfies the constraints (for example an unreachable terminal inventory).
    NoFeasibleSolution,
    /// A [`SolverAdapter`] returned a result that fails [`solve_with_adapter`]'s checks; the
    /// message names the check.
    InvalidAdapterResult(&'static str),
}

impl Display for CombinatorialOptimizationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::DecisionLengthMismatch { expected, found } => {
                write!(f, "decision length mismatch: expected {expected}, found {found}")
            }
            Self::InconsistentInventoryPath { step } => write!(
                f,
                "inventory_path is not the running sum of trades: step {step} does not add up"
            ),
            Self::ObjectiveNotFinite => write!(f, "objective evaluation returned non-finite value"),
            Self::EmptyDomain => write!(f, "decision domain is empty"),
            Self::EnumerationLimitExceeded { limit } => {
                write!(f, "enumeration limit exceeded: more than {limit} candidates")
            }
            Self::NoFeasibleSolution => write!(f, "no feasible solution found"),
            Self::InvalidAdapterResult(msg) => write!(f, "invalid adapter result: {msg}"),
        }
    }
}

impl std::error::Error for CombinatorialOptimizationError {}

/// One integer decision variable: the arithmetic grid `lower, lower + step, ...` up to and
/// including the last point `<= upper`.
///
/// `upper` itself is a candidate only when `upper - lower` is a multiple of `step`.
/// [`DecisionSchema::validate`] rejects `step <= 0` and `lower > upper`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegerVariable {
    /// Smallest value, the first grid point.
    pub lower: i64,
    /// Inclusive upper bound on the grid.
    pub upper: i64,
    /// Grid spacing; must be `> 0`.
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

/// A box of integer decisions: the Cartesian product of the variables' grids (AFML §21.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionSchema {
    /// The decision variables, in the order the objective receives them.
    pub variables: Vec<IntegerVariable>,
    /// Hard cap for exact finite-set enumeration.
    ///
    /// Counts every point in the box, feasible or not. Enforced by [`solve_exact`] (and so by
    /// [`compare_exact_and_adapter`]); [`solve_with_adapter`] ignores it.
    pub max_enumeration: usize,
}

impl DecisionSchema {
    /// Checks the schema and that its box has at most `max_enumeration` points: everything
    /// [`solve_exact`] needs.
    ///
    /// # Errors
    ///
    /// - [`CombinatorialOptimizationError::EmptyDomain`] if `variables` is empty.
    /// - [`CombinatorialOptimizationError::InvalidInput`] if `max_enumeration == 0`, a
    ///   variable has `step <= 0` or `lower > upper`, or the box size overflows `usize`.
    /// - [`CombinatorialOptimizationError::EnumerationLimitExceeded`] if the box has more
    ///   than `max_enumeration` points.
    pub fn validate(&self) -> Result<(), CombinatorialOptimizationError> {
        self.validate_variables()?;
        if self.max_enumeration == 0 {
            return Err(CombinatorialOptimizationError::InvalidInput(
                "max_enumeration must be > 0",
            ));
        }
        let size = self.decision_space_size()?;
        if size > self.max_enumeration {
            return Err(CombinatorialOptimizationError::EnumerationLimitExceeded {
                limit: self.max_enumeration,
            });
        }
        Ok(())
    }

    /// Checks the variables alone: at least one, each with `step > 0` and `lower <= upper`. The
    /// box may be of any size, so this is the check for a [`SolverAdapter`], which does not
    /// enumerate; `max_enumeration` is not read.
    ///
    /// # Errors
    ///
    /// - [`CombinatorialOptimizationError::EmptyDomain`] if `variables` is empty.
    /// - [`CombinatorialOptimizationError::InvalidInput`] if a variable has `step <= 0` or
    ///   `lower > upper`.
    pub fn validate_variables(&self) -> Result<(), CombinatorialOptimizationError> {
        if self.variables.is_empty() {
            return Err(CombinatorialOptimizationError::EmptyDomain);
        }
        for var in &self.variables {
            var.validate()?;
        }
        Ok(())
    }

    /// Whether `decision` is a point of the box: one value per variable, each on its grid
    /// (`lower <= v <= upper` and `v - lower` a multiple of `step`).
    ///
    /// ```
    /// use openquant::combinatorial_optimization::{DecisionSchema, IntegerVariable};
    ///
    /// let schema = DecisionSchema {
    ///     variables: vec![IntegerVariable { lower: 0, upper: 10, step: 3 }],
    ///     max_enumeration: 1,
    /// };
    /// assert!(schema.contains(&[9]));
    /// assert!(!schema.contains(&[10])); // off the grid
    /// assert!(!schema.contains(&[9, 0])); // wrong length
    /// ```
    pub fn contains(&self, decision: &[i64]) -> bool {
        decision.len() == self.variables.len()
            && self.variables.iter().zip(decision).all(|(var, &v)| {
                var.step > 0
                    && var.lower <= v
                    && v <= var.upper
                    && (i128::from(v) - i128::from(var.lower)) % i128::from(var.step) == 0
            })
    }

    /// Number of points in the box: the product of each variable's grid size.
    ///
    /// Does not check `max_enumeration`.
    ///
    /// # Errors
    ///
    /// - [`CombinatorialOptimizationError::EmptyDomain`] if `variables` is empty.
    /// - [`CombinatorialOptimizationError::InvalidInput`] if a variable has `step <= 0` or
    ///   `lower > upper`, or a grid size or the product overflows `usize`.
    ///
    /// ```
    /// use openquant::combinatorial_optimization::{DecisionSchema, IntegerVariable};
    ///
    /// let schema = DecisionSchema {
    ///     variables: vec![
    ///         IntegerVariable { lower: 0, upper: 10, step: 3 }, // 0, 3, 6, 9
    ///         IntegerVariable { lower: -1, upper: 1, step: 1 }, // -1, 0, 1
    ///     ],
    ///     max_enumeration: 5,
    /// };
    /// assert_eq!(schema.decision_space_size(), Ok(12));
    /// assert!(schema.validate().is_err()); // 12 > max_enumeration
    /// ```
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

/// The best decision a solver found over a [`DecisionSchema`].
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationResult {
    /// The best decision, one value per schema variable in schema order.
    pub best_decision: Vec<i64>,
    /// The objective value of `best_decision`.
    pub best_objective: f64,
    /// How many candidates the solver evaluated (the whole box for [`solve_exact`]).
    pub evaluated_candidates: usize,
}

/// An objective over integer decisions, scored by [`solve_exact`] or a [`SolverAdapter`].
pub trait IntegerObjective {
    /// Whether larger or smaller values of [`IntegerObjective::evaluate`] are better.
    fn sense(&self) -> ObjectiveSense;

    /// Scores one decision, given as one value per schema variable in schema order.
    ///
    /// Returning an error aborts the whole search; it does not mark the candidate
    /// infeasible. A non-finite value also aborts [`solve_exact`] with
    /// [`CombinatorialOptimizationError::ObjectiveNotFinite`].
    ///
    /// # Errors
    ///
    /// Implementation-defined.
    fn evaluate(&self, decision: &[i64]) -> Result<f64, CombinatorialOptimizationError>;
}

/// A heuristic or external solver to be run through [`solve_with_adapter`] and scored
/// against [`solve_exact`] with [`compare_exact_and_adapter`].
pub trait SolverAdapter {
    /// Searches `schema` for a good decision under `objective`.
    ///
    /// Called directly, the result is whatever the implementation returns. Through
    /// [`solve_with_adapter`] (and so [`compare_exact_and_adapter`]) it is checked: the
    /// decision must be a point of the box and `best_objective` must match the objective at
    /// that decision.
    ///
    /// # Errors
    ///
    /// Implementation-defined.
    fn solve(
        &self,
        schema: &DecisionSchema,
        objective: &dyn IntegerObjective,
    ) -> Result<OptimizationResult, CombinatorialOptimizationError>;
}

/// The exact optimum and an adapter's answer on the same problem, from
/// [`compare_exact_and_adapter`].
#[derive(Debug, Clone, PartialEq)]
pub struct AdapterComparison {
    /// The global optimum from [`solve_exact`].
    pub exact: OptimizationResult,
    /// The adapter's result, as checked by [`solve_with_adapter`].
    pub adapter: OptimizationResult,
    /// How far short of the exact optimum the adapter fell, in objective units:
    /// `exact - adapter` when maximising and `adapter - exact` when minimising.
    ///
    /// Not floored. It is `>= 0` because [`solve_with_adapter`] has checked that the adapter's
    /// decision is in the box and its value is the objective there, and the exact optimum is
    /// the best value over the whole box; 0 means the adapter found an optimum.
    pub objective_gap_vs_exact: f64,
}

/// Finds the global optimum of `objective` by visiting every point of the schema's box
/// (AFML §21.5).
///
/// Candidates are visited in lexicographic order from the lower bounds without being stored;
/// the comparison is strict, so among tied values the first decision visited wins.
///
/// # Errors
///
/// - Any error of [`DecisionSchema::validate`], including
///   [`CombinatorialOptimizationError::EnumerationLimitExceeded`] when the box has more than
///   `max_enumeration` points.
/// - [`CombinatorialOptimizationError::ObjectiveNotFinite`] if the objective returns NaN or
///   an infinity for any candidate.
/// - Any error returned by [`IntegerObjective::evaluate`], unchanged.
///
/// # Examples
///
/// See the [module documentation](self).
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
    // Defensive: `validate` already rejects `lower > upper`, so every grid holds at least
    // `lower`, and the non-empty box below always yields a best decision. `EmptyDomain` and
    // `NoFeasibleSolution` stay (both are reachable elsewhere) instead of an `expect`.
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

/// Runs `adapter` on `schema` and checks its answer.
///
/// The schema is checked with [`DecisionSchema::validate_variables`], so the box may be of
/// any size: `max_enumeration` is an enumeration cap and does not apply to a solver that
/// does not enumerate. The adapter's result is then checked: `best_decision` must be a point
/// of the box ([`DecisionSchema::contains`]), and `best_objective` must equal the objective
/// re-evaluated at `best_decision` to within `1e-9` relative (at least `1e-9` absolute). The
/// returned `best_objective` is that re-evaluated value; `evaluated_candidates` is the
/// adapter's own count.
///
/// # Errors
///
/// - Any error of [`DecisionSchema::validate_variables`].
/// - Any error returned by [`SolverAdapter::solve`], unchanged.
/// - [`CombinatorialOptimizationError::InvalidAdapterResult`] if the decision is not a point
///   of the box, or the reported objective does not match the re-evaluated one.
/// - [`CombinatorialOptimizationError::ObjectiveNotFinite`] if the objective at the decision
///   is not finite, or an error of [`IntegerObjective::evaluate`], unchanged.
pub fn solve_with_adapter(
    schema: &DecisionSchema,
    objective: &dyn IntegerObjective,
    adapter: &dyn SolverAdapter,
) -> Result<OptimizationResult, CombinatorialOptimizationError> {
    schema.validate_variables()?;
    let mut result = adapter.solve(schema, objective)?;
    if !schema.contains(&result.best_decision) {
        return Err(CombinatorialOptimizationError::InvalidAdapterResult(
            "best_decision is not a point of the schema's box",
        ));
    }
    let value = objective.evaluate(&result.best_decision)?;
    if !value.is_finite() {
        return Err(CombinatorialOptimizationError::ObjectiveNotFinite);
    }
    let tolerance = 1e-9 * value.abs().max(1.0);
    let error = (result.best_objective - value).abs();
    // A NaN report gives a NaN error, which must be rejected too.
    if error.is_nan() || error > tolerance {
        return Err(CombinatorialOptimizationError::InvalidAdapterResult(
            "best_objective does not match the objective at best_decision",
        ));
    }
    result.best_objective = value;
    Ok(result)
}

/// Runs [`solve_exact`] and `adapter` on the same problem and reports how far short of the
/// exact optimum the adapter fell. The box must fit `max_enumeration`, since it is enumerated.
///
/// # Errors
///
/// Any error of [`solve_exact`] (checked first) or of [`solve_with_adapter`].
///
/// ```
/// use openquant::combinatorial_optimization::{
///     compare_exact_and_adapter, CombinatorialOptimizationError, DecisionSchema,
///     IntegerObjective, IntegerVariable, ObjectiveSense, OptimizationResult, SolverAdapter,
/// };
///
/// // Minimise |x - 7| over 0..=9.
/// struct Distance;
/// impl IntegerObjective for Distance {
///     fn sense(&self) -> ObjectiveSense {
///         ObjectiveSense::Minimize
///     }
///     fn evaluate(&self, d: &[i64]) -> Result<f64, CombinatorialOptimizationError> {
///         Ok((d[0] - 7).abs() as f64)
///     }
/// }
///
/// // A "solver" that always answers with the lower corner of the box.
/// struct LowerCorner;
/// impl SolverAdapter for LowerCorner {
///     fn solve(
///         &self,
///         schema: &DecisionSchema,
///         objective: &dyn IntegerObjective,
///     ) -> Result<OptimizationResult, CombinatorialOptimizationError> {
///         let x: Vec<i64> = schema.variables.iter().map(|v| v.lower).collect();
///         let value = objective.evaluate(&x)?;
///         Ok(OptimizationResult { best_decision: x, best_objective: value, evaluated_candidates: 1 })
///     }
/// }
///
/// # fn main() -> Result<(), CombinatorialOptimizationError> {
/// let schema = DecisionSchema {
///     variables: vec![IntegerVariable { lower: 0, upper: 9, step: 1 }],
///     max_enumeration: 10,
/// };
/// let report = compare_exact_and_adapter(&schema, &Distance, &LowerCorner)?;
/// assert_eq!(report.exact.best_decision, [7]);
/// assert_eq!(report.adapter.best_decision, [0]);
/// assert_eq!(report.objective_gap_vs_exact, 7.0);
/// # Ok(())
/// # }
/// ```
pub fn compare_exact_and_adapter(
    schema: &DecisionSchema,
    objective: &dyn IntegerObjective,
    adapter: &dyn SolverAdapter,
) -> Result<AdapterComparison, CombinatorialOptimizationError> {
    let exact = solve_exact(schema, objective)?;
    let adapter_result = solve_with_adapter(schema, objective, adapter)?;
    let gap = match objective.sense() {
        ObjectiveSense::Maximize => exact.best_objective - adapter_result.best_objective,
        ObjectiveSense::Minimize => adapter_result.best_objective - exact.best_objective,
    };
    Ok(AdapterComparison { exact, adapter: adapter_result, objective_gap_vs_exact: gap })
}

/// The inclusive range of integer trades allowed at one step of a trading trajectory.
///
/// Every integer in `min_trade..=max_trade` is tried, so the range width drives the search
/// cost directly, even when the inventory bounds rule most trades out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TradeBounds {
    /// Smallest allowed trade (negative sells).
    pub min_trade: i64,
    /// Largest allowed trade; must be `>= min_trade`.
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

/// The state space of a single instrument's inventory traded over a fixed number of steps.
///
/// A path is one integer trade per step; the inventory after every trade must stay within
/// `[inventory_min, inventory_max]`, and `terminal_inventory`, if set, is a hard constraint
/// on the final inventory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TradingTrajectorySchema {
    /// Inventory before the first trade.
    pub initial_inventory: i64,
    /// Lowest inventory allowed after any trade (inclusive).
    pub inventory_min: i64,
    /// Highest inventory allowed after any trade (inclusive).
    pub inventory_max: i64,
    /// Allowed trades at each step; its length is the horizon.
    pub step_trade_bounds: Vec<TradeBounds>,
    /// Required final inventory, or `None` to leave it free. Paths ending elsewhere are
    /// discarded, not penalised.
    pub terminal_inventory: Option<i64>,
    /// Cap on the number of **feasible** paths; a schema with exactly this many is accepted.
    ///
    /// Infeasible partial paths explored by the search do not count towards it.
    pub max_paths: usize,
}

impl TradingTrajectorySchema {
    /// Checks the bounds and constraints of the schema. Does not count paths.
    ///
    /// # Errors
    ///
    /// - [`CombinatorialOptimizationError::InvalidInput`] if `inventory_min > inventory_max`,
    ///   `initial_inventory` or `terminal_inventory` lies outside
    ///   `[inventory_min, inventory_max]`, `max_paths == 0`, or a step has
    ///   `min_trade > max_trade`.
    /// - [`CombinatorialOptimizationError::EmptyDomain`] if `step_trade_bounds` is empty.
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

    /// Number of trading steps, `step_trade_bounds.len()`.
    pub fn horizon(&self) -> usize {
        self.step_trade_bounds.len()
    }
}

/// One feasible trading path: the trades and the inventory they produce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TradingTrajectoryPath {
    /// Trade at each step, in time order.
    pub trades: Vec<i64>,
    /// Inventory path includes initial inventory at index 0.
    ///
    /// Entry `t + 1` is the inventory after `trades[t]`, so the length is
    /// `trades.len() + 1`.
    pub inventory_path: Vec<i64>,
}

impl TradingTrajectoryPath {
    /// Number of trading steps, `trades.len()`.
    pub fn horizon(&self) -> usize {
        self.trades.len()
    }
}

/// An objective over whole trading paths, scored by [`solve_trading_trajectory_exact`].
///
/// Path dependence is allowed: the objective sees every trade and inventory. Wrap
/// [`evaluate_trading_path`] for the built-in cost model.
pub trait TradingTrajectoryObjective {
    /// Whether larger or smaller values of [`TradingTrajectoryObjective::evaluate`] are
    /// better.
    fn sense(&self) -> ObjectiveSense;

    /// Scores one path.
    ///
    /// Returning an error, or a non-finite value, aborts the whole search.
    ///
    /// # Errors
    ///
    /// Implementation-defined.
    fn evaluate(&self, path: &TradingTrajectoryPath)
        -> Result<f64, CombinatorialOptimizationError>;
}

/// The best path found by [`solve_trading_trajectory_exact`].
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryOptimizationResult {
    /// The optimal path; among ties, the first enumerated.
    pub best_path: TradingTrajectoryPath,
    /// The objective value of `best_path`.
    pub best_objective: f64,
    /// Number of feasible paths scored.
    pub evaluated_paths: usize,
}

/// Coefficients of the built-in objective [`evaluate_trading_path`].
///
/// Per-step vectors are indexed by step and must have one entry per trade.
#[derive(Debug, Clone, PartialEq)]
pub struct TradingTrajectoryObjectiveConfig {
    /// Expected return per unit of inventory at each step, earned on the inventory held
    /// *after* that step's trade.
    pub expected_returns: Vec<f64>,
    /// Penalty on held inventory each step (path-dependent risk proxy).
    ///
    /// Charged as `risk_aversion * q_t^2` on the post-trade inventory `q_t`; must be finite
    /// and `>= 0`.
    pub risk_aversion: f64,
    /// Per-step linear impact coefficient applied to |trade|.
    ///
    /// Not checked for sign or finiteness (a non-finite result is caught at the end).
    pub impact_coefficients: Vec<f64>,
    /// Fixed per-step ticket cost whenever trade != 0 (non-convex).
    ///
    /// Must be finite and `>= 0`.
    pub fixed_ticket_cost: f64,
    /// Final inventory the terminal penalty pulls towards.
    pub terminal_inventory_target: i64,
    /// Coefficient of the quadratic penalty on `final inventory - terminal_inventory_target`;
    /// must be finite and `>= 0`. A soft counterpart of
    /// [`TradingTrajectorySchema::terminal_inventory`], which is a hard constraint.
    pub terminal_inventory_penalty: f64,
}

/// Lists every feasible path of `schema` by depth-first search.
///
/// Trades at each step are tried from `min_trade` upwards, so paths come out in
/// lexicographic order of their trades. Every path is held in memory.
///
/// # Errors
///
/// - Any error of [`TradingTrajectorySchema::validate`].
/// - [`CombinatorialOptimizationError::EnumerationLimitExceeded`] if there are more than
///   `max_paths` feasible paths.
/// - [`CombinatorialOptimizationError::NoFeasibleSolution`] if no path satisfies the
///   inventory bounds and the terminal constraint.
///
/// ```
/// use openquant::combinatorial_optimization::{
///     enumerate_trading_paths, CombinatorialOptimizationError, TradeBounds,
///     TradingTrajectorySchema,
/// };
///
/// # fn main() -> Result<(), CombinatorialOptimizationError> {
/// // Two steps, one unit either way, long-only up to 2, flat at both ends.
/// let mut schema = TradingTrajectorySchema {
///     initial_inventory: 0,
///     inventory_min: 0,
///     inventory_max: 2,
///     step_trade_bounds: vec![TradeBounds { min_trade: -1, max_trade: 1 }; 2],
///     terminal_inventory: Some(0),
///     max_paths: 2,
/// };
/// let paths = enumerate_trading_paths(&schema)?;
/// let trades: Vec<_> = paths.iter().map(|p| p.trades.clone()).collect();
/// assert_eq!(trades, [vec![0, 0], vec![1, -1]]);
/// assert_eq!(paths[1].inventory_path, [0, 1, 0]);
///
/// schema.max_paths = 1;
/// assert_eq!(
///     enumerate_trading_paths(&schema),
///     Err(CombinatorialOptimizationError::EnumerationLimitExceeded { limit: 1 })
/// );
/// # Ok(())
/// # }
/// ```
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

/// Finds the optimal trading path by scoring every feasible path of `schema`.
///
/// Calls [`enumerate_trading_paths`], so memory grows with the number of feasible paths.
/// The comparison is strict: among tied scores the first path enumerated (the one that
/// trades lowest first) wins.
///
/// # Errors
///
/// - Any error of [`enumerate_trading_paths`].
/// - [`CombinatorialOptimizationError::ObjectiveNotFinite`] if the objective returns NaN or
///   an infinity for any path.
/// - Any error returned by [`TradingTrajectoryObjective::evaluate`], unchanged.
///
/// ```
/// use openquant::combinatorial_optimization::{
///     evaluate_trading_path, solve_trading_trajectory_exact, CombinatorialOptimizationError,
///     ObjectiveSense, TradeBounds, TradingTrajectoryObjective,
///     TradingTrajectoryObjectiveConfig, TradingTrajectoryPath, TradingTrajectorySchema,
/// };
///
/// struct NetPnl(TradingTrajectoryObjectiveConfig);
/// impl TradingTrajectoryObjective for NetPnl {
///     fn sense(&self) -> ObjectiveSense {
///         ObjectiveSense::Maximize
///     }
///     fn evaluate(
///         &self,
///         path: &TradingTrajectoryPath,
///     ) -> Result<f64, CombinatorialOptimizationError> {
///         evaluate_trading_path(path, &self.0)
///     }
/// }
///
/// # fn main() -> Result<(), CombinatorialOptimizationError> {
/// let schema = TradingTrajectorySchema {
///     initial_inventory: 0,
///     inventory_min: 0,
///     inventory_max: 3,
///     step_trade_bounds: vec![TradeBounds { min_trade: -3, max_trade: 3 }; 5],
///     terminal_inventory: Some(0),
///     max_paths: 10_000,
/// };
/// let solve = |fixed_ticket_cost| {
///     let objective = NetPnl(TradingTrajectoryObjectiveConfig {
///         expected_returns: vec![0.02, 0.02, -0.01, 0.02, 0.0],
///         risk_aversion: 0.001,
///         impact_coefficients: vec![0.002; 5],
///         fixed_ticket_cost,
///         terminal_inventory_target: 0,
///         terminal_inventory_penalty: 0.0,
///     });
///     solve_trading_trajectory_exact(&schema, &objective)
/// };
///
/// // Free to trade: sell out ahead of the dip and buy back after it.
/// let free = solve(0.0)?;
/// assert_eq!(free.best_path.trades, [3, 0, -3, 3, -3]);
/// assert_eq!(free.evaluated_paths, 256);
/// // A fixed ticket cost of 0.03 makes dodging the dip not worth two more tickets.
/// assert_eq!(solve(0.03)?.best_path.trades, [3, 0, 0, 0, -3]);
/// # Ok(())
/// # }
/// ```
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
    // Defensive: `enumerate_trading_paths` never returns an empty list, so a best path exists.
    let idx = best_idx.ok_or(CombinatorialOptimizationError::NoFeasibleSolution)?;
    Ok(TrajectoryOptimizationResult {
        best_path: paths[idx].clone(),
        best_objective: best_value,
        evaluated_paths: paths.len(),
    })
}

/// The built-in path objective: expected P&L net of inventory risk, linear impact and a
/// fixed ticket cost, minus a quadratic terminal penalty.
///
/// ```text
/// J = sum_t ( q_t r_t - lambda q_t^2 - c_t |dq_t| - kappa 1[dq_t != 0] ) - eta (q_T - q*)^2
/// ```
///
/// where `q_t` is the inventory *after* trade `dq_t` at step `t`, `r_t` is
/// `expected_returns[t]`, `lambda` is `risk_aversion`, `c_t` is `impact_coefficients[t]`,
/// `kappa` is `fixed_ticket_cost` and `eta` is `terminal_inventory_penalty` around
/// `q* = terminal_inventory_target`. Larger is better, so wrap it in a
/// [`TradingTrajectoryObjective`] with [`ObjectiveSense::Maximize`]. The ticket cost `kappa`
/// is what makes the problem non-convex. Impact is linear in `|dq_t|`, not the square-root
/// cost of AFML §21.3.
///
/// `path` must be self-consistent: `inventory_path[0]` is the starting inventory and each
/// later entry is the previous one plus that step's trade, as [`enumerate_trading_paths`]
/// builds them.
///
/// # Errors
///
/// - [`CombinatorialOptimizationError::InvalidInput`] if `inventory_path.len() !=
///   trades.len() + 1`, or `risk_aversion`, `fixed_ticket_cost`,
///   `terminal_inventory_penalty` or an entry of `impact_coefficients` is negative or
///   non-finite.
/// - [`CombinatorialOptimizationError::DecisionLengthMismatch`] if `expected_returns` or
///   `impact_coefficients` does not have one entry per trade.
/// - [`CombinatorialOptimizationError::InconsistentInventoryPath`] if `inventory_path` is not
///   the running sum of `trades`.
/// - [`CombinatorialOptimizationError::ObjectiveNotFinite`] if the result is NaN or infinite.
///
/// `final inventory - terminal_inventory_target` is formed in `i128`, so it cannot overflow
/// for any pair of `i64` values.
///
/// ```
/// use openquant::combinatorial_optimization::{
///     evaluate_trading_path, TradingTrajectoryObjectiveConfig, TradingTrajectoryPath,
/// };
///
/// // Buy 2, sell 1: inventory 0 -> 2 -> 1.
/// let path = TradingTrajectoryPath { trades: vec![2, -1], inventory_path: vec![0, 2, 1] };
/// let cfg = TradingTrajectoryObjectiveConfig {
///     expected_returns: vec![0.01, 0.02],
///     risk_aversion: 0.001,
///     impact_coefficients: vec![0.001, 0.002],
///     fixed_ticket_cost: 0.005,
///     terminal_inventory_target: 0,
///     terminal_inventory_penalty: 0.01,
/// };
/// // Step 1: 2 * 0.01 - 0.001 * 4 - 0.001 * 2 - 0.005 = 0.009
/// // Step 2: 1 * 0.02 - 0.001 * 1 - 0.002 * 1 - 0.005 = 0.012
/// // Terminal: -0.01 * (1 - 0)^2 = -0.01
/// let value = evaluate_trading_path(&path, &cfg).unwrap();
/// assert!((value - 0.011).abs() < 1e-12);
/// ```
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
    // A negative impact coefficient would pay the trader to trade.
    if cfg.impact_coefficients.iter().any(|c| !(c.is_finite() && *c >= 0.0)) {
        return Err(CombinatorialOptimizationError::InvalidInput(
            "impact_coefficients must be finite and >= 0",
        ));
    }
    for step in 0..path.horizon() {
        let expected = path.inventory_path[step].checked_add(path.trades[step]);
        if expected != Some(path.inventory_path[step + 1]) {
            return Err(CombinatorialOptimizationError::InconsistentInventoryPath { step });
        }
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

    // Widened: the difference of two i64 values can need 65 bits.
    let terminal_diff =
        i128::from(path.inventory_path[path.horizon()]) - i128::from(cfg.terminal_inventory_target);
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
    if step == schema.horizon() {
        let inventory_final = *inventory_path.last().unwrap_or(&schema.initial_inventory);
        if schema.terminal_inventory.is_some_and(|required| required != inventory_final) {
            return Ok(());
        }
        if out.len() == schema.max_paths {
            return Err(CombinatorialOptimizationError::EnumerationLimitExceeded {
                limit: schema.max_paths,
            });
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
