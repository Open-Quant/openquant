---
title: "combinatorial_optimization"
description: "Exhaustive search over small integer problems, and over single-instrument trading paths with a fixed cost per trade, as an exact baseline for heuristic solvers."
status: authored
last_authored: '2026-09-26'
audience:
  - quant-dev
  - platform-engineering
module: "combinatorial_optimization"
api_surface: "rust-only"
afml_chapter:
  - "21"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 21: §21.2 Combinatorial Optimization; §21.3 The Objective Function; §21.5 An Integer Optimization Approach (Snippets 21.1–21.3)."
  - "Garleanu, N. and Pedersen, L. H. (2013). Dynamic trading with predictable returns and transaction costs. Journal of Finance 68(6), 2309–2340."
  - "Rosenberg, G., Haghnegahdar, P., Goddard, P., Carr, P., Wu, K. and López de Prado, M. (2016). Solving the optimal trading trajectory problem using a quantum annealer. IEEE Journal of Selected Topics in Signal Processing 10(6), 1053–1060."
rust_api:
  - "DecisionSchema"
  - "IntegerVariable"
  - "IntegerObjective"
  - "ObjectiveSense"
  - "solve_exact"
  - "SolverAdapter"
  - "solve_with_adapter"
  - "compare_exact_and_adapter"
  - "OptimizationResult"
  - "AdapterComparison"
  - "TradingTrajectorySchema"
  - "TradeBounds"
  - "TradingTrajectoryPath"
  - "TradingTrajectoryObjective"
  - "TradingTrajectoryObjectiveConfig"
  - "enumerate_trading_paths"
  - "evaluate_trading_path"
  - "solve_trading_trajectory_exact"
  - "CombinatorialOptimizationError"
sidebar:
  badge: Module
---

Some problems have no convexity to exploit. A fixed commission per trade, a lot size, a
threshold rule: each makes the objective jump, and a gradient-based optimiser either cannot
be applied or finds a local answer. AFML's Chapter 21 takes the blunt route. If the choices
are discrete and finite, list every one, score it, and keep the best. Nothing about the
objective needs to be smooth, convex or even continuous, and the answer is the global
optimum. The price is that the number of candidates grows exponentially, which is why the
chapter is about quantum computers.

This module does the listing and scoring on an ordinary CPU, for two shapes of problem, and
lets you check a faster solver against its answer.

## What this module is, and what the chapter is

The chapter's worked problem is a **multi-asset** dynamic allocation. $K$ units of capital
are split among $N$ assets at each of $H$ horizons; the allocations at one horizon are the
*pigeonhole partitions* of $K$ into $N$ slots, each with every sign pattern; a trajectory is
one allocation per horizon; and the trajectory with the best Sharpe ratio, net of a
square-root transaction cost, wins.

**The chapter's method lives in a separate module, `dynamic_allocation`**
(`openquant::dynamic_allocation` in Rust, `openquant.dynamic_allocation` in Python). It
generates the pigeonhole partitions (Snippet 21.1) and the signed weight set $\Omega$ with
gross exposure 1 (Snippet 21.2), and searches all of $\Omega^H$ for the trajectory with the
best Sharpe ratio net of $\tau = \sum c\sqrt{\lvert\Delta\omega\rvert}$ (Snippet 21.3). Its
$\Omega$ drops the book's repeated vectors, with the same optimum. $\lvert\Omega\rvert^H$
grows fast (38 vectors for $N = K = 3$, so 54,872 trajectories over 3 horizons and 79 million
over 5), and a search above `max_trajectories` returns `TooManyTrajectories` instead of
starting. Covariances must be positive definite, and `k` defaults to $N$. It has no page of
its own yet.

This module stays a generic exact-enumeration baseline. It provides the machinery
underneath: exhaustive search over a box of integers with any objective you write, and
exhaustive search over the inventory path of a **single** instrument, with a built-in
objective of its own that is not the chapter's.

## Two search spaces

**`DecisionSchema`** is a list of `IntegerVariable { lower, upper, step }`, each an arithmetic
grid. `solve_exact` visits the Cartesian product in lexicographic order and calls your
`IntegerObjective::evaluate` on each point. It refuses to start if the product has more than
`max_enumeration` points.

**`TradingTrajectorySchema`** describes a position traded over `step_trade_bounds.len()`
steps. Each step allows any integer trade within its `TradeBounds`, the inventory must stay
inside `[inventory_min, inventory_max]` after every trade, and `terminal_inventory`, if set,
is a hard constraint on where it ends. `enumerate_trading_paths` lists every feasible path by
depth-first search; `solve_trading_trajectory_exact` scores them with a
`TradingTrajectoryObjective` you supply. The ready-made score is `evaluate_trading_path`:

$$
J = \sum_{t=1}^{T}\Big( q_t\,r_t \;-\; \lambda\,q_t^2 \;-\; c_t\,\lvert\Delta q_t\rvert \;-\; \kappa\,\mathbf 1[\Delta q_t\neq 0] \Big) \;-\; \eta\,(q_T-q^\ast)^2
$$

where $q_t$ is the inventory *after* the trade $\Delta q_t$ at step $t$, $r_t$ is
`expected_returns[t]`, $\lambda$ is `risk_aversion`, $c_t$ is `impact_coefficients[t]`,
$\kappa$ is `fixed_ticket_cost` and $\eta$ penalises ending away from
`terminal_inventory_target`. The ticket cost $\kappa$ is the term that breaks convexity: it
charges the same for a trade of one unit as for a trade of a hundred.

## A fixed cost changes the path

Five steps, up to three units long, flat at the start and the end. The forecast is two
steps up, one down, one up, one flat. Every path is scored for four ticket costs:

```rust
use openquant::combinatorial_optimization::{
    evaluate_trading_path, solve_trading_trajectory_exact, CombinatorialOptimizationError,
    ObjectiveSense, TradeBounds, TradingTrajectoryObjective, TradingTrajectoryObjectiveConfig,
    TradingTrajectoryPath, TradingTrajectorySchema,
};

struct NetPnl(TradingTrajectoryObjectiveConfig);
impl TradingTrajectoryObjective for NetPnl {
    fn sense(&self) -> ObjectiveSense {
        ObjectiveSense::Maximize
    }
    fn evaluate(
        &self,
        path: &TradingTrajectoryPath,
    ) -> Result<f64, CombinatorialOptimizationError> {
        evaluate_trading_path(path, &self.0)
    }
}

let schema = TradingTrajectorySchema {
    initial_inventory: 0,
    inventory_min: 0,
    inventory_max: 3,
    step_trade_bounds: vec![TradeBounds { min_trade: -3, max_trade: 3 }; 5],
    terminal_inventory: Some(0),
    max_paths: 10_000,
};
let best_trades = |fixed_ticket_cost| {
    let objective = NetPnl(TradingTrajectoryObjectiveConfig {
        expected_returns: vec![0.02, 0.02, -0.01, 0.02, 0.0],
        risk_aversion: 0.001,
        impact_coefficients: vec![0.002; 5],
        fixed_ticket_cost,
        terminal_inventory_target: 0,
        terminal_inventory_penalty: 0.0,
    });
    solve_trading_trajectory_exact(&schema, &objective)
};

// Free to trade: sell out ahead of the dip and buy back after it.
let free = best_trades(0.0)?;
assert_eq!(free.best_path.trades, [3, 0, -3, 3, -3]);
assert_eq!(free.evaluated_paths, 256);
// Dodging the dip takes two more tickets, and at 0.03 they cost more than it saves.
let costly = best_trades(0.03)?;
assert_eq!(costly.best_path.trades, [3, 0, 0, 0, -3]);
assert_eq!(costly.best_path.inventory_path, [0, 3, 3, 3, 3, 0]);
// At 0.1 the round trip does not pay for its two tickets.
assert_eq!(best_trades(0.1)?.best_path.trades, [0, 0, 0, 0, 0]);
```

| `fixed_ticket_cost` | best trades | net objective |
| --- | --- | --- |
| 0 | `[3, 0, -3, 3, -3]` | 0.129 |
| 0.01 | `[3, 0, -3, 3, -3]` | 0.089 |
| 0.03 | `[3, 0, 0, 0, -3]` | 0.042 |
| 0.1 | `[0, 0, 0, 0, 0]` | 0 |

An optimiser that drops the ticket term, as a convex relaxation must, gives the first row's
answer at every cost. Each row is a global optimum over the 256 feasible paths.

## Checking a heuristic against the exact answer

Exhaustive search runs out of room quickly. Its lasting use is as a referee: on instances
small enough to enumerate, a faster method can be scored against the truth. `SolverAdapter` is
the interface for that method, and `compare_exact_and_adapter` runs both and reports how far
short the adapter fell.

`solve_with_adapter` runs an adapter on its own, on a box of any size: `max_enumeration` caps
enumeration and does not apply to it. It checks the answer rather than trusting it. The
decision must be a point of the box, and the reported objective must match the objective
re-evaluated at that decision, or the call fails with `InvalidAdapterResult`. So the gap
`compare_exact_and_adapter` reports is never floored: it is the true shortfall, and it cannot
be negative because the exact optimum is the best value in the same box.

```rust
use openquant::combinatorial_optimization::{
    compare_exact_and_adapter, CombinatorialOptimizationError, DecisionSchema,
    IntegerObjective, IntegerVariable, ObjectiveSense, OptimizationResult, SolverAdapter,
};

// A smooth bowl with one isolated spike: the kind of surface a local search misses.
struct Spike;
impl IntegerObjective for Spike {
    fn sense(&self) -> ObjectiveSense {
        ObjectiveSense::Maximize
    }
    fn evaluate(&self, x: &[i64]) -> Result<f64, CombinatorialOptimizationError> {
        let bowl = -((x[0] - 2).pow(2) + (x[1] - 2).pow(2)) as f64;
        Ok(if x == [8, 8] { bowl + 100.0 } else { bowl })
    }
}

// Coordinate ascent from the lower corner: step one variable by one while it helps.
struct Climb;
impl SolverAdapter for Climb {
    fn solve(
        &self,
        schema: &DecisionSchema,
        objective: &dyn IntegerObjective,
    ) -> Result<OptimizationResult, CombinatorialOptimizationError> {
        let mut x: Vec<i64> = schema.variables.iter().map(|v| v.lower).collect();
        let mut best = objective.evaluate(&x)?;
        let mut evaluated = 1;
        let mut improved = true;
        while improved {
            improved = false;
            for i in 0..x.len() {
                for d in [-1, 1] {
                    let mut y = x.clone();
                    y[i] += d;
                    if y[i] < schema.variables[i].lower || y[i] > schema.variables[i].upper {
                        continue;
                    }
                    let value = objective.evaluate(&y)?;
                    evaluated += 1;
                    if value > best {
                        (x, best, improved) = (y, value, true);
                    }
                }
            }
        }
        Ok(OptimizationResult {
            best_decision: x,
            best_objective: best,
            evaluated_candidates: evaluated,
        })
    }
}

let grid = IntegerVariable { lower: 0, upper: 10, step: 1 };
let schema = DecisionSchema { variables: vec![grid, grid], max_enumeration: 1_000 };
let report = compare_exact_and_adapter(&schema, &Spike, &Climb)?;

assert_eq!(report.exact.best_decision, [8, 8]);
assert_eq!(report.exact.evaluated_candidates, 121);
assert_eq!(report.adapter.best_decision, [2, 2]); // the top of the bowl
assert_eq!(report.objective_gap_vs_exact, 28.0);
```

The climber finds the bowl's peak in a few dozen evaluations and never sees the spike. That
gap of 28 is what an exact baseline is for: without it, the heuristic's answer looks fine.

## What to watch for

- **The search space is exponential.** A trajectory with $T$ steps and $m$ allowed trades per
  step has up to $m^T$ paths: 7 trades over 5 steps is 16,807 before the inventory bounds
  prune it, and over 10 steps it is 282 million. Treat hitting `max_paths` or
  `max_enumeration` as the signal to write an adapter, not to raise the cap. The adapter runs
  through `solve_with_adapter` above the cap; only the comparison against `solve_exact` needs
  a box small enough to enumerate.
- **`enumerate_trading_paths` holds every path in memory**, and
  `solve_trading_trajectory_exact` calls it, so memory grows with the path count, not just
  time. `solve_exact` over a `DecisionSchema` visits candidates without storing them.
- **`max_paths` counts feasible paths, and a schema with exactly that many is accepted.**
  Before this was fixed, a schema with exactly `max_paths` feasible paths was rejected
  whenever the terminal-inventory constraint pruned a path after the last one was found.
  `max_enumeration` counts every point in the box, feasible or not.
- **An objective cannot mark a candidate infeasible.** Returning an error aborts the whole
  search, and so does a non-finite score (`ObjectiveNotFinite`). Constraints that the box or
  the inventory bounds cannot express must be a large finite penalty.
- **Ties go to the first candidate found.** The comparison is strict, and enumeration is in
  lexicographic order from the lower bounds, so among equal scores the smallest decision
  (or the path that trades lowest first) wins.
- **The built-in objective is a single instrument with linear impact.** Impact is
  $c_t\lvert\Delta q_t\rvert$, not the chapter's $c\sqrt{\lvert\Delta\omega\rvert}$, and the
  return is earned on the inventory held *after* each step's trade. For any other cost model,
  implement `TradingTrajectoryObjective` yourself; the enumeration does not care.
- **`terminal_inventory` and `terminal_inventory_penalty` are different things.** The first
  removes paths that end elsewhere; the second only charges them. With both set to the same
  inventory, the penalty never applies.
- **Nothing here runs in parallel.** Part 5 of the book pairs brute force with
  multiprocessing. To split a large enumeration across cores, partition one variable's range
  and solve the pieces with [`hpc_parallel`](/modules/hpc-parallel/).

## Related modules

- [`hpc-parallel`](/modules/hpc-parallel/) — the atoms-and-molecules engine to spread a large
  search over threads.
- [`portfolio-optimization`](/modules/portfolio-optimization/) — static, continuous
  allocation, the convex problem this chapter argues is not enough.
- [`bet-sizing`](/modules/bet-sizing/) — discretised position sizes, the other place this
  library turns a continuous answer into integers.
