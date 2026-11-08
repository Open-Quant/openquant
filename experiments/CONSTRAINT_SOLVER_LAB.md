# Constraint solvers: nonunique fits and a verified fast path

This follow-up separates two operational cases that the covariance robustness lab
exposed. A positive-definite covariance has a unique unrestricted GMV solution;
if that solution is already nonnegative, it is also the unique long-only solution.
A singular covariance can instead have many different long-only minimizers with
the same training objective and different future risks. Solver choice among them
is then part of the estimator. Neither case implies an investment recommendation.

The module imports the frozen covariance-robustness helper and records its hash.
No native OpenQuant optimizer is used. Matrices are deliberately small mathematical
controls, not fitted market models. All variance scores use specified future
covariance; no returns, leverage financing, execution or compounding are modeled.

## A qualified shortcut, not a different allocation rule

`fast_long_only` first requires positive definiteness and normalized condition
number at most 1e12. A supplied unrestricted candidate must be finite, fully
invested and nonnegative, and pass the existing normalized KKT threshold plus
full unrestricted stationarity across all coordinates. When accepted, its weights
are returned unchanged; otherwise the original strict SLSQP method is called.
Constrained KKT conditions alone do not establish that a candidate is the
unrestricted optimum when a boundary constraint is active, so that stronger
stationarity check matters. Invalid, negative or unqualified candidates do not
bypass the solver. This remains numerical qualification, not certified arithmetic.

The mathematical justification is immediate: a global unrestricted minimizer
that belongs to the smaller feasible set minimizes over that set too. Positive
definiteness gives uniqueness. General convex KKT sufficiency is verified in
[Boyd and Vandenberghe §5.5.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
This eliminates unnecessary differences between two numerical implementations of
the same allocation rule. It does not improve that rule's mathematical future risk.

## Zero-training-risk simplex geometry

For training covariance S=F'F, a fully invested nonnegative portfolio has zero
training risk exactly when Fw=0. The feasible set is the compact polytope
`A w=b, w>=0`, where A stacks the budget row and independent factor rows, and
b=(1,0,...,0). Row normalization preserves these equations. The experiment
explicitly enumerates basic feasible supports to obtain vertices; it is limited
to at most eight coordinates and uses declared floating tolerances.

A positive-definite quadratic objective is minimized on this set by enumerating
all nonempty faces and solving each equality-constrained quadratic. The true
minimizer's positive support appears among those faces. Solving rank-deficient
face multiplier equations by least squares and checking their residuals is part
of this explicit reference algorithm, not a replacement covariance estimator.
This is a small numerical reference for the qualified designs, not a general
certified polytope library.

Two secondary objectives have different meanings: squared Euclidean weight norm
specifies a training-only lexicographic tie-break; known future covariance supplies
a diagnostic minimum over the zero-training-risk set. The latter is not an
observable fitting input or a recommended rule. A convex quadratic's maximum
on a compact polytope is attained at a vertex. Its minimum can be interior, so the
minimum vertex score is not substituted for the future minimum.

The fixed designs are:

- F=(1,-1,0,0), future diagonal (1,1,4,16). Zero-risk vertices are
  (.5,.5,0,0), (0,0,1,0), (0,0,0,1), with future risks .5, 4, 16.
  The minimum-norm tie-break is uniform, risk 1.375. The future diagnostic
  minimizer is (16,16,4,1)/37, risk 16/37, below every vertex risk.
- F=(2,-1,0,0), with the same future diagonal. Minimum norm gives
  (3,6,5,5)/19, not equal weights. This also makes a finite ridge perturbation
  visibly different from lexicographic optimization.
- Two factor rows (1,-1,0,0,1,-1) and (0,0,1,-1,1,-1), with future diagonal
  (1,2,4,8,16,32), checking a less trivial equality polytope.

For each design SLSQP starts from uniform weights and every coordinate vertex.
These are fixed starts, not random replications or a probability distribution of
solver behavior. All accepted solutions, feasibility residuals and future scores
are retained. A ridge path adds lambda*max(abs(S))*I for lambda in
{1e-8,1e-4,.01,.1}. Finite ridge changes the primary objective; it is not silently
called an exact tie-break. In the asymmetric control it can permit nonzero
original training exposure. Small-ridge numerical conditioning is also explicit.

## Frozen benchmark and cost scope

Before retained runs, set benchmark_repetitions=1000 and seed=419. Recreate all
36 covariance-robustness cells with the same training streams, including paired
stationary/reversal observations. Compare strict SLSQP with the verified shortcut
on every matrix. The already computed unrestricted candidate is supplied to the
shortcut; this measures incremental cost in a workflow that already needs that
candidate. It is not a standalone solver-speed claim that ignores a required
inverse calculation. Strict/shortcut order alternates by replication to reduce
systematic ordering effects. Timings are in performance, excluded from replay.

Every raw record retains the training hash, branch used, availability, weights
and future scores. Shortcut branches must exactly reuse the supplied weights;
fallback branches are compared against the same strict solver. The synthetic
future-risk differences between mathematically equivalent fits are numerical
implementation effects, not evidence of economic performance. The default CLI
uses only 100 repetitions per cell to keep a routine run modest.

```bash
OPENBLAS_NUM_THREADS=1 python experiments/constraint_solver_lab.py \
  --benchmark-repetitions 1000 --out constraint-solver.json
```

## Retained benchmark findings

The frozen 1,000-replicate grid contains 36,000 training matrices. Independent audit confirmed that all training hashes and strict solver weights exactly match the prior robustness grid. The shortcut reused 7,733 inverse solutions without changing their weights; all 28,267 remaining cases exactly matched the strict fallback. No branch was unavailable. This removes artificial numerical differences between equivalent solutions when the qualified inverse is already available; it does not establish superior statistical risk.

One retained run measured 29.39 seconds for strict calls and 25.48 seconds for shortcut calls (62.21 seconds overall including generation and scoring). These are incremental timings with inverse weights precomputed, not a standalone production benchmark or an uncertainty-qualified speed claim. Workload composition matters: verification overhead remains when the inverse solution is infeasible. All raw outputs and reviewer evidence are retained privately.

Independent exact rational checks give minimum future risks over the zero-training-risk sets of 16/37, 80/169 and 252/443 for the three fixed cases. These are diagnostic truths only. Independent support enumeration also checked 72 random feasible polytopes against a separate constrained optimization implementation, with maximum objective discrepancy 4.54e−14. Floating-point qualification remains the stated limit.
