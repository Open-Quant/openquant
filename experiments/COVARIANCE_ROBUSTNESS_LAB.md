# Covariance estimates under constraints, heavy tails and shifts

This standalone mathematical experiment follows the exact Gaussian covariance
lab. It tests which conclusions survive changed distributions, nonnegative weight
constraints and a known change in future covariance. It calls NumPy/SciPy solvers,
not native OpenQuant allocators. Gaussian and t observations are abstract unbounded
mean-zero vectors; no compounding, financing, transaction costs, leverage limits
on unrestricted fits, or investment performance is modeled.

## Fixed estimators and numerical acceptance

All fits use only the centered training covariance, with denominator T-1:

- Unrestricted sample GMV: normalized S^-1 1. It is unavailable at T<=p, when
  Cholesky fails, or when the normalized covariance condition number exceeds 1e12.
- Equal weights: always a separately specified baseline.
- A fixed 50% blend of sample-GMV weights and equal weights. It requires the
  sample fit; no unavailable sample fit is replaced by equal weight.
- Long-only GMV: minimize w'Sw subject to sum(w)=1 and w>=0 using SLSQP.
- GMV from .5 S + .5 diag(S). Covariance shrinkage and weight blending are
  different estimators; diagonal shrinkage does not imply equal weights.

The long-only objective is divided by max(abs(S)); this changes its scale but not
its minimizers. Zero covariance is allowed. A normalized eigenvalue below
-64*p*machine-epsilon rejects a non-PSD objective; tiny numerical residuals are
tolerated, not treated as certified exact arithmetic. SLSQP uses analytical
objective/constraint derivatives, ftol=1e-14 and maxiter=500. Its success flag is
insufficient: accepted fits must also have normalized budget, nonnegativity,
dual-feasibility and complementarity errors at most 1e-7. Solver exceptions,
malformed outputs and failed checks are retained as unavailable. No clipping,
pseudoinverse, projection, jitter or fallback silently changes the estimator.

For simplex feasibility, gamma=w'Sw and the KKT conditions are Sw-gamma*1>=0
and w_i*(Sw-gamma*1)_i=0. These conditions suffice for a PSD quadratic objective
by convexity. They are derived directly here; [Boyd and Vandenberghe §5.5.3,
printed page 244](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) supplies
the general KKT sufficiency theorem. [SciPy 1.17 SLSQP documentation](https://docs.scipy.org/doc/scipy-1.17.0/reference/optimize.minimize-slsqp.html)
explains the optimizer stopping criteria and why its returned bound multipliers
are not a complete certificate. An independent small-dimensional support-enumeration
solver checks accepted solutions. A pre-grid qualification exposed success flags
with residuals above the independent threshold at ftol=1e-12; ftol was tightened
before retained simulations, with the qualification failures preserved.

## Protocol fixed before retained simulation

Use p=8, T in {4,12,68}, seed=419, and 1,000 independent repetitions per cell.
There are three training distributions and four covariance regimes: 36 cells and
180 method cells. Methods and their 50% parameters are fixed, not selected from
observed outcomes.

Distributions are Gaussian; elliptical t5 obtained with one independent chi-square
scale per row; and independent coordinate t5 followed by the covariance Cholesky
transform. Latent observations are Z*sqrt(3/U), U~chi-square(5), so
E[3/U]=1. The common-row versus separate-coordinate scale changes dependence
and radial tails despite matching covariance. These families do not inherit the
Gaussian Wishart/F reference law. Covariance matching is also checked empirically
in fixed-seed tests; that check does not prove the full distribution.

Regimes are identity covariance; diagonal variances geometrically spaced from 1
to 16; independent repeated hedge blocks [[1,1.5],[1.5,4]]; and diagonal training
variances with their order reversed in the known future covariance. The first
three are stationary. Future covariance enters risk scoring and oracle diagnostics
only. Diagonal stationary and reversal regimes share identical training draws,
so their changed future risks are paired. Changing future covariance while keeping training observations fixed leaves every fitted
weight unchanged.

Population oracles are analytic for these controls. A positive diagonal covariance
has inverse-variance GMV weights, which are already long-only. For m=p/2 hedge
blocks, unrestricted within-block weights are (1.25,-.25), scaled by 1/m across
blocks, with risk .875/m. Long-only weights are (1,0)/m, risk 1/m. Thus forbidding
a negative position removes a genuine population hedge in this model. Both
future unrestricted and future long-only oracle risks are reported so estimation
error is not confused with the cost of the constraint. No method is claimed to
dominate all regimes or to learn the future covariance shift.

## Reporting and limits

Each fit records weights, true future variance, fitted/sample objective, leverage,
weight norm, and failures. Raw derived records, training input hashes, covariance
matrices/hashes, configuration, source hash and NumPy/SciPy versions are retained.
Stage timing is isolated in performance, excluded from exact scientific replay.
Covariance construction, sample inverse, long-only optimization and diagonal-shrink
inversion are timed; equal-weight and weight-blend construction are not separately
timed. Total wall time covers report construction too.
Input matrices are reproducible from seeds; they are not all stored separately.

Means and quantiles describe available fits. When numerical failures occur, this
conditioning can select easier samples. Method and paired-comparison denominators,
failure reasons and unavailable-pair counts are explicit. Risk-mean MCSE is deliberately omitted as a conservative uniform reporting
choice, without relying on Gaussian reference laws for every method. Exact simplex
weights have future risk bounded by the largest future marginal variance, so
long-only risk moments are finite; the omission is not a claim otherwise. Paired lower-risk frequencies are bounded
indicators; their pointwise Wilson intervals are not simultaneous across the grid.
A declared 1e-12-times-equal-risk comparison threshold defines the recorded
indicator. It does not account for optimizer error: when the unrestricted fit is
already feasible, the exact long-only optimum is identical, but small numerical
weight differences can still count as wins. Those near-identical-pair rates cannot
rank the estimators. A separately labeled post-grid sensitivity audit retains all
thresholds and the original results.

```bash
OPENBLAS_NUM_THREADS=1 python experiments/covariance_robustness_lab.py \
  --repetitions 1000 --train-size 12 --out covariance-robustness.json
```

NumPy 2.3.5 and SciPy 1.17.0 were used in qualification. This is research evidence,
not a deployment-ready optimizer API or a guarantee about financial observations.

## Retained grid findings

The declared grid completed 36,000 independent training fits in approximately
50.1 seconds of measured subprocess time, producing 180,000 method records.
Only the prescribed T=4 sample-inverse and weight-blend methods were unavailable
(24,000 method records); all other fits passed their declared checks. Independent
review rescored all future variances, checked all oracle bounds, and recomputed
108 paired summaries. Maximum accepted KKT residual was approximately 7.07e-8.

Gaussian isotropic median future variance at T=12 was .3026 for sample GMV,
.2135 for long-only, .1578 for diagonal covariance shrinkage, and .125 for equal
weight. In the hedge control at T=68, sample GMV median risk .2413 was lower than
long-only .2604: the constraint removes a real population hedge. The paired
long-only lower-risk frequency was .052 (pointwise Wilson interval about
[.0399,.0676]) in that cell, versus .805 at T=12. The associated risk differences
and numerical-accuracy qualification must accompany such frequencies.

The known diagonal reversal exposes model mismatch. At T=68 Gaussian median
future variance was about 2.7433 for sample GMV, 2.7310 for long-only and 1.4925
for the weight blend, versus .7323 for equal weight. This does not recommend
equal weighting universally: in the stationary diagonal cell, corresponding
sample and equal-weight risks were .3779 and .7323. Heavy-tail controls changed
finite-sample risk and concentration without establishing a universal winner.

Rank-deficient T=4 long-only solutions also merit separate interpretation:
multiple portfolios may minimize the training objective, and a solver's choice
among them need not minimize future risk. This motivates a distinct follow-up on
explicit tie-breaking; the present frozen results are not silently replaced.

The post-grid numerical-equivalence audit found 7,733 paired fits where the
unrestricted solution was already nonnegative, so the exact constrained optimum
is the same. Within the Gaussian isotropic T=68 cohort of 983 such fits, apparent
long-only wins were .5351 at the declared 1e-12 relative threshold, .2584 at
1e-8, and zero at 1e-6. Other shifted-covariance cohorts had differences as large
as 1.14e-5 relative to equal-weight risk. All sensitivity thresholds are retained;
none is selected as a universal numerical guard. Reusing a verified feasible
unrestricted optimum is a separate implementation improvement, not a change to
this frozen baseline.
