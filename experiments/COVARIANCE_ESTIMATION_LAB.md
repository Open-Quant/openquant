# Covariance estimation, concentration and population risk

Status: private synthetic research. Owner: OpenQuant maintainers.

This study isolates covariance-estimation error with exact distributional controls.
It asks whether a smaller fitted variance necessarily means less population risk,
and what is gained or lost by shrinking **weights** toward equal weight. It is
separate from the [rolling risk-model lab](RISK_MODEL_LAB.md): that experiment calls
native long-only allocators and tracks holdings and toy costs. This one uses a
single centered Gaussian sample, an explicit unrestricted formula, and known
population covariance. No OpenQuant native implementation is tested.

## Information and allocation contract

There are `p >= 2` assets and `T >= 2` independent zero-mean Gaussian observations
with fixed covariance Sigma. They are abstract unbounded random vectors, not
necessarily economically valid simple returns. No compounded wealth, financing,
execution cost or profitability is calculated.

Sample covariance is the centered cross-product divided by `T-1`. The sample GMV
weights solve `S z = 1` then normalize `z` to sum to one. Weights may be negative
and gross leverage may exceed one. There is no leverage constraint or clipping.
For each predeclared lambda the weight vector is

`w_lambda = (1-lambda) * w_sample + lambda * w_equal`.

This is **weight shrinkage**, not covariance shrinkage, Ledoit–Wolf estimation, or
an intensity estimated from the evaluation sample. Population covariance enters
only reference calculations and risk scoring; it does not enter the sample fit.

At `T <= p`, the centered covariance lacks full rank, so the inverse-defined
sample estimator is unavailable. This does not mean singular quadratic
minimization has no solutions: nullspace or pseudoinverse portfolios require a
separately specified estimator. We add neither a pseudoinverse nor jitter.
The lambda=1 equal-weight baseline remains available even in this case.
Numerical failures are retained, not silently replaced with equal weight.

## Population and finite-sample references

For positive definite Sigma, let

`w_star = Sigma^-1 1 / (1' Sigma^-1 1)` and `v_star = 1 / (1' Sigma^-1 1)`.

The formula and distinction between covariance error and portfolio-risk loss are
consistent with the open-access primary study by
[Holgersson, Karlsson and Stephan](https://link.springer.com/article/10.1007/s10182-018-00349-7).
The [Okhrin–Schmid publisher preview](https://www.sciencedirect.com/science/article/pii/S0304407605001442)
confirms Gaussian finite-sample analysis and the centered `T-1` covariance
convention. Its full theorem proof was not accessed; the following derivation is
independent, rather than an assertion that its paywalled proof was read.

Whiten the data and rotate the budget direction to the first coordinate. The
centered cross-product has a Wishart distribution with `nu=T-1` degrees of freedom.
Partition its remaining `d=p-1` coordinates as a regression design `Z` and first
coordinate as response `y`. The estimated slope `beta=(Z'Z)^-1 Z'y`, conditional
on Z, is Gaussian with covariance `(Z'Z)^-1`. Integrating the Gaussian random
design gives a multivariate t slope with `r=nu-d+1=T-p+1` degrees of freedom and
scale `I/r`. Gaussian/Wishart background is described in
[Stanford's normal-theory notes](https://web.stanford.edu/class/stats305c/lectures/Normal_Theory.html).

The resulting exact model laws for `T>p` are

`population_risk / v_star = 1 + (p-1)/(T-p+1) * F(p-1,T-p+1)`

and

`sample_forecast / v_star = chi_square(T-p)/(T-1)`.

The first is squared estimation error plus oracle risk. The second is the
regression residual Schur complement; its degrees of freedom are
`nu-d=T-p`, not `T-p+1`. Expectations are therefore

`E[population_risk] = v_star * (T-2)/(T-p-1)`, when `T>p+1`,

`E[sample_forecast] = v_star * (T-p)/(T-1)`.

The error in sample GMV weights has zero mean. Writing `v_equal=w_equal' Sigma
w_equal` gives the fixed-weight-blend expectation for `T>p+1` (lambda=1 is
separately equal to `v_equal`, even outside that regime):

`E[risk_lambda] = v_star + (1-lambda)^2 * v_star*(p-1)/(T-p-1)
                 + lambda^2*(v_equal-v_star)`.

This separates estimation variation from the bias of the equal-weight target.
The fitted first-order condition also gives the pathwise forecast identity

`forecast_lambda = (1-lambda^2)*forecast_sample + lambda^2*forecast_equal`.

Under isotropic Sigma=I, `v_star=1/p` and equal weight is the oracle. Every
fully invested vector satisfies

`population_risk = sum(w_i^2) = 1/p + ||w-w_equal||^2`.

For this family, fixed shrinkage scales the F-distributed excess risk by
`(1-lambda)^2`. For heterogeneous variances, squared weight norm is still a
concentration diagnostic, but it no longer equals population risk. No scaled-F
law is claimed for partial shrinkage toward a nonoracle target.

## Infinite moments are results, not numerical failures

For lambda<1 and `T=p+1`, the inverse exists almost surely under the continuous
Gaussian model, but the population-risk **mean is infinite**. The report suppresses
sample-mean inference and retains empirical quantiles and all weights.
For `T=p+2` and `T=p+3`, the risk mean is finite but its variance is infinite;
a conventional finite-variance mean MCSE is suppressed. Finite observed values
cannot establish existence of population moments.

At `T>p+3`, risk has finite variance and a sample-standard-deviation MCSE is
reported across independent replications. It can still be unstable near that
boundary. Absolute-weight/leverage means have different moment requirements;
their MCSE requires `T>p+1`. Bounded negative-weight counts and quantile-exceedance
indicators retain finite-variance Monte Carlo summaries. Equal-weight population
risk is deterministic and retains its finite moments in every setting.

Optional SciPy F and chi-square quantiles supply prespecified .5/.9/.99 thresholds.
Exceedance frequencies have Wilson intervals and Monte Carlo standard errors,
even when the risk mean is infinite. Point-mass equal-weight thresholds have
expected strict exceedance zero, not `1-q`. Intervals across many thresholds,
methods and scenarios are marginal, not simultaneous.

## Frozen controls

The retained grid fixes 5,000 independent replications, seed137, p in {8,32},
T in {p,p+1,p+2,p+3,p+4,2p+4,8p+4}, and lambda in {0,.25,.5,.75,1}.
Each setting uses two families:

- `isotropic`: Sigma=I. Equal weight is the exact oracle; apparent fitted
  improvement can be entirely estimation noise.
- `heterogeneous_diagonal`: variances geometrically spaced from 1 to 16.
  The known oracle uses inverse population variances; equal weight is generally
  suboptimal. This prevents interpreting isotropic results as a universal
  advantage of equal weighting.

Methods use the same sample within a replication. Every family/configuration/
replication has its own deterministic stream, independent of requested family or
lambda order. The grid is not tuned after results; every row-budget and boundary
control is retained. This is not an estimate of a covariance structure in markets.

## Run and artifacts

```bash
OPENBLAS_NUM_THREADS=1 python experiments/covariance_estimation_lab.py \
  --out /tmp/covariance-estimation.json
python -m pytest python/tests/test_covariance_estimation_lab.py -q
```

NumPy runs estimation, moments and empirical diagnostics. SciPy is optional for
theoretical quantiles; its presence/version is recorded, and missing quantiles
remain unavailable. No native extension is imported.

Every replication retains signed weights, budget residual, squared norm, gross
leverage, largest absolute weight, negative-weight count, sample covariance
condition number, population variance and in-sample forecast. Known-oracle risk
and equal-weight risk are recorded per scenario. Paired differences from the
constant equal-weight population risk share the original replication.
Input hashes contain array shape and canonical little-endian float64 bytes.
Source, numerical-library versions, configuration and timing accompany CLI output.

Successful-only means after numerical failures are not unconditional estimates
of the stated exact law. Unavailable counts and reasons stay visible. Native
long-only allocation, leverage caps, trading costs, serial dependence, fat tails,
changing covariances and real data remain outside this study. The theoretical
quantiles are evaluated numerically, not with certified interval arithmetic.

Acceptance tests include diagonal hand calculations, unrestricted signed weights,
centered degrees of freedom, population-truth isolation, scale/translation
invariance, pathwise forecast and concentration identities, exact moment boundaries,
closed-form F/chi-square median checks, unavailable inverse controls, and reproducible
reports. Independent review checks the regression derivation and distributional
representation before retained runs.

## Retained first-grid observations

The frozen 5,000-replication grid illustrates both sides of the tradeoff. For p=8,
selected population-risk means were:

| Covariance | T | Lambda | Observed mean risk | Model expectation | MCSE |
|---|---:|---:|---:|---:|---:|
| Isotropic | 12 | 0 | .40538 | .41667 | .00537 |
| Isotropic | 12 | .5 | .19510 | .19792 | .00134 |
| Isotropic | 12 | 1 | .12500 | .12500 | 0 |
| Heterogeneous diagonal | 12 | 0 | 1.12214 | 1.13803 | .01726 |
| Heterogeneous diagonal | 12 | .5 | .63366 | .63828 | .00449 |
| Heterogeneous diagonal | 12 | 1 | .73226 | .73226 | 0 |
| Heterogeneous diagonal | 68 | 0 | .38204 | .38192 | .00033 |
| Heterogeneous diagonal | 68 | .5 | .44885 | .44925 | .00034 |
| Heterogeneous diagonal | 68 | 1 | .73226 | .73226 | 0 |

These are example rows from the retained complete grid, not a post-grid selection
of lambda. Under isotropy at T=12, the sample portfolio's average fitted variance
was only .04552, despite mean population risk .40538. At T=68 in the heterogeneous
family, shrinking all the way to equal weight increased risk. No one method wins
under every declared covariance and sample-size combination.

At T=p, sample/partial-shrink portfolios were unavailable and equal weight remained
available. At T=p+1, risk means were omitted as specified; finite medians do not
change the infinite-mean reference. All empirical exceedance results are retained.
The largest population-quantile discrepancy in this grid was 2.59 known-binomial
standard errors (isotropic p=32,T=33, .9-quantile exceedance .089 versus .1).
Multiple shrinkage settings share the same underlying exceedance events; these
are neither independent discoveries nor simultaneous confidence statements.

A preliminary run exposed a tiny artificial MCSE for constant equal-weight risk
from floating mean rounding. The final helper recognizes exact constant arrays.
The same predeclared grid was rerun without changing generation or methods;
preliminary source, summary and file hashes are retained separately. Exact-constant
handling does not turn an infinite-moment population into a finite-moment one.
