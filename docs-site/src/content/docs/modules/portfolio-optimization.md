---
title: "portfolio_optimization"
description: "Mean-variance allocation with weight bounds: inverse variance, minimum volatility, maximum Sharpe ratio, and minimum risk for a target return."
status: authored
last_authored: '2026-09-24'
audience:
  - quant-dev
  - platform-engineering
module: "portfolio_optimization"
api_surface: "both"
afml_chapter:
  - "16"
citation:
  - "Markowitz, H. (1952). Portfolio selection. Journal of Finance 7(1), 77–91."
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 16: §16.2 The Problem with Convex Portfolio Optimization; §16.3 Markowitz's Curse."
  - "Stellato, B., Banjac, G., Goulart, P., Bemporad, A. and Boyd, S. (2020). OSQP: an operator splitting solver for quadratic programs. Mathematical Programming Computation 12(4), 637–672."
  - "Michaud, R. O. (1989). The Markowitz optimization enigma: is 'optimized' optimal? Financial Analysts Journal 45(1), 31–42."
rust_api:
  - "allocate_from_inputs"
  - "allocate_with_solution"
  - "allocate_inverse_variance"
  - "allocate_min_vol"
  - "allocate_max_sharpe"
  - "allocate_efficient_risk"
  - "compute_expected_and_covariance"
  - "AllocationOptions"
  - "MeanVariance"
  - "ReturnsMethod"
  - "AllocError"
python_api:
  - "portfolio.allocate_from_inputs"
  - "portfolio.allocate_with_solution"
  - "portfolio.allocate_inverse_variance"
  - "portfolio.allocate_min_vol"
  - "portfolio.allocate_max_sharpe"
  - "portfolio.allocate_efficient_risk"
sidebar:
  badge: Module
---

This is the classical toolbox: given expected returns $\mu$ and a covariance matrix $\Sigma$,
find weights that minimise risk, maximise the Sharpe ratio, or hit a return target at least
risk, subject to bounds on each weight. AFML's Chapter 16 exists because these portfolios are
fragile, and [`hrp`](/modules/hrp/) is the book's alternative. They remain the benchmark
every alternative is measured against, and with sensible bounds they are often good enough.
The Python module is `openquant.portfolio`.

## Four solutions

All four are long-only and fully invested by default, $\sum_i w_i=1$ and $0\le w_i\le 1$.

| `solution` | Problem |
| --- | --- |
| `"inverse_variance"` | weights proportional to the reciprocal of each variance; ignores correlation, then clamps to the bounds |
| `"min_volatility"` | minimise portfolio variance |
| `"max_sharpe"` | maximise excess return over `risk_free_rate` per unit of volatility |
| `"efficient_risk"` | minimise portfolio variance subject to an expected return of at least `target_return` |

Written out, the last is

$$
\min_{w}\; w^\top\Sigma\,w
\quad\text{s.t.}\quad \mu^\top w\ge r^*,\;\; \mathbf 1^\top w=1,\;\; l_i\le w_i\le u_i
$$

and minimum volatility is the same without the return constraint. Maximum Sharpe is not a
quadratic programme as stated, since it is a ratio. It becomes one under the substitution
$y=\kappa w$ with $\kappa>0$: minimise $y^\top\Sigma y$ subject to $(\mu-r_f)^\top y=1$, with
the bounds rewritten as $l_i\,\mathbf 1^\top y\le y_i\le u_i\,\mathbf 1^\top y$ so that they stay
linear, then recover $w=y/\mathbf 1^\top y$.

The three optimisations go to an internal solver, ADMM in the OSQP formulation (Stellato et
al., 2020) followed by an exact solve on the active constraints it identifies. **Bounds are
part of the problem.** The closed-form minimum-variance solution shorts assets, and clipping
it to zero afterwards is not the long-only optimum; the solver finds the constrained one.

Bounds come two ways. `tuple_bounds = (lo, hi)` applies to every asset; `bounds` sets them per
asset by index and takes precedence (a `HashMap<usize, (f64, f64)>` in Rust, a list of
`(index, lo, hi)` in Python).

```python
from openquant import portfolio

# Expected annual returns and an annual covariance matrix for four assets.
names = ["bonds", "equity", "small cap", "gold"]
mu = [0.03, 0.07, 0.09, 0.04]
vol = [0.05, 0.16, 0.22, 0.15]
rho = [[1.0, 0.1, 0.1, 0.1], [0.1, 1.0, 0.8, 0.0], [0.1, 0.8, 1.0, 0.0], [0.1, 0.0, 0.0, 1.0]]
cov = [[rho[i][j] * vol[i] * vol[j] for j in range(4)] for i in range(4)]

def show(label, result):
    weights, risk, ret, _ = result
    cells = "  ".join(f"{round(w, 2) + 0.0:5.2f}" for w in weights)
    print(f"{label:24s} {cells}   return {ret:.3f}  risk {risk:.3f}")

print(" " * 25 + "  ".join(f"{n[:5]:>5s}" for n in names))
show("inverse variance", portfolio.allocate_from_inputs(mu, cov, "inverse_variance"))
show("minimum volatility", portfolio.allocate_from_inputs(mu, cov, "min_volatility"))
show("maximum Sharpe", portfolio.allocate_from_inputs(mu, cov, "max_sharpe", risk_free_rate=0.02))
show("6.5% return, least risk", portfolio.allocate_from_inputs(mu, cov, "efficient_risk", target_return=0.065))
show("same, 35% cap per asset", portfolio.allocate_from_inputs(mu, cov, "efficient_risk", target_return=0.065,
                                                              tuple_bounds=(0.0, 0.35)))

# Markowitz's curse: nudge one expected return by half a percentage point.
base = portfolio.allocate_from_inputs(mu, cov, "max_sharpe", risk_free_rate=0.02)[0]
nudged = portfolio.allocate_from_inputs([0.03, 0.075, 0.09, 0.04], cov, "max_sharpe", risk_free_rate=0.02)[0]
print("equity +0.5pt moves weights by:", "  ".join(f"{round(b - a, 2) + 0.0:+.2f}" for a, b in zip(base, nudged)))
```

```text
                         bonds  equit  small   gold
inverse variance          0.79   0.08   0.04   0.09   return 0.036  risk 0.049
minimum volatility        0.87   0.06   0.00   0.07   return 0.033  risk 0.048
maximum Sharpe            0.55   0.17   0.15   0.14   return 0.047  risk 0.069
6.5% return, least risk   0.14   0.27   0.36   0.23   return 0.065  risk 0.124
same, 35% cap per asset   0.13   0.29   0.35   0.23   return 0.065  risk 0.124
equity +0.5pt moves weights by: -0.02  +0.09  -0.06  +0.00
```

The last line is the argument of AFML §16.3 in one number. Raising the expected return of
equity from 7.0% to 7.5%, a change far smaller than anyone's ability to forecast it, moves
nine points of the portfolio into equity, six of them out of its close substitute. Equity
and small cap are correlated at 0.8, so the optimiser treats them as nearly interchangeable
and swings between them on small differences in $\mu$ (Michaud, 1989, called it error
maximisation). The remedies are the ones on this site: bounds, as in the fifth row; leaving
$\mu$ out altogether, as minimum volatility and [`hrp`](/modules/hrp/) do; or a better
covariance estimate.

## From prices

`allocate_min_vol`, `allocate_max_sharpe`, `allocate_efficient_risk`,
`allocate_inverse_variance` and `allocate_with_solution(prices, solution, options)` take a
matrix of prices, rows by date and columns by asset, and estimate the inputs themselves:
**simple** returns $p_t/p_{t-1}-1$, the same convention as [`cla`](/modules/cla/),
[`hrp`](/modules/hrp/) and [`hcaa`](/modules/hcaa/); an expected return that is either
their mean (`ReturnsMethod::Mean`) or an exponentially weighted mean
(`ReturnsMethod::Exponential { span }`); and their sample covariance. Simple returns are the
consistent choice for a one-period problem on weights, because a portfolio's simple return is
the weighted sum of its assets' simple returns, which is not true of log returns.

**Everything is annual.** The expected returns *and* the covariance are multiplied by the
number of periods in a year, $252/\text{step}$, so `portfolio_return` is
$\mu^\top w$, `portfolio_risk` is the annualised volatility $\sqrt{w^\top\Sigma w}$, and
`portfolio_sharpe` is $(\mu^\top w-r_f)/\sqrt{w^\top\Sigma w}$ in annual units. Give
`risk_free_rate` and `target_return` as annual figures. `compute_expected_and_covariance`
returns the same annualised $\mu$ and $\Sigma$ without solving anything, so passing them to
`allocate_from_inputs` reproduces the result from prices. `resample_by` of `"W"` or `"M"`
keeps every 5th or 21st row first, and the annualisation factor becomes 252/5 or 252/21.

## From Rust

```rust
use nalgebra::DMatrix;
use openquant::portfolio_optimization::{allocate_from_inputs, AllocError, AllocationOptions};

let mu = [0.03, 0.07, 0.09, 0.04];
let vol = [0.05, 0.16, 0.22, 0.15];
let rho = [[1.0, 0.1, 0.1, 0.1], [0.1, 1.0, 0.8, 0.0], [0.1, 0.8, 1.0, 0.0], [0.1, 0.0, 0.0, 1.0]];
let cov = DMatrix::from_fn(4, 4, |i, j| rho[i][j] * vol[i] * vol[j]);

let min_vol = allocate_from_inputs(&mu, &cov, "min_volatility", &AllocationOptions::default())?;
assert!((min_vol.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
assert!(min_vol.weights.iter().all(|w| *w > -1e-9));
assert!((min_vol.portfolio_risk - 0.0476).abs() < 1e-4);

// A return target above the minimum-variance return binds exactly, and costs risk.
let target = AllocationOptions { target_return: 0.065, ..AllocationOptions::default() };
let efficient = allocate_from_inputs(&mu, &cov, "efficient_risk", &target)?;
assert!((efficient.portfolio_return - 0.065).abs() < 1e-7);
assert!(efficient.portfolio_risk > min_vol.portfolio_risk);

// With a 35% cap the most any portfolio can return is 6.8%, so 7% is infeasible...
let capped = AllocationOptions { target_return: 0.07, tuple_bounds: Some((0.0, 0.35)), ..AllocationOptions::default() };
assert!(matches!(
    allocate_from_inputs(&mu, &cov, "efficient_risk", &capped),
    Err(AllocError::OptimizationFailed(_))
));
// ...and bounds that cannot sum to one are rejected before solving.
let impossible = AllocationOptions { tuple_bounds: Some((0.3, 0.4)), ..AllocationOptions::default() };
assert!(matches!(
    allocate_from_inputs(&mu, &cov, "min_volatility", &impossible),
    Err(AllocError::InfeasibleBounds { .. })
));
```

## What to watch for

- **With `allocate_from_inputs` the units are yours.** `portfolio_sharpe` is
  $(\mu^\top w-r_f)/\sqrt{w^\top\Sigma w}$ in whatever units $\mu$, $\Sigma$ and
  `risk_free_rate` were given in, so they must agree: an annual $\mu$ with a daily $\Sigma$
  overstates the Sharpe ratio by $\sqrt{252}$. The weights do not depend on the scale of
  $\Sigma$, only the reported risk and Sharpe ratio do.
- **`portfolio_sharpe` is reported for every solution**, not only `"max_sharpe"`; it is the
  Sharpe ratio of the portfolio returned, against `risk_free_rate`. It is 0 only when the
  portfolio's risk is 0. Before [#110](https://github.com/Open-Quant/openquant/issues/110)
  was fixed it was 0 for the other three solutions, and from prices it divided an annual
  return by a daily volatility, so it read about 16 times ($\sqrt{252}$) too high.
- **It is in-sample.** The maximum-Sharpe portfolio's `portfolio_sharpe` is the best ratio on
  the history it was fitted to, by construction, and overstates what the portfolio will earn.
- **A target return that cannot be met is reported as `OptimizationFailed`**, with the text
  "no portfolio satisfies the constraints". The same error covers a solver that did not
  converge. Bounds that cannot sum to one are caught earlier as `InfeasibleBounds`, with the
  sums in the message.
- **`"efficient_risk"` treats the target as a floor.** A target below the minimum-variance
  portfolio's return gives the minimum-variance portfolio, not a deliberately worse one.
- **`"max_sharpe"` needs at least one asset above the risk-free rate**, and says so
  otherwise.
- **The solver is dense and meant for tens of assets.** Each iteration is a dense
  back-substitution, and it has been tested at that scale. Thousands of assets call for a
  sparse solver.
- **The covariance is the plain sample covariance.** With more assets than a few dozen
  observations per asset it is poorly conditioned, and every weakness described above gets
  worse. Shrink it, or denoise it, before passing it to `allocate_from_inputs`.

## Related modules

- [`cla`](/modules/cla/) — the whole efficient frontier from the same inputs, by the
  critical line algorithm. From the same prices, or the same $\mu$ and $\Sigma$, the two
  modules find the same maximum-Sharpe portfolio.
- [`hrp`](/modules/hrp/), [`hcaa`](/modules/hcaa/) — allocation without inverting $\Sigma$
  or estimating $\mu$.
- [`risk-metrics`](/modules/risk-metrics/) — tail risk of the resulting portfolio.
- [`codependence`](/modules/codependence/), [`onc`](/modules/onc/) — structure in the
  correlation matrix, for constraints by cluster.
