---
title: "cla"
description: "The Critical Line Algorithm: the exact long-only efficient frontier as a sequence of turning points, with no general-purpose optimiser."
status: authored
last_authored: '2026-09-21'
audience:
  - quant-dev
  - platform-engineering
module: "cla"
api_surface: "both"
afml_chapter:
  - "16"
citation:
  - "Markowitz, H. (1956). The optimization of a quadratic function subject to linear constraints. Naval Research Logistics Quarterly 3(1–2), 111–133."
  - "Bailey, D. H. and López de Prado, M. (2013). An open-source implementation of the critical-line algorithm for portfolio optimization. Algorithms 6(1), 169–196."
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 16: §16.2 The Problem with Convex Portfolio Optimization; §16.3 Markowitz's Curse."
rust_api:
  - "CLA"
  - "WeightBounds"
  - "AssetPricesInput"
  - "AssetPrices"
  - "ReturnsEstimation"
  - "covariance"
  - "ClaError"
python_api:
  - "cla.allocate_cla"
sidebar:
  badge: Module
---

A quadratic programming solver answers one question at a time: the best portfolio for *this*
target return, or *this* risk aversion. Markowitz's Critical Line Algorithm (1956) answers
all of them at once. It computes the entire efficient frontier of a portfolio with bounded
weights, exactly, in a finite number of steps, and it was designed for this problem and no
other. Bailey and López de Prado (2013) published the open implementation this module
follows, and AFML's Chapter 16 uses CLA as the mean-variance benchmark against which
[`hrp`](/modules/hrp/) is judged.

## The idea

Fix which assets sit at a bound and which are *free*. Among the free assets the optimal
weights are a linear function of the risk-aversion parameter $\lambda$, so as $\lambda$ falls
from infinity to zero the optimal portfolio moves along a straight line in weight space, the
**critical line**, until some asset hits a bound or some bounded asset wants to leave one.
That portfolio is a **turning point**. The free set changes by one asset, and a new line
begins.

The frontier is therefore piecewise: between two consecutive turning points every efficient
portfolio is a convex combination of them. The algorithm starts at the maximum-return
portfolio, $\lambda=\infty$, and walks down to the minimum-variance portfolio, $\lambda=0$,
solving a small linear system at each step. A handful of turning points describes the whole
frontier exactly.

## What `allocate` returns

`CLA::allocate(prices, expected_returns, covariance, resample_by, solution)` takes either a
price history or $\mu$ and $\Sigma$ directly, and one of four `solution` names:

| `solution` | `weights` holds |
| --- | --- |
| `"cla_turning_points"` (default) | every turning point, from maximum return to minimum variance |
| `"min_volatility"` | the one turning point with the least variance |
| `"max_sharpe"` | the frontier portfolio with the highest ratio of return to volatility |
| `"efficient_frontier"` | about 100 portfolios spread along the segments, with `efficient_frontier_means` and `efficient_frontier_sigma` filled in |

`lambdas`, `gammas` and `free_weights` always describe the turning points, whichever solution
was asked for. Bounds are a `WeightBounds::Tuple(lo, hi)` for all assets or
`WeightBounds::Lists(lows, highs)` per asset; from Python, `weight_bounds_lower` and
`weight_bounds_upper`.

```python
from openquant import cla, portfolio

# The four assets of the portfolio_optimization page.
names = ["bonds", "equity", "small cap", "gold"]
mu = [0.03, 0.07, 0.09, 0.04]
vol = [0.05, 0.16, 0.22, 0.15]
rho = [[1.0, 0.1, 0.1, 0.1], [0.1, 1.0, 0.8, 0.0], [0.1, 0.8, 1.0, 0.0], [0.1, 0.0, 0.0, 1.0]]
cov = [[rho[i][j] * vol[i] * vol[j] for j in range(4)] for i in range(4)]

def stats(w):
    ret = sum(a * b for a, b in zip(w, mu))
    risk = sum(w[i] * cov[i][j] * w[j] for i in range(4) for j in range(4)) ** 0.5
    return ret, risk

points = cla.allocate_cla(expected_returns=mu, covariance_matrix=cov)
print("turning point       " + "  ".join(f"{n[:5]:>5s}" for n in names) + "   return    risk")
for k, w in enumerate(points["weights"]):
    ret, risk = stats(w)
    print(f"{k:13d}       " + "  ".join(f"{round(x, 2) + 0.0:5.2f}" for x in w) + f"   {ret:6.3f}  {risk:6.3f}")

by_cla = cla.allocate_cla(expected_returns=mu, covariance_matrix=cov, solution="max_sharpe")["weights"][0]
by_qp = portfolio.allocate_from_inputs(mu, cov, "max_sharpe")[0]
gap = max(abs(a - b) for a, b in zip(by_cla, by_qp))
print("max Sharpe: CLA and the QP solver agree to 1e-8:", gap < 1e-8)
```

```text
turning point       bonds  equit  small   gold   return    risk
            0        0.00   0.00   1.00   0.00    0.090   0.220
            1        0.00   0.00   1.00   0.00    0.090   0.220
            2        0.00   0.09   0.91   0.00    0.088   0.212
            3        0.00   0.31   0.44   0.25    0.071   0.144
            4        0.82   0.10   0.00   0.08    0.035   0.048
            5        0.87   0.06   0.00   0.07    0.033   0.048
max Sharpe: CLA and the QP solver agree to 1e-8: True
```

Read the table as a story of assets entering. The frontier starts fully in small cap, the
highest-return asset. Equity enters at point 2, gold at point 3, and bonds at point 4, by
which time small cap has been driven to zero and *leaves*. Point 5 is the minimum-variance
portfolio. Any efficient portfolio is a blend of two neighbouring rows: one returning 5% is
part of the way from row 3 to row 4.

The last line is an independent check. [`portfolio_optimization`](/modules/portfolio-optimization/)
reaches its answers by an iterative solver and this module by linear algebra along the
critical line, and on the same inputs they agree to better than one part in a hundred million.

<figure>
<img class="dark:sl-hidden" src="/figures/ch16-frontier-light.svg" alt="Expected return against volatility for four assets. A curve runs from the minimum-variance portfolio at about 5 percent volatility and 3.3 percent return up to small cap alone at 22 percent volatility and 9 percent return. Turning points are marked along it. Equity and gold lie well to the right of the curve, bonds just below its lower end, and the maximum-Sharpe portfolio sits close to the minimum-variance end." />
<img class="light:sl-hidden" src="/figures/ch16-frontier-dark.svg" alt="Expected return against volatility for four assets. A curve runs from the minimum-variance portfolio at about 5 percent volatility and 3.3 percent return up to small cap alone at 22 percent volatility and 9 percent return. Turning points are marked along it. Equity and gold lie well to the right of the curve, bonds just below its lower end, and the maximum-Sharpe portfolio sits close to the minimum-variance end." />
<figcaption>The frontier of the example, from the <code>"efficient_frontier"</code> solution, with its turning points. Single assets other than the highest-returning one lie inside it.</figcaption>
</figure>

## From Rust

```rust
use nalgebra::DMatrix;
use openquant::cla::{ClaError, WeightBounds, CLA};

let mu = DMatrix::from_column_slice(4, 1, &[0.03, 0.07, 0.09, 0.04]);
let vol = [0.05, 0.16, 0.22, 0.15];
let rho = [[1.0, 0.1, 0.1, 0.1], [0.1, 1.0, 0.8, 0.0], [0.1, 0.8, 1.0, 0.0], [0.1, 0.0, 0.0, 1.0]];
let cov = DMatrix::from_fn(4, 4, |i, j| rho[i][j] * vol[i] * vol[j]);

let mut cla = CLA::new(WeightBounds::Tuple(0.0, 1.0), "mean");
cla.allocate(None, Some(&mu), Some(&cov), None, None)?;

// The walk starts at the highest-return asset and ends at minimum variance, where lambda is 0.
assert_eq!(cla.weights.first().unwrap(), &vec![0.0, 0.0, 1.0, 0.0]);
assert_eq!(*cla.lambdas.last().unwrap(), 0.0);
// Every turning point is a feasible portfolio.
for w in &cla.weights {
    assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert!(w.iter().all(|x| (-1e-9..=1.0 + 1e-9).contains(x)));
}

// With a 40% cap the frontier starts from the best feasible corner instead.
let mut capped = CLA::new(WeightBounds::Tuple(0.0, 0.4), "mean");
capped.allocate(None, Some(&mu), Some(&cov), None, Some("min_volatility"))?;
assert!(capped.weights[0].iter().all(|x| *x <= 0.4 + 1e-9));

assert!(matches!(
    cla.allocate(None, Some(&mu), Some(&cov), None, Some("max_return")),
    Err(ClaError::UnknownSolution(_))
));
```

## What to watch for

- **From prices, CLA uses simple returns and
  [`portfolio_optimization`](/modules/portfolio-optimization/) uses log returns**, so the same
  price history gives different maximum-Sharpe portfolios in the two modules. Given the same
  $\mu$ and $\Sigma$ they agree, as the example shows
  ([#110](https://github.com/Open-Quant/openquant/issues/110)).
- **`"max_sharpe"` has no risk-free rate.** It maximises $\mu^\top w/\sigma$, the tangency
  portfolio for a rate of zero, which for low-yielding safe assets lands near the
  minimum-variance end, as in the figure. For a non-zero rate, subtract it from the expected
  returns before calling, or use `portfolio_optimization`.
- **The first turning point appears twice.** The walk records its starting portfolio and then
  the first turning point proper, which is the same portfolio with a finite $\lambda$; the
  first `lambda` is infinite. Skip index 0 when plotting against $\lambda$.
- **Bounds that cannot sum to one are reported as `DimensionMismatch`**, which is misleading.
  If that error appears with correctly sized inputs, check that the lower bounds sum to at
  most 1 and the upper bounds to at least 1.
- **It is exact, and it inherits every weakness of its inputs.** CLA removes solver error,
  not estimation error. The frontier's upper end is always the single asset with the highest
  expected return, whatever its risk, and small changes in $\mu$ move the turning points a
  long way; see [Markowitz's curse](/modules/portfolio-optimization/#four-solutions).
- **The covariance matrix must be positive definite among the free assets.** With more
  assets than observations it is singular and the linear systems along the line fail. Shrink
  it first, or use [`hrp`](/modules/hrp/), which does not invert anything.
- **`resample_by` is positional** — every 5th or 21st row — and produced scrambled matrices
  before [#93](https://github.com/Open-Quant/openquant/issues/93) was fixed.
- **Methods beginning with an underscore are public for testing parity with the reference
  implementation**, not for use. `_purge_num_err` and `_purge_excess` are already called by
  `allocate`.

## Related modules

- [`portfolio-optimization`](/modules/portfolio-optimization/) — single portfolios by
  quadratic programming, with a risk-free rate and per-asset bounds by index.
- [`hrp`](/modules/hrp/), [`hcaa`](/modules/hcaa/) — what AFML proposes instead.
- [`risk-metrics`](/modules/risk-metrics/) — variance and tail risk of a chosen portfolio.
