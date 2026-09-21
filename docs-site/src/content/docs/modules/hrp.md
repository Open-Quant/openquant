---
title: "hrp"
description: "Hierarchical Risk Parity: portfolio weights from a clustering of the correlation matrix, with no matrix inversion."
status: authored
last_authored: '2026-09-21'
audience:
  - quant-dev
  - platform-engineering
module: "hrp"
api_surface: "both"
afml_chapter:
  - "16"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 16: §16.2 The Problem with Convex Portfolio Optimization; §16.3 Markowitz's Curse; §16.4 From Geometric to Hierarchical Relationships; §16.4.1 Tree Clustering (Snippet 16.1); §16.4.2 Quasi-Diagonalization (Snippet 16.2); §16.4.3 Recursive Bisection (Snippet 16.3); §16.6 Out-of-Sample Monte Carlo Simulations."
  - "López de Prado, M. (2016). Building diversified portfolios that outperform out of sample. Journal of Portfolio Management 42(4), 59–69."
rust_api:
  - "HierarchicalRiskParity"
  - "HrpDendrogram"
  - "HrpError"
python_api:
  - "hrp.allocate_hrp"
sidebar:
  badge: Module
---

Mean-variance optimisation needs the inverse of the covariance matrix, and the inverse is
least stable exactly when it matters most. The more correlated the assets, the closer the
matrix is to singular, the more its inverse amplifies estimation error, and the more
diversification is needed: AFML calls this Markowitz's curse (§16.3). The optimiser responds
to small changes in the inputs with large changes in the weights, and out of sample it often
loses to equal weighting.

Hierarchical Risk Parity (López de Prado, 2016) avoids the inversion altogether. It uses the
covariance matrix twice, once to decide which assets are alike and once to read variances off
it, and never solves a linear system. It needs no expected returns. The result is not optimal
in sample by any criterion, and in the chapter's Monte Carlo experiments (§16.6) it has lower
out-of-sample variance than the minimum-variance portfolio that is.

## Three steps

**Tree clustering** (§16.4.1). Turn correlations into distances,
$d_{ij}=\sqrt{\tfrac12(1-\rho_{ij})}$, and build a single-linkage tree: repeatedly merge the
two closest clusters, where the distance between clusters is that of their closest members.

**Quasi-diagonalisation** (§16.4.2). Read the leaves of the tree from left to right. Reordering
the covariance matrix that way puts similar assets next to each other, so the large
covariances lie along the diagonal. No values change; this is a permutation.
`ordered_indices` holds it.

**Recursive bisection** (§16.4.3). Start with every weight at 1. Split the ordered list in
half. Give each half the variance it would have under inverse-variance weights,
$\tilde V=\tilde w^\top \Sigma\,\tilde w$ with $\tilde w_i\propto 1/\Sigma_{ii}$, and scale the
two halves' weights by

$$
\alpha \;=\; 1-\frac{\tilde V_{\text{left}}}{\tilde V_{\text{left}}+\tilde V_{\text{right}}}
\qquad\text{and}\qquad 1-\alpha
$$

so the riskier half gets less. Recurse into each half until every piece is a single asset.
Weights are positive and sum to 1 by construction.

## What the hierarchy buys

The example gives nine assets the *same* volatility, so any difference between methods comes
from correlation alone. Six are equities sharing one factor, two are bonds sharing another,
and one is a commodity on its own.

```python
import random

from openquant import hrp

# Nine assets with the same volatility: six equities that share a factor, two bonds that share
# another, and one commodity on its own. Correlation within a group is about 0.8.
rng = random.Random(2)
sizes = {"equity": 6, "bond": 2, "commodity": 1}
names = [f"{g}_{i}" for g, k in sizes.items() for i in range(k)]
returns = []
for _ in range(750):
    row = []
    for k in sizes.values():
        factor = rng.gauss(0, 0.01)
        row += [0.9 * factor + 0.45 * rng.gauss(0, 0.01) for _ in range(k)]
    returns.append(row)

weights, order = hrp.allocate_hrp(names, asset_returns=returns)
print("leaf order:", " ".join(names[i] for i in order))

var = [sum(r[j] ** 2 for r in returns) / len(returns) for j in range(9)]
ivp = [(1 / v) / sum(1 / u for u in var) for v in var]
print("group        inverse-variance      HRP")
for g in sizes:
    pick = [i for i, n in enumerate(names) if n.startswith(g)]
    print(f"{g:10s}   {sum(ivp[i] for i in pick):16.3f}   {sum(weights[i] for i in pick):6.3f}")
```

```text
leaf order: commodity_0 bond_0 bond_1 equity_1 equity_5 equity_4 equity_0 equity_2 equity_3
group        inverse-variance      HRP
equity                  0.679    0.478
bond                    0.214    0.341
commodity               0.107    0.181
```

Inverse-variance weighting sees nine equally risky assets and gives each a ninth, which puts
68% of the portfolio on one factor because that factor happens to have six tickers. HRP sees
that the six are one bet and cuts it to 48%. A minimum-variance optimiser would go further,
and would do so by inverting a matrix in which six columns are nearly collinear.

The leaf order also shows the method's known weakness. Bisection cuts the *list* in half, not
the *tree*: nine leaves split four and five, which puts `equity_1` in the first half with the
bonds and the commodity, away from its own cluster. That is why equities end up at 48% rather
than the third that three equal clusters would suggest. [`hcaa`](/modules/hcaa/) and
López de Prado's later nested clustered optimisation exist to address this.

## From Rust

```rust
use nalgebra::DMatrix;
use openquant::hrp::{HierarchicalRiskParity, HrpError};

// Assets 0 and 1 are nearly the same bet; asset 2 is independent. All have variance 0.04.
let covariance = DMatrix::from_row_slice(3, 3, &[
    0.040, 0.036, 0.000,
    0.036, 0.040, 0.000,
    0.000, 0.000, 0.040,
]);
let names: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();

let mut model = HierarchicalRiskParity::new();
model.allocate(&names, None, None, Some(&covariance), None, false)?;

// The tree joins a and b first, so they are adjacent in the leaf order.
assert_eq!(model.clusters[0], [0, 1]);
let position = |asset: usize| model.ordered_indices.iter().position(|&i| i == asset).unwrap();
assert_eq!(position(0).abs_diff(position(1)), 1);

assert!((model.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
assert!(model.weights.iter().all(|w| *w > 0.0));
// The independent asset gets more than either member of the correlated pair.
assert!(model.weights[2] > model.weights[0] && model.weights[2] > model.weights[1]);

assert_eq!(model.allocate(&names, None, None, None, None, false), Err(HrpError::NoData));
```

`allocate` takes whichever of prices, returns or a covariance matrix you have. If returns are
given, prices are ignored; a supplied covariance matrix is always used as is. After it
returns, `weights`, `ordered_indices`, `clusters` (the merge list, scipy-style: ids below $n$
are assets, $n+k$ is the cluster made by merge $k$), and the seriated correlation and
distance matrices are public fields.

## What to watch for

- **The tree is built on correlation distance directly.** AFML's Snippet 16.1 hands scipy the
  square distance matrix, which scipy treats as coordinates, so the book clusters on the
  Euclidean distance *between columns* of the distance matrix. This implementation, like
  mlfinlab, passes pairwise distances. The two usually give the same tree and need not.
- **`use_shrinkage` is a fixed 10% shrink of the off-diagonal terms**, not a Ledoit–Wolf or
  OAS estimator. For a real shrinkage estimate, compute the covariance yourself and pass it
  in.
- **Returns are simple returns** and the covariance is the sample covariance, unannualised.
  HRP weights do not depend on the scale, so annualisation is immaterial here.
- **`resample_by` keeps every 5th (`"W"`) or 21st (`"M"`) row.** It is positional, not
  calendar-aware. Before [#93](https://github.com/Open-Quant/openquant/issues/93) was fixed
  this path scrambled the matrix; weights from older versions with resampling are wrong.
- **`plot_clusters` does not return a dendrogram.** `ivl` and `leaves` are the real leaf
  order; `icoord` and `dcoord` are placeholders with every link drawn at height 1. To draw
  the tree, use `clusters` with the seriated distances.
- **Single linkage chains.** One asset moderately close to two groups can glue them together
  early, and the rest of the tree inherits it. Look at `seriated_correlations` before
  trusting the weights; a blocky diagonal means the ordering worked.
- **Long-only, fully invested, no constraints.** There is no way to bound a weight or target
  a volatility. For that see
  [`portfolio-optimization`](/modules/portfolio-optimization/).

## Related modules

- [`hcaa`](/modules/hcaa/) — the same tree with other allocation metrics.
- [`onc`](/modules/onc/) — choose the number of clusters instead of bisecting blindly.
- [`codependence`](/modules/codependence/) — distances other than correlation.
- [`cla`](/modules/cla/), [`portfolio-optimization`](/modules/portfolio-optimization/) — the
  convex optimisers HRP is an answer to.
