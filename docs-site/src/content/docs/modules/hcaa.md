---
title: "hcaa"
description: "Hierarchical allocation down the cluster tree, cut into a chosen number of clusters, with a choice of risk measure: variance, standard deviation, expected shortfall, conditional drawdown, Sharpe ratio or equal splits."
status: authored
last_authored: '2026-09-25'
audience:
  - quant-dev
  - platform-engineering
module: "hcaa"
api_surface: "both"
afml_chapter:
  - "16"
citation:
  - "Raffinot, T. (2017). Hierarchical clustering-based asset allocation. Journal of Portfolio Management 44(2), 89–99."
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 16, §16.4 From Geometric to Hierarchical Relationships."
  - "Tibshirani, R., Walther, G. and Hastie, T. (2001). Estimating the number of clusters in a data set via the gap statistic. Journal of the Royal Statistical Society B 63(2), 411–423."
rust_api:
  - "HierarchicalClusteringAssetAllocation"
  - "HcaaError"
python_api:
  - "hcaa.allocate_hcaa"
sidebar:
  badge: Module
---

[HRP](/modules/hrp/) splits risk between the two halves of a clustered asset list in
proportion to their variance. Variance is one view of risk, and for assets with fat left
tails it is a generous one. Raffinot's Hierarchical Clustering-based Asset Allocation (2017)
generalises the idea: keep the tree, and let the quantity that is balanced between branches
be whatever risk measure suits the portfolio.

:::note[What is and is not implemented]
The published method cuts the tree into an optimal number of clusters, found by the gap
statistic, and allocates down the tree to those clusters and then within them. This module
allocates down the tree and cuts it at `optimal_num_clusters`, but does not estimate that
number: with no value given it does not cut, and every merge in the tree is split. Choose the
count yourself, for example with [`onc`](/modules/onc/). Before
[#108](https://github.com/Open-Quant/openquant/issues/108) was fixed the count was ignored and
the leaf list was halved at its midpoint, as HRP does.
:::

## The metrics

The tree is single linkage on the pairwise correlation distance, the tree of HRP's
`distance="correlation"` option. HRP's default instead clusters on the distance between rows
of the distance matrix, as AFML's Snippet 16.4 does (see
[which distance is clustered](/modules/hrp/#which-distance-is-clustered)). Weight starts at
the root and is
handed down it. At each of the top $k-1$ merges, where $k$ is `optimal_num_clusters`, the
node's weight is split between its two children, the left scaled by $\alpha$ and the right by
$1-\alpha$. Below that cut each subtree is one cluster, and its weight is shared equally among
its assets under `"equal_weighting"` and by inverse variance under every other metric. To score
a side, its assets are combined with inverse-variance weights into one return series or one
variance, and:

| `allocation_metric` | $\alpha$ |
| --- | --- |
| `"minimum_variance"` | one minus the left half's share of the two variances |
| `"minimum_standard_deviation"` | the same with standard deviations |
| `"expected_shortfall"` | the same with each half's expected shortfall at `confidence_level` |
| `"conditional_drawdown_risk"` | the same with each half's conditional drawdown at `confidence_level` |
| `"sharpe_ratio"` | the left half's share of the two Sharpe ratios |
| `"equal_weighting"` | one half, always |

The first four give less to the riskier side. `"sharpe_ratio"` gives *more* to the better
side and needs expected returns: pass `expected_asset_returns`, or prices, from which they
are estimated as a mean or (`calculate_expected_returns = "exponential"`) an exponentially
weighted mean of returns. When the two Sharpe ratios do not give a share between 0 and 1 —
one of them is negative — that split falls back to minimum variance.

The two tail metrics need the return history and raise `MissingReturnsForTailRisk` if only a
covariance matrix was supplied. `confidence_level` is the tail probability, 0.05 by default
from Python.

```python
import random

from openquant import hcaa

# Nine assets in three groups of three. Bonds are quiet, equities are not, and commodities
# are volatile and occasionally gap down by 8%.
rng = random.Random(2)
groups = {"bond": 0.004, "equity": 0.012, "commodity": 0.016}
names = [f"{g}_{i}" for g in groups for i in range(3)]
returns = []
for _ in range(750):
    row = []
    for g, vol in groups.items():
        factor = rng.gauss(0, vol)
        if g == "commodity" and rng.random() < 0.01:
            factor -= 0.08
        row += [factor + rng.gauss(0, vol * 0.5) for _ in range(3)]
    returns.append(row)

print("metric                         bond   equity   commodity")
for metric in ("equal_weighting", "minimum_variance", "minimum_standard_deviation",
               "expected_shortfall", "conditional_drawdown_risk"):
    weights, _ = hcaa.allocate_hcaa(names, asset_returns=returns, allocation_metric=metric)
    by_group = [sum(w for n, w in zip(names, weights) if n.startswith(g)) for g in groups]
    print(f"{metric:28s} {by_group[0]:6.3f}   {by_group[1]:6.3f}   {by_group[2]:9.3f}")

# Cut into three clusters, weights are equal inside each. With no cut, the tree splits
# inside the groups too.
for k in (3, None):
    weights, _ = hcaa.allocate_hcaa(names, asset_returns=returns, allocation_metric="equal_weighting",
                                    optimal_num_clusters=k)
    print(f"equal weighting, {k} clusters:", " ".join(f"{w:.3f}" for w in weights))
```

```text
metric                         bond   equity   commodity
equal_weighting               0.500    0.250       0.250
minimum_variance              0.867    0.092       0.041
minimum_standard_deviation    0.719    0.169       0.112
expected_shortfall            0.749    0.161       0.090
conditional_drawdown_risk     0.745    0.194       0.062
equal weighting, 3 clusters: 0.167 0.167 0.167 0.083 0.083 0.083 0.083 0.083 0.083
equal weighting, None clusters: 0.125 0.125 0.250 0.062 0.062 0.125 0.062 0.125 0.062
```

The tree's top split puts the bonds on one side and equities with commodities on the other,
and the next split separates those two. Read down the commodity column. Judged by standard
deviation, commodities get 11%. Expected shortfall notices the gaps and trims them to 9%;
conditional drawdown, in which losses on consecutive days accumulate, to 6%, moving the
difference to equities. That is the case for the module: the same tree, with the risk measure
that matches what you are afraid of.

Minimum variance is far more concentrated than the rest, 87% in bonds. Variance is quadratic
in volatility, so a fourfold difference in volatility becomes a sixteenfold difference in the
split. Standard deviation is usually the more sensible default.

The last two lines show the cut. With three clusters, each group gets the weight the two top
splits give it, bonds a half and the others a quarter, shared equally inside. With no cut the
splits continue inside each group, which depends on the order in which single linkage happened
to merge three nearly identical assets, and the weights inside a group come out unequal.

## From Rust

```rust
use nalgebra::DMatrix;
use openquant::hcaa::{HcaaError, HierarchicalClusteringAssetAllocation};

let covariance = DMatrix::from_row_slice(3, 3, &[
    0.010, 0.008, 0.000,
    0.008, 0.010, 0.000,
    0.000, 0.000, 0.040,
]);
let names: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();
let mut model = HierarchicalClusteringAssetAllocation::new("mean");

// Arguments: names, prices, returns, covariance, expected returns, metric, confidence level,
// cluster count (None: no cut), resampling.
model.allocate(&names, None, None, Some(&covariance), None, "minimum_variance", 0.05, None, None)?;
let by_variance = model.weights.clone();
model.allocate(&names, None, None, Some(&covariance), None, "minimum_standard_deviation", 0.05, None, None)?;

// The volatile asset gets more under standard deviation than under variance.
assert!(model.weights[2] > by_variance[2]);
assert!((model.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);

// Tail metrics need returns, not just a covariance matrix.
assert_eq!(
    model.allocate(&names, None, None, Some(&covariance), None, "expected_shortfall", 0.05, None, None),
    Err(HcaaError::MissingReturnsForTailRisk)
);
assert!(matches!(
    model.allocate(&names, None, None, Some(&covariance), None, "kelly", 0.05, None, None),
    Err(HcaaError::UnknownAllocationMetric(_))
));
```

## What to watch for

- **Choose `optimal_num_clusters`.** There is no gap statistic; the default is no cut, and
  then splits run all the way down to single assets, where they reflect noise in the merge
  order more than structure.
- **`"equal_weighting"` is equal per split, not per asset or per cluster.** Each of the top
  $k-1$ merges gives half to each child, so a cluster's weight halves with every level it sits
  below the root: in the example, a half for the bonds and a quarter each for the others.
- **Expected shortfall and conditional drawdown are computed here, not in
  [`risk-metrics`](/modules/risk-metrics/).** Both are historical estimates from the half's
  inverse-variance portfolio. The drawdown measure builds a wealth curve from the returns and
  averages the worst drawdowns, which is the correct construction; it is unrelated to the
  `risk_metrics` function of the same name, which has a known defect.
- **The Python default metric is `"equal_weighting"`**, which is the least useful of the six
  for the reason above. Pass `allocation_metric` explicitly.
- **Tail metrics on short histories are noisy.** A 5% tail of 750 days is 38 observations,
  and one of them can decide a split. The weights move more from sample to sample than the
  variance-based ones do.
- **`"sharpe_ratio"` brings back the problem HRP was built to avoid.** It depends on expected
  returns, the least reliable input in portfolio construction, and a mean of daily returns is
  a very noisy estimate of one.
- **Results differ from [HRP](/modules/hrp/) even with `"minimum_variance"`.** HRP halves the
  ordered leaf list at its midpoint, which can cut across a branch of the tree; this module
  splits only at the tree's own branches. They agree when every branch happens to divide
  the list in half and HRP is given the same tree (`distance="correlation"`).

## Related modules

- [`hrp`](/modules/hrp/) — the method this extends.
- [`onc`](/modules/onc/) — a way to choose `optimal_num_clusters`.
- [`risk-metrics`](/modules/risk-metrics/) — the same risk measures for a single return
  series.
- [`codependence`](/modules/codependence/) — alternatives to correlation distance.
