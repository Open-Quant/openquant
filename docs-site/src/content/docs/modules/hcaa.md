---
title: "hcaa"
description: "Hierarchical allocation with a choice of risk measure: variance, standard deviation, expected shortfall, conditional drawdown, Sharpe ratio or equal splits."
status: authored
last_authored: '2026-09-21'
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

:::caution[Status: this is HRP with a choice of metric, not Raffinot's HCAA]
The published method cuts the tree into an optimal number of clusters (by the gap statistic),
allocates between those clusters, and then within each. This implementation does neither.
`optimal_num_clusters` is accepted and ignored, and allocation bisects the ordered leaf list
at its midpoint exactly as HRP does. With `"minimum_variance"` its weights equal
[`hrp`](/modules/hrp/)'s to the last bit. What the module adds to HRP is the other five
metrics. Tracked in [#108](https://github.com/Open-Quant/openquant/issues/108).
:::

## The metrics

The tree and leaf order are HRP's: single linkage on correlation distance, then
quasi-diagonalisation. At each bisection the left half is scaled by $\alpha$ and the right by
$1-\alpha$. Within a half, assets are combined with inverse-variance weights to form one
return series or one variance, and:

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

three, _ = hcaa.allocate_hcaa(names, asset_returns=returns, allocation_metric="minimum_variance",
                              optimal_num_clusters=3)
eight, _ = hcaa.allocate_hcaa(names, asset_returns=returns, allocation_metric="minimum_variance",
                              optimal_num_clusters=8)
print("optimal_num_clusters changes the weights:", three != eight)
```

```text
metric                         bond   equity   commodity
equal_weighting               0.375    0.250       0.375
minimum_variance              0.853    0.089       0.058
minimum_standard_deviation    0.659    0.167       0.174
expected_shortfall            0.690    0.164       0.146
conditional_drawdown_risk     0.669    0.213       0.118
optimal_num_clusters changes the weights: False
```

Read down the commodity column. Judged by standard deviation, commodities and equities are
about equally risky and get 17% each. Expected shortfall notices the gaps and trims
commodities to 15%; conditional drawdown, in which losses on consecutive days accumulate, to
12%, moving the difference to equities. That is the case for the module: the same tree,
with the risk measure that matches what you are afraid of.

Minimum variance is far more concentrated than the rest, 85% in bonds. Variance is quadratic
in volatility, so a fourfold difference in volatility becomes a sixteenfold difference in the
split. Standard deviation is usually the more sensible default.

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
// cluster count (ignored), resampling.
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

- **Everything in the status note.** Do not describe results from this module as HCAA.
- **`"equal_weighting"` is not equal weight.** It splits one half each way at every
  bisection, so an asset's weight depends on how deep it sits in the halving: with nine
  assets some get an eighth and some a sixteenth, and three equal-sized groups come out at
  0.375, 0.25 and 0.375. In Raffinot's method, equal weighting is across the clusters found
  by the cut.
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
- **A bisection that cuts across a cluster** is inherited from [HRP](/modules/hrp/#what-the-hierarchy-buys):
  the list is halved by position, not at the tree's own branches.

## Related modules

- [`hrp`](/modules/hrp/) — the method this extends, and what `"minimum_variance"` reproduces.
- [`onc`](/modules/onc/) — a way to choose the number of clusters, which this module would
  need in order to implement the published method.
- [`risk-metrics`](/modules/risk-metrics/) — the same risk measures for a single return
  series.
- [`codependence`](/modules/codependence/) — alternatives to correlation distance.
