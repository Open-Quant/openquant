---
title: "hcaa"
description: "Hierarchical allocation down the cluster tree, cut into a chosen number of clusters, with a choice of risk measure: variance, standard deviation, expected shortfall, conditional drawdown, Sharpe ratio or equal splits."
status: authored
last_authored: '2026-09-26'
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
  - "Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. Journal of the American Statistical Association 58(301), 236–244."
  - "Tibshirani, R., Walther, G. and Hastie, T. (2001). Estimating the number of clusters in a data set via the gap statistic. Journal of the Royal Statistical Society B 63(2), 411–423."
rust_api:
  - "HierarchicalClusteringAssetAllocation"
  - "HcaaError"
  - "HcaaDistance"
  - "HcaaLinkage"
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

The tree is built with Ward linkage (see [which linkage](#which-linkage)) on the correlation
distance $d_{ij}=\sqrt{2(1-\rho_{ij})}$, by default on $d$ itself as pairwise distances (see
[which distance is clustered](#which-distance-is-clustered)). It is not HRP's tree, which is
single linkage on the distance of distances.
Weight starts at the root and is handed down it. At each of the top $k-1$ merges, where $k$ is `optimal_num_clusters`, the
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
side and needs expected returns: pass `expected_asset_returns`, or a return history
(`asset_returns` or prices), from which they are estimated as a mean or
(`calculate_expected_returns = "exponential"`) an exponentially weighted mean of returns,
annualised by 252. Only a covariance matrix is not enough and raises
`MissingExpectedReturnsForSharpe`. The share of two Sharpe ratios means something only when
neither is negative: with both negative it still lands between 0 and 1 but favours the
*worse* side (−1 against −3 would give the better side a quarter). So whenever either
Sharpe ratio is negative, or both are 0, that split falls back to minimum variance.

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
splits continue inside each group, which depends on the order in which the tree happened to
merge three nearly identical assets, and the weights inside a group come out unequal.

## Which distance is clustered

`distance=` (Python) or `HcaaDistance` (Rust, set with `with_distance` or the `distance` field)
chooses the matrix the tree is built on. The options are the same as
[HRP's](/modules/hrp/#which-distance-is-clustered); the default is not.

| `distance=` (Python) | `HcaaDistance::` (Rust) | Tree built on |
| --- | --- | --- |
| `"correlation"` (default) | `Correlation` (default) | $d_{ij}$, pairwise, as mlfinlab's HCAA |
| `"distance_of_distances"` | `DistanceOfDistances` | $\tilde d_{ij}=\sqrt{\sum_n (d_{ni}-d_{nj})^2}$, as AFML Snippet 16.4 and HRP's default |

**Why pairwise is the default.** mlfinlab's HCAA, the implementation this module was ported
from, calls `linkage(squareform(d))`, which clusters on $d$ itself. Raffinot's method builds
on Mantegna's (1999) correlation tree, which is built on $d$ directly. Implementations disagree, though: the R package
HierPortfolios clusters its HCAA on $\tilde d$, and the paper's full text was not available to
check which it prescribes. Under single linkage the book's Monte Carlo agrees with the choice:
the pairwise tree gives HCAA lower out-of-sample variance, unlike HRP. Under Ward, the default
linkage, the two matrices give nearly the same results (see [which linkage](#which-linkage)).

AFML §16.5 Monte Carlo (Snippets 16.4 and 16.5: 10 assets, 260-day window, monthly rebalance,
10,000 runs, seeds `[51, k]`, the generator and backtest of the
[HRP runbook](/runbooks/hrp-vs-ivp-cla-oos/)), SYNTHETIC data, HCAA with `"minimum_variance"`
and **single** linkage (the default before
[#197](https://github.com/Open-Quant/openquant/issues/197)):

| | $d$ (default) | $\tilde d$ | $d$, 5 clusters | $\tilde d$, 5 clusters |
| --- | ---: | ---: | ---: | ---: |
| HCAA mean OOS variance ×1e4 | **3.049** | 3.154 | 3.275 | 3.364 |
| IVP / HCAA, mean OOS variance | **1.625** | 1.571 | 1.513 | 1.473 |
| CLA / HCAA, mean OOS variance | **1.672** | 1.616 | 1.557 | 1.516 |
| HRP ($\tilde d$, its default) / HCAA | **1.178** | 1.139 | 1.097 | 1.068 |
| HRP ($d$) / HCAA | **1.250** | 1.208 | 1.164 | 1.133 |
| runs where HCAA beats IVP | 87.8% | **90.4%** | 87.0% | 90.1% |
| runs where HCAA beats CLA | **73.9%** | 72.2% | 70.9% | 69.7% |
| HCAA turnover per rebalance | 0.107 | 0.087 | 0.110 | **0.077** |
| HCAA effective number of assets | 6.78 | 7.06 | 6.90 | 7.15 |

For reference, HRP's mean OOS variance ×1e4 is 3.592 ($\tilde d$) and 3.812 ($d$), CLA's 5.098
and IVP's 4.955; turnover is 0.122 and 0.216 for HRP, 0.187 for CLA and 0.074 for IVP. The
HRP figures reproduce that page's table exactly. Head to head, $\tilde d$ has 3.4% more mean OOS
variance than $d$ with no cut (paired mean log ratio +0.068, t = 34; $d$ lower in 66% of runs)
and 2.7% more with 5 clusters (t = 30). With `"minimum_standard_deviation"` the gap is much
larger: 3.757 for $d$ against 5.025 for $\tilde d$, which is no better than IVP. What
$\tilde d$ buys is stability: about 20% less turnover with `"minimum_variance"` and 40% less
with `"minimum_standard_deviation"` (0.126 against 0.208). The 10 bps cost model changes no
variance at 4 significant digits.

In this simulation every HCAA variant with `"minimum_variance"` has lower OOS variance than
HRP. That is one synthetic design with five independent assets and five noisy copies, where the
tree's own branches are the right split and HRP's midpoint split of the leaf list is not.

## Which linkage

The tree repeatedly merges the two closest clusters. `linkage=` (Python) or `HcaaLinkage`
(Rust, set with `with_linkage` or the `linkage` field) says how the distance between two
clusters is measured from the distances between their assets. The updates are scipy's
`linkage(method=...)`, and every combination with `distance=` is pinned against scipy.

| `linkage=` (Python) | `HcaaLinkage::` (Rust) | Distance between clusters $A$ and $B$ |
| --- | --- | --- |
| `"ward"` (default) | `Ward` (default) | the increase in within-cluster sum of squares from merging them, $\sqrt{2\lvert A\rvert\lvert B\rvert/(\lvert A\rvert+\lvert B\rvert)}\,\lVert c_A-c_B\rVert$ for centroids $c$ |
| `"average"` | `Average` | the mean of $d_{ab}$ over $a\in A$, $b\in B$ |
| `"complete"` | `Complete` | the largest $d_{ab}$ |
| `"single"` | `Single` | the smallest $d_{ab}$ (HRP's linkage, and this module's before #197) |

**What Ward means here.** Ward's criterion is defined for points in Euclidean space, and
scipy's `method="ward"` (R's `ward.D2`) applies its update to whatever distances it is given.
Both matrices this module clusters are genuinely Euclidean. $d_{ij}=\sqrt{2(1-\rho_{ij})}$ is
the distance between the two assets' return series after each is demeaned and scaled to unit
length (factor the correlation matrix as $XX^\top$ and the rows of $X$ are such points), and
$\tilde d$ is the Euclidean distance between the columns of $d$. So Ward here is Ward's
minimum-variance method on actual points, not an approximation; the fixture generator checks
this by running scipy's Ward on the points $X$ and getting the same tree. This is what
mlfinlab's HCAA does (`linkage(squareform(d), method="ward")`) for $d$, and what R
HierPortfolios' HCAA does (`hclust(dist(D), "ward.D2")`) for $\tilde d$.

**Why Ward is the default.** Ward is the linkage of the published method and of every HCAA
reference implementation we could read: mlfinlab's `hcaa.py` (`linkage="ward"`, passed to
scipy), R HierPortfolios' `HCAA_Portfolio` (`linkage = "ward"`, run as `ward.D2`) and
jduarte00's implementation. It builds compact clusters of similar size. Single linkage
"chains": assets join a growing cluster one at a time, so the tree is deep and lopsided and
the top splits isolate one asset against the rest, which under `"equal_weighting"` or
`"minimum_standard_deviation"` can hand that one asset a large share of the capital. Before
[#197](https://github.com/Open-Quant/openquant/issues/197) this module used single linkage.
**Pass `linkage="single"` to reproduce results from before #197.**

**The trade-off depends on the universe.**

- **Block-clustered universes** (assets in a few groups of related exposures): Ward's mean
  out-of-sample variance is 6–45% lower than single linkage's, with lower drawdown and a 2–4×
  higher effective number of assets.
- **AFML's Snippet 16.4 design** (independent assets plus noisy near-duplicates): single
  linkage has about 3% (`"minimum_variance"`) to 28% (`"minimum_standard_deviation"`) lower
  out-of-sample variance. Ward isolates the largest group of near-duplicates at the root and,
  under `"equal_weighting"` or `"minimum_standard_deviation"`, gives that group about half the
  capital.
- **Complete linkage** is close to Ward throughout.

If your universe contains many near-copies of one exposure (share classes, several trackers of
one index), deduplicate it, use `"minimum_variance"`, or pass `linkage="complete"` or
`linkage="single"`.

**The AFML design in detail.** The table below is the book's Monte Carlo from the section above
(SYNTHETIC, 10,000 runs, seeds `[51, k]`). It is the design most favourable to single linkage:
its true tree is a few tight pairs and singletons, which single linkage recovers well and Ward's
preference for balanced clusters does not. Mean OOS variance ×1e4, with turnover per rebalance
in brackets:

| `allocation_metric`, cut | distance | single | complete | average | **ward** (default) |
| --- | --- | ---: | ---: | ---: | ---: |
| minimum variance, no cut | $d$ (default) | **3.049** (0.107) | 3.060 (0.103) | 3.057 (0.104) | 3.141 (0.086) |
| minimum variance, no cut | $\tilde d$ | 3.154 (0.087) | **3.140** (0.086) | 3.144 (0.085) | 3.145 (**0.085**) |
| minimum variance, 5 clusters | $d$ | **3.275** (0.110) | 3.284 (0.106) | 3.285 (0.107) | 3.356 (0.076) |
| minimum variance, 5 clusters | $\tilde d$ | 3.364 (0.077) | **3.345** (0.076) | 3.356 (0.075) | 3.357 (**0.074**) |
| minimum standard deviation, no cut | $d$ | 3.757 (0.208) | **3.648** (0.174) | 3.694 (0.184) | 4.792 (0.123) |
| minimum standard deviation, no cut | $\tilde d$ | 5.025 (0.126) | 4.910 (0.126) | 5.010 (0.120) | 5.046 (**0.117**) |

For reference: HRP 3.592 ($\tilde d$, its default) and 3.812 ($d$), CLA 5.098, IVP 4.955.
The single-linkage column reproduces the table above exactly.

- **With `"minimum_variance"`, Ward costs about 3% of variance and saves about 20% of turnover.**
  On $d$ its mean OOS variance is 3.0% above single linkage's (paired mean log ratio +0.068,
  t = 32; Ward lower in 34% of runs), and its turnover is 0.086 against 0.107. Against IVP
  (1.578×, Ward lower in 91% of runs) and CLA (1.623×) it is still far ahead, and it stays below
  HRP (HRP $\tilde d$ / HCAA = 1.144).
- **With `"minimum_standard_deviation"` the cost is large here.** Ward on $d$ gives 4.792, 28%
  above single linkage's 3.757 and only slightly better than IVP (4.955), because about half
  the capital goes to the near-duplicates Ward isolates at the root. Its turnover is 41% lower
  (0.123 against 0.208).
- **Under Ward the choice of distance hardly matters** (3.141 against 3.145, 4.792 against
  5.046): Ward's balanced tree on $d$ looks like the tree on $\tilde d$.
- **Effective number of assets** rises from 6.78 (single, $d$) to 7.07 (Ward); IVP's is 8.26.
- The 10 bps cost model changes no variance at 4 significant digits.

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
  averages the worst drawdowns. `risk_metrics`' function of the same name does the same from
  a wealth curve you pass it, but takes the upper-tail level (0.95) where this module takes
  the tail probability (0.05).
- **The Python default metric is `"equal_weighting"`**, which is the least useful of the six
  for the reason above. Pass `allocation_metric` explicitly.
- **Tail metrics on short histories are noisy.** A 5% tail of 750 days is 38 observations,
  and one of them can decide a split. The weights move more from sample to sample than the
  variance-based ones do.
- **`"sharpe_ratio"` brings back the problem HRP was built to avoid.** It depends on expected
  returns, the least reliable input in portfolio construction, and a mean of daily returns is
  a very noisy estimate of one.
- **The default tree is not HRP's.** HCAA uses Ward linkage on pairwise distances by default;
  HRP uses single linkage on distances between columns of the distance matrix. Pass
  `linkage="single", distance="distance_of_distances"` for HRP's tree.
- **The default linkage changed from single to Ward**
  ([#197](https://github.com/Open-Quant/openquant/issues/197)). Weights and leaf order change
  for most inputs; pass `linkage="single"` (`HcaaLinkage::Single`) to reproduce earlier
  results. Ward does better on block-clustered universes; on universes full of near-copies of
  one exposure single linkage can do better (see [which linkage](#which-linkage)).
- **Results differ from [HRP](/modules/hrp/) even with `"minimum_variance"`.** HRP halves the
  ordered leaf list at its midpoint, which can cut across a branch of the tree; this module
  splits only at the tree's own branches. They agree when every branch happens to divide
  the list in half and both use the same tree (the same `distance=`, and `linkage="single"`).

## Related modules

- [`hrp`](/modules/hrp/) — the method this extends.
- [`onc`](/modules/onc/) — a way to choose `optimal_num_clusters`.
- [`risk-metrics`](/modules/risk-metrics/) — the same risk measures for a single return
  series.
- [`codependence`](/modules/codependence/) — alternatives to correlation distance.
