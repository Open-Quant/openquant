---
title: "onc"
description: "Optimal Number of Clusters: partition a correlation matrix with k-means, choosing the number of clusters by silhouette quality."
status: authored
last_authored: '2026-09-24'
audience:
  - quant-dev
  - platform-engineering
module: "onc"
api_surface: "both"
citation:
  - "López de Prado, M. (2020). Machine Learning for Asset Managers. Cambridge University Press. Chapter 4, Optimal Clustering: §4.4 Optimal Number of Clusters; §4.4.1 Observations Matrix; §4.4.2 Base Clustering (Snippet 4.1); §4.4.3 Higher-Level Clustering (Snippet 4.2); §4.5 Experimental Results."
  - "López de Prado, M. and Lewis, M. J. (2019). Detection of false investment strategies using unsupervised learning methods. Quantitative Finance 19(9), 1555–1565."
  - "Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics 20, 53–65."
rust_api:
  - "get_onc_clusters"
  - "check_improve_clusters"
  - "OncResult"
  - "OncError"
python_api:
  - "onc.get_onc_clusters"
sidebar:
  badge: Module
---

Most clustering algorithms need to be told how many clusters to find, and the answer is
usually the thing you wanted to learn. How many distinct bets are in this portfolio? How many
genuinely different strategies are in these two hundred backtests? The Optimal Number of
Clusters algorithm (López de Prado and Lewis, 2019; *Machine Learning for Asset Managers*,
Chapter 4) answers by trying every count and keeping the partition whose clusters are most
clearly separated. It is not from AFML. Its two uses in this library's workflow are grouping
substitutable features before measuring
[importance](/modules/feature-importance/) and counting the *effective* number of trials for
a [deflated Sharpe ratio](/modules/backtest-statistics/#deflating-for-the-trials-you-ran).

## The algorithm

**Observations.** Convert the correlation matrix to the distance
$d_{ij}=\sqrt{\tfrac12(1-\rho_{ij})}$. Each item is then represented by *its row of that
matrix*, its distances to every other item, so two items are close when they relate to
everything else in the same way. That is more robust than comparing the pair alone.

**Quality.** For an item $i$, let $a_i$ be its mean distance to the other members of its
cluster and $b_i$ its mean distance to the members of the nearest other cluster. Its
silhouette (Rousseeuw, 1987) is

$$
S_i \;=\; \frac{b_i-a_i}{\max(a_i,\,b_i)}
$$

which is near 1 for an item deep inside a well-separated cluster and negative for one that
sits closer to a neighbouring cluster. The quality of a whole partition is the $t$-statistic
of the silhouettes, $q=\mathrm{E}[S_i]/\sqrt{\mathrm{V}[S_i]}$: high when silhouettes are
large *and* uniformly so.

**Base clustering.** Run k-means for every $k$ from 2 to $N-1$, `repeat` times each with
different initialisations, and keep the partition with the highest $q$.

**Higher-level clustering.** Compute $q$ per cluster. Clusters below the average are pooled
and the whole procedure is run again on just their members, on the view that a poor cluster
may be several real ones merged. The re-clustered partition is kept if it scores better.

`get_onc_clusters(corr, repeat)` returns an `OncResult`: `clusters`, a map from cluster label
to member indices; `silhouette_scores`, one per item in the original order; and
`ordered_correlation`, the matrix permuted so that clusters are contiguous.

## Recovering planted clusters

```python
import random

from openquant import onc

# Twelve series in three planted groups of sizes 5, 4 and 3, shuffled so that the correlation
# matrix shows no structure as given.
rng = random.Random(7)
group_of = [0] * 5 + [1] * 4 + [2] * 3
rng.shuffle(group_of)
series = []
factors = [[rng.gauss(0, 1) for _ in range(600)] for _ in range(3)]
for g in group_of:
    series.append([0.8 * f + 0.6 * rng.gauss(0, 1) for f in factors[g]])

def corr(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5

matrix = [[corr(a, b) for b in series] for a in series]
result = onc.get_onc_clusters(matrix, 5)

print("planted groups:", group_of)
for label, members in sorted(result["clusters"].items()):
    print(f"cluster {label}: members {members}  planted group {sorted({group_of[i] for i in members})}")
silhouettes = result["silhouette_scores"]
print(f"mean silhouette {sum(silhouettes) / len(silhouettes):.2f}")
```

```text
planted groups: [1, 2, 0, 2, 1, 0, 2, 0, 0, 1, 0, 1]
cluster 0: members [1, 3, 6]  planted group [2]
cluster 1: members [2, 5, 7, 8, 10]  planted group [0]
cluster 2: members [0, 4, 9, 11]  planted group [1]
mean silhouette 0.48
```

Nothing told the algorithm to look for three clusters, or that their sizes differ. It
returned three, and each holds exactly the members of one planted group.

<figure>
<img class="dark:sl-hidden" src="/figures/mlam4-onc-light.svg" alt="Two heat maps of the same twelve by twelve correlation matrix. As given, high correlations are scattered across the matrix with no visible pattern. Reordered by ONC cluster, they form three solid blocks along the diagonal, of sizes three, five and four, with near-zero correlation everywhere else." />
<img class="light:sl-hidden" src="/figures/mlam4-onc-dark.svg" alt="Two heat maps of the same twelve by twelve correlation matrix. As given, high correlations are scattered across the matrix with no visible pattern. Reordered by ONC cluster, they form three solid blocks along the diagonal, of sizes three, five and four, with near-zero correlation everywhere else." />
<figcaption>The example's correlation matrix, before and after. The right-hand panel is <code>ordered_correlation</code>.</figcaption>
</figure>

## From Rust

```rust
use nalgebra::DMatrix;
use openquant::onc::{get_onc_clusters, OncError};

// Two blocks of three: 0.8 within a block, 0.1 across.
let block = |i: usize| i / 3;
let corr = DMatrix::from_fn(6, 6, |i, j| {
    if i == j { 1.0 } else if block(i) == block(j) { 0.8 } else { 0.1 }
});

let result = get_onc_clusters(&corr, 3)?;
assert_eq!(result.clusters.len(), 2);
let mut found: Vec<Vec<usize>> = result.clusters.values().cloned().collect();
found.sort();
assert_eq!(found, vec![vec![0, 1, 2], vec![3, 4, 5]]);

// One silhouette per item, all clearly positive for a clean partition.
assert_eq!(result.silhouette_scores.len(), 6);
assert!(result.silhouette_scores.iter().all(|s| *s > 0.5));

assert_eq!(get_onc_clusters(&corr, 0).unwrap_err(), OncError::InvalidRepeat);
```

## What to watch for

- **The higher-level step runs only on messy inputs.** Re-clustering happens when more than
  two clusters score below the average $t$-statistic, which clean inputs like the example never
  reach. The re-clustered partition is kept only if its mean cluster $t$-statistic beats that of
  the clusters it replaced, as in Snippet 4.2. Until
  [#107](https://github.com/Open-Quant/openquant/issues/107) was fixed the comparison was
  inverted, so on matrices that reached this step ONC returned the worse of its two partitions.
- **Results are reproducible, and not tunable.** k-means is seeded from a fixed value, the
  repetition number and $k$, so the same matrix always gives the same answer. `repeat` adds
  initialisations; there is no seed parameter to vary.
- **Cost grows as the cube of the number of items or worse.** Every $k$ up to $N-1$ is tried,
  `repeat` times, and each silhouette pass is quadratic. A few hundred items is comfortable;
  thousands is not, and the recursion multiplies it.
- **It always returns at least two clusters.** There is no "one cluster" outcome, so a matrix
  with no structure still comes back partitioned. A low mean silhouette, or a silhouette
  $t$-statistic near zero, is the sign that the clusters are not real.
- **Negative correlation is distance, not similarity.** With $d=\sqrt{\tfrac12(1-\rho)}$ a
  pair at $\rho=-1$ is as far apart as possible. If a strategy and its mirror image should
  count as the same bet, take absolute correlations first.
- **The input must be a correlation matrix**: square, with at least two rows. Values are
  clamped to $[-1,1]$ but symmetry and a unit diagonal are not checked. From Python it is a
  list of lists, and `clusters` comes back as a dict keyed by label.

## Related modules

- [`codependence`](/modules/codependence/) — the distance used here, and alternatives that
  see non-linear dependence.
- [`hrp`](/modules/hrp/), [`hcaa`](/modules/hcaa/) — allocation over a hierarchical tree
  rather than a flat partition.
- [`feature-importance`](/modules/feature-importance/) — cluster features, then measure
  importance per cluster.
- [`backtest-statistics`](/modules/backtest-statistics/) — the number of clusters of trial
  returns is the $N$ to deflate by.
