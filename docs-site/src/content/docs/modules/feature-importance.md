---
title: "feature_importance"
description: "MDI, MDA and SFI feature importance, and a PCA cross-check, for models validated on purged folds."
status: authored
last_authored: '2026-09-24'
audience:
  - quant-dev
  - platform-engineering
module: "feature_importance"
api_surface: "rust-only"
afml_chapter:
  - "8"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 8: §8.2 The Importance of Feature Importance; §8.3.1 Mean Decrease Impurity (Snippet 8.2); §8.3.2 Mean Decrease Accuracy (Snippet 8.3); §8.4.1 Single Feature Importance (Snippet 8.4); §8.4.2 Orthogonal Features (Snippets 8.5–8.6); §8.6 Experiments with Synthetic Data."
  - "Breiman, L. (2001). Random forests. Machine Learning 45(1), 5–32."
rust_api:
  - "mean_decrease_impurity"
  - "mean_decrease_accuracy"
  - "single_feature_importance"
  - "get_orthogonal_features"
  - "feature_pca_analysis"
  - "plot_feature_importance"
  - "ImportanceStats"
  - "PcaCorrelation"
  - "FeatureImportanceError"
sidebar:
  badge: Module
---

AFML's first law of backtesting (§8.2) is that a backtest is not a research tool; feature
importance is. A backtest tells you that some combination of features and rules made money
on one path. Importance tells you *which* features the model leans on, which can be checked
against what you believe about the market, and which survives the model being refitted.

Chapter 8 gives three methods, and the reason to have three is that each one is wrong in a
different way. This module implements them for any model behind the
[`SimpleClassifier`](/modules/cross-validation/#scoring) trait. The Python counterpart,
with its own model, is [`feature-diagnostics`](/modules/feature-diagnostics/).

| | In- or out-of-sample | What it can be fooled by |
| --- | --- | --- |
| **MDI** | in-sample | noise features always get some share; tree ensembles only |
| **MDA** | out-of-sample | correlated features hide each other (substitution) |
| **SFI** | out-of-sample | features that matter only jointly score nothing |

## The three methods

**Mean decrease impurity** (§8.3.1). In a tree ensemble, each split reduces impurity; MDI
credits that reduction to the splitting feature and averages over trees. This crate has no
tree learner, so `mean_decrease_impurity(per_tree_importances, names)` is the *aggregation
step* of Snippet 8.2: pass one row of importances per tree, from whatever forest you
trained, and it returns the mean and its standard error per feature, scaled so the means sum
to 1. Following the snippet, **a zero is treated as missing, not as zero** — see the pitfall
below.

**Mean decrease accuracy** (§8.3.2). Fit on each training fold, score the test fold, then
shuffle one feature's column in the test fold and score again. With $s_k$ the score of fold
$k$ and $s_{k,j}$ the score with feature $j$ damaged,

$$
\mathrm{MDA}_j \;=\; \frac1K\sum_{k=1}^{K}\frac{s_k-s_{k,j}}{s^{\max}-s_{k,j}}
$$

where $s^{\max}$ is the best attainable score: 0 for negative log loss, 1 for accuracy and F1.
A value of 1 means damaging the feature destroyed everything the model had; 0 means the model
did not need it; negative means the model did better without it. Scores here *are* weighted
by `sample_weight` on the test fold, as Snippet 8.3 does. The last argument is the seed for
the shuffles: the same seed gives the same result, and a different seed a slightly different
one.

**Single feature importance** (§8.4.1). Cross-validate the model on each feature alone.
There is nothing to substitute for, so correlated features cannot hide each other. The value
returned is the raw cross-validated score, not a ratio: for negative log loss compare it with
$-\ln 2\approx-0.693$, the score of a coin flip.

```rust
use chrono::{Duration, NaiveDate};
use openquant::cross_validation::{PurgedKFold, Scoring, SimpleClassifier};
use openquant::feature_importance::{mean_decrease_accuracy, single_feature_importance};

/// Scores by the difference of class means, squashed by a sigmoid of sharpness `k`.
struct MeanDiff { k: f64, w: Vec<f64>, b: f64 }

impl SimpleClassifier for MeanDiff {
    fn fit(&mut self, x: &[Vec<f64>], y: &[f64], _sample_weight: Option<&[f64]>) {
        let m = x[0].len();
        let (mut pos, mut neg, mut n_pos, mut n_neg) = (vec![0.0; m], vec![0.0; m], 0.0, 0.0);
        for (row, label) in x.iter().zip(y) {
            let (sum, count) =
                if *label > 0.5 { (&mut pos, &mut n_pos) } else { (&mut neg, &mut n_neg) };
            *count += 1.0;
            row.iter().enumerate().for_each(|(j, v)| sum[j] += v);
        }
        self.w = (0..m).map(|j| pos[j] / n_pos - neg[j] / n_neg).collect();
        self.b = -(0..m).map(|j| self.w[j] * (pos[j] / n_pos + neg[j] / n_neg) / 2.0).sum::<f64>();
    }
    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let z = |r: &Vec<f64>| r.iter().zip(&self.w).map(|(a, b)| a * b).sum::<f64>() + self.b;
        x.iter().map(|r| 1.0 / (1.0 + (-self.k * z(r)).exp())).collect()
    }
}

// Reproducible noise in [-1, 1) from a hash, so the example needs no RNG.
fn noise(i: usize, salt: u64) -> f64 {
    let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt.wrapping_mul(0xD1B5_4A32_D192_ED03);
    h ^= h >> 31;
    h = h.wrapping_mul(0x7FB5_D329_728E_A185);
    h ^= h >> 27;
    (h >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

// 600 samples: one strong feature, one weak, one irrelevant.
let n = 600;
let x: Vec<Vec<f64>> = (0..n).map(|i| vec![noise(i, 1), noise(i, 2), noise(i, 3)]).collect();
let y: Vec<f64> = (0..n)
    .map(|i| f64::from(u8::from(x[i][0] + 0.5 * x[i][1] + 0.4 * noise(i, 9) > 0.0)))
    .collect();
let names: Vec<String> = ["strong", "weak", "noise"].map(String::from).to_vec();

let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 0, 0).unwrap();
let info: Vec<_> =
    (0..n as i64).map(|i| (open + Duration::hours(i), open + Duration::hours(i + 3))).collect();
let splits = PurgedKFold::new(5, info, 0.0)?.split(n)?;

let mut model = MeanDiff { k: 4.0, w: vec![], b: 0.0 };
let mda = mean_decrease_accuracy(&mut model, &x, &y, &names, &splits, None, Scoring::NegLogLoss, 42)?;
let sfi = single_feature_importance(&mut model, &x, &y, &names, &splits, None, Scoring::NegLogLoss)?;

assert!((mda["strong"].mean - 0.725).abs() < 1e-3);
assert!((mda["weak"].mean - 0.187).abs() < 1e-3);
assert!(mda["noise"].mean.abs() < 0.01);
// Alone, the irrelevant feature scores a coin flip; the strong one is far better.
assert!((sfi["noise"].mean + 0.696).abs() < 1e-3);
assert!((sfi["strong"].mean + 0.330).abs() < 1e-3);
```

Printed to three decimals with their standard errors, the two results are:

```text
strong  MDA +0.725 ± 0.017   SFI -0.330 ± 0.007
weak    MDA +0.187 ± 0.018   SFI -0.666 ± 0.009
noise   MDA -0.003 ± 0.003   SFI -0.696 ± 0.001
```

The `±` is `ImportanceStats::std`, which despite the name is the standard error of the mean
across folds. Each method follows its snippet: MDI and MDA divide pandas' sample standard
deviation (ddof 1) by $\sqrt{n}$, SFI divides numpy's population deviation (ddof 0).

## The PCA cross-check

§8.4.2 proposes a sanity check that needs no labels. PCA ranks directions in feature space by
variance, without knowing what is being predicted. If a supervised importance ranking agrees
with the unsupervised one, that is weak evidence the model has not simply overfit.
`get_orthogonal_features(rows, variance_thresh)` standardises the features and projects them
onto the leading eigenvectors that explain `variance_thresh` of the variance;
`feature_pca_analysis(rows, importance, variance_thresh)` correlates an importance vector
with the eigenvector loadings and returns Pearson, Spearman, Kendall and weighted-Kendall
coefficients. They match `scipy.stats` as Snippet 8.6 calls it: with more than one retained
component the importance vector is repeated once per component and so is full of ties, and
Spearman uses average ranks and Kendall is tau-b, as scipy's are. The weighted Kendall is
`scipy.stats.weightedtau(importance, 1 / pca_rank)`, with hyperbolic weights by rank, and
`pca_rank` gives tied loadings their average rank. Where scipy would return NaN because an
input is constant, the rank coefficients here return 0.

## What to watch for

- **MDA of a very persistent feature is still somewhat understated.** The shuffle happens
  within each test fold, as in Snippet 8.3. When a feature moves so slowly that one fold
  spans only a few of its swings, the fold's values are bunched together, and shuffling
  among them damages less than shuffling the whole sample would. On 2,000 rows in five
  folds, an informative AR(1) feature scores about the same as an i.i.d. one up to an
  autocorrelation of 0.95, and about 70% of it at 0.99. Before
  [#98](https://github.com/Open-Quant/openquant/issues/98) was fixed the column was rotated
  by one row instead of shuffled, which left such a feature almost unchanged and scored it at
  about 3% of the i.i.d. value.
- **MDI treats zero as missing.** Snippet 8.2 does this because it trains with
  `max_features=1`, where a zero means "this feature was never offered to the tree". With any
  other setting a zero means "offered and useless", and dropping it inflates the mean: a
  feature given `[0.0, 0.2, 0.0]` by three trees is averaged as 0.2, not 0.067. Either train
  the forest as the book does or replace zeros with a tiny positive number first.
- **A standard error from one fold or one tree is reported as 0**, not NaN as pandas would
  give. With MDI, a feature that only one tree split on also gets 0.
- **SFI scores on unweighted test folds.** It delegates to
  [`ml_cross_val_score`](/modules/cross-validation/#what-to-watch-for), which passes weights
  to `fit` only. MDA weights both.
- **`plot_feature_importance` writes a CSV.** The name is mlfinlab's; there is no plotting in
  the Rust crate. With `output_path = None` it does nothing.
- **Substitution is not solved by any single method.** Two correlated features split MDI,
  hide each other in MDA and each look strong in SFI. Read the three together, or
  orthogonalise first and accept that principal components have no names.

## Related modules

- [`feature-diagnostics`](/modules/feature-diagnostics/) — the same ideas from Python, with a
  substitution-effect report.
- [`cross-validation`](/modules/cross-validation/) — the purged folds MDA and SFI run on.
- [`fingerprint`](/modules/fingerprint/) — *how* the model uses a feature, not just how much.
- [`codependence`](/modules/codependence/) and [`onc`](/modules/onc/) — cluster correlated
  features before measuring importance.
