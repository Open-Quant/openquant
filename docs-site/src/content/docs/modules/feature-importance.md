---
title: "feature_importance"
description: "MDI, MDA and SFI feature importance, and a PCA cross-check, for models validated on purged folds."
status: authored
last_authored: '2026-09-25'
audience:
  - quant-dev
  - platform-engineering
module: "feature_importance"
api_surface: "both"
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
python_api:
  - "feature_importance.mean_decrease_impurity"
  - "feature_importance.mean_decrease_accuracy"
  - "feature_importance.single_feature_importance"
  - "feature_importance.mda_from_probabilities"
  - "feature_importance.sfi_from_probabilities"
sidebar:
  badge: Module
---

AFML's first law of backtesting (§8.2) is that a backtest is not a research tool; feature
importance is. A backtest tells you that some combination of features and rules made money
on one path. Importance tells you *which* features the model leans on, which can be checked
against what you believe about the market, and which survives the model being refitted.

Chapter 8 gives three methods, and the reason to have three is that each one is wrong in a
different way. This module implements them for any model behind the
[`SimpleClassifier`](/modules/cross-validation/#scoring) trait, and from Python for any model
with `fit` and `predict_proba` (see [From Python](#from-python)).
[`feature-diagnostics`](/modules/feature-diagnostics/) is an older pure-Python version with its
own linear model.

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
damage one feature's column in the test fold and score again. With $s_k$ the score of fold
$k$ and $s_{k,j}$ the score with feature $j$ damaged,

$$
\mathrm{MDA}_j \;=\; \frac1K\sum_{k=1}^{K}\frac{s_k-s_{k,j}}{s^{\max}-s_{k,j}}
$$

where $s^{\max}$ is the best attainable score: 0 for negative log loss, 1 for accuracy and F1.
A value of 1 means damaging the feature destroyed everything the model had; 0 means the model
did not need it; negative means the model did better without it. Scores here *are* weighted
by `sample_weight` on the test fold, as Snippet 8.3 does.

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
let mda = mean_decrease_accuracy(&mut model, &x, &y, &names, &splits, None, Scoring::NegLogLoss)?;
let sfi = single_feature_importance(&mut model, &x, &y, &names, &splits, None, Scoring::NegLogLoss)?;

assert!((mda["strong"].mean - 0.751).abs() < 1e-3);
assert!((mda["weak"].mean - 0.234).abs() < 1e-3);
assert!(mda["noise"].mean.abs() < 0.01);
// Alone, the irrelevant feature scores a coin flip; the strong one is far better.
assert!((sfi["noise"].mean + 0.696).abs() < 1e-3);
assert!((sfi["strong"].mean + 0.330).abs() < 1e-3);
```

Printed to three decimals with their standard errors, the two results are:

```text
strong  MDA +0.751 ± 0.006   SFI -0.330 ± 0.007
weak    MDA +0.234 ± 0.024   SFI -0.666 ± 0.009
noise   MDA -0.008 ± 0.002   SFI -0.696 ± 0.001
```

The `±` is `ImportanceStats::std`, which despite the name is the standard error of the mean
across folds.

## The PCA cross-check

§8.4.2 proposes a sanity check that needs no labels. PCA ranks directions in feature space by
variance, without knowing what is being predicted. If a supervised importance ranking agrees
with the unsupervised one, that is weak evidence the model has not simply overfit.
`get_orthogonal_features(rows, variance_thresh)` standardises the features and projects them
onto the leading eigenvectors that explain `variance_thresh` of the variance;
`feature_pca_analysis(rows, importance, variance_thresh)` correlates an importance vector
with the eigenvector loadings and returns Pearson, Spearman, Kendall and weighted-Kendall
coefficients.

Treat the rank coefficients as unreliable for now: with more than one retained component the
Spearman and Kendall values mishandle ties, and the weighted Kendall is not
`scipy.stats.weightedtau`, which is what the book uses
([#94](https://github.com/Open-Quant/openquant/issues/94)). Pearson is unaffected.

## From Python

`openquant.feature_importance` scores with the Rust functions above, but the model stays in
Python. MDA and SFI always run on purged k-fold splits built from the label spans `t0` and
`t1`, which are required arguments: called without them, every function raises. There is no
argument for passing folds of your own.

| Function | Model | Result |
| --- | --- | --- |
| `mean_decrease_impurity(per_tree_importances, feature_names=None)` | none: one row per tree, e.g. `[t.feature_importances_ for t in forest.estimators_]` | MDI |
| `mean_decrease_accuracy(estimator, X, y, t0, t1, *, n_splits, pct_embargo, scoring, sample_weight, seed)` | any object with `fit(X, y, sample_weight=...)` and `predict_proba(X)`; copied per fold with `sklearn.base.clone` when scikit-learn is installed | MDA, each test column shuffled with `numpy.random.default_rng(seed)` |
| `single_feature_importance(estimator, X, y, t0, t1, ...)` | the same, fitted on one column at a time | SFI |
| `mda_from_probabilities(y, t0, t1, base_proba, permuted_proba, *, n_splits, ...)` | none: out-of-sample probabilities you computed on `purged_kfold_splits(t0, t1, n_splits, pct_embargo)` | MDA |
| `sfi_from_probabilities(y, t0, t1, proba, *, n_splits, ...)` | the same, one column per single-feature model | SFI |

The estimator-driven functions fit in Python and pass the probabilities to the last two, which
rebuild the purged folds and hand the Rust MDA and SFI a stand-in classifier that plays the
probabilities back. Each result maps a feature name to `{"mean", "std"}`, `std` being the
standard error, in the order of `feature_names`. Labels must be 0/1 and `scoring` is
`"neg_log_loss"`, `"accuracy"` or `"f1"`.

```python
import numpy as np
from openquant import feature_importance as fi


class NearestMean:
    """A tiny classifier with the scikit-learn interface: fit and predict_proba."""

    def fit(self, X, y, sample_weight=None):
        self.w = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
        return self

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-4.0 * X @ self.w))
        return np.column_stack([1.0 - p, p])


rng = np.random.default_rng(0)
n = 400
X = rng.normal(size=(n, 3))  # f0 drives the label; f1 and f2 are noise
y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(float)
t0 = np.arange(n)  # the bar of each event ...
t1 = t0 + 5        # ... and the bar its label resolves on

mda = fi.mean_decrease_accuracy(NearestMean(), X, y, t0, t1, n_splits=5, pct_embargo=0.01)
sfi = fi.single_feature_importance(NearestMean(), X, y, t0, t1, n_splits=5, pct_embargo=0.01)
mdi = fi.mean_decrease_impurity([[0.6, 0.3, 0.1], [0.5, 0.3, 0.2], [0.7, 0.2, 0.1]])
for name in ("f0", "f1", "f2"):
    print(name, round(mda[name]["mean"], 3), round(sfi[name]["mean"], 3), round(mdi[name]["mean"], 3))
```

```text
f0 0.86 -0.282 0.6
f1 -0.036 -0.735 0.267
f2 -0.019 -0.708 0.133
```

The three columns are MDA, SFI and (from made-up per-tree importances) MDI. SFI is scored by
negative log loss, so every value is negative and the least negative feature is best.

## What to watch for

- **MDA here does not shuffle; it shifts the column by one row.** That keeps the result
  deterministic, but a shifted copy of a *persistent* feature is almost the same feature, so
  its measured importance collapses. With a feature autocorrelation of 0.99 the Python
  implementation, which shares the design, reports 0.009 where a true shuffle gives 0.155
  ([#98](https://github.com/Open-Quant/openquant/issues/98)). The example above uses
  independent draws, where a shift is as good as a shuffle. **On real bar features, do not
  trust a low MDA from this function until that issue is closed.**
- **MDI treats zero as missing.** Snippet 8.2 does this because it trains with
  `max_features=1`, where a zero means "this feature was never offered to the tree". With any
  other setting a zero means "offered and useless", and dropping it inflates the mean: a
  feature given `[0.0, 0.2, 0.0]` by three trees is averaged as 0.2, not 0.067. Either train
  the forest as the book does or replace zeros with a tiny positive number first.
- **Standard errors are computed with a population standard deviation** (divide by $n$).
  Snippets 8.2 and 8.3 use pandas' sample deviation; the reported error is too small by
  $\sqrt{(n-1)/n}$, 18% with three folds and 11% with five (#94).
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
