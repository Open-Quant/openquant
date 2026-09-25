---
title: "cross_validation"
description: "Purged k-fold cross-validation with an embargo, for labels that overlap in time."
status: authored
last_authored: '2026-09-24'
audience:
  - quant-dev
  - platform-engineering
module: "cross_validation"
api_surface: "rust-only"
afml_chapter:
  - "7"
  - "12"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 7: §7.3 Why K-Fold CV Fails in Finance; §7.4.1 Purging the Training Set (Snippet 7.1); §7.4.2 Embargo (Snippet 7.2); §7.4.3 The Purged K-Fold Class (Snippet 7.3); §7.5 Bugs in Sklearn's Cross-Validation (Snippet 7.4). Chapter 12: §12.4 The Combinatorial Purged Cross-Validation Method."
rust_api:
  - "PurgedKFold"
  - "ml_get_train_times"
  - "ml_cross_val_score"
  - "SimpleClassifier"
  - "Scoring"
  - "TrainTestSplit"
  - "PurgedSplit"
  - "PurgedSplitDiagnostics"
  - "CpcvSplit"
  - "CpcvPath"
  - "naive_kfold_splits"
  - "count_train_test_overlaps"
  - "CrossValidationError"
sidebar:
  badge: Module
---

k-fold cross-validation estimates out-of-sample performance on the assumption that a training
observation tells the model nothing about a test observation beyond what the model
generalises. With financial labels that assumption fails in a specific, mechanical way (AFML
§7.3). A label formed at bar $t$ and resolved at bar $t+h$ is a function of the prices in
between. If a training label's span overlaps a test label's span, the two share returns: the
model has been shown part of the answer, and the fold's score is inflated. Shuffling makes it
worse, since it scatters such pairs across every fold boundary.

This module is Rust-only. Python bindings are tracked in
[#42](https://github.com/Open-Quant/openquant/issues/42).

## Purging and embargo

**Purging** (§7.4.1) removes from the training set every label whose span
$[t_{i,0},\,t_{i,1}]$ intersects the window the test labels cover. For a test fold that
window runs from the first test label's start to the *latest* end among the fold's labels,
which with variable-length labels need not be the last label's. A training label is dropped
if it starts inside the window, ends inside it, or envelops it.

**Embargo** (§7.4.2) removes a further stretch of training labels just *after* the test
window. Purging handles overlap in the labels; the embargo handles what leaks through the
features, which are usually serially correlated — a moving average computed a few bars after
the test window still contains test-window prices.

`PurgedKFold::new(n_splits, samples_info_sets, pct_embargo)` takes one `(start, end)` pair
per sample, in time order, and `split(n_samples)` returns `(train_indices, test_indices)` for
each fold. Folds are contiguous blocks; nothing is shuffled.

<figure>
<img class="dark:sl-hidden" src="/figures/ch7-purged-fold-light.svg" alt="Forty labels in time order for one fold of five. Labels 16 to 23 are the test fold. Three labels on each side of it are purged because their four-bar spans overlap the test window. A further three on each side are embargoed. The remaining twenty labels are training data." />
<img class="light:sl-hidden" src="/figures/ch7-purged-fold-dark.svg" alt="Forty labels in time order for one fold of five. Labels 16 to 23 are the test fold. Three labels on each side of it are purged because their four-bar spans overlap the test window. A further three on each side are embargoed. The remaining twenty labels are training data." />
<figcaption>The third fold of the example below at <code>pct_embargo = 0.15</code>. Twenty of the 32 non-test labels survive.</figcaption>
</figure>

## Two ways this differs from the book

Both are visible in the figure, and both make the split more conservative than Snippet 7.3,
never less.

1. **The embargo is applied on both sides of the test fold.** AFML embargoes only what
   follows a test set, because only later features can contain test-window prices. Here the
   same number of samples is also removed *before* the fold.
2. **The embargo is counted from the fold's edges, not from the end of the purged zone.**
   Snippet 7.3 resumes training `embargo` samples after the last test label's end. Here it is
   `embargo` samples after the last test *sample*. When labels span at least as many samples
   as the embargo, the embargo therefore removes nothing that purging had not already
   removed.

The example shows the second point: forty hourly labels, each resolved three hours after it
starts, five folds, third fold.

```rust
use chrono::{Duration, NaiveDate};
use openquant::cross_validation::{CrossValidationError, PurgedKFold};

let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 0, 0).unwrap();
let info_sets: Vec<_> =
    (0..40).map(|i| (open + Duration::hours(i), open + Duration::hours(i + 3))).collect();

let train_of = |pct_embargo: f64| -> Result<Vec<usize>, CrossValidationError> {
    Ok(PurgedKFold::new(5, info_sets.clone(), pct_embargo)?.split(40)?[2].0.clone())
};

// Test fold is samples 16-23. Purging alone removes 13-15 and 24-26.
let purged: Vec<usize> = (0..=12).chain(27..=39).collect();
assert_eq!(train_of(0.0)?, purged);
// An embargo of ceil(0.07 * 40) = 3 samples falls entirely inside the purged zone.
assert_eq!(train_of(0.07)?, purged);
// Six samples reach past it, on both sides.
assert_eq!(train_of(0.15)?, (0..=9).chain(30..=39).collect::<Vec<usize>>());
```

```text
embargo 0.00: test 16-23  train 0-12, 27-39  (26 of 32 kept)
embargo 0.07: test 16-23  train 0-12, 27-39  (26 of 32 kept)
embargo 0.15: test 16-23  train 0-9, 30-39  (20 of 32 kept)
```

The text block is the output of `cargo run --example docs_cross_validation`, and
`test_docs_page_example_values` pins the same three index sets. To get an embargo of $e$
samples *beyond* the purge in the book's sense, set `pct_embargo` to $(e+\ell)/n$, with
$\ell$ the longest label span in samples. Both points are recorded on
[#94](https://github.com/Open-Quant/openquant/issues/94).

`pct_embargo` is a fraction of the *whole sample count*, rounded up: 0.01 on 5,000 samples
is 50 samples. AFML suggests a value around 0.01. `new` rejects values outside $[0, 1)$, and
information sets that end before they start.

## Diagnostics and combinatorial splits

`split_with_diagnostics(n_samples)` returns the same folds as `split`, each as a
`PurgedSplit` whose `diagnostics` say why each excluded sample was excluded.
`purged_indices` overlap the test window. `embargo_indices` lie inside an embargo window,
whether or not they were also purged, so an embargo that adds nothing shows up as
`embargo_indices` contained in `purged_indices`. `test_ranges` gives the test set as ranges.
Training is every sample in none of the three. `overlap_count_after_purge` counts training
labels that still intersect a test label. It is always 0, and is there so a pipeline can
assert it.

`cpcv_splits(n_samples, k)` is combinatorial purged CV (AFML §12.4). It returns one
`CpcvSplit` for each of the $\binom{N}{k}$ ways to test $k$ of the $N$ folds at once, in
lexicographic order of `test_fold_ids`. Adjacent test folds form one block, and each block is
purged and embargoed like a `split` fold, so `k = 1` gives back `split_with_diagnostics`.
`cpcv_paths(k)` returns the $\varphi[N,k] = \frac{k}{N}\binom{N}{k}$ backtest paths those splits
make. Path $j$ takes, for each fold, the $j$-th split that tests it: `split_for_fold[g]` is that
split's `split_id`. The numbering matches [`backtesting-engine`](/modules/backtesting-engine/)'s
`run_cpcv`, which also scores returns along the paths.

`naive_kfold_splits` is the unpurged baseline §7.3 warns against, and
`count_train_test_overlaps(info_sets, train, test)` counts training samples whose span
intersects some test sample's. Together they measure the leak that purging removes.

```rust
use chrono::{Duration, NaiveDate};
use openquant::cross_validation::{count_train_test_overlaps, naive_kfold_splits, PurgedKFold};

let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 0, 0).unwrap();
let info_sets: Vec<_> =
    (0..40).map(|i| (open + Duration::hours(i), open + Duration::hours(i + 3))).collect();
let cv = PurgedKFold::new(5, info_sets.clone(), 0.15)?;

// The third fold of the example above, with the reason for each exclusion.
let fold = &cv.split_with_diagnostics(40)?[2];
assert_eq!(fold.diagnostics.test_ranges, vec![(16, 24)]);
assert_eq!(fold.diagnostics.purged_indices, vec![13, 14, 15, 24, 25, 26]);
// Six samples each side; 13-15 and 24-26 were purged as well.
let embargoed: Vec<usize> = (10..=15).chain(24..=29).collect();
assert_eq!(fold.diagnostics.embargo_indices, embargoed);

// N = 5, k = 2: C(5, 2) = 10 splits and 2/5 * 10 = 4 paths.
assert_eq!(cv.cpcv_splits(40, 2)?.len(), 10);
assert_eq!(cv.cpcv_paths(2)?.len(), 4);

// Unpurged, the same fold trains on 6 labels that overlap the test window.
let (train, test) = &naive_kfold_splits(40, 5)?[2];
assert_eq!(count_train_test_overlaps(&info_sets, train, test)?, 6);
```

## Scoring

`ml_cross_val_score(classifier, x, y, sample_weight, splits, scoring)` fits on each training
set and scores each test set. It takes any `SimpleClassifier` — a two-method trait, `fit`
and `predict_proba` — and one of three `Scoring` rules:

| `Scoring` | Value per fold |
| --- | --- |
| `Accuracy` | share of test samples where `proba ≥ 0.5` matches the label |
| `NegLogLoss` | $\frac1n\sum_i\bigl[y_i\ln p_i+(1-y_i)\ln(1-p_i)\bigr]$, with $p_i$ clipped to $[10^{-15},\,1-10^{-15}]$; higher is better |
| `F1` | harmonic mean of precision and recall for the positive class; 0 when there are no positive predictions |

AFML's advice (§7.5 and Chapter 9) is to prefer log loss for anything that will be sized by
probability: accuracy scores a confident wrong call the same as a hesitant one, and
[bet sizing](/modules/bet-sizing/) does not.

```rust
use chrono::{Duration, NaiveDate};
use openquant::cross_validation::{ml_cross_val_score, PurgedKFold, Scoring, SimpleClassifier};

/// Predicts the training base rate, whatever the features. A floor for any real model.
struct BaseRate(f64);

impl SimpleClassifier for BaseRate {
    fn fit(&mut self, _x: &[Vec<f64>], y: &[f64], _sample_weight: Option<&[f64]>) {
        self.0 = y.iter().sum::<f64>() / y.len() as f64;
    }
    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        vec![self.0; x.len()]
    }
}

let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 0, 0).unwrap();
let info_sets: Vec<_> =
    (0..40).map(|i| (open + Duration::hours(i), open + Duration::hours(i + 3))).collect();
let x: Vec<Vec<f64>> = (0..40).map(|i| vec![f64::from(i)]).collect();
let y: Vec<f64> = (0..40).map(|i| f64::from(u8::from(i % 4 == 0))).collect();

let splits = PurgedKFold::new(5, info_sets, 0.0)?.split(40)?;
let scores = ml_cross_val_score(&mut BaseRate(0.0), &x, &y, None, &splits, Scoring::NegLogLoss);

// A quarter of labels are positive, so every fold scores near ln-loss of p = 0.25.
let entropy = -(0.25 * 0.25f64.ln() + 0.75 * 0.75f64.ln());
assert_eq!(scores.len(), 5);
assert!(scores.iter().all(|s| (s + entropy).abs() < 0.002));
```

## What to watch for

- **Sample weights are used to fit, not to score.** Snippet 7.4 exists because scikit-learn's
  `cross_val_score` did exactly this; AFML's version passes the test-fold weights to the
  metric. Here `sample_weight` reaches `fit` only, so every test sample counts equally. If
  your weights matter — and with return attribution they differ by an order of magnitude —
  compute the weighted score yourself from `splits`.
- **`ml_cross_val_score` does not validate.** It returns `Vec<f64>`, not `Result`. An index
  in `splits` beyond `x` panics, and an empty test set yields `NaN`. Build `splits` with
  `PurgedKFold` and pass the same `x` and `y` it was sized for.
- **`samples_info_sets` must be in time order, one per row of `x`.** Folds are blocks of
  consecutive indices; the purge compares timestamps but the embargo counts positions.
  Unsorted input purges correctly and embargoes nonsense.
- **Purging can empty a training set.** With long labels and many folds, the purged zone can
  swallow a small dataset. Check `train.len()` per fold; a fold trained on a handful of
  samples produces a score, not a meaningful one.
- **`ml_get_train_times` is the purge alone**, on timestamps, for one or more test windows
  (Snippet 7.1). It applies no embargo and nothing else in the crate calls it; use it when
  you build your own splits, for instance several disjoint test blocks at once.
- **One path is not a backtest.** Purged k-fold gives one out-of-sample prediction per
  sample, hence one performance path. `cpcv_splits` and `cpcv_paths` give $\varphi[N,k]$ of
  them, and [`backtesting-engine`](/modules/backtesting-engine/)'s `run_cpcv` scores each.
- **CPCV grows fast.** $\binom{N}{k}$ splits each hold their own index vectors: $N = 10$,
  $k = 5$ is 252 splits. Only an overflowing count is an error (`TooManySplits`).

## Related modules

- [`sampling`](/modules/sampling/) and [`sample-weights`](/modules/sample-weights/) — the
  in-sample half of the same overlap problem.
- [`backtesting-engine`](/modules/backtesting-engine/) — walk-forward, purged CV and CPCV
  over these splits.
- [`hyperparameter-tuning`](/modules/hyperparameter-tuning/) — grid and randomised search
  under `PurgedKFold`.
- [`feature-importance`](/modules/feature-importance/) — MDA scores features on purged folds.
