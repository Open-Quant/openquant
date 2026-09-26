---
title: "sb_bagging"
description: "A bagging ensemble that draws each estimator's sample with the sequential bootstrap, around a deliberately simple one-feature base learner."
status: authored
last_authored: '2026-09-25'
audience:
  - quant-dev
  - platform-engineering
module: "sb_bagging"
api_surface: "both"
afml_chapter:
  - "4"
  - "6"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 4, §4.5 Bagging Classifiers and Uniqueness; §4.5.1 Sequential Bootstrap. Chapter 6, §6.3 Bootstrap Aggregation."
  - "Breiman, L. (1996). Bagging predictors. Machine Learning 24(2), 123–140."
rust_api:
  - "SequentiallyBootstrappedBaggingClassifier"
  - "SequentiallyBootstrappedBaggingRegressor"
  - "MaxSamples"
  - "MaxFeatures"
  - "SbBaggingError"
python_api:
  - "sb_bagging.fit_predict_sb_classifier"
  - "sb_bagging.fit_predict_sb_regressor"
  - "sb_bagging.SequentiallyBootstrappedBaggingClassifier"
sidebar:
  badge: Module
---

Bagging (Breiman, 1996) fits many copies of a weak learner, each on a bootstrap sample of the
training set, and averages them. The variance reduction depends on the copies being
different, and with overlapping financial labels a uniform bootstrap does not make them
different enough: each sample is full of near-duplicates, and the rows left out of it
resemble the rows left in, which also inflates out-of-bag accuracy (AFML §4.5). The remedy is
to draw each estimator's sample with the
[sequential bootstrap](/modules/sampling/#the-sequential-bootstrap). This module is the
ensemble built around that idea, a port of mlfinlab's
`SequentiallyBootstrappedBaggingClassifier` and `SequentiallyBootstrappedBaggingRegressor`.

:::caution[Status: the sampling is real, the base learner is a sketch]
Each estimator's sample is drawn with the sequential bootstrap, seeded by `random_state`.
Measured on 60 overlapping labels, the average uniqueness of the estimators' samples is
0.261, against 0.254 for a uniform bootstrap. The base learner, though, is fixed: a
one-feature stump or a one-feature least-squares line. Use this module to study what
sequential sampling does to an ensemble; for a production model, pass
[`sampling.seq_bootstrap`](/modules/sampling/#the-sequential-bootstrap) indices to your own
learners.
:::

## What it does today

For each of `n_estimators` estimators, `fit(x, y, ind_mat, sample_weight)`:

1. draws `max_features` column indices and **keeps only the first**;
2. draws `max_samples` label indices with the sequential bootstrap over `ind_mat`;
3. fits the base learner on that one feature over those rows, weighting each draw by its
   row's `sample_weight` if one is given (a row drawn twice counts twice).

The classifier's base learner is a decision stump whose threshold is the *mean* of the
feature over the sample — not a fitted split — predicting whichever side of it had the
higher rate of `y == 1`. The regressor's is a one-variable least-squares line. `predict`
takes a majority vote (ties go to class 1) or the mean of the lines.

`ind_mat` is the bars × labels indicator matrix from
[`sampling.get_ind_matrix`](/modules/sampling/#concurrency-and-uniqueness); its columns must
correspond one-to-one with the rows of `x`, and a different count is `DimensionMismatch`.
`sample_weight`, if given, needs one finite, non-negative weight per row, not all zero.

```python
import random

from openquant import sampling, sb_bagging

# 120 labels of 8 bars, one starting every 2 bars. One feature carries signal, one is noise.
rng = random.Random(3)
n = 120
spans = [(2 * i, 2 * i + 7) for i in range(n)]
ind_mat = sampling.get_ind_matrix(spans, list(range(2 * n + 8)))
signal = [rng.gauss(0, 1) for _ in range(n)]
x = [[s, rng.gauss(0, 1)] for s in signal]
y = [int(s + rng.gauss(0, 0.8) > 0) for s in signal]

fit = sb_bagging.fit_predict_sb_classifier(x, y, ind_mat, n_estimators=50, random_state=7)
predictions = fit["predictions"]  # a list of 0/1 ints
in_sample = sum(p == t for p, t in zip(predictions, y)) / n
print(f"in-sample accuracy {in_sample:.3f}   out-of-bag accuracy {fit['oob_score']:.3f}")

again = sb_bagging.fit_predict_sb_classifier(x, y, ind_mat, n_estimators=50, random_state=7)
print("same random_state, same predictions:", again["predictions"] == predictions)
```

```text
in-sample accuracy 0.725   out-of-bag accuracy 0.650
same random_state, same predictions: True
```

`oob_score` scores each row only with the estimators that did not draw it, so it sits below
the in-sample figure. It is still not an honest estimate of generalisation when labels
overlap (see below); score the model under
[purged cross-validation](/modules/cross-validation/) for that.

The Python functions fit and predict on the same `x` in one call and return a dict with
`predictions` and `oob_score` (always computed; `None` only if every estimator drew every row).
`sample_weight` is passed through to `fit`.

### Predicting new rows

To score rows the model was not fitted on — a test fold, or tomorrow's events — use
`sb_bagging.SequentiallyBootstrappedBaggingClassifier`. It takes `n_estimators`,
`max_samples`, `max_features` (both fractions), `bootstrap_features`, `oob_score` (default
`False`) and `random_state` (default 42), and has:

- `fit(x, y, ind_mat, sample_weight=None)`, which returns the model. `ind_mat` covers only
  the training labels: one column per row of `x`.
- `predict(x)`, a list of 0/1 ints.
- `predict_proba(x)`, one `[P(y=0), P(y=1)]` row per input row, with columns in the order of
  `classes_` (`[0, 1]`), the scikit-learn shape. `np.asarray(...)` gives an `n × 2` array. The
  stumps have no probabilities of their own, so `P(y=1)` is the share of estimators voting 1:
  a multiple of `1 / n_estimators`, and `predict` is 1 exactly when it is at least 0.5.
- `classes_`, `n_features_in_`, `oob_score_` (`None` unless `oob_score=True`) and
  `estimators_samples_`.

`predict` and `predict_proba` raise `ValueError` before a successful `fit`, or for a matrix
with a different column count from the training one. The same `random_state` and inputs
reproduce the fit, and with `oob_score=True` the model gives the same predictions and
`oob_score` as `fit_predict_sb_classifier`.

Out-of-fold probabilities under [purged k-fold](/modules/cross-validation/): fit on each
training fold with that fold's columns of the indicator matrix, and predict its test fold.

```python
import random

import numpy as np
from openquant import sampling, sb_bagging
from openquant.cross_validation import purged_kfold_splits

# The same 120 labels as above.
rng = random.Random(3)
n = 120
spans = [(2 * i, 2 * i + 7) for i in range(n)]
ind_mat = np.asarray(sampling.get_ind_matrix(spans, list(range(2 * n + 8))))
signal = [rng.gauss(0, 1) for _ in range(n)]
x = np.array([[s, rng.gauss(0, 1)] for s in signal])
y = np.array([int(s + rng.gauss(0, 0.8) > 0) for s in signal])

t0, t1 = [s for s, _ in spans], [e for _, e in spans]
oof = np.empty((n, 2))
for train, test in purged_kfold_splits(t0, t1, n_splits=5, pct_embargo=0.01):
    model = sb_bagging.SequentiallyBootstrappedBaggingClassifier(n_estimators=50, random_state=7)
    model.fit(x[train].tolist(), y[train].tolist(), ind_mat[:, train].tolist())
    oof[test] = model.predict_proba(x[test].tolist())

print("rows sum to 1:", bool(np.allclose(oof.sum(axis=1), 1.0)))
print(f"out-of-fold accuracy {np.mean((oof[:, 1] >= 0.5) == y):.3f}")
```

```text
rows sum to 1: True
out-of-fold accuracy 0.717
```

Only the classifier has a model object; the regressor is still reached through
`fit_predict_sb_regressor` from Python.

## From Rust

```rust
use nalgebra::DMatrix;
use openquant::sampling::get_ind_matrix;
use openquant::sb_bagging::{
    MaxSamples, SbBaggingError, SequentiallyBootstrappedBaggingClassifier,
};

let n = 40;
let spans: Vec<(usize, usize)> = (0..n).map(|i| (2 * i, 2 * i + 5)).collect();
let bars: Vec<usize> = (0..2 * n + 6).collect();
let ind_mat = get_ind_matrix(&spans, &bars)?;

// One feature; the class is its sign.
let x = DMatrix::from_fn(n, 1, |r, _| r as f64 - 19.5);
let y: Vec<u8> = (0..n).map(|r| u8::from(r >= 20)).collect();

let mut model = SequentiallyBootstrappedBaggingClassifier::new(7);
model.n_estimators = 25;
model.max_samples = MaxSamples::Float(0.5);
model.fit(&x, &y, &ind_mat, None)?;

assert_eq!(model.estimators_samples.len(), 25);
assert_eq!(model.estimators_samples[0].len(), 20);
let fresh = DMatrix::from_row_slice(2, 1, &[-15.0, 15.0]);
assert_eq!(model.predict(&fresh)?, vec![0, 1]);

// Settings are validated at fit time.
model.max_samples = MaxSamples::Float(1.5);
assert_eq!(model.fit(&x, &y, &ind_mat, None), Err(SbBaggingError::MaxSamplesOutOfRange));
```

Configuration is by public field after `new(random_state)`: `n_estimators` (default 10),
`max_samples` and `max_features` (an absolute `Int` or a `Float` fraction, default all),
`bootstrap_features`, `oob_score` and `warm_start`. With `warm_start`, a second `fit` keeps
the existing estimators and adds up to the new `n_estimators`; lowering it is
`DecreasingEstimators`, and combining it with `oob_score` is `WarmStartWithOob`.
`estimators_samples` holds the row indices each estimator was trained on. With `oob_score`,
`fit` sets `oob_score_value` to the accuracy (classifier) or R² (regressor) over the rows at
least one estimator did not draw, each predicted only by those estimators; it is `None` if
every row was drawn by every estimator. The same `random_state` and inputs reproduce a fit.

## What to watch for

- **The base learner is the weak point.** A result from this module is a result from
  sequentially bootstrapped bagging of one-feature stumps or lines, not of a real model.
- **Out-of-bag rows are still close to in-bag rows.** Sequential sampling makes the samples
  more unique, but a held-out label still overlaps drawn labels in time, so `oob_score`
  remains optimistic when labels overlap (AFML §4.5). Treat it as a sanity check and score
  the model under [purged cross-validation](/modules/cross-validation/).
- **`max_features` above one column changes nothing** except the random stream, because only
  the first sampled feature is used. With several informative features each estimator sees
  one of them, chosen at random.
- **Labels are `u8` with 1 as the positive class.** Anything other than 1 counts as negative
  when the stump picks its side, so a −1/+1 encoding must be mapped to 0/1 first.
- **The sequential draw is expensive.** Each draw rescans the whole indicator matrix, so one
  estimator costs on the order of bars × labels × `max_samples` operations. Thousands of
  labels over many estimators is slow; lower `max_samples` or bootstrap within blocks.

## Related modules

- [`sampling`](/modules/sampling/) — the indicator matrix and the sequential bootstrap itself.
- [`sample-weights`](/modules/sample-weights/) — the other correction for overlapping labels.
- [`ensemble-methods`](/modules/ensemble-methods/) — bagging and boosting diagnostics.
- [`cross-validation`](/modules/cross-validation/) — how to score an ensemble when labels
  overlap.
