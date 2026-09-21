---
title: "sb_bagging"
description: "A bagging ensemble intended to draw each estimator's sample with the sequential bootstrap. Read the status note before using it."
status: authored
last_authored: '2026-09-20'
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

:::caution[Status: this module does not yet do what its name says]
As implemented today, **every estimator is trained on a uniform bootstrap sample**, not a
sequential one. `fit` fills `seq_bootstrap`'s warm-up list with as many uniform random label
indices as the sample is long, so the uniqueness-weighted draw is never used. Measured on 60
overlapping labels, the average uniqueness of the classifier's samples is 0.255 — the same
as a uniform bootstrap (0.255) and below `sampling::seq_bootstrap` (0.261).

Three further gaps: the reported `oob_score` is in-sample accuracy, `sample_weight` is
accepted and ignored, and the base learner is a fixed one-feature model. All are tracked in
[#90](https://github.com/Open-Quant/openquant/issues/90). Until it is closed, treat this
module as an interface sketch, and get the real thing by passing
[`sampling.seq_bootstrap`](/modules/sampling/) indices to your own learners.
:::

## What it does today

For each of `n_estimators` estimators, `fit(x, y, ind_mat, sample_weight)`:

1. draws `max_features` column indices and **keeps only the first**;
2. draws `max_samples` row indices (uniformly, per the note above);
3. fits the base learner on that one feature over those rows.

The classifier's base learner is a decision stump whose threshold is the *mean* of the
feature over the sample — not a fitted split — predicting whichever side of it had the
higher rate of `y == 1`. The regressor's is a one-variable least-squares line. `predict`
takes a majority vote (ties go to class 1) or the mean of the lines.

`ind_mat` is the bars × labels indicator matrix from
[`sampling.get_ind_matrix`](/modules/sampling/#concurrency-and-uniqueness); its columns must
correspond one-to-one with the rows of `x`.

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
predictions = list(fit["predictions"])  # the binding returns bytes; list() gives 0/1 ints
in_sample = sum(p == t for p, t in zip(predictions, y)) / n
print(f"in-sample accuracy {in_sample:.3f}   reported oob_score {fit['oob_score']:.3f}")

again = sb_bagging.fit_predict_sb_classifier(x, y, ind_mat, n_estimators=50, random_state=7)
print("same random_state, same predictions:", list(again["predictions"]) == predictions)
```

```text
in-sample accuracy 0.733   reported oob_score 0.733
same random_state, same predictions: True
```

The two numbers on the first line are equal because they are the same computation. Do not
read `oob_score` as an estimate of generalisation; score the model under
[purged cross-validation](/modules/cross-validation/) instead.

The Python functions fit and predict on the same `x` in one call and return a dict with
`predictions` and `oob_score`. There is no way to predict on new rows from Python; that needs
the Rust types.

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
`estimators_samples` holds the row indices each estimator was trained on.

## What to watch for

- **Everything in the status note.** In particular, a result from this module is a result
  from ordinary bagging of one-feature stumps.
- **A mismatched `ind_mat` panics.** If `ind_mat` has more label columns than `x` has rows,
  `fit` indexes past the end of `x` and panics; from Python that surfaces as a
  `PanicException`, not a `ValueError`. Fewer columns than rows fails silently instead: the
  extra rows are never sampled.
- **`max_features` above one column changes nothing** except the random stream, because only
  the first sampled feature is used. With several informative features each estimator sees
  one of them, chosen at random.
- **Labels are `u8` with 1 as the positive class.** Anything other than 1 counts as negative
  when the stump picks its side, so a −1/+1 encoding must be mapped to 0/1 first.
- **`random_state` reproduces a fit today only because the draws are uniform.** Once the
  sequential draw is restored, reproducibility depends on `seq_bootstrap` becoming seedable,
  which is part of the same issue.

## Related modules

- [`sampling`](/modules/sampling/) — the indicator matrix and the sequential bootstrap itself.
- [`sample-weights`](/modules/sample-weights/) — the other correction for overlapping labels.
- [`ensemble-methods`](/modules/ensemble-methods/) — bagging and boosting diagnostics.
- [`cross-validation`](/modules/cross-validation/) — how to score an ensemble when labels
  overlap.
