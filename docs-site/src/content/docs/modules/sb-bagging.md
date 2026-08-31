---
title: "sb_bagging"
description: "Sequentially bootstrapped bagging classifiers/regressors."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-31'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "sb_bagging"
api_surface: "both"
rust_api:
  - "SequentiallyBootstrappedBaggingClassifier"
  - "SequentiallyBootstrappedBaggingRegressor"
  - "MaxSamples"
  - "MaxFeatures"
sidebar:
  badge: Module
---

## Concept Overview

Bagging in which the resampling respects label overlap. The standard bootstrap assumes IID draws; with triple-barrier labels whose spans overlap, an IID bag is full of near-duplicates, the base learners end up correlated, and the variance reduction bagging promises never materialises. Sequential bootstrap instead draws each index with probability proportional to its average uniqueness *given what has already been drawn*, so each bag is as close to independent as the data permits.

## When to Use

Use it in place of ordinary bagging whenever the labels come from `labeling` — that is, whenever observations overlap in time. Measure the benefit rather than assuming it: `ensemble_methods::average_pairwise_prediction_correlation` will tell you whether the base learners actually decorrelated, and if rho is still high the extra sampling cost bought nothing. Note that `new()` takes the random seed, not the ensemble size: `n_estimators` defaults to 10 and must be set explicitly.

## Mathematical Foundations

### Bagging Predictor

$$\hat f(x)=\frac{1}{B}\sum_{b=1}^{B} f_b(x)$$

where $B$ = `n_estimators` and $f_b$ is the base learner fitted to the $b$-th resample. Note that $B$ defaults to $10$, not to the constructor argument, which is the random seed.

### Sequential Bootstrap Draw

$$\Pr\!\left[i\mid\varphi\right]=\frac{\bar u_i(\varphi)}{\sum_j \bar u_j(\varphi)},\qquad \bar u_i(\varphi)=\frac{1}{|T_i|}\sum_{t\in T_i}\frac{1}{1+c_t(\varphi)}$$

where $\varphi$ is the set of indices drawn so far, $T_i$ the bars spanned by observation $i$'s label, and $c_t(\varphi)$ the number of already-drawn observations whose label also covers bar $t$. Drawing an observation that overlaps what is already in the bag drives $\bar u_i$ down, so the next draw prefers something disjoint — this is what stops the standard IID bootstrap from silently resampling the same overlapping event $B$ times. Probabilities are recomputed after every draw. See [`sampling`](/modules/sampling/) for the uniqueness machinery.

## Usage Examples

### Rust

#### Instantiate SB bagging classifier

```rust
use openquant::sb_bagging::SequentiallyBootstrappedBaggingClassifier;

// The single constructor argument is `random_state` — NOT the ensemble size.
// n_estimators defaults to 10 and has to be set explicitly.
let mut bag = SequentiallyBootstrappedBaggingClassifier::new(42);
bag.n_estimators = 100;
bag.oob_score = true;

println!("{} estimators, seed {}", bag.n_estimators, bag.random_state);
```

## API Reference

### Python API

- `sb_bagging.fit_predict_sb_classifier`
- `sb_bagging.fit_predict_sb_regressor`

### Rust API

- `SequentiallyBootstrappedBaggingClassifier`
- `SequentiallyBootstrappedBaggingRegressor`
- `MaxSamples`
- `MaxFeatures`

## Risk Notes and Caveats

- Sequential bootstrap improves diversity under event overlap.
- Tune max_samples/max_features with out-of-sample monitoring.

## Related Modules

- [`sampling`](/modules/sampling/)
- [`ensemble-methods`](/modules/ensemble-methods/)
- [`sample-weights`](/modules/sample-weights/)
- [`labeling`](/modules/labeling/)
- [`cross-validation`](/modules/cross-validation/)
