---
title: "ensemble_methods"
description: "Diagnostics for bagged ensembles: how much variance averaging removes, given how correlated the estimators are."
status: authored
last_authored: '2026-09-20'
audience:
  - quant-dev
  - platform-engineering
module: "ensemble_methods"
api_surface: "both"
afml_chapter:
  - "6"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 6: §6.2 The Three Sources of Errors; §6.3 Bootstrap Aggregation; §6.3.1 Variance Reduction; §6.3.2 Improved Accuracy; §6.3.3 Observation Redundancy; §6.5 Boosting; §6.6 Bagging vs. Boosting in Finance."
  - "Breiman, L. (1996). Bagging predictors. Machine Learning 24(2), 123–140."
rust_api:
  - "bagging_ensemble_variance"
  - "average_pairwise_prediction_correlation"
  - "bias_variance_noise"
  - "aggregate_regression_mean"
  - "aggregate_classification_vote"
  - "aggregate_classification_probability_mean"
  - "bootstrap_sample_indices"
  - "sequential_bootstrap_sample_indices"
  - "recommend_bagging_vs_boosting"
  - "EnsembleError"
python_api:
  - "ensemble.bagging_ensemble_variance"
  - "ensemble.average_pairwise_prediction_correlation"
  - "ensemble.bias_variance_noise"
  - "ensemble.aggregate_regression_mean"
  - "ensemble.aggregate_classification_vote"
  - "ensemble.aggregate_classification_probability_mean"
  - "ensemble.bootstrap_sample_indices"
  - "ensemble.sequential_bootstrap_sample_indices"
  - "ensemble.recommend_bagging_vs_boosting"
sidebar:
  badge: Module
---

This module does not train ensembles. It answers the question that comes before training
one: *will averaging help here?* AFML's answer in Chapter 6 is a single formula, and most of
the module is that formula and the quantities that go into it. The Python module is named
`openquant.ensemble`; the Rust module is `ensemble_methods`.

## Why bagging works, and when it does not

Average $N$ estimators whose predictions each have variance $\bar\sigma^2$ and whose pairwise
correlation averages $\bar\rho$. The variance of the average is (§6.3.1)

$$
\mathrm{V}\Bigl[\frac1N\sum_{i=1}^{N}\varphi_i\Bigr]
\;=\; \bar\sigma^2\Bigl(\bar\rho + \frac{1-\bar\rho}{N}\Bigr)
$$

The second term is what more estimators buy, and it goes to zero. The first term is a floor
that no number of estimators moves. With independent estimators variance falls as $1/N$; at
$\bar\rho=0.9$, a thousand estimators remove at most a tenth of it.

<figure>
<img class="dark:sl-hidden" src="/figures/ch6-bagging-variance-light.svg" alt="Variance of a bagged ensemble relative to a single estimator, plotted against the number of estimators from 1 to 50, for correlations 0, 0.3, 0.6 and 0.9. Each curve falls quickly over the first ten estimators and then flattens at a floor equal to its correlation." />
<img class="light:sl-hidden" src="/figures/ch6-bagging-variance-dark.svg" alt="Variance of a bagged ensemble relative to a single estimator, plotted against the number of estimators from 1 to 50, for correlations 0, 0.3, 0.6 and 0.9. Each curve falls quickly over the first ten estimators and then flattens at a floor equal to its correlation." />
<figcaption>Each curve flattens at ρ̄. Past a few dozen estimators, the only lever left is making them less alike.</figcaption>
</figure>

This is why overlapping labels matter so much to bagging in finance (§6.3.3). Bootstrap
samples drawn from redundant observations are nearly the same sample, so the estimators
fitted to them are nearly the same estimator, and $\bar\rho$ is high before any modelling
choice has been made. The levers are the ones in Chapter 4: fewer overlapping labels,
[sequential bootstrapping](/modules/sampling/#the-sequential-bootstrap), a `max_samples`
set to the [average uniqueness](/modules/sampling/#concurrency-and-uniqueness), and feature
subsampling.

The example builds 25 models whose errors share a common component, measures $\bar\rho$ from
their predictions, and checks the formula against the variance actually realised by the
average.

```python
import random
from statistics import pvariance

from openquant import ensemble

# 25 models forecast 4,000 targets. Each model's error is a shared component plus its own,
# mixed so that any two models' errors correlate at about rho. Each has error variance 1.
rng = random.Random(11)
n_models, n_obs = 25, 4000
truth = [rng.gauss(0, 1) for _ in range(n_obs)]

print(" rho   measured corr   formula   realised")
for rho in (0.0, 0.3, 0.9):
    shared = [rng.gauss(0, 1) for _ in range(n_obs)]
    errors = [[rho**0.5 * s + (1 - rho) ** 0.5 * rng.gauss(0, 1) for s in shared]
              for _ in range(n_models)]
    corr = ensemble.average_pairwise_prediction_correlation(errors)
    formula = ensemble.bagging_ensemble_variance(1.0, corr, n_models)
    predictions = [[t + e for t, e in zip(truth, row)] for row in errors]
    bagged = ensemble.aggregate_regression_mean(predictions)
    realised = pvariance([b - t for b, t in zip(bagged, truth)])
    print(f"{rho:4.1f}   {round(corr, 3) + 0.0:13.3f}   {formula:7.3f}   {realised:8.3f}")

decision = ensemble.recommend_bagging_vs_boosting(0.62, 0.9, 0.4, 1.0, n_models)
print(decision["recommended"], f"{decision['expected_variance_reduction']:.3f}")
```

```text
 rho   measured corr   formula   realised
 0.0           0.000     0.040      0.040
 0.3           0.292     0.321      0.317
 0.9           0.902     0.906      0.920
boosting 0.096
```

Twenty-five independent models cut error variance to 4% of a single model's. Twenty-five
models correlated at 0.9 cut it to 92%. Note what the correlation was measured on: the models'
*errors*. `average_pairwise_prediction_correlation` takes whatever rows it is given, and on
raw predictions of a common target it mostly measures that they all track the target. Pass
residuals, or predictions on a de-meaned target, to get the $\bar\rho$ the formula means.

## The other functions

| Function | What it returns |
| --- | --- |
| `aggregate_regression_mean(rows)` | element-wise mean of the models' predictions |
| `aggregate_classification_vote(rows)` | majority vote over 0/1 labels; **a tie goes to 1** |
| `aggregate_classification_probability_mean(rows, threshold)` | mean probability per observation, and the label at `threshold` |
| `bootstrap_sample_indices(n, size, seed)` | `size` uniform draws from `0..n`, reproducible |
| `bias_variance_noise(y_true, rows)` | squared bias, variance, and MSE across models, averaged over observations |
| `recommend_bagging_vs_boosting(...)` | the formula above plus a rule-of-thumb label |

`recommend_bagging_vs_boosting` returns `Boosting` if base accuracy is below 0.55, *or*
$\bar\rho\ge 0.75$, *or* label redundancy is at least 0.70, and `Bagging` otherwise. The
reasoning follows §6.6 — bagging addresses variance and overfitting, boosting addresses bias,
and bagging cannot rescue a learner that is barely better than chance (§6.3.2). **The three
cut-offs are this library's, not the book's.** AFML's own conclusion is that bagging is
generally preferable in finance because overfitting is the greater danger; treat the label as
a prompt to look at the inputs. The numbers that carry information are
`expected_bagging_variance` and `expected_variance_reduction`.

## From Rust

```rust
use openquant::ensemble_methods::{
    aggregate_classification_vote, bagging_ensemble_variance, recommend_bagging_vs_boosting,
    EnsembleError, EnsembleMethod,
};

// The floor: at rho = 0.9, a thousand estimators remove under a tenth of the variance.
let v = bagging_ensemble_variance(1.0, 0.9, 1_000)?;
assert!((v - 0.9001).abs() < 1e-12);
// Independent estimators: 1/N.
assert!((bagging_ensemble_variance(2.0, 0.0, 50)? - 0.04).abs() < 1e-12);

// Two of three vote 1; a 1-1 tie also resolves to 1.
assert_eq!(aggregate_classification_vote(&[vec![1, 0], vec![1, 0], vec![0, 0]])?, vec![1, 0]);
assert_eq!(aggregate_classification_vote(&[vec![1], vec![0]])?, vec![1]);
assert_eq!(aggregate_classification_vote(&[vec![2]]), Err(EnsembleError::NonBinaryLabels));

let decision = recommend_bagging_vs_boosting(0.62, 0.30, 0.40, 1.0, 25)?;
assert_eq!(decision.recommended, EnsembleMethod::Bagging);
assert!((decision.expected_bagging_variance - 0.328).abs() < 1e-12);
```

## What to watch for

- **`sequential_bootstrap_sample_indices` draws a uniform bootstrap.** To make the result
  depend on `seed`, it fills `seq_bootstrap`'s warm-up list with as many uniform indices as
  the sample is long, so the uniqueness-weighted draw never happens. For a given seed it
  returns exactly what `bootstrap_sample_indices` returns. Use
  [`sampling.seq_bootstrap`](/modules/sampling/) directly until
  [#90](https://github.com/Open-Quant/openquant/issues/90) is closed.
- **`noise` in `bias_variance_noise` is always zero.** It is computed as
  MSE − bias² − variance, and with bias and variance defined against the observed `y_true`
  those three satisfy the identity exactly. Separating irreducible noise from bias needs the
  noiseless target, which only a simulation has. Read `bias_sq` as "bias plus noise".
  Also on #90.
- **The formula assumes equal variances and one average correlation.** A few strong models
  among many weak ones, or clusters of near-identical models, break it; the realised column
  in the example is the honest check.
- **From Python, vote and label outputs are `bytes`.** Wrap them in `list()`.
  `bias_variance_noise` returns a plain tuple `(bias_sq, variance, noise, mse)`, and
  `recommend_bagging_vs_boosting` a dict.
- **Majority vote discards confidence.** Averaging probabilities keeps it, and the averaged
  probability is what [`bet-sizing`](/modules/bet-sizing/) wants as input.

## Related modules

- [`sampling`](/modules/sampling/) — uniqueness and the sequential bootstrap, which attack
  $\bar\rho$ at the source.
- [`sb-bagging`](/modules/sb-bagging/) — a bagging ensemble over overlapping labels.
- [`cross-validation`](/modules/cross-validation/) — score the ensemble without leakage.
- [`feature-importance`](/modules/feature-importance/) — what the ensemble learned from.
