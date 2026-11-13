---
title: "feature_diagnostics"
description: "Python feature-importance reports on purged folds: MDA, SFI, a coefficient-based stand-in for MDI, PCA orthogonalisation, substitution effects and a pre-model feature screen."
status: authored
last_authored: '2026-11-13'
audience:
  - quant-dev
  - platform-engineering
module: "feature_diagnostics"
api_surface: "python-only"
afml_chapter:
  - "8"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 8: §8.3 Feature Importance with Substitution Effects (Snippets 8.2–8.3); §8.4 Feature Importance without Substitution Effects (Snippets 8.4–8.6); §8.6 Experiments with Synthetic Data."
python_api:
  - "feature_diagnostics.mda_importance"
  - "feature_diagnostics.sfi_importance"
  - "feature_diagnostics.mdi_importance"
  - "feature_diagnostics.orthogonalize_features_pca"
  - "feature_diagnostics.substitution_effect_report"
  - "feature_diagnostics.feature_screen_report"
sidebar:
  badge: Module
---

This is the notebook-facing side of AFML Chapter 8: hand it a feature matrix and labels and
it returns importance tables ready to read or plot. Its Python implementation is
**independent of the Rust [`feature_importance`](/modules/feature-importance/) module** — it
does not call it, and it brings its own model. Read that page for what MDI, MDA and SFI mean
and how each can mislead; this page covers what the Python functions do with them.

## The model is fixed

The importance estimators and substitution report use the same model: a logistic
regression of the 0/1 label on the features, fitted by Newton's method (iteratively
reweighted least squares) with an L2 penalty of 0.001 on every coefficient, including
the intercept. `sample_weight` weights each row's log-likelihood. Fitting is implemented in Python; the
module also uses NumPy for PCA and Polars for tables. This fixed model determines what
the importance scores can reveal.

Importance is always importance *to a model*. A feature that matters through a threshold, an
interaction, or a non-monotone effect can be missed unless the input features represent
that effect. Such a feature may still receive importance through correlations with
other columns; a low score does not establish that it contains no predictive information.

Use these reports for screening. Measure importance for the model you will trade with
[`feature_importance`](/modules/feature-importance/) or your own code.

The four importance and substitution entry points require finite labels equal to
`0.0` or `1.0`; NaN, infinities, fractional labels and other class encodings raise
`ValueError` before fitting. Single-class binary inputs remain valid. Feature names
must be unique after conversion to strings, including overrides supplied to
`feature_screen_report` for a DataFrame. Duplicate names raise `ValueError` rather
than combining different columns' rejection reasons. The screen's optional `y`
is checked for matching length only; it is not used to fit a classifier.

## Purged folds are mandatory

`mda_importance`, `sfi_importance` and `substitution_effect_report` cross-validate on purged
k-fold splits and need `event_end_indices`: for each row, the row index at which its label
is resolved. Omit it and they raise, with an explanation, rather than quietly running
unpurged. `allow_unpurged=True` is the explicit opt-out, and the returned `cv` block then
records `"purged": False`. This Python splitter removes `ceil(n * pct_embargo)` rows
before each test fold and after its latest label end. The pre-fold window is counted back
from the first test row. The post-fold window starts at the first row whose start is
strictly after every test label has resolved, matching the Rust `PurgedKFold` rule
([Snippet 7.3](/modules/cross-validation/#where-the-embargo-starts)). The Python splitter
retains its additional pre-fold embargo; Rust applies only the post-label window.

For example, with test rows 0–3, labels ending at rows 2–5 and a two-row embargo, purging
removes rows through 5 and the post-label embargo removes rows 6–7. Earlier versions
counted the right embargo from the last test row instead, so those rows could remain in
training. Correcting this can change feature scores, and long label horizons or large
embargoes can now produce the existing empty-training-fold `ValueError`. Zero embargo
and point labels (`event_end_indices[i] == i`) keep their previous splits.

## Weights and numerical limits

Weights must be finite and nonnegative, with at least one positive value overall.
Individual zero weights are allowed, but every training and test fold must retain positive
weight. Each bootstrap draw in `mdi_importance` must also retain positive weight; a draw
containing only zero-weight rows raises `ValueError` rather than being silently redrawn.

Exceptional weighted score arithmetic uses exact rational sums when floating-point
aggregates overflow, tiny log-loss totals underflow, or the F1 precision-recall product
becomes subnormal. Log loss still averages the same clipped, rounded per-row logarithms;
this does not make the logarithm itself exact.

For finite features and binary labels, fitting can retry with rational accumulation and
linear-system elimination after nonfinite arithmetic or a singular floating-point solve.
Very large weights also trigger checks for material loss-tail cancellation. Recovery
preserves the supplied weights and the original ridge penalty: normalising the weights
would change their strength relative to that fixed penalty and therefore change the fit.

This is not an arbitrary-precision logistic regression. Exponential and logarithmic
calculations still use binary64. For absolute logits up to 4096, subnormal or zero
exponential tails are recovered from eight normal-range factors; larger logits
retain the zero-tail limit. Recovery restarts from zero with the same 50-iteration budget;
extreme separable problems can reach that limit before reaching the ridge optimum.
Its stopping check requires small coefficient changes and small logit changes for each
observation, so one enormous logit cannot hide changes in the other rows.

Prediction retries a nonfinite dot product, or a finite one whose estimated roundoff
exceeds `1e-10 * max(1, abs(logit))`, with exact rational arithmetic when the input
features and stored coefficients, including the intercept, are finite. The estimate uses
the absolute product magnitudes to detect potentially material cancellation; it is a
diagnostic policy, not a universal prediction-error bound. Recovery then rounds the logit
and applies the floating-point sigmoid. This evaluates the stored model; it does not
recover precision already lost in its coefficients. Severely ill-conditioned features can
therefore still lose prediction accuracy. Finite inputs alone do not guarantee convergence
or accurate coefficients.

Recovery can be expensive even on small data. Development probes measured an exceptional
80-row, 8-feature fit at about 1.6 seconds after the exact solver speedup, and recovered
scoring on 4,096 rows at up to 26 milliseconds (roughly 5–20 times the ordinary score
path). Recovery-heavy prediction on 4,096 rows took about 27–28 milliseconds. Relative to
the preceding nonfinite-only prediction check, the cancellation diagnostic increased
ordinary prediction time to about 1.25–1.46 times the baseline across 16–512 features.
Complete ordinary MDA and SFI reports, where fitting dominates, measured about 0.96–1.09
times the baseline across the tested workloads, including 32 features; ratios below one
are measurement noise, not claimed speedups. These are measurements on a shared machine,
not performance bounds. Importance reports repeat fits across folds, features and bootstrap
draws, so exceptional fitting costs can compound.

## Three rankings of the same features

The data follow AFML's synthetic experiment (§8.6) in miniature: two informative features, a
noisy copy of the first, and two of pure noise.

```python
import random

from openquant import feature_diagnostics as fd

# 1,200 samples. Each label resolves five rows after it starts, so neighbours overlap.
rng = random.Random(8)
n = 1200
a = [rng.gauss(0, 1) for _ in range(n)]
b = [rng.gauss(0, 1) for _ in range(n)]
names = ["a", "a_copy", "b", "noise_1", "noise_2"]
X = [[a[i], a[i] + rng.gauss(0, 0.2), b[i], rng.gauss(0, 1), rng.gauss(0, 1)] for i in range(n)]
y = [1.0 if a[i] + 0.6 * b[i] + rng.gauss(0, 0.7) > 0 else 0.0 for i in range(n)]
ends = [min(i + 5, n - 1) for i in range(n)]

common = dict(feature_names=names, event_end_indices=ends, n_splits=5, pct_embargo=0.01)
mdi = {r["feature"]: r["mean"] for r in fd.mdi_importance(X, y, feature_names=names)["records"]}
mda = {r["feature"]: r["mean"] for r in fd.mda_importance(X, y, **common)["records"]}
sfi = {r["feature"]: r["mean"] for r in fd.sfi_importance(X, y, **common)["records"]}

print("feature      MDI     MDA     SFI")
for f in names:
    print(f"{f:8s}  {mdi[f]:6.3f}  {mda[f] + 0.0:6.3f}  {sfi[f]:6.3f}")

pair = fd.substitution_effect_report(X, y, **common)["pair_records"][0]
print(f"{pair['feature_a']} ~ {pair['feature_b']}: corr {pair['corr']:.3f}, "
      f"singly {pair['single_sum']:.3f}, jointly {pair['group_importance']:.3f}")
```

```text
feature      MDI     MDA     SFI
a          0.538   0.626  -0.474
a_copy     0.093   0.016  -0.482
b          0.337   0.410  -0.633
noise_1    0.018  -0.001  -0.696
noise_2    0.015  -0.005  -0.695
a ~ a_copy: corr 0.981, singly 0.641, jointly 0.656
```

Read across the `a_copy` row. MDA says it is nearly worthless, 0.016 against 0.626 for `a`.
SFI says it is nearly as good as `a`: −0.482 against −0.474. Both are right. Given `a`, the model
has no use for a noisier copy of it, and damaging the copy costs nothing; on its own, the
copy carries almost all of `a`'s information. This is the substitution effect, and it is the
reason to assess removals by refitting and reevaluating the remaining set. Removing `a`
can increase the credit assigned to `a_copy`, but its new rank must be measured.

<figure>
<img class="dark:sl-hidden" src="/figures/ch8-importance-light.svg" alt="Three bar charts of importance for five features. Under the coefficient-based MDI, feature a leads, b follows, and a_copy is small. Under MDA, a dominates, b is about two-thirds of it, and a_copy is near zero. Under SFI, measured as improvement over a coin flip, a and a_copy are nearly equal, b is clearly positive and both noise features are marginally negative." />
<img class="light:sl-hidden" src="/figures/ch8-importance-dark.svg" alt="Three bar charts of importance for five features. Under the coefficient-based MDI, feature a leads, b follows, and a_copy is small. Under MDA, a dominates, b is about two-thirds of it, and a_copy is near zero. Under SFI, measured as improvement over a coin flip, a and a_copy are nearly equal, b is clearly positive and both noise features are marginally negative." />
<figcaption>Same data, same model, three answers. SFI is drawn as the gain over a coin flip's log loss of −0.693.</figcaption>
</figure>

SFI levels here are cross-validated negative log losses. A constant probability 0.5
scores −0.693; with imbalanced labels, also compare against a baseline probability fitted
on the training fold rather than treating a fair coin as the only relevant benchmark. The
informative features beat it, `b` by less because it carries less of the signal on its own;
the noise features land just below it, which is what fitting pure noise costs out of sample.

## What each function returns

All return a `dict`. The importance functions share a shape: `table` (a Polars frame with
`feature`, `mean`, `std`, `stderr`, sorted by `mean`), `records` (the same as a list of
dicts), `viz_payload` (for [`viz`](/modules/viz/)), and for the cross-validated ones a `cv`
block recording the split settings and `mean_base_score`.

| Function | Notes |
| --- | --- |
| `mda_importance` | `scoring` is `"neg_log_loss"` (default), `"accuracy"` or `"f1"` (threshold 0.5); test-fold scores use `sample_weight` |
| `sfi_importance` | raw cross-validated score per feature; compare with −0.693 for log loss |
| `mdi_importance` | **not MDI** — see below |
| `orthogonalize_features_pca` | standardise, then project on the components explaining `variance_threshold` (default 0.95) |
| `substitution_effect_report` | for each pair with `abs(corr) ≥ corr_threshold` (default 0.9): the sum of their single MDAs, the MDA of damaging both at once, and their ratio; plus MDA on PCA-orthogonalised features |
| `feature_screen_report` | no model: coverage, variance and pairwise correlation checks, with `selected_features` and `rejection_reasons` |

In the substitution report, `dilution_ratio` is joint importance over the sum of single
importances, and `flag_substitution_risk` is set when it exceeds 1.15. In the example
it is 1.02 and unflagged: the linear model concentrated on `a` and never split credit with
the copy. Other models can share credit differently, so a tree ensemble requires its own
measurement. The report is only as sensitive as the fixed model lets it be, and 1.15 is a
heuristic flag threshold rather than a significance level.

## Validate values as well as shapes

The importance helpers convert `X`, `y` and `sample_weight` to floats and check
rectangular shape and matching lengths. Feature and label conversion alone does not
establish finite feature values or binary labels; weights receive the additional checks
described above. Supply one sample per row, one
feature per column, finite `X`, labels of exactly `0` or `1`, and finite nonnegative
weights. Keep row ordering identical across these inputs and the event end indices.
The standalone PCA helper additionally rejects nonfinite `X`; do not assume that
validation applies to every importance helper.

Every fitted and scored fold needs positive total weight. A positive total over the
whole dataset is insufficient when weights are concentrated in a few rows. Inputs with
no positive weight, and train/test folds without positive weight, raise `ValueError`.
This avoids reporting a perfect-looking zero log loss without weighted evidence:

```python
from openquant import feature_diagnostics as fd

try:
    fd.sfi_importance(
        [[float(i)] for i in range(6)], [0, 1, 0, 1, 0, 1],
        sample_weight=[0.0] * 6, event_end_indices=list(range(6)),
        n_splits=2, pct_embargo=0.0,
    )
except ValueError as exc:
    assert "positive value" in str(exc)
else:
    raise AssertionError("zero-weight inputs must be rejected")
print("zero-weight SFI is rejected")
```

```text
zero-weight SFI is rejected
```

Check weight support on the actual training and test partitions before interpreting a
report; do not replace missing weights with zeros simply to make lengths match. If valid
weights still produce an unusable fold, revise the sampling or split design explicitly.
These are caller-side checks, not guarantees that the current API rejects all invalid
values. See [input-error diagnostics](/setup/input-errors/) for the distinction between
successful execution and valid numerical inputs.

## What to watch for

- **`mdi_importance` is not mean decrease impurity.** There are no trees. It bootstraps the
  rows `n_estimators` times, fits the linear model to each, and averages the normalised
  absolute coefficients. That is a reasonable in-sample ranking *if the features are on a
  common scale* — it does not standardise them. Changing units changes coefficient
  magnitudes and the effect of the L2 penalty, so a basis-point feature cannot be compared
  directly with a percentage feature. Fit any scaling on training data only.
- **Permutation depends on the distribution within each test fold.** The column is shuffled
  within each test fold, as in Snippet 8.3, with `random.Random(seed)` (`seed=42` by
  default, also taken by `substitution_effect_report`, and echoed in `cv["seed"]`). When a
  feature moves so slowly that one fold spans only a few of its swings, shuffling among the
  fold's bunched-together values can damage it less than shuffling the whole sample would.
  There is no universal autocorrelation cutoff
  below which the effect is negligible. Before
  [#98](https://github.com/Open-Quant/openquant/issues/98) was fixed the column was rotated
  by a few rows instead, and such a feature scored 0.103 instead of 0.646.
- **The correlation screen prioritizes coverage, then standard deviation.** It greedily
  compares `(coverage, std)` for each correlated pair, keeping the earlier feature on a
  tie. With equal coverage, a noisier or differently scaled column can win. The screen
  does not use `y` to select features (it only checks a supplied label vector's conversion
  and length); fit screening on training data and evaluate the selected set on
  held-out data rather than interpreting the rule as predictive ranking.
- **Full-sample preprocessing is not purged by the splitter.** The optional PCA branch in
  `substitution_effect_report` fits scaling and PCA on all supplied rows before running
  MDA folds. Its component comparison is descriptive, not a fully out-of-sample estimate.
  The report also chooses correlated pairs using all supplied rows.
  For evaluation, fit transforms separately on each training fold and apply them to its
  test fold. The standalone PCA helper returns fitted means, scales and components; it
  does not create a cross-validation pipeline.
- **Error bars are descriptive.** For MDA and SFI, `stderr` is the sample deviation across
  fold values divided by their count's square root. Overlapping training sets make folds
  dependent; this is not automatically a calibrated confidence interval. MDI applies the
  same formula across its bootstrap fits, describing resampling variation rather than
  out-of-sample predictive uncertainty.
- **Python fitting and repeated permutations can be expensive.** Benchmark the intended
  row/feature/fold counts; the Rust feature-importance API uses a caller-supplied model,
  so switching APIs is not a numerically identical replacement.
- **`event_end_indices` are row positions, not timestamps**, and must be at or after each
  row's own index. Values past the last row are clipped.

## Related modules

- [`feature-importance`](/modules/feature-importance/) — the methods themselves, for any
  model, in Rust.
- [`cross-validation`](/modules/cross-validation/) — purging and embargo.
- [`labeling`](/modules/labeling/) — where `event_end_indices` come from.
- [`viz`](/modules/viz/) — renders the `viz_payload` each report carries.
