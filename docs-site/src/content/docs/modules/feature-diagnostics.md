---
title: "feature_diagnostics"
description: "Python feature-importance reports on purged folds: MDA, SFI, a coefficient-based stand-in for MDI, PCA orthogonalisation, substitution effects and a pre-model feature screen."
status: authored
last_authored: '2026-09-20'
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
it returns importance tables ready to read or plot. It is written in pure Python and is
**independent of the Rust [`feature_importance`](/modules/feature-importance/) module** — it
does not call it, and it brings its own model. Read that page for what MDI, MDA and SFI mean
and how each can mislead; this page covers what the Python functions do with them.

## The model is fixed

Every function here fits the same model: ridge-regularised least squares of the 0/1 label
on the features. It is fast and has no dependencies, and it is the first thing to know about
any number this module reports, for two reasons.

Importance is always importance *to a model*. A feature that matters through a threshold, an
interaction, or any non-monotone effect is invisible to a linear model and will be scored as
noise here.

:::caution[The model's probabilities are miscalibrated]
The fitted value is passed through a sigmoid although least squares already produces a number
on the probability scale, so predictions are squeezed into roughly 0.4–0.8. On the balanced
data below the model predicts class 1 for 93% of rows. `scoring="accuracy"` and `"f1"`
therefore measure little more than class balance, and every log-loss figure is compressed
toward −0.69. *Rankings* under `neg_log_loss` are still informative; the *levels* are not.
Tracked in [#99](https://github.com/Open-Quant/openquant/issues/99).
:::

Use these reports for screening. Measure importance for the model you will trade with
[`feature_importance`](/modules/feature-importance/) or your own code.

## Purged folds are mandatory

`mda_importance`, `sfi_importance` and `substitution_effect_report` cross-validate on purged
k-fold splits and need `event_end_indices`: for each row, the row index at which its label
is resolved. Omit it and they raise, with an explanation, rather than quietly running
unpurged. `allow_unpurged=True` is the explicit opt-out, and the returned `cv` block then
records `"purged": False`. The splitter embargoes both sides of each test fold, like the
[Rust one](/modules/cross-validation/#two-ways-this-differs-from-the-book).

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
a          0.524   0.108  -0.645
a_copy     0.103   0.011  -0.647
b          0.336   0.042  -0.698
noise_1    0.020  -0.000  -0.724
noise_2    0.017   0.000  -0.724
a ~ a_copy: corr 0.981, singly 0.119, jointly 0.121
```

Read across the `a_copy` row. MDA says it is nearly worthless, 0.011 against 0.108 for `a`.
SFI says it is as good as `a`: −0.647 against −0.645. Both are right. Given `a`, the model
has no use for a noisier copy of it, and damaging the copy costs nothing; on its own, the
copy carries almost all of `a`'s information. This is the substitution effect, and it is the
reason to never drop a feature on one method's say-so: remove `a` from this set and `a_copy`
becomes the most important feature in it.

<figure>
<img class="dark:sl-hidden" src="/figures/ch8-importance-light.svg" alt="Three bar charts of importance for five features. Under the coefficient-based MDI, feature a leads, b follows, and a_copy is small. Under MDA, a dominates, b is about 40 percent of it, and a_copy is near zero. Under SFI, measured as improvement over a coin flip, a and a_copy are nearly equal, b is marginally negative and both noise features are clearly negative." />
<img class="light:sl-hidden" src="/figures/ch8-importance-dark.svg" alt="Three bar charts of importance for five features. Under the coefficient-based MDI, feature a leads, b follows, and a_copy is small. Under MDA, a dominates, b is about 40 percent of it, and a_copy is near zero. Under SFI, measured as improvement over a coin flip, a and a_copy are nearly equal, b is marginally negative and both noise features are clearly negative." />
<figcaption>Same data, same model, three answers. SFI is drawn as the gain over a coin flip's log loss of −0.693; its levels are depressed by the calibration problem above.</figcaption>
</figure>

Do not read the SFI *levels*. `b` scores −0.698, marginally worse than a coin flip's −0.693,
although it is genuinely informative, and the noise features score −0.724. Both are symptoms
of the miscalibration above rather than facts about the features: a model whose
probabilities cannot go below 0.4 pays a log-loss penalty on every negative. The ordering —
`a` and `a_copy` together, then `b`, then noise — is the part to trust.

## What each function returns

All return a `dict`. The importance functions share a shape: `table` (a Polars frame with
`feature`, `mean`, `std`, `stderr`, sorted by `mean`), `records` (the same as a list of
dicts), `viz_payload` (for [`viz`](/modules/viz/)), and for the cross-validated ones a `cv`
block recording the split settings and `mean_base_score`.

| Function | Notes |
| --- | --- |
| `mda_importance` | `scoring` is `"neg_log_loss"` (default), `"accuracy"` or `"f1"` — use the default until #99 is closed; test-fold scores use `sample_weight` |
| `sfi_importance` | raw cross-validated score per feature; compare with −0.693 for log loss |
| `mdi_importance` | **not MDI** — see below |
| `orthogonalize_features_pca` | standardise, then project on the components explaining `variance_threshold` (default 0.95) |
| `substitution_effect_report` | for each pair with `abs(corr) ≥ corr_threshold` (default 0.9): the sum of their single MDAs, the MDA of damaging both at once, and their ratio; plus MDA on PCA-orthogonalised features |
| `feature_screen_report` | no model: coverage, variance and pairwise correlation checks, with `selected_features` and `rejection_reasons` |

In the substitution report, `dilution_ratio` is joint importance over the sum of single
importances, and `flag_substitution_risk` is set when it exceeds 1.15. In the example
it is 1.02 and unflagged: the linear model concentrated on `a` and never split credit with
the copy. A tree ensemble on the same data would split it, and the ratio would be well above
one. The report is only as sensitive as the fixed model lets it be.

## What to watch for

- **`mdi_importance` is not mean decrease impurity.** There are no trees. It bootstraps the
  rows `n_estimators` times, fits the linear model to each, and averages the normalised
  absolute coefficients. That is a reasonable in-sample ranking *if the features are on a
  common scale* — it does not standardise them, so a feature measured in basis points will
  outrank the same feature measured in percent. Standardise first, or ignore this column.
- **MDA shifts rather than shuffles.** The column is rotated by `fold_index + 1` rows. For
  serially correlated features that barely disturbs them and importance is badly
  understated: 0.009 instead of 0.155 at an autocorrelation of 0.99
  ([#98](https://github.com/Open-Quant/openquant/issues/98)). The example above uses
  independent draws, which hides the problem. Real features are not independent draws.
- **`feature_screen_report` keeps the higher-variance feature of a correlated pair.** On the
  example it rejects `a` and keeps `a_copy`, the noisier one, because noise adds variance.
  The screen never looks at `y`, so it cannot know better. Use it to find the pairs; choose
  which member to keep yourself.
- **It is pure Python and quadratic in places.** Fine for a few thousand rows and a few dozen
  features in a notebook. For more, use the Rust module.
- **`event_end_indices` are row positions, not timestamps**, and must be at or after each
  row's own index. Values past the last row are clipped.

## Related modules

- [`feature-importance`](/modules/feature-importance/) — the methods themselves, for any
  model, in Rust.
- [`cross-validation`](/modules/cross-validation/) — purging and embargo.
- [`labeling`](/modules/labeling/) — where `event_end_indices` come from.
- [`viz`](/modules/viz/) — renders the `viz_payload` each report carries.
