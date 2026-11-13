---
title: "fingerprint"
description: "Decompose what a fitted model has learned into linear, non-linear and pairwise-interaction effects per feature."
status: authored
last_authored: '2026-11-13'
audience:
  - quant-dev
  - platform-engineering
module: "fingerprint"
api_surface: "rust-only"
citation:
  - "Li, Y., Turkington, D. and Yazdani, A. (2020). Beyond the black box: an intuitive approach to investment prediction with machine learning. Journal of Financial Data Science 2(1), 61–75."
  - "Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of Statistics 29(5), 1189–1232. (Partial dependence.)"
rust_api:
  - "RegressionModelFingerprint"
  - "ClassificationModelFingerprint"
  - "RegressionPredictor"
  - "ClassificationPredictor"
  - "Effect"
  - "PairwiseEffect"
  - "FingerprintError"
sidebar:
  badge: Module
---

[Feature importance](/modules/feature-importance/) says *how much* a model relies on a
feature. It does not say *how*. A feature the model uses as a straight line and one it uses
as a U-shape can have the same importance while expressing different relationships. Either
shape can reflect useful structure or an artefact of the fitted model and data; neither
alone establishes an economic explanation. The model fingerprint of Li, Turkington and
Yazdani (2020) separates the two, and adds
a third part for pairs of features that only matter together. It is not from AFML; the
implementation is a port of mlfinlab's.

It needs nothing from the model but predictions, so it works on anything: implement
`RegressionPredictor::predict` or `ClassificationPredictor::predict_proba` (both take rows
and return one number per row).

## The decomposition

Everything is built on the **partial dependence** function (Friedman, 2001). For feature $k$
and a value $v$, set column $k$ to $v$ in every row of the data, predict, and average:

$$
\hat f_k(v) \;=\; \frac1N\sum_{i=1}^{N} \hat f\bigl(x_{i,1},\dots,x_{i,k-1},\,v,\,x_{i,k+1},\dots\bigr)
$$

`fit` evaluates this at `num_values` quantiles of each feature, from its minimum to its
maximum. Let $\ell_k(v)=a_k+b_k v$ be the least-squares line through those points and
$\bar f_k$ their mean. Then, averaging over the grid,

$$
\text{linear}_k = \operatorname{mean}_v\bigl|\ell_k(v)-\bar f_k\bigr|,
\qquad
\text{non-linear}_k = \operatorname{mean}_v\bigl|\hat f_k(v)-\ell_k(v)\bigr|
$$

The linear effect is how far the fitted line moves the prediction; the non-linear effect is
what the line leaves unexplained. For a pair $(k,l)$, the joint partial dependence
$\hat f_{k,l}(v,w)$ is computed on the full grid of both features, and the pairwise effect is
the mean absolute part of it that the two single-feature curves cannot account for:

$$
\text{pairwise}_{k,l} = \operatorname{mean}_{v,w}\Bigl|\hat f_{k,l}(v,w)-\bar f_{k,l}
-\bigl(\hat f_k(v)-\bar f_k\bigr)-\bigl(\hat f_l(w)-\bar f_l\bigr)\Bigr|
$$

All three are in the units of the prediction, so they can be compared with each other.
`Effect::raw` holds them; `Effect::norm` rescales each nonzero family to sum to 1.
An all-zero family stays zero. Normalised effects compare features within a family, not
linear against non-linear families; use raw effects for the latter. These absolute summaries
are not an additive decomposition of prediction variance or an out-of-sample performance score.

<img src="/figures/modules/fingerprint.svg" alt="Idealized linear dependence 2v, quadratic dependence v squared with a flat fitted line at one third, and interaction residual vw whose slope changes with w." />

[Open the full-size figure](/figures/modules/fingerprint.svg) to inspect labels or zoom on a narrow screen.

*Analytic illustration for $2x_0+x_1^2+x_0x_2$ with independent, continuous
uniform features on $[-1,1]$. The panels show curves, not effect-size estimates:
centered linear dependence, quadratic dependence with its population least-squares
line, and the joint interaction residual. Averaging the last panel over a symmetric
$x_2$ hides the interaction. These idealized curves are not output from `fit`:
the finite sample and quantile grid below give slightly different effects.*

## On a model whose answer is known

The model below is a formula, so the right decomposition can be read off it:
$2x_0$ is linear, $x_1^2$ is purely non-linear on a symmetric feature, and $x_0x_2$ is a pure
interaction.

```rust
use openquant::fingerprint::{RegressionModelFingerprint, RegressionPredictor};

struct Known;
impl RegressionPredictor for Known {
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|r| 2.0 * r[0] + r[1] * r[1] + r[0] * r[2]).collect()
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

let x: Vec<Vec<f64>> = (0..600).map(|i| vec![noise(i, 1), noise(i, 2), noise(i, 3)]).collect();

let mut fingerprint = RegressionModelFingerprint::new();
fingerprint.fit(&Known, &x, 20, Some(&[(0, 2), (0, 1)]))?;
let (linear, non_linear, pairwise) = fingerprint.get_effects()?;
let pairwise = pairwise.expect("pairs were requested");

// x0 carries the linear effect: mean |2v| over [-1, 1] is about 1.
assert!((linear.raw[&0] - 1.058).abs() < 1e-3);
assert!(linear.norm[&0] > 0.98);
// x1 is the only non-linear feature; its linear effect is small on this nearly symmetric sample.
assert!(non_linear.norm[&1] > 0.999);
assert!(linear.raw[&1] < 0.01);
// x2 has near-zero individual effects and interacts with x0. Pair keys use "(k, l)".
assert!(linear.raw[&2] < 0.02 && non_linear.raw[&2] < 1e-12);
assert!((pairwise.raw["(0, 2)"] - 0.281).abs() < 1e-3);
assert!(pairwise.raw["(0, 1)"] < 1e-12);
```

The pairwise result reveals the interaction that the single-feature partial-dependence
curves nearly miss in this sample. The small empirical means leave small linear effects,
so $x_2$'s individual effect is near zero rather than exactly zero. This does not imply that
all importance methods miss it: permutation importance can expose a model's use of the
interaction. The pairwise effect of 0.28 is in the same prediction units as $x_1$'s non-linear effect.

## What to watch for

- **Partial dependence assumes features can be varied independently.** Setting one column to
  a value while leaving the others alone creates rows that may never occur: a 5% daily return
  with bottom-decile volatility. The model is evaluated out of its training distribution and
  the curve can show behaviour it never exhibits on real data. Correlated features are where
  the fingerprint is least trustworthy; orthogonalise, or read those curves sceptically.
- **Pairs are opt-in and expensive.** Single effects cost `features × num_values` full
  predictions over the data; each pair costs `num_values²` more. With `num_values = 50`
  (mlfinlab's default) that is 2,500 passes per pair. Pass the pairs you have a reason to
  suspect, not all of them.
- **The grid uses linearly interpolated sample quantiles, including both endpoints.** It
  is not an equally spaced value grid. Every grid point receives equal weight, including
  repeated quantile values and the extreme endpoints. Outliers can therefore change the
  fitted line and effect sizes; choose a representative reference dataset in the same
  feature representation the fitted model expects.
- **Validate the predictor and requested pairs yourself.** Return exactly one finite
  prediction per row (probabilities in $[0,1]$ for classification), and use distinct feature
  indices within bounds. The implementation does not validate these contracts: invalid
  indices can panic, while a prediction-count mismatch can distort averages. Single-feature
  curves divide by the returned prediction count; pairwise curves divide by the input row count.
  Supply finite feature values and deterministic predictions so repeated evaluations are comparable.
- **The linear effect depends on the feature's range, by design.** A feature with a small
  slope and a wide range can have a larger linear effect than one with a steep slope and a
  narrow range. That is the right answer to "how much does it move predictions", and the
  wrong one to "how sensitive is the model".
- **Classification works on probabilities**, so effects are in probability units and are
  compressed near 0 and 1. A feature that moves log-odds a great deal for confident
  predictions will look minor.
- **`plot_effects` returns three summary strings**, not a plot. There is no plotting in the
  Rust crate.

## Related modules

- [`feature-importance`](/modules/feature-importance/) — how much each feature matters, to
  pair with how it matters.
- [`feature-diagnostics`](/modules/feature-diagnostics/) — importance reports from Python.
- [`codependence`](/modules/codependence/) — find the correlated features whose partial
  dependence to distrust.
