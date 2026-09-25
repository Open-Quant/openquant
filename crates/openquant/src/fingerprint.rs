//! Model fingerprints (Li, Turkington and Yazdani, 2020): decompose what a fitted model has
//! learned into linear, non-linear and pairwise-interaction effects per feature.
//!
//! Not from AFML; a port of mlfinlab's implementation, complementary to the feature
//! importance of AFML chapter 8 (importance says *how much* a feature matters, the
//! fingerprint says *how*). It needs only predictions: implement [`RegressionPredictor`] or
//! [`ClassificationPredictor`] for your model.
//!
//! For each feature `k`, the partial dependence `f_k(v)` (Friedman, 2001) is the mean
//! prediction with column `k` set to `v` in every row, evaluated at `num_values` quantiles
//! from the feature's minimum to its maximum. With `l_k` the least-squares line through
//! those points and `f_k_bar` their mean:
//!
//! - linear effect = mean over the grid of `|l_k(v) - f_k_bar|`;
//! - non-linear effect = mean over the grid of `|f_k(v) - l_k(v)|`;
//! - pairwise effect of `(k, l)` = mean over the joint grid of the part of the joint partial
//!   dependence that the two centred single-feature curves do not explain.
//!
//! All effects are in the units of the prediction ([`Effect::raw`]); [`Effect::norm`]
//! rescales each family to sum to 1. Features are identified by column index; pairs by the
//! string `"(k, l)"`.
//!
//! ```
//! use openquant::fingerprint::{RegressionModelFingerprint, RegressionPredictor};
//!
//! // 2 x0 is linear, x1^2 is non-linear on a symmetric grid, x0 x2 is a pure interaction.
//! struct Known;
//! impl RegressionPredictor for Known {
//!     fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
//!         x.iter().map(|r| 2.0 * r[0] + r[1] * r[1] + r[0] * r[2]).collect()
//!     }
//! }
//!
//! # fn main() -> Result<(), openquant::fingerprint::FingerprintError> {
//! // Every combination of 11 evenly spaced values in [-1, 1].
//! let grid: Vec<f64> = (0..=10).map(|i| f64::from(i) / 5.0 - 1.0).collect();
//! let mut x = Vec::new();
//! for &a in &grid {
//!     for &b in &grid {
//!         for &c in &grid {
//!             x.push(vec![a, b, c]);
//!         }
//!     }
//! }
//!
//! let mut fingerprint = RegressionModelFingerprint::new();
//! fingerprint.fit(&Known, &x, 11, Some(&[(0, 2), (0, 1)]))?;
//! let (linear, non_linear, pairwise) = fingerprint.get_effects()?;
//! let pairwise = pairwise.expect("pairs were requested");
//!
//! // Mean |2 v| over the grid is 12/11.
//! assert!((linear.raw[&0] - 12.0 / 11.0).abs() < 1e-9);
//! assert!(linear.raw[&1] < 1e-9 && non_linear.norm[&1] > 0.999);
//! assert!(linear.raw[&2] < 1e-9 && non_linear.raw[&2] < 1e-9);
//! // The interaction is x0 x2 itself: mean |v w| = (6/11)^2.
//! assert!((pairwise.raw["(0, 2)"] - 36.0 / 121.0).abs() < 1e-9);
//! assert!(pairwise.raw["(0, 1)"] < 1e-9);
//! # Ok(())
//! # }
//! ```
#![deny(missing_docs)]

use std::collections::BTreeMap;

/// Errors returned by the fingerprint types.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FingerprintError {
    /// `get_effects` or `plot_effects` was called before a successful `fit`.
    #[error("fit must be called before get_effects")]
    NotFitted,
    /// The named input is empty.
    #[error("{0} cannot be empty")]
    Empty(&'static str),
    /// The named parameter violates its requirement.
    #[error("{name} must be {requirement}")]
    Invalid {
        /// Parameter name.
        name: &'static str,
        /// What it must satisfy.
        requirement: &'static str,
    },
    /// The rows of `x` have no columns.
    #[error("x must have at least one feature")]
    NoFeatures,
    /// The rows of `x` differ in length.
    #[error("ragged x rows")]
    RaggedX,
}

/// Per-feature effects, keyed by feature (column) index.
#[derive(Clone, Debug, Default)]
pub struct Effect {
    /// Effects in the units of the prediction.
    pub raw: BTreeMap<usize, f64>,
    /// Effects divided by their sum (all zero if the sum is zero).
    pub norm: BTreeMap<usize, f64>,
}

/// Pairwise-interaction effects, keyed by the string `"(k, l)"`.
#[derive(Clone, Debug, Default)]
pub struct PairwiseEffect {
    /// Effects in the units of the prediction.
    pub raw: BTreeMap<String, f64>,
    /// Effects divided by their sum (all zero if the sum is zero).
    pub norm: BTreeMap<String, f64>,
}

/// A fitted regression model, as seen by [`RegressionModelFingerprint`].
pub trait RegressionPredictor {
    /// Predicts one value per row of `x` (rows are observations, columns features).
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64>;
}

/// A fitted classifier, as seen by [`ClassificationModelFingerprint`].
pub trait ClassificationPredictor {
    /// Predicts one probability per row of `x` (typically of the positive class).
    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64>;
}

/// Fingerprint of a regression model; see the [module docs](self) for the decomposition.
#[derive(Clone, Debug, Default)]
pub struct RegressionModelFingerprint {
    linear_effect: Option<Effect>,
    non_linear_effect: Option<Effect>,
    pair_wise_effect: Option<PairwiseEffect>,
}

/// Fingerprint of a classifier, computed on predicted probabilities, so effects are in
/// probability units and compressed near 0 and 1.
#[derive(Clone, Debug, Default)]
pub struct ClassificationModelFingerprint {
    linear_effect: Option<Effect>,
    non_linear_effect: Option<Effect>,
    pair_wise_effect: Option<PairwiseEffect>,
}

impl RegressionModelFingerprint {
    /// Creates an unfitted fingerprint.
    pub fn new() -> Self {
        Self::default()
    }

    /// Computes the linear, non-linear and (optionally) pairwise effects of `model.predict`
    /// on `x`.
    ///
    /// `x` has one row per observation and one column per feature; `num_values` is the number
    /// of quantile grid points per feature; `pairwise_combinations` lists the feature pairs
    /// `(k, l)` whose interaction to measure. Cost is `features * num_values` predictions
    /// over `x`, plus `num_values^2` per pair.
    ///
    /// # Errors
    ///
    /// - [`FingerprintError::Empty`] if `x` is empty.
    /// - [`FingerprintError::Invalid`] if `num_values < 2`.
    /// - [`FingerprintError::NoFeatures`] if the rows have no columns.
    /// - [`FingerprintError::RaggedX`] if the rows differ in length.
    ///
    /// # Panics
    ///
    /// Panics if a pair names a feature index `>= x[0].len()`.
    pub fn fit<M: RegressionPredictor>(
        &mut self,
        model: &M,
        x: &[Vec<f64>],
        num_values: usize,
        pairwise_combinations: Option<&[(usize, usize)]>,
    ) -> Result<(), FingerprintError> {
        let (lin, nonlin, pair) =
            fit_impl(|data| model.predict(data), x, num_values, pairwise_combinations)?;
        self.linear_effect = Some(lin);
        self.non_linear_effect = Some(nonlin);
        self.pair_wise_effect = pair;
        Ok(())
    }

    /// Returns `(linear, non_linear, pairwise)` from the last [`fit`](Self::fit);
    /// `pairwise` is `None` unless pairs were requested.
    ///
    /// # Errors
    ///
    /// [`FingerprintError::NotFitted`] before a successful fit.
    pub fn get_effects(
        &self,
    ) -> Result<(&Effect, &Effect, Option<&PairwiseEffect>), FingerprintError> {
        let lin = self.linear_effect.as_ref().ok_or(FingerprintError::NotFitted)?;
        let nonlin = self.non_linear_effect.as_ref().ok_or(FingerprintError::NotFitted)?;
        Ok((lin, nonlin, self.pair_wise_effect.as_ref()))
    }

    /// Returns one summary line per effect family (e.g. `"linear:3 features"`); there is no
    /// plotting in the Rust crate.
    ///
    /// # Errors
    ///
    /// [`FingerprintError::NotFitted`] before a successful fit.
    pub fn plot_effects(&self) -> Result<Vec<String>, FingerprintError> {
        let (lin, nonlin, pair) = self.get_effects()?;
        let mut lines = vec![
            format!("linear:{} features", lin.raw.len()),
            format!("nonlinear:{} features", nonlin.raw.len()),
        ];
        if let Some(p) = pair {
            lines.push(format!("pairwise:{} pairs", p.raw.len()));
        }
        Ok(lines)
    }
}

impl ClassificationModelFingerprint {
    /// Creates an unfitted fingerprint.
    pub fn new() -> Self {
        Self::default()
    }

    /// Computes the linear, non-linear and (optionally) pairwise effects of
    /// `model.predict_proba` on `x`.
    ///
    /// `x` has one row per observation and one column per feature; `num_values` is the number
    /// of quantile grid points per feature; `pairwise_combinations` lists the feature pairs
    /// `(k, l)` whose interaction to measure. Cost is `features * num_values` predictions
    /// over `x`, plus `num_values^2` per pair.
    ///
    /// # Errors
    ///
    /// - [`FingerprintError::Empty`] if `x` is empty.
    /// - [`FingerprintError::Invalid`] if `num_values < 2`.
    /// - [`FingerprintError::NoFeatures`] if the rows have no columns.
    /// - [`FingerprintError::RaggedX`] if the rows differ in length.
    ///
    /// # Panics
    ///
    /// Panics if a pair names a feature index `>= x[0].len()`.
    pub fn fit<M: ClassificationPredictor>(
        &mut self,
        model: &M,
        x: &[Vec<f64>],
        num_values: usize,
        pairwise_combinations: Option<&[(usize, usize)]>,
    ) -> Result<(), FingerprintError> {
        let (lin, nonlin, pair) =
            fit_impl(|data| model.predict_proba(data), x, num_values, pairwise_combinations)?;
        self.linear_effect = Some(lin);
        self.non_linear_effect = Some(nonlin);
        self.pair_wise_effect = pair;
        Ok(())
    }

    /// Returns `(linear, non_linear, pairwise)` from the last [`fit`](Self::fit);
    /// `pairwise` is `None` unless pairs were requested.
    ///
    /// # Errors
    ///
    /// [`FingerprintError::NotFitted`] before a successful fit.
    pub fn get_effects(
        &self,
    ) -> Result<(&Effect, &Effect, Option<&PairwiseEffect>), FingerprintError> {
        let lin = self.linear_effect.as_ref().ok_or(FingerprintError::NotFitted)?;
        let nonlin = self.non_linear_effect.as_ref().ok_or(FingerprintError::NotFitted)?;
        Ok((lin, nonlin, self.pair_wise_effect.as_ref()))
    }

    /// Returns one summary line per effect family (e.g. `"linear:3 features"`); there is no
    /// plotting in the Rust crate.
    ///
    /// # Errors
    ///
    /// [`FingerprintError::NotFitted`] before a successful fit.
    pub fn plot_effects(&self) -> Result<Vec<String>, FingerprintError> {
        let (lin, nonlin, pair) = self.get_effects()?;
        let mut lines = vec![
            format!("linear:{} features", lin.raw.len()),
            format!("nonlinear:{} features", nonlin.raw.len()),
        ];
        if let Some(p) = pair {
            lines.push(format!("pairwise:{} pairs", p.raw.len()));
        }
        Ok(lines)
    }
}

fn fit_impl<F>(
    predictor: F,
    x: &[Vec<f64>],
    num_values: usize,
    pairwise_combinations: Option<&[(usize, usize)]>,
) -> Result<(Effect, Effect, Option<PairwiseEffect>), FingerprintError>
where
    F: Fn(&[Vec<f64>]) -> Vec<f64>,
{
    if x.is_empty() {
        return Err(FingerprintError::Empty("x"));
    }
    if num_values < 2 {
        return Err(FingerprintError::Invalid { name: "num_values", requirement: ">= 2" });
    }
    let n_features = x[0].len();
    if n_features == 0 {
        return Err(FingerprintError::NoFeatures);
    }
    if x.iter().any(|r| r.len() != n_features) {
        return Err(FingerprintError::RaggedX);
    }

    let feature_values = get_feature_values(x, num_values);
    let partial_dep = get_individual_partial_dependence(&predictor, x, &feature_values);
    let linear_raw = get_linear_effect(&feature_values, &partial_dep);
    let nonlin_raw = get_non_linear_effect(&feature_values, &partial_dep);
    let linear = Effect { norm: normalize_usize_map(&linear_raw), raw: linear_raw };
    let non_linear = Effect { norm: normalize_usize_map(&nonlin_raw), raw: nonlin_raw };

    let pair = pairwise_combinations.map(|pairs| {
        let raw =
            get_pairwise_effect(pairs, &predictor, x, num_values, &feature_values, &partial_dep);
        PairwiseEffect { norm: normalize_string_map(&raw), raw }
    });

    Ok((linear, non_linear, pair))
}

fn get_feature_values(x: &[Vec<f64>], num_values: usize) -> Vec<Vec<f64>> {
    let n_features = x[0].len();
    let mut out = vec![vec![0.0; num_values]; n_features];
    for j in 0..n_features {
        let mut col: Vec<f64> = x.iter().map(|r| r[j]).collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for (k, q) in (0..num_values).map(|i| i as f64 / (num_values - 1) as f64).enumerate() {
            out[j][k] = quantile_sorted(&col, q);
        }
    }
    out
}

fn get_individual_partial_dependence<F>(
    predictor: &F,
    x: &[Vec<f64>],
    feature_values: &[Vec<f64>],
) -> Vec<Vec<f64>>
where
    F: Fn(&[Vec<f64>]) -> Vec<f64>,
{
    let n_features = x[0].len();
    let num_values = feature_values[0].len();
    let mut out = vec![vec![0.0; num_values]; n_features];
    for j in 0..n_features {
        for (k, &xk) in feature_values[j].iter().enumerate() {
            let mut x_mod = x.to_vec();
            for row in &mut x_mod {
                row[j] = xk;
            }
            let pred = predictor(&x_mod);
            out[j][k] = pred.iter().sum::<f64>() / pred.len() as f64;
        }
    }
    out
}

fn get_linear_effect(
    feature_values: &[Vec<f64>],
    partial_dep: &[Vec<f64>],
) -> BTreeMap<usize, f64> {
    let mut store = BTreeMap::new();
    for j in 0..feature_values.len() {
        let x = &feature_values[j];
        let y = &partial_dep[j];
        let (a, b) = ols_line(x, y);
        let y_mean = y.iter().sum::<f64>() / y.len() as f64;
        let effect = x.iter().map(|v| (a + b * *v - y_mean).abs()).sum::<f64>() / x.len() as f64;
        store.insert(j, effect);
    }
    store
}

fn get_non_linear_effect(
    feature_values: &[Vec<f64>],
    partial_dep: &[Vec<f64>],
) -> BTreeMap<usize, f64> {
    let mut store = BTreeMap::new();
    for j in 0..feature_values.len() {
        let x = &feature_values[j];
        let y = &partial_dep[j];
        let (a, b) = ols_line(x, y);
        let effect = x.iter().zip(y.iter()).map(|(vx, vy)| (a + b * *vx - *vy).abs()).sum::<f64>()
            / x.len() as f64;
        store.insert(j, effect);
    }
    store
}

fn get_pairwise_effect<F>(
    pairs: &[(usize, usize)],
    predictor: &F,
    x: &[Vec<f64>],
    num_values: usize,
    feature_values: &[Vec<f64>],
    partial_dep: &[Vec<f64>],
) -> BTreeMap<String, f64>
where
    F: Fn(&[Vec<f64>]) -> Vec<f64>,
{
    let mut store = BTreeMap::new();
    for &(k, l) in pairs {
        let yk_centered = center(&partial_dep[k]);
        let yl_centered = center(&partial_dep[l]);
        let mut vals = Vec::with_capacity(num_values * num_values);

        for (ik, &xk) in feature_values[k].iter().enumerate() {
            for (il, &xl) in feature_values[l].iter().enumerate() {
                let mut x_mod = x.to_vec();
                for row in &mut x_mod {
                    row[k] = xk;
                    row[l] = xl;
                }
                let ykl = predictor(&x_mod).iter().sum::<f64>() / x.len() as f64;
                vals.push((ykl, yk_centered[ik], yl_centered[il]));
            }
        }

        let mean_ykl = vals.iter().map(|(v, _, _)| *v).sum::<f64>() / vals.len() as f64;
        let mut acc = 0.0;
        for (ykl, yk, yl) in vals {
            acc += (ykl - mean_ykl - yk - yl).abs();
        }
        store.insert(format!("({k}, {l})"), acc / (num_values * num_values) as f64);
    }
    store
}

fn center(v: &[f64]) -> Vec<f64> {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    v.iter().map(|x| *x - m).collect()
}

fn normalize_usize_map(effect: &BTreeMap<usize, f64>) -> BTreeMap<usize, f64> {
    let sum: f64 = effect.values().sum();
    if sum == 0.0 {
        return effect.keys().map(|k| (*k, 0.0)).collect();
    }
    effect.iter().map(|(k, v)| (*k, *v / sum)).collect()
}

fn normalize_string_map(effect: &BTreeMap<String, f64>) -> BTreeMap<String, f64> {
    let sum: f64 = effect.values().sum();
    if sum == 0.0 {
        return effect.keys().map(|k| (k.clone(), 0.0)).collect();
    }
    effect.iter().map(|(k, v)| (k.clone(), *v / sum)).collect()
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let q = q.clamp(0.0, 1.0);
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo])
    }
}

fn ols_line(x: &[f64], y: &[f64]) -> (f64, f64) {
    let mx = x.iter().sum::<f64>() / x.len() as f64;
    let my = y.iter().sum::<f64>() / y.len() as f64;
    let mut cov = 0.0;
    let mut varx = 0.0;
    for (xi, yi) in x.iter().zip(y.iter()) {
        cov += (*xi - mx) * (*yi - my);
        varx += (*xi - mx).powi(2);
    }
    let b = if varx > 0.0 { cov / varx } else { 0.0 };
    let a = my - b * mx;
    (a, b)
}
