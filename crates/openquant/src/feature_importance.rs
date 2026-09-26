//! Feature importance for models validated on purged folds (AFML Chapter 8).
//!
//! - [`mean_decrease_impurity`] (MDI, §8.3.1, Snippet 8.2) — in-sample; aggregates per-tree
//!   impurity importances you computed elsewhere (this crate has no tree learner). Zeros are
//!   treated as missing, as in the snippet.
//! - [`mean_decrease_accuracy`] (MDA, §8.3.2, Snippet 8.3) — out-of-sample; relative loss of
//!   test-fold score when one feature column is shuffled, `(s_k - s_kj) / (s_max - s_kj)` with
//!   `s_max` = 0 for negative log loss and 1 for accuracy/F1.
//! - [`single_feature_importance`] (SFI, §8.4.1, Snippet 8.4) — out-of-sample; the raw
//!   cross-validated score of the model fitted on each feature alone.
//!   [`single_feature_importance_from_proba`] scores out-of-sample probabilities computed
//!   elsewhere, with sample-weighted scoring.
//! - [`get_orthogonal_features`] and [`feature_pca_analysis`] (§8.4.2, Snippets 8.5–8.6) —
//!   PCA-orthogonalised features, and rank correlations between an importance vector and the
//!   PCA loadings as an unsupervised sanity check.
//!
//! Conventions:
//! - Feature matrices are row-major `Vec<Vec<f64>>`: one row per sample, one column per
//!   feature, columns in the order of `feature_names`. Labels are `0.0`/`1.0`.
//! - `splits` are `(train_indices, test_indices)` pairs into the rows, typically from
//!   [`crate::cross_validation::PurgedKFold`].
//! - Results map feature name to [`ImportanceStats`], whose `std` is the **standard error of
//!   the mean**, not a standard deviation: MDI and MDA divide the sample deviation (ddof 1) by
//!   `sqrt(n)`, SFI the population deviation (ddof 0). A standard error from one fold or tree is
//!   0, not `NaN`.
//! - Each method is fooled differently: MDI gives noise features a share, MDA lets correlated
//!   features hide each other (substitution), SFI misses features that matter only jointly.
//!
//! ```
//! use openquant::feature_importance::{mean_decrease_impurity, FeatureImportanceError};
//!
//! # fn main() -> Result<(), FeatureImportanceError> {
//! let per_tree = vec![vec![0.6, 0.3, 0.1], vec![0.5, 0.3, 0.2], vec![0.7, 0.2, 0.1]];
//! let names: Vec<String> = ["f0", "f1", "f2"].map(String::from).to_vec();
//! let mdi = mean_decrease_impurity(&per_tree, &names)?;
//! // Means 0.6, 0.267, 0.133 already sum to 1.
//! assert!((mdi["f0"].mean - 0.6).abs() < 1e-12);
//! // Sample deviation of (0.6, 0.5, 0.7) is 0.1; its standard error over 3 trees is 0.1 / sqrt(3).
//! assert!((mdi["f0"].std - 0.1 / 3f64.sqrt()).abs() < 1e-12);
//! # Ok(())
//! # }
//! ```
#![deny(missing_docs)]

use std::collections::BTreeMap;

use nalgebra::{DMatrix, SymmetricEigen};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::cross_validation::{
    check_prediction_count, check_score_inputs, ml_cross_val_score, CrossValidationError, Scoring,
    SimpleClassifier,
};

/// Errors returned by the feature-importance functions.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FeatureImportanceError {
    /// [`plot_feature_importance`] could not write its CSV; carries the I/O error message.
    #[error("failed to write output file: {0}")]
    WriteOutput(String),
    /// The named input is empty.
    #[error("{0} cannot be empty")]
    Empty(&'static str),
    /// A per-tree importance row does not have one entry per feature name.
    #[error("importance row length mismatch")]
    ImportanceRowLengthMismatch,
    /// The named input does not have one entry per feature.
    #[error("{0} length mismatch")]
    LengthMismatch(&'static str),
    /// PCA feature rows do not all have the same length.
    #[error("ragged feature rows")]
    RaggedFeatureRows,
    /// `x` or `y` is empty.
    #[error("x and y cannot be empty")]
    EmptyXy,
    /// `x` and `y` have different numbers of rows.
    #[error("x/y length mismatch")]
    XyLengthMismatch,
    /// The rows of `x` do not all have the same length.
    #[error("ragged x rows")]
    RaggedX,
    /// `splits`, `sample_weight` or the model's predictions do not fit `x`: a split index is
    /// not a row, `sample_weight` does not have one entry per row, or the model returned the
    /// wrong number of predictions for a test fold. See the wrapped error.
    #[error(transparent)]
    CrossValidation(#[from] CrossValidationError),
}

/// Importance of one feature.
#[derive(Clone, Copy, Debug, Default)]
pub struct ImportanceStats {
    /// Mean importance across trees (MDI) or folds (MDA, SFI).
    pub mean: f64,
    /// **Standard error** of `mean` (despite the name): the deviation divided by `sqrt(n)`.
    pub std: f64,
}

/// Correlations between an importance vector and PCA loadings, from
/// [`feature_pca_analysis`] (AFML Snippet 8.6). Each is 0 where SciPy would return `NaN`
/// because an input is constant.
#[derive(Clone, Copy, Debug, Default)]
pub struct PcaCorrelation {
    /// Pearson correlation of the (repeated) importances with `|eigenvector * eigenvalue|`.
    pub pearson: f64,
    /// Spearman correlation (average ranks for ties), as `scipy.stats.spearmanr`.
    pub spearman: f64,
    /// Kendall tau-b, as `scipy.stats.kendalltau`.
    pub kendall: f64,
    /// `scipy.stats.weightedtau(importance, 1 / pca_rank)` with hyperbolic weights, where
    /// `pca_rank` ranks each feature's summed absolute loading (1 = largest, ties averaged).
    pub weighted_kendall_rank: f64,
}

/// Mean decrease impurity aggregation (AFML §8.3.1, Snippet 8.2).
///
/// `per_tree_importances` holds one row per tree, each with one impurity importance per
/// feature in `feature_names` order (e.g. scikit-learn's `feature_importances_` of every
/// estimator). Following the snippet, **a zero is treated as missing** (the snippet trains with
/// `max_features = 1`, where 0 means "never offered to the tree"); with other settings this
/// inflates the mean, so replace zeros with a tiny positive number if 0 means "useless". Per
/// feature the mean over non-missing trees is taken and its standard error is the sample
/// deviation (ddof 1) times `n_trees^-0.5`, with `n_trees` counting every tree. Means and
/// standard errors are then divided by the sum of the means, so the means sum to 1. A feature
/// that is zero in every tree gets 0; if no mean is positive, everything is 0.
///
/// # Errors
///
/// - [`FeatureImportanceError::Empty`] if `per_tree_importances` or `feature_names` is empty.
/// - [`FeatureImportanceError::ImportanceRowLengthMismatch`] if a row's length differs from
///   `feature_names.len()`.
///
/// ```
/// use openquant::feature_importance::mean_decrease_impurity;
///
/// // The zero is dropped: f0 averages 0.5 over one tree, f1 averages 0.75 over two.
/// let per_tree = vec![vec![0.0, 1.0], vec![0.5, 0.5]];
/// let names: Vec<String> = ["f0", "f1"].map(String::from).to_vec();
/// let mdi = mean_decrease_impurity(&per_tree, &names).unwrap();
/// assert!((mdi["f0"].mean - 0.5 / 1.25).abs() < 1e-12);
/// assert!((mdi["f1"].mean - 0.75 / 1.25).abs() < 1e-12);
/// assert_eq!(mdi["f0"].std, 0.0); // one observation
/// ```
pub fn mean_decrease_impurity(
    per_tree_importances: &[Vec<f64>],
    feature_names: &[String],
) -> Result<BTreeMap<String, ImportanceStats>, FeatureImportanceError> {
    if per_tree_importances.is_empty() {
        return Err(FeatureImportanceError::Empty("per_tree_importances"));
    }
    let n_features = feature_names.len();
    if n_features == 0 {
        return Err(FeatureImportanceError::Empty("feature_names"));
    }
    if per_tree_importances.iter().any(|r| r.len() != n_features) {
        return Err(FeatureImportanceError::ImportanceRowLengthMismatch);
    }

    let mut means = vec![0.0; n_features];
    let mut stderrs = vec![0.0; n_features];
    for j in 0..n_features {
        let col: Vec<f64> = per_tree_importances
            .iter()
            .map(|r| if r[j] == 0.0 { f64::NAN } else { r[j] })
            .collect();
        // Snippet 8.2: pandas `df0.std()`, the sample deviation (ddof = 1).
        let (m, s) = nan_mean_std(&col, 1);
        means[j] = m;
        stderrs[j] = s * (per_tree_importances.len() as f64).powf(-0.5);
    }

    let denom: f64 = means.iter().filter(|v| v.is_finite()).sum();
    let mut out = BTreeMap::new();
    for (j, name) in feature_names.iter().enumerate() {
        let mean = if denom > 0.0 && means[j].is_finite() { means[j] / denom } else { 0.0 };
        let std = if denom > 0.0 && stderrs[j].is_finite() { stderrs[j] / denom } else { 0.0 };
        out.insert(name.clone(), ImportanceStats { mean, std });
    }
    Ok(out)
}

/// Mean decrease accuracy (AFML Snippet 8.3): for each split, fit on the train rows, score the
/// test rows, then score them again with one feature column shuffled; importance is the relative
/// loss of score. Shuffles draw from a `StdRng` seeded with `seed`, so a given seed always
/// gives the same result.
///
/// For each split in `splits` (`(train_indices, test_indices)` into the rows of `x`), `model`
/// is fitted on the training rows (with their `sample_weight`), the test rows are scored, and
/// then each feature column is shuffled in turn within the test rows and scored again. The
/// per-fold importance is `(base - perm) / (0 - perm)` for [`Scoring::NegLogLoss`] and
/// `(base - perm) / (1 - perm)` for [`Scoring::Accuracy`] and [`Scoring::F1`]; it is 0 when
/// the denominator is 0 or the ratio is not finite. Test-fold scores **are** weighted by
/// `sample_weight`. Accuracy and F1 use [`SimpleClassifier::predict`] (so an override is
/// honoured), negative log loss uses [`SimpleClassifier::predict_proba`]. The result is the mean
/// over folds and its standard error (sample deviation, ddof 1, over `sqrt(n_folds)`).
///
/// 1 means damaging the feature destroyed everything the model had, 0 that the model did not
/// need it, negative that it did better without it. Because the shuffle stays within each test
/// fold, a very persistent feature is somewhat understated.
///
/// # Errors
///
/// - [`FeatureImportanceError::EmptyXy`] if `x` or `y` is empty.
/// - [`FeatureImportanceError::XyLengthMismatch`] if `x.len() != y.len()`.
/// - [`FeatureImportanceError::LengthMismatch`] (`"feature_names"`) if the first row of `x`
///   does not have one entry per feature name.
/// - [`FeatureImportanceError::RaggedX`] if the rows of `x` differ in length.
/// - [`FeatureImportanceError::CrossValidation`] wrapping
///   [`CrossValidationError::SplitIndexOutOfRange`] if a split index is not a row of `x`,
///   [`CrossValidationError::LengthMismatch`] if `sample_weight` does not have one entry per
///   row, or [`CrossValidationError::PredictionCountMismatch`] if `model` does not return one
///   prediction per test row.
///
/// ```
/// use openquant::cross_validation::{Scoring, SimpleClassifier};
/// # use openquant::feature_importance::FeatureImportanceError;
///
/// /// A fixed rule: P(y = 1) is the first column. `fit` learns nothing.
/// struct FirstColumn;
/// impl SimpleClassifier for FirstColumn {
///     fn fit(&mut self, _x: &[Vec<f64>], _y: &[f64], _w: Option<&[f64]>) {}
///     fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
///         x.iter().map(|r| r[0]).collect()
///     }
/// }
///
/// let y = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
/// let splits = vec![((4..8).collect::<Vec<_>>(), (0..4).collect()), ((0..4).collect(), (4..8).collect())];
/// let names: Vec<String> = ["signal", "noise"].map(String::from).to_vec();
///
/// let x: Vec<Vec<f64>> = y.iter().enumerate().map(|(i, v)| vec![*v, i as f64 / 8.0]).collect();
/// let mda = openquant::feature_importance::mean_decrease_accuracy(
///     &mut FirstColumn, &x, &y, &names, &splits, None, Scoring::Accuracy, 7,
/// )?;
/// // The model never reads the second column, so shuffling it changes nothing.
/// assert_eq!(mda["noise"].mean, 0.0);
/// assert_eq!(mda["noise"].std, 0.0);
/// // Shuffling the first column costs every correct prediction it moves: (1 - s) / (1 - s) = 1.
/// assert_eq!(mda["signal"].mean, 1.0);
/// # Ok::<(), FeatureImportanceError>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn mean_decrease_accuracy<C: SimpleClassifier>(
    model: &mut C,
    x: &[Vec<f64>],
    y: &[f64],
    feature_names: &[String],
    splits: &[(Vec<usize>, Vec<usize>)],
    sample_weight: Option<&[f64]>,
    scoring: Scoring,
    seed: u64,
) -> Result<BTreeMap<String, ImportanceStats>, FeatureImportanceError> {
    validate_xy(x, y, feature_names)?;
    check_score_inputs(x.len(), y.len(), sample_weight, splits)?;

    let n_features = feature_names.len();
    let mut per_feature = vec![Vec::new(); n_features];
    let mut rng = StdRng::seed_from_u64(seed);

    for (train_idx, test_idx) in splits {
        let x_train = rows(x, train_idx);
        let y_train = vals(y, train_idx);
        let sw_train = sample_weight.map(|sw| vals(sw, train_idx));
        model.fit(&x_train, &y_train, sw_train.as_deref());

        let x_test = rows(x, test_idx);
        let y_test = vals(y, test_idx);
        let sw_test = sample_weight.map(|sw| vals(sw, test_idx));

        let base = score_model(model, &x_test, &y_test, sw_test.as_deref(), scoring)?;

        for (j, scores) in per_feature.iter_mut().enumerate() {
            let mut x_perm = x_test.clone();
            permute_col(&mut x_perm, j, &mut rng);
            let perm = score_model(model, &x_perm, &y_test, sw_test.as_deref(), scoring)?;
            let imp = match scoring {
                Scoring::NegLogLoss => {
                    if -perm == 0.0 {
                        0.0
                    } else {
                        (base - perm) / (-perm)
                    }
                }
                Scoring::Accuracy | Scoring::F1 => {
                    if (1.0 - perm).abs() < 1e-12 {
                        0.0
                    } else {
                        (base - perm) / (1.0 - perm)
                    }
                }
            };
            scores.push(if imp.is_finite() { imp } else { 0.0 });
        }
    }

    Ok(pack_stats(feature_names, &per_feature))
}

/// Single feature importance (AFML §8.4.1, Snippet 8.4): cross-validate `clf` on each feature
/// alone.
///
/// For each feature, `clf` is fitted and scored with
/// [`ml_cross_val_score`] on the one-column matrix over `splits`. The value is the **raw** cross-validated score, not a ratio: for negative
/// log loss compare it with `-ln 2 ≈ -0.693`, a coin flip. `sample_weight` is passed to `fit`
/// only; test folds are scored unweighted. The standard error is the population deviation
/// (ddof 0) over `sqrt(n_folds)`.
///
/// # Errors
///
/// - [`FeatureImportanceError::EmptyXy`] if `x` or `y` is empty.
/// - [`FeatureImportanceError::XyLengthMismatch`] if `x.len() != y.len()`.
/// - [`FeatureImportanceError::LengthMismatch`] (`"feature_names"`) if the first row of `x`
///   does not have one entry per feature name.
/// - [`FeatureImportanceError::RaggedX`] if the rows of `x` differ in length.
/// - [`FeatureImportanceError::CrossValidation`] wrapping the [`ml_cross_val_score`] error if
///   a split index is not a row of `x`, `sample_weight` does not have one entry per row, or
///   `clf` does not return one prediction per test row.
///
/// ```
/// use openquant::cross_validation::{Scoring, SimpleClassifier};
/// # use openquant::feature_importance::FeatureImportanceError;
///
/// /// A fixed rule: P(y = 1) is the first column. `fit` learns nothing.
/// struct FirstColumn;
/// impl SimpleClassifier for FirstColumn {
///     fn fit(&mut self, _x: &[Vec<f64>], _y: &[f64], _w: Option<&[f64]>) {}
///     fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
///         x.iter().map(|r| r[0]).collect()
///     }
/// }
///
/// let y = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
/// let splits = vec![((4..8).collect::<Vec<_>>(), (0..4).collect()), ((0..4).collect(), (4..8).collect())];
/// let names: Vec<String> = ["signal", "noise"].map(String::from).to_vec();
///
/// // Feature 0 is the label itself; feature 1 is its opposite.
/// let x: Vec<Vec<f64>> = y.iter().map(|v| vec![*v, 1.0 - v]).collect();
/// let sfi = openquant::feature_importance::single_feature_importance(
///     &mut FirstColumn, &x, &y, &names, &splits, None, Scoring::Accuracy,
/// )?;
/// assert_eq!(sfi["signal"].mean, 1.0);
/// assert_eq!(sfi["noise"].mean, 0.0);
/// assert_eq!(sfi["signal"].std, 0.0);
/// # Ok::<(), FeatureImportanceError>(())
/// ```
pub fn single_feature_importance<C: SimpleClassifier>(
    clf: &mut C,
    x: &[Vec<f64>],
    y: &[f64],
    feature_names: &[String],
    splits: &[(Vec<usize>, Vec<usize>)],
    sample_weight: Option<&[f64]>,
    scoring: Scoring,
) -> Result<BTreeMap<String, ImportanceStats>, FeatureImportanceError> {
    validate_xy(x, y, feature_names)?;
    let mut out = BTreeMap::new();
    for (j, name) in feature_names.iter().enumerate() {
        let xj: Vec<Vec<f64>> = x.iter().map(|r| vec![r[j]]).collect();
        let scores = ml_cross_val_score(clf, &xj, y, sample_weight, splits, scoring)?;
        // Snippet 8.4 takes `.std()` of the numpy array cvScore returns: ddof = 0.
        let (mean, std) = mean_std(&scores, 0);
        out.insert(
            name.clone(),
            ImportanceStats { mean, std: std * (scores.len() as f64).powf(-0.5) },
        );
    }
    Ok(out)
}

/// Single feature importance from out-of-sample probabilities computed elsewhere, with
/// weighted scoring (AFML §8.4.1, Snippets 8.4 and 7.4).
///
/// `proba[j][i]` is the probability of class 1 for sample `i` from a model fitted on feature
/// `j` alone, on the fold of `splits` whose test set holds `i`. Each fold's test rows are
/// scored as [`mean_decrease_accuracy`] scores them: weighted by `sample_weight` when given
/// (weighted accuracy, weighted mean log loss, F1 from weighted counts), as AFML's `cvScore`
/// passes the weights to the scorer. Accuracy and F1 threshold the probabilities at 0.5. The
/// statistic is otherwise [`single_feature_importance`]'s: the raw mean score over folds, with
/// the population deviation (ddof 0) over `sqrt(n_folds)` as its standard error. Without
/// weights the two functions agree; [`single_feature_importance`] itself passes the weights to
/// `fit` only, which here happened on the caller's side.
///
/// Probabilities are used as given (log loss clips them to `[1e-15, 1 - 1e-15]`); validate
/// them first if they may be out of range.
///
/// # Errors
///
/// - [`FeatureImportanceError::EmptyXy`] if `y` is empty.
/// - [`FeatureImportanceError::LengthMismatch`] (`"proba"`) if `proba` does not have one
///   column per feature name, or a column does not have one value per label.
/// - [`FeatureImportanceError::CrossValidation`] if `sample_weight` does not have one entry
///   per label or a split index is out of range.
///
/// ```
/// use openquant::cross_validation::Scoring;
/// use openquant::feature_importance::single_feature_importance_from_proba;
/// # use openquant::feature_importance::FeatureImportanceError;
///
/// let y = [1.0, 0.0, 1.0, 0.0];
/// let splits = vec![(vec![2, 3], vec![0, 1]), (vec![0, 1], vec![2, 3])];
/// let names = vec!["f".to_string()];
/// // Right on samples 0 and 2, wrong on 1 and 3.
/// let proba = vec![vec![0.9, 0.8, 0.9, 0.8]];
/// let plain =
///     single_feature_importance_from_proba(&y, &proba, &names, &splits, None, Scoring::Accuracy)?;
/// assert_eq!(plain["f"].mean, 0.5);
/// // Weighting the correct samples 3:1 gives 0.75 in each fold.
/// let w = [3.0, 1.0, 3.0, 1.0];
/// let weighted = single_feature_importance_from_proba(
///     &y, &proba, &names, &splits, Some(&w), Scoring::Accuracy,
/// )?;
/// assert_eq!(weighted["f"].mean, 0.75);
/// # Ok::<(), FeatureImportanceError>(())
/// ```
pub fn single_feature_importance_from_proba(
    y: &[f64],
    proba: &[Vec<f64>],
    feature_names: &[String],
    splits: &[(Vec<usize>, Vec<usize>)],
    sample_weight: Option<&[f64]>,
    scoring: Scoring,
) -> Result<BTreeMap<String, ImportanceStats>, FeatureImportanceError> {
    if y.is_empty() {
        return Err(FeatureImportanceError::EmptyXy);
    }
    if proba.len() != feature_names.len() || proba.iter().any(|col| col.len() != y.len()) {
        return Err(FeatureImportanceError::LengthMismatch("proba"));
    }
    check_score_inputs(y.len(), y.len(), sample_weight, splits)?;

    /// Plays back one feature's probabilities; each test row carries its sample index.
    struct Column<'a>(&'a [f64]);
    impl SimpleClassifier for Column<'_> {
        fn fit(&mut self, _x: &[Vec<f64>], _y: &[f64], _sample_weight: Option<&[f64]>) {}
        fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
            x.iter().map(|row| self.0[row[0] as usize]).collect()
        }
    }

    let mut out = BTreeMap::new();
    for (name, column) in feature_names.iter().zip(proba) {
        let model = Column(column);
        let mut scores = Vec::with_capacity(splits.len());
        for (_, test) in splits {
            // Indices up to 2^53 are exact as f64.
            let x_test: Vec<Vec<f64>> = test.iter().map(|&i| vec![i as f64]).collect();
            let y_test = vals(y, test);
            let sw_test = sample_weight.map(|sw| vals(sw, test));
            scores.push(score_model(&model, &x_test, &y_test, sw_test.as_deref(), scoring)?);
        }
        // As in `single_feature_importance`: Snippet 8.4's `.std()` of a numpy array, ddof 0.
        let (mean, std) = mean_std(&scores, 0);
        out.insert(
            name.clone(),
            ImportanceStats { mean, std: std * (scores.len() as f64).powf(-0.5) },
        );
    }
    Ok(out)
}

/// Orthogonal features by PCA (AFML §8.4.2, Snippet 8.5).
///
/// `feature_rows` is row-major (one row per sample). Each column is standardised (population
/// deviation; a constant column becomes 0), the eigenvectors of `Z'Z` are sorted by
/// descending eigenvalue, and the smallest leading set whose cumulative share of the
/// eigenvalues reaches `variance_thresh` is kept (at least one; all of them if
/// `variance_thresh > 1`). Returns `Z` projected onto those eigenvectors: one row per sample,
/// one column per kept component. Eigenvector signs are arbitrary. An empty `feature_rows`
/// returns an empty vector.
///
/// # Errors
///
/// - [`FeatureImportanceError::RaggedFeatureRows`] if the rows differ in length.
/// - [`FeatureImportanceError::Empty`] (`"feature columns"`) if `feature_rows` is non-empty
///   but its rows have no columns.
///
/// ```
/// use openquant::feature_importance::get_orthogonal_features;
///
/// // Two perfectly correlated columns carry one component.
/// let rows = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
/// let pcs = get_orthogonal_features(&rows, 0.95).unwrap();
/// assert!(pcs.iter().all(|r| r.len() == 1));
/// // Standardised columns are (-sqrt(1.5), 0, sqrt(1.5)); the component is their sum / sqrt(2).
/// assert!((pcs[0][0].abs() - 3f64.sqrt()).abs() < 1e-12);
/// assert!(pcs[1][0].abs() < 1e-12);
/// ```
pub fn get_orthogonal_features(
    feature_rows: &[Vec<f64>],
    variance_thresh: f64,
) -> Result<Vec<Vec<f64>>, FeatureImportanceError> {
    if feature_rows.is_empty() {
        return Ok(Vec::new());
    }
    let (_, evec, x_std) = compute_pca(feature_rows, variance_thresh)?;
    Ok((to_dmatrix(&x_std) * evec).row_iter().map(|r| r.iter().copied().collect()).collect())
}

/// Correlate an importance vector with PCA loadings (AFML §8.4.2, Snippet 8.6).
///
/// The PCA is that of [`get_orthogonal_features`] with `variance_thresh`. With `k` kept
/// components, the loadings `|eigenvector_ij * eigenvalue_j|` (all features, component by
/// component) are correlated with `feature_importance_mean` repeated `k` times, giving Pearson,
/// Spearman and Kendall tau-b as `scipy.stats` computes them (average ranks for the ties the
/// repetition creates). The weighted Kendall compares the importances with `1 / pca_rank`, where
/// `pca_rank` ranks each feature's summed absolute loading. Agreement between the supervised
/// and unsupervised rankings is weak evidence that the model has not simply overfit.
///
/// # Errors
///
/// - [`FeatureImportanceError::Empty`] (`"feature_rows"`) if `feature_rows` is empty.
/// - [`FeatureImportanceError::LengthMismatch`] (`"feature_importance_mean"`) if the
///   importance vector does not have one entry per column of the first row.
/// - [`FeatureImportanceError::RaggedFeatureRows`] if the rows differ in length.
/// - [`FeatureImportanceError::Empty`] (`"feature columns"`) if the rows have no columns (and
///   `feature_importance_mean` is empty).
///
/// ```
/// use openquant::feature_importance::feature_pca_analysis;
///
/// // Columns 0 and 1 are identical and orthogonal to column 2: eigenvalues 8, 4, 0.
/// let rows = vec![
///     vec![1.0, 1.0, 1.0],
///     vec![1.0, 1.0, -1.0],
///     vec![-1.0, -1.0, 1.0],
///     vec![-1.0, -1.0, -1.0],
/// ];
/// // A 0.5 threshold keeps only the first component, loadings (8/sqrt 2, 8/sqrt 2, 0).
/// let corr = feature_pca_analysis(&rows, &[0.5, 0.4, 0.1], 0.5).unwrap();
/// assert!((corr.pearson - 0.970725).abs() < 1e-6);
/// assert!((corr.spearman - 0.75f64.sqrt()).abs() < 1e-12);
/// assert!((corr.kendall - 2.0 / 6f64.sqrt()).abs() < 1e-12);
/// assert!((corr.weighted_kendall_rank - 0.768706).abs() < 1e-6);
/// ```
pub fn feature_pca_analysis(
    feature_rows: &[Vec<f64>],
    feature_importance_mean: &[f64],
    variance_thresh: f64,
) -> Result<PcaCorrelation, FeatureImportanceError> {
    if feature_rows.is_empty() {
        return Err(FeatureImportanceError::Empty("feature_rows"));
    }
    let n_features = feature_rows[0].len();
    if feature_importance_mean.len() != n_features {
        return Err(FeatureImportanceError::LengthMismatch("feature_importance_mean"));
    }

    let (eval, evec, _) = compute_pca(feature_rows, variance_thresh)?;

    let pcs = eval.len();
    let mut all_eigs = Vec::with_capacity(n_features * pcs);
    for c in 0..pcs {
        for r in 0..n_features {
            all_eigs.push((evec[(r, c)] * eval[c]).abs());
        }
    }
    let mut repeated_imp = Vec::with_capacity(n_features * pcs);
    for _ in 0..pcs {
        repeated_imp.extend_from_slice(feature_importance_mean);
    }

    let pearson = pearson_corr(&repeated_imp, &all_eigs);
    let spearman = spearman_corr(&repeated_imp, &all_eigs);
    let kendall = kendall_tau(&repeated_imp, &all_eigs);

    let mut pca_strength = vec![0.0; n_features];
    for r in 0..n_features {
        let mut s = 0.0;
        for c in 0..pcs {
            s += (evec[(r, c)] * eval[c]).abs();
        }
        pca_strength[r] = s;
    }
    let pca_rank = rank_desc(&pca_strength);
    let inv_rank: Vec<f64> = pca_rank.iter().map(|r| 1.0 / r).collect();
    let weighted = weighted_kendall_tau(feature_importance_mean, &inv_rank);

    Ok(PcaCorrelation { pearson, spearman, kendall, weighted_kendall_rank: weighted })
}

/// Write importances to a CSV file (the name is mlfinlab's; nothing is plotted).
///
/// With `output_path = Some(path)` it writes `oob_score,<oob>`, `oos_score,<oos>`, a header
/// `feature,mean,std`, and one line per feature in name order, overwriting `path`. With
/// `None` it does nothing.
///
/// # Errors
///
/// [`FeatureImportanceError::WriteOutput`] if the file cannot be written.
pub fn plot_feature_importance(
    importance: &BTreeMap<String, ImportanceStats>,
    oob_score: f64,
    oos_score: f64,
    output_path: Option<&str>,
) -> Result<(), FeatureImportanceError> {
    if let Some(path) = output_path {
        let mut s = format!("oob_score,{oob_score}\noos_score,{oos_score}\nfeature,mean,std\n");
        for (k, v) in importance {
            s.push_str(&format!("{k},{},{}\n", v.mean, v.std));
        }
        std::fs::write(path, s).map_err(|e| FeatureImportanceError::WriteOutput(e.to_string()))?;
    }
    Ok(())
}

/// PCA output: `(eigenvalues, eigenvectors, standardized feature rows)`.
type PcaDecomposition = (Vec<f64>, DMatrix<f64>, Vec<Vec<f64>>);

fn compute_pca(
    feature_rows: &[Vec<f64>],
    variance_thresh: f64,
) -> Result<PcaDecomposition, FeatureImportanceError> {
    if feature_rows.iter().any(|r| r.len() != feature_rows[0].len()) {
        return Err(FeatureImportanceError::RaggedFeatureRows);
    }
    if feature_rows[0].is_empty() {
        return Err(FeatureImportanceError::Empty("feature columns"));
    }
    let x_std = standardize(feature_rows);
    let x = to_dmatrix(&x_std);
    let dot = x.transpose() * &x;
    let eig = SymmetricEigen::new(dot);

    let mut idx: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    idx.sort_by(|&a, &b| {
        eig.eigenvalues[b].partial_cmp(&eig.eigenvalues[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut eval = Vec::with_capacity(idx.len());
    let mut evec_cols = Vec::with_capacity(idx.len());
    for i in idx {
        eval.push(eig.eigenvalues[i]);
        evec_cols.push(eig.eigenvectors.column(i).clone_owned());
    }

    let total: f64 = eval.iter().sum();
    let mut cum = 0.0;
    let mut dim = 0usize;
    if total > 0.0 {
        for (i, v) in eval.iter().enumerate() {
            cum += *v;
            dim = i;
            if cum / total >= variance_thresh {
                break;
            }
        }
    }
    let kept = dim + 1;
    eval.truncate(kept);
    let evec = DMatrix::<f64>::from_columns(&evec_cols[..kept]);
    Ok((eval, evec, x_std))
}

fn validate_xy(
    x: &[Vec<f64>],
    y: &[f64],
    feature_names: &[String],
) -> Result<(), FeatureImportanceError> {
    if x.is_empty() || y.is_empty() {
        return Err(FeatureImportanceError::EmptyXy);
    }
    if x.len() != y.len() {
        return Err(FeatureImportanceError::XyLengthMismatch);
    }
    if x[0].len() != feature_names.len() {
        return Err(FeatureImportanceError::LengthMismatch("feature_names"));
    }
    if x.iter().any(|r| r.len() != x[0].len()) {
        return Err(FeatureImportanceError::RaggedX);
    }
    Ok(())
}

fn rows(x: &[Vec<f64>], idx: &[usize]) -> Vec<Vec<f64>> {
    idx.iter().map(|i| x[*i].clone()).collect()
}

fn vals(v: &[f64], idx: &[usize]) -> Vec<f64> {
    idx.iter().map(|i| v[*i]).collect()
}

fn score_model<C: SimpleClassifier>(
    model: &C,
    x_test: &[Vec<f64>],
    y_test: &[f64],
    sample_weight: Option<&[f64]>,
    scoring: Scoring,
) -> Result<f64, CrossValidationError> {
    Ok(match scoring {
        Scoring::Accuracy => {
            let pred = model.predict(x_test);
            check_prediction_count(y_test.len(), pred.len())?;
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..y_test.len() {
                let w = sample_weight.map(|sw| sw[i]).unwrap_or(1.0);
                den += w;
                if (pred[i] - y_test[i]).abs() < 1e-12 {
                    num += w;
                }
            }
            if den > 0.0 {
                num / den
            } else {
                0.0
            }
        }
        Scoring::NegLogLoss => {
            let probs = model.predict_proba(x_test);
            check_prediction_count(y_test.len(), probs.len())?;
            let mut loss = 0.0;
            let mut den = 0.0;
            let eps = 1e-15;
            for i in 0..y_test.len() {
                let w = sample_weight.map(|sw| sw[i]).unwrap_or(1.0);
                let p = probs[i].clamp(eps, 1.0 - eps);
                loss += w * (-(y_test[i] * p.ln() + (1.0 - y_test[i]) * (1.0 - p).ln()));
                den += w;
            }
            if den > 0.0 {
                -(loss / den)
            } else {
                0.0
            }
        }
        Scoring::F1 => {
            let pred = model.predict(x_test);
            check_prediction_count(y_test.len(), pred.len())?;
            let mut tp = 0.0;
            let mut fp = 0.0;
            let mut fnn = 0.0;
            for i in 0..y_test.len() {
                let w = sample_weight.map(|sw| sw[i]).unwrap_or(1.0);
                let p_pos = pred[i] > 0.5;
                let y_pos = y_test[i] > 0.5;
                if p_pos && y_pos {
                    tp += w;
                } else if p_pos && !y_pos {
                    fp += w;
                } else if !p_pos && y_pos {
                    fnn += w;
                }
            }
            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let recall = if tp + fnn > 0.0 { tp / (tp + fnn) } else { 0.0 };
            if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            }
        }
    })
}

/// Shuffles one column of `x` in place, as AFML Snippet 8.3 does with `np.random.shuffle`.
/// A shuffle (unlike a rotation) breaks the feature-label link however persistent the feature is.
fn permute_col(x: &mut [Vec<f64>], col: usize, rng: &mut StdRng) {
    let mut values: Vec<f64> = x.iter().map(|row| row[col]).collect();
    values.shuffle(rng);
    for (row, v) in x.iter_mut().zip(values) {
        row[col] = v;
    }
}

fn pack_stats(feature_names: &[String], values: &[Vec<f64>]) -> BTreeMap<String, ImportanceStats> {
    let mut out = BTreeMap::new();
    for (j, name) in feature_names.iter().enumerate() {
        // Snippet 8.3: `imp.std()` on a pandas DataFrame, the sample deviation (ddof = 1).
        let (m, s) = mean_std(&values[j], 1);
        let mean = if m.is_finite() { m } else { 0.0 };
        let std = if s.is_finite() { s * (values[j].len() as f64).powf(-0.5) } else { 0.0 };
        out.insert(name.clone(), ImportanceStats { mean, std });
    }
    out
}

fn nan_mean_std(v: &[f64], ddof: usize) -> (f64, f64) {
    let vals: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    mean_std(&vals, ddof)
}

/// Mean and standard deviation with `ddof` delta degrees of freedom (divide by `n - ddof`).
/// The deviation is 0 when there are no more than `ddof` values (pandas would give NaN).
fn mean_std(v: &[f64], ddof: usize) -> (f64, f64) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    if v.len() <= ddof {
        return (mean, 0.0);
    }
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len() - ddof) as f64;
    (mean, var.sqrt())
}

fn standardize(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = rows.len();
    let m = rows[0].len();
    let mut means = vec![0.0; m];
    for r in rows {
        for j in 0..m {
            means[j] += r[j];
        }
    }
    for v in &mut means {
        *v /= n as f64;
    }
    let mut stds = vec![0.0; m];
    for r in rows {
        for j in 0..m {
            stds[j] += (r[j] - means[j]).powi(2);
        }
    }
    for s in &mut stds {
        *s = (*s / n as f64).sqrt();
    }

    rows.iter()
        .map(|r| {
            (0..m).map(|j| if stds[j] > 0.0 { (r[j] - means[j]) / stds[j] } else { 0.0 }).collect()
        })
        .collect()
}

fn to_dmatrix(rows: &[Vec<f64>]) -> DMatrix<f64> {
    let n = rows.len();
    let m = rows[0].len();
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    DMatrix::<f64>::from_row_slice(n, m, &flat)
}

fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }
    let mx = x.iter().sum::<f64>() / x.len() as f64;
    let my = y.iter().sum::<f64>() / y.len() as f64;
    let mut num = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        num += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx == 0.0 || vy == 0.0 {
        0.0
    } else {
        num / (vx.sqrt() * vy.sqrt())
    }
}

/// Ranks with 1 for the largest value; tied values share the average of their ranks, as
/// pandas `rank(ascending=False)` and `scipy.stats.rankdata(-v)` do.
fn rank_desc(values: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut rank = vec![0.0; values.len()];
    let mut first = 0;
    while first < idx.len() {
        let mut last = first;
        while last + 1 < idx.len() && values[idx[last + 1]] == values[idx[first]] {
            last += 1;
        }
        // positions first..=last hold ranks first+1 ..= last+1
        let avg = (first + last) as f64 / 2.0 + 1.0;
        for i in &idx[first..=last] {
            rank[*i] = avg;
        }
        first = last + 1;
    }
    rank
}

/// `scipy.stats.spearmanr`: Pearson correlation of average ranks.
fn spearman_corr(x: &[f64], y: &[f64]) -> f64 {
    pearson_corr(&rank_desc(x), &rank_desc(y))
}

/// `scipy.stats.kendalltau` (the default tau-b): `(C - D) / sqrt((P - T_x) (P - T_y))`, where
/// `P` counts all pairs and `T_x`, `T_y` the pairs tied in x and in y. 0 when either input is
/// constant (scipy returns NaN).
fn kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let (mut s, mut untied_x, mut untied_y) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        for j in (i + 1)..x.len() {
            let sx = sign(x[i] - x[j]);
            let sy = sign(y[i] - y[j]);
            s += sx * sy;
            untied_x += sx.abs();
            untied_y += sy.abs();
        }
    }
    if untied_x == 0.0 || untied_y == 0.0 {
        0.0
    } else {
        s / (untied_x.sqrt() * untied_y.sqrt())
    }
}

/// `scipy.stats.weightedtau` with its defaults (AFML Snippet 8.6): Vigna's weighted tau with
/// additive hyperbolic weights, averaged over ranking the elements by `(x, y)` and by `(y, x)`.
/// 0 when either input is constant (scipy returns NaN).
fn weighted_kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    match (weighted_tau_ranked(x, y), weighted_tau_ranked(y, x)) {
        (Some(a), Some(b)) => (a + b) / 2.0,
        _ => 0.0,
    }
}

/// One half of `weightedtau`: the element with the largest `x` (ties broken by larger `y`, then
/// by larger index, as scipy's reversed `lexsort` does) has rank 0 and weight 1, the next
/// weight 1/2, and so on. A pair weighs the sum of its two elements' weights. The result is
/// `sum_pairs w * sgn(dx) * sgn(dy) / sqrt(sum_{dx != 0} w * sum_{dy != 0} w)`.
fn weighted_tau_ranked(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        x[a].partial_cmp(&x[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(y[a].partial_cmp(&y[b]).unwrap_or(std::cmp::Ordering::Equal))
            .then(a.cmp(&b))
    });
    let mut weight = vec![0.0; n];
    for (rank, i) in order.iter().rev().enumerate() {
        weight[*i] = 1.0 / (rank as f64 + 1.0);
    }

    let (mut s, mut untied_x, mut untied_y) = (0.0, 0.0, 0.0);
    for i in 0..n {
        for j in (i + 1)..n {
            let w = weight[i] + weight[j];
            let sx = sign(x[i] - x[j]);
            let sy = sign(y[i] - y[j]);
            s += w * sx * sy;
            untied_x += w * sx.abs();
            untied_y += w * sy.abs();
        }
    }
    if untied_x == 0.0 || untied_y == 0.0 {
        return None;
    }
    Some((s / (untied_x.sqrt() * untied_y.sqrt())).clamp(-1.0, 1.0))
}

/// -1, 0 or 1. Unlike `f64::signum`, 0.0 maps to 0 so that ties count as ties.
fn sign(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Both inputs have ties: x at 0.1 and 0.3, y at 2.0 and 3.0. Expected values from scipy 1.18:
    //   spearmanr(x, y) = 0.5454545454545454
    //   kendalltau(x, y) = 0.3846153846153847   (tau-b: S = 5 over 15 pairs, 2 tied in each)
    //   weightedtau(x, y) = 0.23914127716864048
    const X: [f64; 6] = [0.3, 0.1, 0.3, 0.2, 0.1, 0.4];
    const Y: [f64; 6] = [2.0, 1.0, 3.0, 3.0, 0.5, 2.0];

    #[test]
    fn average_ranks_for_ties() {
        assert_eq!(rank_desc(&X), vec![2.5, 5.5, 2.5, 4.0, 5.5, 1.0]);
    }

    #[test]
    fn rank_correlations_match_scipy_with_ties() {
        assert!((spearman_corr(&X, &Y) - 0.5454545454545454).abs() < 1e-12);
        assert!((kendall_tau(&X, &Y) - 5.0 / 13.0).abs() < 1e-12);
        assert!((weighted_kendall_tau(&X, &Y) - 0.23914127716864048).abs() < 1e-12);
    }

    #[test]
    fn rank_correlations_of_a_constant_are_zero() {
        let c = [1.0; 6];
        assert_eq!(kendall_tau(&c, &Y), 0.0);
        assert_eq!(weighted_kendall_tau(&c, &Y), 0.0);
    }
}
