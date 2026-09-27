//! Value tests for `feature_importance`. Expected numbers are hand-worked from AFML snippets
//! 8.2-8.4 in the comments, or come from `tests/fixtures/feature_importance/pca_reference.json`
//! (written by `generate.py` next to it, using numpy + scipy.stats only).
//!
//! The pre-existing `tests/feature_importance.rs` asserts orderings and `is_finite()`; see
//! `docs/test-sensitivity-audit.md`.

use openquant::cross_validation::{Scoring, SimpleClassifier};
use openquant::feature_importance::{
    feature_pca_analysis, get_orthogonal_features, mean_decrease_accuracy, mean_decrease_impurity,
    single_feature_importance,
};
use serde::Deserialize;
use std::path::Path;

fn names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("f{i}")).collect()
}

/// P(y = 1) = 0.9 when the FIRST column it is shown is positive, else 0.1. `fit` is a no-op, so
/// every score below can be worked out by hand.
struct SignOfFirstColumn;

impl SimpleClassifier for SignOfFirstColumn {
    fn fit(&mut self, _x: &[Vec<f64>], _y: &[f64], _sample_weight: Option<&[f64]>) {}
    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|r| if r[0] > 0.0 { 0.9 } else { 0.1 }).collect()
    }
}

// ---------------------------------------------------------------------------------------------
// MDI (AFML snippet 8.2)
// ---------------------------------------------------------------------------------------------

/// Three trees, two features:
///     tree 0: (0.6, 0.4)   tree 1: (0.8, 0.2)   tree 2: (0.7, 0.3)
/// Column means (0.7, 0.3) already sum to one, so the normalised means are (0.7, 0.3).
#[test]
fn mdi_means_hand_worked() {
    let imp = vec![vec![0.6, 0.4], vec![0.8, 0.2], vec![0.7, 0.3]];
    let mdi = mean_decrease_impurity(&imp, &names(2)).unwrap();
    assert!((mdi["f0"].mean - 0.7).abs() < 1e-15, "{}", mdi["f0"].mean);
    assert!((mdi["f1"].mean - 0.3).abs() < 1e-15, "{}", mdi["f1"].mean);
}

/// A zero importance means "this tree never split on the feature" and is excluded from that
/// feature's mean (snippet 8.2: `df0.replace(0, np.nan)`).
///     tree 0: (1.0, 0.0)   tree 1: (0.5, 0.5)
/// f0 mean = 0.75, f1 mean = 0.5 (one tree only); normalised by 1.25 -> (0.6, 0.4).
#[test]
fn mdi_ignores_zero_importances_hand_worked() {
    let imp = vec![vec![1.0, 0.0], vec![0.5, 0.5]];
    let mdi = mean_decrease_impurity(&imp, &names(2)).unwrap();
    assert!((mdi["f0"].mean - 0.6).abs() < 1e-15, "{}", mdi["f0"].mean);
    assert!((mdi["f1"].mean - 0.4).abs() < 1e-15, "{}", mdi["f1"].mean);
}

/// Snippet 8.2 reports `df0.std() * df0.shape[0] ** -0.5`, and `df0` is a pandas DataFrame, whose
/// `.std()` is the SAMPLE standard deviation (ddof = 1). For the three trees above each column
/// has deviations (-0.1, 0.1, 0.0):
///     sample std = sqrt(0.02 / 2) = 0.1,   standard error = 0.1 / sqrt(3) = 0.0577350269...
/// (A population std would give sqrt(0.02 / 3) / sqrt(3) = 0.0471404521...)
#[test]
fn mdi_standard_error_uses_sample_std_hand_worked() {
    let imp = vec![vec![0.6, 0.4], vec![0.8, 0.2], vec![0.7, 0.3]];
    let mdi = mean_decrease_impurity(&imp, &names(2)).unwrap();
    let want = 0.1 / 3f64.sqrt();
    assert!((mdi["f0"].std - want).abs() < 1e-12, "{} vs {want}", mdi["f0"].std);
    assert!((mdi["f1"].std - want).abs() < 1e-12, "{} vs {want}", mdi["f1"].std);
}

/// Convention-free facts about the same three trees: the two columns are mirror images
/// (0.6/0.4, 0.8/0.2, 0.7/0.3) so their standard errors are equal, and the value must lie between
/// the ddof=0 answer (0.04714...) and the ddof=1 answer (0.05773...) derived above. The bracket
/// comes from those two closed forms, not from the library's output; it rejects e.g. a missing
/// 1/sqrt(n_trees) factor (0.0816 or 0.1) or a 1/n_trees factor (0.0272).
#[test]
fn mdi_standard_error_is_symmetric_for_mirrored_columns() {
    let imp = vec![vec![0.6, 0.4], vec![0.8, 0.2], vec![0.7, 0.3]];
    let mdi = mean_decrease_impurity(&imp, &names(2)).unwrap();
    assert!(mdi["f0"].std > 0.0);
    assert!((mdi["f0"].std - mdi["f1"].std).abs() < 1e-15);
    assert!(mdi["f0"].std > 0.047 && mdi["f0"].std < 0.058, "{}", mdi["f0"].std);
}

// ---------------------------------------------------------------------------------------------
// MDA (AFML snippet 8.3) and SFI (snippet 8.4)
// ---------------------------------------------------------------------------------------------

type Splits = Vec<(Vec<usize>, Vec<usize>)>;

/// MDA shuffles each test-fold column with a seeded RNG (snippet 8.3's `np.random.shuffle`). In
/// the data below every test fold has two rows, so a shuffle either swaps a column's two values
/// or leaves them in place, and only one of those choices changes any score: whether fold A's f0
/// is swapped (fold B scores 0 for f0 in either order, and f1 is never read). So under every
/// seed each MDA below takes one of two hand-worked values, and which one depends only on that
/// coin flip. The tests check every seed in `SEEDS` against the value for its outcome and that
/// both outcomes occur, which keeps them independent of the random stream: they used to pin one
/// seed picked for rand 0.8's shuffle, and rand 0.9 changed the shuffle (#219).
const SEEDS: std::ops::Range<u64> = 0..32;

/// Two folds whose test sets have two rows each, so "permute the column" can only mean "swap the
/// two values" or "leave them".
///
///            f0   f1   y
///   row 0    +1   -1   1      fold A tests rows 0,1
///   row 1    -1   +1   0
///   row 2    +1   +1   1      fold B tests rows 2,3
///   row 3    -1   +1   1
///
/// The classifier looks at f0 only.
fn mda_data() -> (Vec<Vec<f64>>, Vec<f64>, Splits) {
    let x = vec![vec![1.0, -1.0], vec![-1.0, 1.0], vec![1.0, 1.0], vec![-1.0, 1.0]];
    let y = vec![1.0, 0.0, 1.0, 1.0];
    let splits = vec![(vec![2, 3], vec![0, 1]), (vec![0, 1], vec![2, 3])];
    (x, y, splits)
}

/// `(mean, std)` of f0's and f1's MDA under each seed of `SEEDS`.
fn mda_by_seed(scoring: Scoring) -> Vec<((f64, f64), (f64, f64))> {
    let (x, y, splits) = mda_data();
    SEEDS
        .map(|seed| {
            let mda = mean_decrease_accuracy(
                &mut SignOfFirstColumn,
                &x,
                &y,
                &names(2),
                &splits,
                None,
                scoring,
                seed,
            )
            .unwrap();
            ((mda["f0"].mean, mda["f0"].std), (mda["f1"].mean, mda["f1"].std))
        })
        .collect()
}

/// Checks that every seed gives one of the two hand-worked outcomes and that both occur.
fn assert_two_outcomes(got: &[f64], unswapped: f64, swapped: f64, tol: f64) {
    for (seed, g) in SEEDS.zip(got) {
        assert!(
            (g - unswapped).abs() < tol || (g - swapped).abs() < tol,
            "seed {seed}: {g}, expected {unswapped} or {swapped}"
        );
    }
    assert!(got.iter().any(|g| (g - swapped).abs() < tol), "no seed swapped fold A: {got:?}");
    assert!(got.iter().any(|g| (g - unswapped).abs() < tol), "every seed swapped: {got:?}");
}

/// Accuracy scoring, importance = (base - permuted) / (1 - permuted):
///   fold A: base predictions (1,0) vs y (1,0) -> 1.0
///           f0 swapped -> predictions (0,1) -> 0.0: importance(f0) = (1 - 0) / (1 - 0) = 1
///           f0 in place -> 1.0: denominator 0, importance 0 (as documented)
///   fold B: base predictions (1,0) vs y (1,1) -> 0.5; f0 either way -> 0.5
///           importance(f0) = 0 / 0.5 = 0
///   mean importance(f0) = 0.5 if fold A was swapped, else 0. f1 is never read, so its
///   importance is exactly 0 in both folds.
#[test]
fn mda_accuracy_hand_worked() {
    let runs = mda_by_seed(Scoring::Accuracy);
    let f0: Vec<f64> = runs.iter().map(|r| r.0 .0).collect();
    assert_two_outcomes(&f0, 0.0, 0.5, 1e-15);
    for (f0, f1) in runs {
        assert!(f0.0.is_finite());
        assert_eq!(f1, (0.0, 0.0));
    }
}

/// Negative log-loss scoring, importance = (base - permuted) / (-permuted), i.e. with
/// L = -score: (L_perm - L_base) / L_perm.
///   fold A: base probabilities (0.9, 0.1) vs y (1,0): L_base = -ln 0.9
///           swapped          (0.1, 0.9) vs y (1,0): L_perm = -ln 0.1
///           importance(f0) = 1 - ln 0.9 / ln 0.1 (0 if left in place)
///   fold B: y = (1,1), so swapping the two probabilities leaves the loss unchanged: 0
///   mean importance(f0) = (1 - ln 0.9 / ln 0.1) / 2 = 0.4771212547... if fold A was swapped
#[test]
fn mda_neg_log_loss_hand_worked() {
    let runs = mda_by_seed(Scoring::NegLogLoss);
    let want = (1.0 - 0.9f64.ln() / 0.1f64.ln()) / 2.0;
    let f0: Vec<f64> = runs.iter().map(|r| r.0 .0).collect();
    // two logs, a mean of two and a ratio: rounding only
    assert_two_outcomes(&f0, 0.0, want, 1e-14);
    assert!(runs.iter().all(|r| r.1 .0 == 0.0));
}

/// Snippet 8.3 ends with `imp.std() * imp.shape[0] ** -0.5` on a pandas DataFrame (ddof = 1).
/// When fold A is swapped the accuracy importances of f0 over the two folds are (1, 0): sample
/// std = sqrt(0.5), standard error = sqrt(0.5) / sqrt(2) = 0.5. (Population std would give
/// 0.5 / sqrt(2) = 0.3536.) Unswapped, they are (0, 0) and the standard error is 0.
#[test]
fn mda_standard_error_uses_sample_std_hand_worked() {
    let runs = mda_by_seed(Scoring::Accuracy);
    let std: Vec<f64> = runs.iter().map(|r| r.0 .1).collect();
    assert_two_outcomes(&std, 0.0, 0.5, 1e-12);
    // The standard error goes with the mean: 0.5 exactly when the mean is 0.5.
    for (seed, r) in SEEDS.zip(&runs) {
        assert_eq!(r.0 .0 == 0.5, (r.0 .1 - 0.5).abs() < 1e-12, "seed {seed}: {:?}", r.0);
    }
}

/// SFI scores each feature alone (the classifier is shown a one-column matrix).
///   f0 alone: fold A accuracy 1.0, fold B accuracy 0.5      -> mean 0.75
///   f1 alone: fold A column (-1,+1) -> predictions (0,1) vs y (1,0) -> 0.0
///             fold B column (+1,+1) -> predictions (1,1) vs y (1,1) -> 1.0 -> mean 0.5
/// Snippet 8.4 takes `.std()` of the numpy array returned by cvScore (ddof = 0) times n^-0.5:
///   f0: std 0.25 -> 0.25 / sqrt(2);   f1: std 0.5 -> 0.5 / sqrt(2)
#[test]
fn sfi_accuracy_hand_worked() {
    let (x, y, splits) = mda_data();
    let sfi = single_feature_importance(
        &mut SignOfFirstColumn,
        &x,
        &y,
        &names(2),
        &splits,
        None,
        Scoring::Accuracy,
    )
    .unwrap();
    assert!((sfi["f0"].mean - 0.75).abs() < 1e-15);
    assert!((sfi["f1"].mean - 0.5).abs() < 1e-15);
    assert!((sfi["f0"].std - 0.25 / 2f64.sqrt()).abs() < 1e-15, "{}", sfi["f0"].std);
    assert!((sfi["f1"].std - 0.5 / 2f64.sqrt()).abs() < 1e-15, "{}", sfi["f1"].std);
}

// ---------------------------------------------------------------------------------------------
// Orthogonal features (AFML snippet 8.5)
// ---------------------------------------------------------------------------------------------

/// Two features with a known correlation. x = (1,2,3,4), y = (1,3,2,4): deviations
/// (-1.5,-0.5,0.5,1.5) and (-1.5,0.5,-0.5,1.5), cross product 4, sums of squares 5 and 5, so
/// rho = 0.8. For standardised Z the matrix Z'Z is proportional to [[1, rho], [rho, 1]] with
/// eigenvalues proportional to (1 + rho, 1 - rho) = (1.8, 0.2) and eigenvectors (1,1)/sqrt2 and
/// (1,-1)/sqrt2. Hence, independent of the ddof used to standardise and of eigenvector signs:
///   * PC1 is proportional to zx + zy, i.e. to (-3, 0, 0, 3):   PC1 = (-c, 0, 0, c)
///   * PC2 is proportional to zx - zy, i.e. to (0, -1, 1, 0):   PC2 = (0, -d, d, 0)
///   * |PC1|^2 / |PC2|^2 = 1.8 / 0.2 = 9
///   * PC1 explains 90% of the variance: a 0.85 threshold keeps one column, 0.95 keeps two.
#[test]
fn orthogonal_features_two_correlated_columns_closed_form() {
    let rows: Vec<Vec<f64>> = [(1.0, 1.0), (2.0, 3.0), (3.0, 2.0), (4.0, 4.0)]
        .iter()
        .map(|(a, b)| vec![*a, *b])
        .collect();

    let both = get_orthogonal_features(&rows, 0.95).unwrap();
    assert_eq!(both[0].len(), 2);
    let pc1: Vec<f64> = both.iter().map(|r| r[0]).collect();
    let pc2: Vec<f64> = both.iter().map(|r| r[1]).collect();
    // a 2x2 symmetric eigenproblem on O(1) numbers: 1e-12 is far above its rounding error
    let tol = 1e-12;
    assert!(pc1[1].abs() < tol && pc1[2].abs() < tol, "{pc1:?}");
    assert!((pc1[0] + pc1[3]).abs() < tol && pc1[3].abs() > 1.0, "{pc1:?}");
    assert!(pc2[0].abs() < tol && pc2[3].abs() < tol, "{pc2:?}");
    assert!((pc2[1] + pc2[2]).abs() < tol && pc2[1].abs() > 0.1, "{pc2:?}");
    let ss = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
    assert!((ss(&pc1) / ss(&pc2) - 9.0).abs() < 1e-10, "ratio {}", ss(&pc1) / ss(&pc2));

    let one = get_orthogonal_features(&rows, 0.85).unwrap();
    assert_eq!(one[0].len(), 1, "90% explained by PC1 must satisfy an 85% threshold");
}

// ---------------------------------------------------------------------------------------------
// Feature importance vs PCA (AFML 8.4.2), against scipy.stats
// ---------------------------------------------------------------------------------------------

#[derive(Deserialize)]
struct PcaCase {
    variance_thresh: f64,
    pearson: f64,
    spearman: f64,
    kendall: f64,
    weighted_kendall_rank: f64,
}

#[derive(Deserialize)]
struct PcaReference {
    x: Vec<Vec<f64>>,
    importance: Vec<f64>,
    one_component: PcaCase,
    several_components: PcaCase,
}

fn pca_reference() -> PcaReference {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/feature_importance/pca_reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// Tolerance against scipy: both sides solve a 5x5 symmetric eigenproblem (LAPACK vs nalgebra's
/// Jacobi-style solver) on a well-separated spectrum; eigenvectors agree to ~1e-13 and Pearson
/// is a smooth function of them. 1e-9 leaves four decades of slack over that and is still far
/// below any real discrepancy (a wrong formula moves these by 1e-2 or more).
const SCIPY_TOL: f64 = 1e-9;

#[test]
fn pca_pearson_matches_scipy() {
    let r = pca_reference();
    for case in [&r.one_component, &r.several_components] {
        let got = feature_pca_analysis(&r.x, &r.importance, case.variance_thresh).unwrap();
        assert!(
            (got.pearson - case.pearson).abs() < SCIPY_TOL,
            "thresh {}: pearson {} vs scipy {}",
            case.variance_thresh,
            got.pearson,
            case.pearson
        );
    }
}

/// With one component kept there are no ties, so Spearman and Kendall are unambiguous.
#[test]
fn pca_rank_correlations_match_scipy_without_ties() {
    let r = pca_reference();
    let case = &r.one_component;
    let got = feature_pca_analysis(&r.x, &r.importance, case.variance_thresh).unwrap();
    assert!(
        (got.spearman - case.spearman).abs() < SCIPY_TOL,
        "{} vs {}",
        got.spearman,
        case.spearman
    );
    assert!((got.kendall - case.kendall).abs() < SCIPY_TOL, "{} vs {}", got.kendall, case.kendall);
}

/// With several components the importance vector is tiled and therefore tied. scipy (and
/// mlfinlab, which calls it) uses average ranks for Spearman and tau-b for Kendall.
#[test]
fn pca_rank_correlations_match_scipy_with_ties() {
    let r = pca_reference();
    let case = &r.several_components;
    let got = feature_pca_analysis(&r.x, &r.importance, case.variance_thresh).unwrap();
    assert!(
        (got.spearman - case.spearman).abs() < SCIPY_TOL,
        "{} vs {}",
        got.spearman,
        case.spearman
    );
    assert!((got.kendall - case.kendall).abs() < SCIPY_TOL, "{} vs {}", got.kendall, case.kendall);
}

/// AFML 8.4.2 / mlfinlab use scipy.stats.weightedtau (Vigna's additive hyperbolic weighting by
/// rank).
#[test]
fn pca_weighted_kendall_matches_scipy_weightedtau() {
    let r = pca_reference();
    for case in [&r.one_component, &r.several_components] {
        let got = feature_pca_analysis(&r.x, &r.importance, case.variance_thresh).unwrap();
        assert!(
            (got.weighted_kendall_rank - case.weighted_kendall_rank).abs() < SCIPY_TOL,
            "thresh {}: {} vs scipy {}",
            case.variance_thresh,
            got.weighted_kendall_rank,
            case.weighted_kendall_rank
        );
    }
}
