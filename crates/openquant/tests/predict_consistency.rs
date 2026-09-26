//! `SimpleClassifier::predict` is the hard-label rule everywhere (#186 item 28).
//!
//! Mean decrease accuracy scored accuracy and F1 with `predict`, but `ml_cross_val_score`, and
//! through it single feature importance, thresholded `predict_proba` at 0.5 and ignored an
//! overridden `predict`.

use openquant::cross_validation::{ml_cross_val_score, Scoring, SimpleClassifier};
use openquant::feature_importance::single_feature_importance;

/// Low probabilities, but a decision rule that always says 1 (as with a tuned threshold).
struct LowProbAlwaysPositive;

impl SimpleClassifier for LowProbAlwaysPositive {
    fn fit(&mut self, _x: &[Vec<f64>], _y: &[f64], _sample_weight: Option<&[f64]>) {}
    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        vec![0.2; x.len()]
    }
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        vec![1.0; x.len()]
    }
}

#[test]
fn hard_label_scores_use_an_overridden_predict() {
    let x: Vec<Vec<f64>> = (0..8).map(|i| vec![f64::from(i), f64::from(i % 3)]).collect();
    let y = vec![1.0; 8];
    let splits =
        vec![((0..4).collect::<Vec<_>>(), (4..8).collect()), ((4..8).collect(), (0..4).collect())];
    let clf = &mut LowProbAlwaysPositive;

    // `predict` says 1 for every row and every label is 1: perfect accuracy and F1. With the
    // old 0.5 threshold on `predict_proba` both were 0.
    for scoring in [Scoring::Accuracy, Scoring::F1] {
        assert_eq!(ml_cross_val_score(clf, &x, &y, None, &splits, scoring).unwrap(), [1.0, 1.0]);
        let names = vec!["a".to_string(), "b".to_string()];
        let sfi = single_feature_importance(clf, &x, &y, &names, &splits, None, scoring).unwrap();
        assert!(sfi.values().all(|s| s.mean == 1.0));
    }

    // Log loss still uses the probabilities: -ln(0.2) per row.
    let nll = ml_cross_val_score(clf, &x, &y, None, &splits, Scoring::NegLogLoss).unwrap();
    assert!(nll.iter().all(|s| (s - 0.2f64.ln()).abs() < 1e-12));
}
