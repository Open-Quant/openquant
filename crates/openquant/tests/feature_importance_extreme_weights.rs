use openquant::cross_validation::Scoring;
use openquant::feature_importance::single_feature_importance_from_proba;

#[test]
fn weighted_scores_are_invariant_to_equal_positive_weight_scale() {
    let y = [1.0, 0.0, 1.0, 0.0];
    let probabilities = vec![vec![0.9, 0.8, 0.9, 0.8]];
    let names = vec!["f".to_owned()];
    let splits = vec![(vec![2, 3], vec![0, 1]), (vec![0, 1], vec![2, 3])];
    // Each fold has one true positive and one false positive; log loss is
    // the equally weighted arithmetic mean of -ln(.9) and -ln(.2).
    for (scoring, expected) in [
        (Scoring::Accuracy, 0.5),
        (Scoring::F1, 2.0 / 3.0),
        (Scoring::NegLogLoss, (0.9_f64.ln() + 0.2_f64.ln()) / 2.0),
    ] {
        for scale in [1.0, 1e308, 1e-308, f64::from_bits(1)] {
            let scores = single_feature_importance_from_proba(
                &y,
                &probabilities,
                &names,
                &splits,
                Some(&[scale; 4]),
                scoring,
            )
            .unwrap();
            assert!(
                (scores["f"].mean - expected).abs() < 1e-14,
                "scale={scale}, expected={expected}, got={}",
                scores["f"].mean
            );
            assert_eq!(scores["f"].std, 0.0);
        }
    }
}

#[test]
fn unequal_extreme_weights_retain_relative_importance() {
    let y = [1.0, 0.0, 1.0, 0.0];
    let probabilities = vec![vec![0.9, 0.8, 0.9, 0.8]];
    let names = vec!["f".to_owned()];
    let splits = vec![(vec![2, 3], vec![0, 1]), (vec![0, 1], vec![2, 3])];
    for scale in [1.0, 4e307, 1e-308, f64::from_bits(1)] {
        let weights = [3.0 * scale, scale, 3.0 * scale, scale];
        for (scoring, expected) in [
            (Scoring::Accuracy, 0.75),
            (Scoring::F1, 6.0 / 7.0),
            (Scoring::NegLogLoss, (3.0 * 0.9_f64.ln() + 0.2_f64.ln()) / 4.0),
        ] {
            let scores = single_feature_importance_from_proba(
                &y,
                &probabilities,
                &names,
                &splits,
                Some(&weights),
                scoring,
            )
            .unwrap();
            assert!(
                (scores["f"].mean - expected).abs() < 1e-14,
                "scale={scale}, expected={expected}, got={}",
                scores["f"].mean
            );
        }
    }
}

#[test]
fn f1_ignores_true_negative_weight_when_rescaling() {
    let names = vec!["f".to_owned()];
    let splits = vec![(vec![], vec![0, 1, 2])];
    for (y, p, expected) in [
        ([0.0, 1.0, 0.0], vec![0.1, 0.9, 0.1], 1.0),
        ([0.0, 1.0, 0.0], vec![0.1, 0.9, 0.9], 2.0 / 3.0),
        ([0.0, 1.0, 1.0], vec![0.1, 0.9, 0.1], 2.0 / 3.0),
    ] {
        let result = single_feature_importance_from_proba(
            &y,
            &[p],
            &names,
            &splits,
            Some(&[1e308, 1e-308, 1e-308]),
            Scoring::F1,
        )
        .unwrap();
        assert!((result["f"].mean - expected).abs() < 1e-14);
    }
}

#[test]
fn zero_and_invalid_weight_behavior_is_preserved() {
    let names = vec!["f".to_owned()];
    let splits = vec![(vec![], vec![0, 1])];
    for (weights, expected) in [
        ([0.0, 0.0], 0.0),
        ([-1e308, 1.5e308], -2.0),
        ([f64::NAN, 1.0], 0.0),
        ([f64::INFINITY, 1.0], f64::NAN),
    ] {
        let result = single_feature_importance_from_proba(
            &[1.0, 0.0],
            &[vec![0.9, 0.8]],
            &names,
            &splits,
            Some(&weights),
            Scoring::Accuracy,
        )
        .unwrap();
        if expected.is_nan() {
            assert!(result["f"].mean.is_nan());
        } else {
            assert!((result["f"].mean - expected).abs() < 1e-14);
        }
    }
}

#[test]
fn collective_small_weights_preserve_subnormal_scores() {
    let mut y = vec![0.0];
    y.extend([1.0; 100]);
    let probabilities = vec![vec![0.9; 101]];
    let mut weights = vec![1e308];
    weights.extend([1e-16; 100]);
    let names = vec!["f".to_owned()];
    let splits = vec![(vec![], (0..101).collect())];
    for (scoring, expected) in [(Scoring::Accuracy, 1e-322), (Scoring::F1, 2e-322)] {
        let result = single_feature_importance_from_proba(
            &y,
            &probabilities,
            &names,
            &splits,
            Some(&weights),
            scoring,
        )
        .unwrap();
        assert_eq!(result["f"].mean, expected);
    }
}

#[test]
fn f1_preserves_tiny_positive_precision_and_recall() {
    for scale in [1e160, 1e161, 1e200] {
        let result = single_feature_importance_from_proba(
            &[1.0, 0.0, 1.0],
            &[vec![0.9, 0.9, 0.1]],
            &["f".to_owned()],
            &[(vec![], vec![0, 1, 2])],
            Some(&[1.0, scale, scale]),
            Scoring::F1,
        )
        .unwrap();
        assert!((result["f"].mean * scale - 1.0).abs() < 1e-14);
    }
}

#[test]
fn f1_retains_score_when_precision_itself_rounds_to_zero() {
    let result = single_feature_importance_from_proba(
        &[1.0, 0.0],
        &[vec![0.9, 0.9]],
        &["f".to_owned()],
        &[(vec![], vec![0, 1])],
        Some(&[f64::from_bits(1), 3.0]),
        Scoring::F1,
    )
    .unwrap();
    assert_eq!(result["f"].mean, f64::from_bits(1));
}
