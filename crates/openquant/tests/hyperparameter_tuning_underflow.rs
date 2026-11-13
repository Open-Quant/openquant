use openquant::hyperparameter_tuning::{classification_score, SearchScoring};

#[test]
fn tiny_positive_weights_preserve_negative_log_loss() {
    // With one row its positive weight cancels analytically.
    for weight in [f64::from_bits(1), 1e-320, 1e-310] {
        for probability in [0.9_f64, 0.75, 0.5, 0.1] {
            let score = classification_score(
                &[1.0],
                &[probability],
                Some(&[weight]),
                SearchScoring::NegLogLoss,
            )
            .unwrap();
            assert!(
                (score - probability.ln()).abs() < 1e-14,
                "weight={weight}, probability={probability}, score={score}"
            );
        }
    }
}

#[test]
fn subnormal_weights_preserve_unequal_ratios_and_skip_zero_weights() {
    // The exact binary64 weights are integer multiples of the smallest subnormal.
    let expected = (0.75_f64.ln() + 3.0 * 0.5_f64.ln()) / 4.0;
    let score = classification_score(
        &[1.0, 0.0, 1.0],
        &[0.75, 0.5, 0.0],
        Some(&[f64::from_bits(1), f64::from_bits(3), 0.0]),
        SearchScoring::NegLogLoss,
    )
    .unwrap();
    assert!((score - expected).abs() < 1e-14, "score={score}, expected={expected}");
}

#[test]
fn tiny_normal_weights_preserve_clipped_near_one_losses() {
    let expected = (1.0_f64 - 1e-15).ln();
    for weight in [f64::MIN_POSITIVE, 1e-300, 1e-151] {
        let score =
            classification_score(&[1.0], &[1.0], Some(&[weight]), SearchScoring::NegLogLoss)
                .unwrap();
        assert!(
            (score - expected).abs() <= expected.abs() * 1e-14,
            "weight={weight}, score={score}, expected={expected}"
        );
    }
}

#[test]
fn ordinary_weighted_log_loss_keeps_original_rounding() {
    let labels = [1.0, 0.0, 1.0];
    let probabilities = [0.75_f64, 0.25, 0.9];
    let weights = [1.0, 3.0, 2.0];
    let mut loss = 0.0;
    let mut total = 0.0;
    for i in 0..labels.len() {
        let y = labels[i];
        let p = probabilities[i];
        loss += -weights[i] * (y * p.ln() + (1.0 - y) * (1.0 - p).ln());
        total += weights[i];
    }
    let actual =
        classification_score(&labels, &probabilities, Some(&weights), SearchScoring::NegLogLoss)
            .unwrap();
    assert_eq!(actual.to_bits(), (-(loss / total)).to_bits());
}

#[test]
fn tiny_weight_recovery_does_not_relax_overflow_errors() {
    use openquant::hyperparameter_tuning::TuningError;
    assert_eq!(
        classification_score(
            &[1.0, 1.0],
            &[0.75, 0.75],
            Some(&[f64::MAX, f64::MAX]),
            SearchScoring::NegLogLoss
        ),
        Err(TuningError::NonFiniteSampleWeightSum)
    );
    assert_eq!(
        classification_score(&[1.0], &[0.01], Some(&[f64::MAX]), SearchScoring::NegLogLoss),
        Err(TuningError::NonFiniteScore)
    );
}
