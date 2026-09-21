//! Value tests for the scoring layer and the log-uniform sampler of `hyperparameter_tuning`.
//! Expected numbers are hand-worked in the comments from the textbook definitions (the same ones
//! sklearn's accuracy_score / balanced_accuracy_score / log_loss implement).
//!
//! The pre-existing `tests/hyperparameter_tuning.rs` never checks an accuracy value (replacing
//! accuracy by the constant 1234.5 passes) and checks the sampler only for range membership.
//! See `docs/test-sensitivity-audit.md`.

use openquant::hyperparameter_tuning::{classification_score, sample_log_uniform, SearchScoring};
use rand::rngs::StdRng;
use rand::SeedableRng;

const Y: [f64; 5] = [1.0, 1.0, 1.0, 0.0, 0.0];
const P: [f64; 5] = [0.9, 0.6, 0.2, 0.4, 0.7];
// predictions at the 0.5 cut: 1, 1, 0, 0, 1  ->  correct: yes, yes, no, yes, no

/// accuracy = 3/5; recall(1) = 2/3, recall(0) = 1/2, balanced accuracy = (2/3 + 1/2)/2 = 7/12.
#[test]
fn unweighted_accuracy_and_balanced_accuracy_hand_worked() {
    let acc = classification_score(&Y, &P, None, SearchScoring::Accuracy).unwrap();
    let bal = classification_score(&Y, &P, None, SearchScoring::BalancedAccuracy).unwrap();
    assert!((acc - 0.6).abs() < 1e-15, "{acc}");
    assert!((bal - 7.0 / 12.0).abs() < 1e-15, "{bal}");
}

/// Log-loss: the probability assigned to the true class is 0.9, 0.6, 0.2, 0.6, 0.3, so
/// neg_log_loss = (ln 0.9 + ln 0.6 + ln 0.2 + ln 0.6 + ln 0.3) / 5 = ln(0.01944) / 5 = -0.78808...
#[test]
fn unweighted_neg_log_loss_hand_worked() {
    let nll = classification_score(&Y, &P, None, SearchScoring::NegLogLoss).unwrap();
    let want = (0.9f64 * 0.6 * 0.2 * 0.6 * 0.3).ln() / 5.0;
    // five logs of O(1) numbers summed: rounding only
    assert!((nll - want).abs() < 1e-14, "{nll} vs {want}");
    assert!((want - (-0.788_08)).abs() < 1e-5, "hand value check: {want}");
}

/// Weights (1, 2, 3, 4, 0); the last sample drops out entirely.
///   accuracy          = (1 + 2 + 4) / 10 = 0.7
///   recall(1)         = (1 + 2) / 6 = 0.5;  recall(0) = 4 / 4 = 1;  balanced = 0.75
///   neg_log_loss      = (1 ln 0.9 + 2 ln 0.6 + 3 ln 0.2 + 4 ln 0.6) / 10
#[test]
fn weighted_scores_hand_worked() {
    let w = [1.0, 2.0, 3.0, 4.0, 0.0];
    let acc = classification_score(&Y, &P, Some(&w), SearchScoring::Accuracy).unwrap();
    let bal = classification_score(&Y, &P, Some(&w), SearchScoring::BalancedAccuracy).unwrap();
    let nll = classification_score(&Y, &P, Some(&w), SearchScoring::NegLogLoss).unwrap();
    assert!((acc - 0.7).abs() < 1e-15, "{acc}");
    assert!((bal - 0.75).abs() < 1e-15, "{bal}");
    let want = (0.9f64.ln() + 6.0 * 0.6f64.ln() + 3.0 * 0.2f64.ln()) / 10.0;
    assert!((nll - want).abs() < 1e-14, "{nll} vs {want}");
}

/// If X is log-uniform on [a, b] then ln X is uniform on [ln a, ln b]:
///   E[ln X] = (ln a + ln b) / 2,  Var[ln X] = (ln b - ln a)^2 / 12,  P(X < sqrt(ab)) = 1/2.
/// With a = 1e-3, b = 1e1: mean -2.302585, sd 2.658785, median 0.1.
/// N = 20000 draws: the standard error of the mean is 2.6588 / sqrt(20000) = 0.0188 and of the
/// median fraction sqrt(0.25 / 20000) = 0.00354. Tolerances are 5 standard errors (a correct
/// sampler fails with probability < 1e-6; the seed is fixed so the test is deterministic anyway).
/// A plain uniform sampler on [a, b] would put ~1% of draws below 0.1 and fail by a mile.
#[test]
fn log_uniform_sampler_has_log_uniform_moments() {
    let (a, b) = (1e-3f64, 1e1f64);
    let n = 20_000usize;
    let mut rng = StdRng::seed_from_u64(40);
    let logs: Vec<f64> = (0..n).map(|_| sample_log_uniform(a, b, &mut rng).unwrap().ln()).collect();
    assert!(logs.iter().all(|l| *l >= a.ln() && *l <= b.ln()));

    let mean = logs.iter().sum::<f64>() / n as f64;
    let var = logs.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let sd_true = (b.ln() - a.ln()) / 12f64.sqrt();
    let se_mean = sd_true / (n as f64).sqrt();
    assert!((mean - (a.ln() + b.ln()) / 2.0).abs() < 5.0 * se_mean, "mean of ln X = {mean}");

    // Var of the sample variance of a uniform: (mu4 - sigma^4) / n with mu4 = 9/5 sigma^4
    let se_var = sd_true.powi(2) * (0.8 / n as f64).sqrt();
    assert!((var - sd_true.powi(2)).abs() < 5.0 * se_var, "var of ln X = {var}");

    let below_median = logs.iter().filter(|l| **l < (a * b).sqrt().ln()).count() as f64 / n as f64;
    assert!((below_median - 0.5).abs() < 5.0 * (0.25 / n as f64).sqrt(), "{below_median}");
}
