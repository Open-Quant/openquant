use std::collections::BTreeMap;

use chrono::NaiveDateTime;
use openquant::cross_validation::SimpleClassifier;
use openquant::hyperparameter_tuning::{
    classification_score, grid_search, randomized_search, sample_log_uniform, HyperParamValue,
    ParamSet, RandomParamDistribution, SearchData, SearchScoring,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn make_series(
    start: &str,
    periods: usize,
    freq_minutes: i64,
) -> Vec<(NaiveDateTime, NaiveDateTime)> {
    let start_dt = NaiveDateTime::parse_from_str(start, "%Y-%m-%d %H:%M:%S").unwrap();
    (0..periods)
        .map(|i| {
            let idx = start_dt + chrono::Duration::minutes(i as i64 * freq_minutes);
            let val = idx + chrono::Duration::minutes(3);
            (idx, val)
        })
        .collect()
}

struct ThresholdClassifier {
    threshold: f64,
    sharpness: f64,
    trained_prior: f64,
}

impl ThresholdClassifier {
    fn from_params(params: &ParamSet) -> Self {
        let threshold = params.get("threshold").and_then(HyperParamValue::as_f64).unwrap_or(0.5);
        let sharpness = params.get("sharpness").and_then(HyperParamValue::as_f64).unwrap_or(6.0);

        Self { threshold, sharpness, trained_prior: 0.5 }
    }
}

impl SimpleClassifier for ThresholdClassifier {
    fn fit(&mut self, _x: &[Vec<f64>], y: &[f64], sample_weight: Option<&[f64]>) {
        let mut total_w = 0.0;
        let mut pos_w = 0.0;
        for (i, yi) in y.iter().enumerate() {
            let w = sample_weight.map(|sw| sw[i]).unwrap_or(1.0);
            total_w += w;
            if *yi == 1.0 {
                pos_w += w;
            }
        }
        self.trained_prior = if total_w > 0.0 { pos_w / total_w } else { 0.5 };
    }

    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let z = (row[0] - self.threshold) * self.sharpness;
                let logistic = 1.0 / (1.0 + (-z).exp());
                // Blend threshold behavior with class prior learned from weighted fit.
                (0.85 * logistic + 0.15 * self.trained_prior).clamp(0.0, 1.0)
            })
            .collect()
    }
}

#[test]
fn test_grid_search_with_purged_kfold_and_embargo() {
    let n = 120usize;
    let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64 / (n as f64 - 1.0)]).collect();
    let y: Vec<f64> = x.iter().map(|v| if v[0] >= 0.7 { 1.0 } else { 0.0 }).collect();
    let sample_weight: Vec<f64> = y.iter().map(|yi| if *yi == 1.0 { 4.0 } else { 1.0 }).collect();
    let info_sets = make_series("2019-01-01 00:00:00", n, 1);

    let mut param_grid = BTreeMap::new();
    param_grid.insert(
        "threshold".to_string(),
        vec![HyperParamValue::Float(0.5), HyperParamValue::Float(0.7), HyperParamValue::Float(0.9)],
    );
    param_grid.insert(
        "sharpness".to_string(),
        vec![HyperParamValue::Float(4.0), HyperParamValue::Float(8.0)],
    );

    let result = grid_search(
        ThresholdClassifier::from_params,
        &param_grid,
        SearchData {
            x: &x,
            y: &y,
            sample_weight: Some(&sample_weight),
            samples_info_sets: &info_sets,
        },
        4,
        0.02,
        SearchScoring::NegLogLoss,
    )
    .unwrap();

    assert_eq!(result.trials.len(), 6);
    assert!(result.best_score.is_finite());

    let best_threshold =
        result.best_params.get("threshold").and_then(HyperParamValue::as_f64).unwrap();
    assert!((best_threshold - 0.7).abs() < 1e-9);
}

#[test]
fn test_randomized_search_seeded_deterministic_and_log_uniform() {
    let n = 90usize;
    let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64 / (n as f64 - 1.0)]).collect();
    let y: Vec<f64> = x.iter().map(|v| if v[0] >= 0.65 { 1.0 } else { 0.0 }).collect();
    let info_sets = make_series("2019-01-01 00:00:00", n, 1);

    let mut param_space = BTreeMap::new();
    param_space.insert(
        "threshold".to_string(),
        RandomParamDistribution::Uniform { low: 0.45, high: 0.85 },
    );
    param_space.insert(
        "sharpness".to_string(),
        RandomParamDistribution::LogUniform { low: 1e-1, high: 2e1 },
    );

    let run = || {
        randomized_search(
            ThresholdClassifier::from_params,
            &param_space,
            12,
            42,
            SearchData { x: &x, y: &y, sample_weight: None, samples_info_sets: &info_sets },
            3,
            0.01,
            SearchScoring::BalancedAccuracy,
        )
        .unwrap()
    };

    let first = run();
    let second = run();
    assert_eq!(first.best_params, second.best_params);
    assert!((first.best_score - second.best_score).abs() < 1e-12);
    assert_eq!(first.trials.len(), 12);
    assert_eq!(first.trials, second.trials);

    let mut rng = StdRng::seed_from_u64(7);
    let s1 = sample_log_uniform(1e-3, 1e1, &mut rng).unwrap();
    let s2 = sample_log_uniform(1e-3, 1e1, &mut rng).unwrap();
    assert!(s1 >= 1e-3 && s1 <= 1e1);
    assert!(s2 >= 1e-3 && s2 <= 1e1);
    assert!((s1 - s2).abs() > 1e-12);
}

#[test]
fn test_scoring_layer_handles_imbalance_weighted_neg_log_loss_and_metrics() {
    // Strongly imbalanced labels: 95% class 0.
    let mut y = vec![0.0; 95];
    y.extend(vec![1.0; 5]);

    // Majority-like predictions: high confidence toward class 0 for all points.
    let probs = vec![0.1; 100];

    let accuracy = classification_score(&y, &probs, None, SearchScoring::Accuracy).unwrap();
    let balanced = classification_score(&y, &probs, None, SearchScoring::BalancedAccuracy).unwrap();
    let unweighted_nll = classification_score(&y, &probs, None, SearchScoring::NegLogLoss).unwrap();

    // Upweight minority class mistakes.
    let weights: Vec<f64> = y.iter().map(|yi| if *yi == 1.0 { 20.0 } else { 1.0 }).collect();
    let weighted_nll =
        classification_score(&y, &probs, Some(&weights), SearchScoring::NegLogLoss).unwrap();

    assert!(accuracy > 0.9);
    assert!(balanced < accuracy);
    // Weighting minority class should penalize this classifier's cross-entropy.
    assert!(weighted_nll < unweighted_nll);
}
