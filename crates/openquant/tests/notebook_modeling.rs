use chrono::{Duration, NaiveDateTime};
use nalgebra::DMatrix;

use openquant::notebook_modeling::{
    fit_bagged_model, fit_meta_model, fit_primary_model, BaggedModelConfig, ModelSelectionMetric,
    ModelingLabError, PrimaryModelConfig,
};
use openquant::sampling::get_ind_matrix;
use openquant::sb_bagging::{MaxFeatures, MaxSamples};

fn synthetic_dataset(n: usize) -> (Vec<Vec<f64>>, Vec<u8>, Vec<(NaiveDateTime, NaiveDateTime)>) {
    let start = NaiveDateTime::parse_from_str("2022-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();

    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut info_sets = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f64;
        let f0 = (t / 11.0).sin();
        let f1 = (t / 17.0).cos();
        let f2 = (t / 5.0).sin() * 0.3;
        let score = 1.4 * f0 + 0.9 * f1 - 0.2 * f2;
        x.push(vec![f0, f1, f2, score]);
        y.push(if score > 0.05 { 1 } else { 0 });

        let s = start + Duration::minutes(i as i64);
        let e = s + Duration::minutes(15);
        info_sets.push((s, e));
    }

    (x, y, info_sets)
}

fn build_square_indicator(n: usize, horizon: usize) -> Vec<Vec<u8>> {
    let labels: Vec<(usize, usize)> = (0..n)
        .map(|i| {
            let end = (i + horizon).min(n.saturating_sub(1));
            (i, end)
        })
        .collect();
    let bars: Vec<usize> = (0..n).collect();
    get_ind_matrix(&labels, &bars)
}

fn to_matrix(x: &[Vec<f64>]) -> DMatrix<f64> {
    DMatrix::from_fn(x.len(), x[0].len(), |r, c| x[r][c])
}

#[test]
fn test_fit_primary_and_meta_model_with_purged_cv_metrics() {
    let (x, y, info_sets) = synthetic_dataset(180);
    let cfg = PrimaryModelConfig::default();

    let primary = fit_primary_model(&x, &y, &info_sets, &cfg).unwrap();
    assert_eq!(primary.cv_metrics.accuracy_scores.len(), cfg.cv.n_splits);
    assert!(primary.cv_metrics.accuracy_mean > 0.5);

    let primary_probs = primary.predict_proba(&x).unwrap();
    let meta = fit_meta_model(&x, &primary_probs, &y, &info_sets, &cfg).unwrap();
    assert_eq!(meta.model.cv_metrics.accuracy_scores.len(), cfg.cv.n_splits);
    assert!(meta.model.cv_metrics.accuracy_mean > 0.5);
}

#[test]
fn test_bagged_model_rejects_oob_only_selection_when_overlap_risk_exists() {
    let (x, y, info_sets) = synthetic_dataset(120);
    let x_mat = to_matrix(&x);
    let ind = build_square_indicator(120, 12);

    let cfg = BaggedModelConfig {
        overlap_risk: true,
        selection_metric: ModelSelectionMetric::OobScore,
        n_estimators: 16,
        max_samples: MaxSamples::Float(1.0),
        max_features: MaxFeatures::Float(0.75),
        ..BaggedModelConfig::default()
    };

    let err = fit_bagged_model(&x_mat, &y, &info_sets, &ind, &cfg).unwrap_err();
    assert_eq!(err, ModelingLabError::CvRequiredUnderOverlapRisk);
}

#[test]
fn test_bagged_model_improves_uniqueness_vs_vanilla_bootstrap_baseline() {
    let (x, y, info_sets) = synthetic_dataset(180);
    let x_mat = to_matrix(&x);
    let ind = build_square_indicator(180, 20);

    let cfg = BaggedModelConfig {
        overlap_risk: true,
        selection_metric: ModelSelectionMetric::CvAccuracy,
        n_estimators: 24,
        max_samples: MaxSamples::Float(1.0),
        max_features: MaxFeatures::Float(0.75),
        uniqueness_trials: 64,
        random_state: 7,
        ..BaggedModelConfig::default()
    };

    let result = fit_bagged_model(&x_mat, &y, &info_sets, &ind, &cfg).unwrap();
    assert!(result.cv_metrics.accuracy_mean > 0.5);
    assert!(result.resolved_max_samples < x_mat.nrows());
    assert!(
        result.diversity.sequential_uniqueness_mean >= result.diversity.vanilla_uniqueness_mean,
        "sequential={} vanilla={}",
        result.diversity.sequential_uniqueness_mean,
        result.diversity.vanilla_uniqueness_mean
    );
}
