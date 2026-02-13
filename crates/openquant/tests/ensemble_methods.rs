use openquant::ensemble_methods::{
    aggregate_classification_probability_mean, aggregate_classification_vote,
    aggregate_regression_mean, average_pairwise_prediction_correlation, bagging_ensemble_variance,
    bias_variance_noise, bootstrap_sample_indices, recommend_bagging_vs_boosting,
    sequential_bootstrap_sample_indices, EnsembleMethod,
};

#[test]
fn test_bias_variance_noise_decomposition() {
    let y = vec![1.0, 0.0, 1.0, 0.0];
    let preds = vec![vec![0.9, 0.1, 0.8, 0.2], vec![0.8, 0.2, 0.7, 0.3], vec![1.0, 0.0, 0.9, 0.1]];

    let out = bias_variance_noise(&y, &preds).unwrap();
    assert!(out.bias_sq >= 0.0);
    assert!(out.variance >= 0.0);
    assert!(out.noise >= 0.0);
    assert!(out.mse >= 0.0);

    let lhs = out.bias_sq + out.variance + out.noise;
    assert!((lhs - out.mse).abs() < 1e-10);
}

#[test]
fn test_bootstrap_and_sequential_bootstrap_shapes() {
    let b = bootstrap_sample_indices(10, 6, 7).unwrap();
    assert_eq!(b.len(), 6);
    assert!(b.iter().all(|v| *v < 10));

    let ind_mat = vec![vec![1, 0, 1, 0], vec![0, 1, 0, 1], vec![1, 1, 0, 0]];
    let sb = sequential_bootstrap_sample_indices(&ind_mat, 8, 11).unwrap();
    assert_eq!(sb.len(), 8);
    assert!(sb.iter().all(|v| *v < ind_mat[0].len()));
}

#[test]
fn test_aggregation_helpers() {
    let reg = aggregate_regression_mean(&[vec![1.0, 3.0], vec![3.0, 1.0]]).unwrap();
    assert_eq!(reg, vec![2.0, 2.0]);

    let vote =
        aggregate_classification_vote(&[vec![1, 0, 1], vec![1, 1, 0], vec![0, 1, 1]]).unwrap();
    assert_eq!(vote, vec![1, 1, 1]);

    let (prob, labels) = aggregate_classification_probability_mean(
        &[vec![0.9, 0.2], vec![0.7, 0.4], vec![0.8, 0.3]],
        0.5,
    )
    .unwrap();
    assert!((prob[0] - 0.8).abs() < 1e-12);
    assert!((prob[1] - 0.3).abs() < 1e-12);
    assert_eq!(labels, vec![1, 0]);
}

#[test]
fn test_variance_reduction_and_redundancy_failure_mode() {
    let low_corr = bagging_ensemble_variance(1.0, 0.0, 10).unwrap();
    assert!((low_corr - 0.1).abs() < 1e-12);

    let high_corr = bagging_ensemble_variance(1.0, 0.95, 10).unwrap();
    assert!(high_corr > 0.9);
    assert!(high_corr > low_corr);
}

#[test]
fn test_pairwise_correlation_and_strategy_recommendation() {
    let weak_preds = vec![
        vec![0.50, 0.52, 0.48, 0.50],
        vec![0.51, 0.53, 0.49, 0.51],
        vec![0.49, 0.51, 0.47, 0.49],
    ];
    let corr = average_pairwise_prediction_correlation(&weak_preds).unwrap();
    assert!(corr > 0.95);

    let weak = recommend_bagging_vs_boosting(0.53, corr, 0.8, 1.0, 16).unwrap();
    assert_eq!(weak.recommended, EnsembleMethod::Boosting);

    let strong_diverse = recommend_bagging_vs_boosting(0.68, 0.15, 0.25, 1.0, 16).unwrap();
    assert_eq!(strong_diverse.recommended, EnsembleMethod::Bagging);
    assert!(strong_diverse.expected_variance_reduction > 0.0);
}
