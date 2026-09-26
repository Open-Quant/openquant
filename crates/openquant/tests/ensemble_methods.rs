use openquant::ensemble_methods::{
    aggregate_classification_probability_mean, aggregate_classification_vote,
    aggregate_regression_mean, average_pairwise_prediction_correlation, bagging_ensemble_variance,
    bias_variance_noise, bootstrap_sample_indices, recommend_bagging_vs_boosting,
    sequential_bootstrap_sample_indices, EnsembleError, EnsembleMethod,
};
use openquant::sampling::{get_ind_mat_average_uniqueness, get_ind_matrix};

#[test]
fn test_bias_variance_noise_decomposition() {
    let y = vec![1.0, 0.0, 1.0, 0.0];
    let preds = vec![vec![0.9, 0.1, 0.8, 0.2], vec![0.8, 0.2, 0.7, 0.3], vec![1.0, 0.0, 0.9, 0.1]];

    // Without the noiseless target there is no noise estimate; bias_sq absorbs it and
    // bias_sq + variance == mse holds exactly.
    let out = bias_variance_noise(&y, &preds, None).unwrap();
    assert!(out.bias_sq >= 0.0);
    assert!(out.variance >= 0.0);
    assert!(out.mse >= 0.0);
    assert_eq!(out.noise, None);
    assert!((out.bias_sq + out.variance - out.mse).abs() < 1e-12);
    assert!((out.bias_sq - 0.025).abs() < 1e-12);
    assert!((out.variance - 0.02 / 3.0).abs() < 1e-12);

    // With the target: bias is measured against it and noise against the labels.
    let target = vec![0.9, 0.1, 0.9, 0.1];
    let out = bias_variance_noise(&y, &preds, Some(&target)).unwrap();
    // mean prediction [0.9, 0.1, 0.8, 0.2] vs target: squared bias mean([0, 0, 0.01, 0.01]).
    assert!((out.bias_sq - 0.005).abs() < 1e-12);
    assert!((out.variance - 0.02 / 3.0).abs() < 1e-12);
    // y - target = [0.1, -0.1, 0.1, -0.1].
    assert!((out.noise.unwrap() - 0.01).abs() < 1e-12);
    // MSE is still against y_true, so it matches the no-target call.
    assert!((out.mse - (0.025 + 0.02 / 3.0)).abs() < 1e-12);
}

/// Least-squares line through (x, y).
fn fit_line(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
    let slope = sxy / sxx;
    (my - slope * mx, slope)
}

#[test]
fn test_bias_variance_noise_recovers_known_label_noise() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rand_distr::{Distribution, Normal};

    // AFML §6.2 simulation: y = f(x) + eps with Var(eps) = sigma^2 known. Each model is a line
    // fitted to its own noisy training sample of a curved f, so the ensemble has real bias and
    // real variance. The models are evaluated on fresh test labels.
    let sigma = 0.5;
    let sigma_sq = sigma * sigma;
    let f = |x: f64| (2.0 * std::f64::consts::PI * x).sin();
    let normal = Normal::new(0.0, sigma).unwrap();
    let mut rng = StdRng::seed_from_u64(128);

    let n_test = 400;
    let x_test: Vec<f64> = (0..n_test).map(|i| (i as f64 + 0.5) / n_test as f64).collect();
    let y_expected: Vec<f64> = x_test.iter().map(|&x| f(x)).collect();

    let n_models = 50;
    let n_train = 30;
    let preds: Vec<Vec<f64>> = (0..n_models)
        .map(|_| {
            let x: Vec<f64> = (0..n_train).map(|_| rng.gen::<f64>()).collect();
            let y: Vec<f64> = x.iter().map(|&v| f(v) + normal.sample(&mut rng)).collect();
            let (a, b) = fit_line(&x, &y);
            x_test.iter().map(|&v| a + b * v).collect()
        })
        .collect();

    // Average over many independent draws of the test labels: noise -> sigma^2 and
    // bias^2 + variance + noise -> mse (the expectation identity).
    let reps = 200;
    let (mut noise_sum, mut gap_sum, mut bias_sum, mut var_sum) = (0.0, 0.0, 0.0, 0.0);
    for rep in 0..reps {
        let y_true: Vec<f64> = y_expected.iter().map(|&m| m + normal.sample(&mut rng)).collect();
        let out = bias_variance_noise(&y_true, &preds, Some(&y_expected)).unwrap();
        let noise = out.noise.expect("noise is reported when y_expected is given");
        noise_sum += noise;
        gap_sum += out.mse - (out.bias_sq + out.variance + noise);
        bias_sum += out.bias_sq;
        var_sum += out.variance;

        if rep == 0 {
            // A single draw: noise within ~4 standard errors (sigma^2 * sqrt(2 / n) = 0.018).
            assert!((noise - sigma_sq).abs() < 0.07, "noise {noise} vs {sigma_sq}");
            // Without the target, noise is not reported and bias_sq absorbs it.
            let observed = bias_variance_noise(&y_true, &preds, None).unwrap();
            assert_eq!(observed.noise, None);
            assert!((observed.mse - out.mse).abs() < 1e-12);
            assert!((observed.variance - out.variance).abs() < 1e-12);
            assert!((observed.bias_sq - (out.bias_sq + noise)).abs() < 0.1);
        }
    }
    let reps = reps as f64;
    let mean_noise = noise_sum / reps;
    let mean_gap = gap_sum / reps;
    // Standard error of mean_noise is sigma^2 * sqrt(2 / (n_test * reps)) ~ 0.00125.
    assert!((mean_noise - sigma_sq).abs() < 0.006, "mean noise {mean_noise} vs {sigma_sq}");
    assert!(mean_gap.abs() < 0.01, "mse - (bias^2 + var + noise) averaged {mean_gap}");
    // The line cannot fit a full sine period, so bias dominates; both terms are material.
    assert!(bias_sum / reps > 0.1);
    assert!(var_sum / reps > 0.005);
}

#[test]
fn test_bias_variance_noise_validates_lengths() {
    let y = vec![1.0, 0.0, 1.0];
    let preds = vec![vec![0.9, 0.1, 0.8], vec![0.8, 0.2, 0.7]];
    assert_eq!(
        bias_variance_noise(&y, &preds, Some(&[1.0, 0.0])),
        Err(EnsembleError::LengthMismatch("y_expected"))
    );
    assert_eq!(
        bias_variance_noise(&y, &[vec![0.9, 0.1]], None),
        Err(EnsembleError::LengthMismatch("prediction"))
    );
    assert_eq!(bias_variance_noise(&[], &preds, None), Err(EnsembleError::Empty("y_true")));
    assert_eq!(
        bias_variance_noise(&y, &[], None),
        Err(EnsembleError::Empty("per_model_predictions"))
    );
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
fn test_sequential_bootstrap_indices_are_uniqueness_weighted() {
    // 40 labels of 10 bars, one starting every 3 bars.
    let n = 40;
    let spans: Vec<(usize, usize)> = (0..n).map(|i| (3 * i, 3 * i + 9)).collect();
    let bars: Vec<usize> = (0..3 * n + 10).collect();
    let ind = get_ind_matrix(&spans, &bars).unwrap();
    let uniqueness = |drawn: &[usize]| {
        let sub: Vec<Vec<u8>> =
            ind.iter().map(|row| drawn.iter().map(|&c| row[c]).collect()).collect();
        get_ind_mat_average_uniqueness(&sub).unwrap()
    };

    assert_eq!(
        sequential_bootstrap_sample_indices(&ind, n, 5).unwrap(),
        sequential_bootstrap_sample_indices(&ind, n, 5).unwrap()
    );
    assert_ne!(
        sequential_bootstrap_sample_indices(&ind, n, 5).unwrap(),
        bootstrap_sample_indices(n, n, 5).unwrap()
    );

    let seeds = 0..300u64;
    let sequential = seeds
        .clone()
        .map(|s| uniqueness(&sequential_bootstrap_sample_indices(&ind, n, s).unwrap()))
        .sum::<f64>()
        / 300.0;
    let standard =
        seeds.map(|s| uniqueness(&bootstrap_sample_indices(n, n, s).unwrap())).sum::<f64>() / 300.0;
    // Uniform ~0.300 and sequential ~0.312 (AFML §4.5.4 setup, standard errors near 0.0005).
    assert!(sequential > standard + 0.006, "sequential {sequential:.4} vs uniform {standard:.4}");
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

/// #186 item 21: only the *averaged* probabilities were range-checked, so 1.4 and -0.4 were
/// accepted because they average to 0.5.
#[test]
fn test_probability_mean_rejects_each_out_of_range_input() {
    let invalid = EnsembleError::Invalid { name: "probabilities", requirement: "in [0,1]" };
    for bad in [1.4, -0.4, f64::NAN] {
        let rows = [vec![bad, 0.2], vec![1.0 - bad, 0.4]];
        assert_eq!(
            aggregate_classification_probability_mean(&rows, 0.5).unwrap_err(),
            invalid,
            "{bad}"
        );
    }
}

/// #186 item 21: a `NaN` (or infinite) single-estimator variance used to pass and return `NaN`.
#[test]
fn test_bagging_variance_rejects_non_finite_variance() {
    for bad in [f64::NAN, f64::INFINITY, -1.0] {
        assert_eq!(
            bagging_ensemble_variance(bad, 0.5, 10).unwrap_err(),
            EnsembleError::Invalid {
                name: "single_estimator_variance",
                requirement: "finite and non-negative",
            },
            "{bad}"
        );
    }
}
