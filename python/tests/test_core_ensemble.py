import math
import random
import statistics

import pytest
from openquant import ensemble


def test_bias_variance_noise_decomposition():
    # Mirrors crates/openquant/tests/ensemble_methods.rs::
    # test_bias_variance_noise_decomposition
    y_true = [1.0, 0.0, 1.0, 0.0]
    predictions = [[0.9, 0.1, 0.8, 0.2], [0.8, 0.2, 0.7, 0.3], [1.0, 0.0, 0.9, 0.1]]

    bias_sq, variance, noise, mse = ensemble.bias_variance_noise(y_true, predictions)

    # Without the noiseless target, noise is not reported and bias_sq absorbs it.
    assert noise is None
    assert abs((bias_sq + variance) - mse) < 1e-12
    # Worked by hand: the mean prediction is [0.9, 0.1, 0.8, 0.2], so the squared bias is
    # mean([0.01, 0.01, 0.04, 0.04]); each point's predictions spread by +-0.1 around the
    # mean, a population variance of 0.02 / 3.
    assert bias_sq == pytest.approx(0.025, abs=1e-12)
    assert variance == pytest.approx(0.02 / 3.0, abs=1e-12)

    # With the target, bias is measured against it and noise = mean((y_true - target)^2).
    target = [0.9, 0.1, 0.9, 0.1]
    bias_sq, variance, noise, mse = ensemble.bias_variance_noise(
        y_true, predictions, y_expected=target
    )
    assert bias_sq == pytest.approx(0.005, abs=1e-12)
    assert variance == pytest.approx(0.02 / 3.0, abs=1e-12)
    assert noise == pytest.approx(0.01, abs=1e-12)
    assert mse == pytest.approx(0.025 + 0.02 / 3.0, abs=1e-12)


def test_bias_variance_noise_recovers_known_label_noise():
    # Mirrors crates/openquant/tests/ensemble_methods.rs::
    # test_bias_variance_noise_recovers_known_label_noise, with Python's RNG: lines fitted to
    # noisy samples of a sine, scored on fresh labels with known noise variance sigma^2.
    rng = random.Random(128)
    sigma = 0.5
    f = lambda x: math.sin(2 * math.pi * x)  # noqa: E731
    x_test = [(i + 0.5) / 400 for i in range(400)]
    y_expected = [f(x) for x in x_test]
    predictions = []
    for _ in range(50):
        x = [rng.random() for _ in range(30)]
        y = [f(v) + rng.gauss(0.0, sigma) for v in x]
        slope, intercept = statistics.linear_regression(x, y)
        predictions.append([intercept + slope * v for v in x_test])

    noises, gaps = [], []
    for _ in range(200):
        y_true = [m + rng.gauss(0.0, sigma) for m in y_expected]
        bias_sq, variance, noise, mse = ensemble.bias_variance_noise(
            y_true, predictions, y_expected=y_expected
        )
        noises.append(noise)
        gaps.append(mse - (bias_sq + variance + noise))

    assert statistics.fmean(noises) == pytest.approx(sigma**2, abs=0.006)
    assert abs(statistics.fmean(gaps)) < 0.01


def test_bias_variance_noise_rejects_mismatched_target():
    with pytest.raises(ValueError, match="y_expected"):
        ensemble.bias_variance_noise([1.0, 0.0], [[0.9, 0.1]], y_expected=[1.0])


def test_bootstrap_and_sequential_bootstrap_indices():
    # Mirrors crates/openquant/tests/ensemble_methods.rs::
    # test_bootstrap_and_sequential_bootstrap_shapes
    indices = ensemble.bootstrap_sample_indices(10, 6, 7)
    assert len(indices) == 6
    assert all(0 <= i < 10 for i in indices)
    assert indices == ensemble.bootstrap_sample_indices(10, 6, 7)

    ind_mat = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]]
    sequential = ensemble.sequential_bootstrap_sample_indices(ind_mat, 8, 11)
    assert len(sequential) == 8
    assert all(0 <= i < len(ind_mat[0]) for i in sequential)
    assert sequential == ensemble.sequential_bootstrap_sample_indices(ind_mat, 8, 11)


def test_aggregation_helpers():
    # Mirrors crates/openquant/tests/ensemble_methods.rs::test_aggregation_helpers
    assert ensemble.aggregate_regression_mean([[1.0, 3.0], [3.0, 1.0]]) == [2.0, 2.0]

    # Label vectors cross the boundary as `bytes` (Rust Vec<u8>), hence list().
    vote = ensemble.aggregate_classification_vote([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
    assert list(vote) == [1, 1, 1]

    probabilities, labels = ensemble.aggregate_classification_probability_mean(
        [[0.9, 0.2], [0.7, 0.4], [0.8, 0.3]], 0.5
    )
    assert abs(probabilities[0] - 0.8) < 1e-12
    assert abs(probabilities[1] - 0.3) < 1e-12
    assert list(labels) == [1, 0]


def test_bagging_variance_reduction():
    # Mirrors crates/openquant/tests/ensemble_methods.rs::
    # test_variance_reduction_and_redundancy_failure_mode
    low_corr = ensemble.bagging_ensemble_variance(1.0, 0.0, 10)
    assert abs(low_corr - 0.1) < 1e-12

    high_corr = ensemble.bagging_ensemble_variance(1.0, 0.95, 10)
    assert high_corr > 0.9
    assert high_corr > low_corr
    # AFML 6.2: sigma^2 * (rho + (1 - rho) / N)
    assert high_corr == pytest.approx(0.95 + 0.05 / 10, abs=1e-12)


def test_pairwise_correlation_and_strategy_recommendation():
    # Mirrors crates/openquant/tests/ensemble_methods.rs::
    # test_pairwise_correlation_and_strategy_recommendation
    weak_predictions = [
        [0.50, 0.52, 0.48, 0.50],
        [0.51, 0.53, 0.49, 0.51],
        [0.49, 0.51, 0.47, 0.49],
    ]
    corr = ensemble.average_pairwise_prediction_correlation(weak_predictions)
    assert corr > 0.95
    # The three rows differ only by a constant shift, so they are perfectly correlated.
    assert corr == pytest.approx(1.0, abs=1e-9)

    weak = ensemble.recommend_bagging_vs_boosting(0.53, corr, 0.8, 1.0, 16)
    assert weak["recommended"] == "boosting"

    strong_diverse = ensemble.recommend_bagging_vs_boosting(0.68, 0.15, 0.25, 1.0, 16)
    assert strong_diverse["recommended"] == "bagging"
    assert strong_diverse["expected_variance_reduction"] > 0.0
    assert strong_diverse["expected_bagging_variance"] == pytest.approx(0.15 + 0.85 / 16)


def test_ensemble_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="length mismatch"):
        ensemble.aggregate_regression_mean([[1.0, 2.0], [1.0]])
    with pytest.raises(ValueError, match="cannot be empty"):
        ensemble.aggregate_regression_mean([])
    with pytest.raises(ValueError, match="n_estimators"):
        ensemble.bagging_ensemble_variance(1.0, 0.0, 0)
    with pytest.raises(ValueError, match="must be > 0"):
        ensemble.bootstrap_sample_indices(0, 5, 1)
