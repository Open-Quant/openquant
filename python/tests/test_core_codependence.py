import math

import pytest

from _core_fixtures import load_csv_columns

from openquant import codependence


def _load_series():
    return load_csv_columns("codependence/random_state_42.csv", ["x", "y_1", "y_2"])


def _corrcoef(x, y):
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    var_x = sum((a - mean_x) ** 2 for a in x)
    var_y = sum((b - mean_y) ** 2 for b in y)
    return cov / math.sqrt(var_x * var_y)


def test_correlation_distances_match_mlfinlab():
    # Mirrors crates/openquant/tests/codependence.rs::test_correlations
    x, y_1, y_2 = _load_series()

    assert abs(codependence.angular_distance(x, y_1) - 0.6703650607372927) < 1e-6
    assert abs(codependence.absolute_angular_distance(x, y_1) - 0.6703650607372927) < 1e-6
    assert abs(codependence.squared_angular_distance(x, y_1) - 0.7034750294490113) < 1e-6
    assert abs(codependence.distance_correlation(x, y_1) - 0.529291364408913) < 1e-6
    assert abs(codependence.distance_correlation(x, y_2) - 0.5216239463593741) < 1e-6


def test_information_metrics_match_mlfinlab():
    # Mirrors crates/openquant/tests/codependence.rs::test_information_metrics
    x, y_1, _ = _load_series()

    assert abs(codependence.get_mutual_info(x, y_1) - 0.5228688725834145) < 1e-6
    assert abs(codependence.get_mutual_info(x, y_1, normalize=True) - 0.6409642333987833) < 1e-6
    assert abs(codependence.get_mutual_info(x, y_1, n_bins=10) - 0.6264238716396385) < 1e-6

    voi = codependence.variation_of_information_score
    assert abs(voi(x, y_1) - 1.425767548566149) < 1e-6
    assert abs(voi(x, y_1, normalize=True) - 0.7316744843171117) < 1e-6
    assert abs(voi(x, y_1, n_bins=10) - 1.4184909443978817) < 1e-6


def test_optimal_number_of_bins_match_mlfinlab():
    # Mirrors crates/openquant/tests/codependence.rs::test_number_of_bins
    x, y_1, _ = _load_series()

    assert codependence.get_optimal_number_of_bins(len(x)) == 15
    assert codependence.get_optimal_number_of_bins(len(x), _corrcoef(x, y_1)) == 9


def test_codependence_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="different lengths"):
        codependence.angular_distance([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="different lengths"):
        codependence.get_mutual_info([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="too short"):
        codependence.distance_correlation([], [])
    with pytest.raises(ValueError, match="too short"):
        codependence.get_optimal_number_of_bins(0)
