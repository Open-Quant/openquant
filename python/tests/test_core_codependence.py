import math

import pytest
from _core_fixtures import load_csv_columns, load_json
from openquant import codependence

# Computed from the published definitions by tests/fixtures/codependence/generate.py
# (MLAM ch. 3 snippets 3.1-3.3; Szekely, Rizzo & Bakirov 2007). Same operations up to
# summation order and the same histogram bin assignment, so only rounding remains.
REF = load_json("codependence/reference.json")
TOL = 1e-12


def _load_series():
    return load_csv_columns("codependence/random_state_42.csv", ["x", "y_1", "y_2"])


def _corrcoef(x, y):
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    var_x = sum((a - mean_x) ** 2 for a in x)
    var_y = sum((b - mean_y) ** 2 for b in y)
    return cov / math.sqrt(var_x * var_y)


def test_correlation_distances_match_reference():
    # Mirrors crates/openquant/tests/codependence.rs::test_correlations
    x, y_1, y_2 = _load_series()

    assert codependence.angular_distance(x, y_1) == pytest.approx(
        REF["angular_distance_x_y1"], abs=TOL
    )
    assert codependence.absolute_angular_distance(x, y_1) == pytest.approx(
        REF["absolute_angular_distance_x_y1"], abs=TOL
    )
    assert codependence.squared_angular_distance(x, y_1) == pytest.approx(
        REF["squared_angular_distance_x_y1"], abs=TOL
    )
    assert codependence.distance_correlation(x, y_1) == pytest.approx(
        REF["distance_correlation_x_y1"], abs=TOL
    )
    assert codependence.distance_correlation(x, y_2) == pytest.approx(
        REF["distance_correlation_x_y2"], abs=TOL
    )


def test_information_metrics_match_reference():
    # Mirrors crates/openquant/tests/codependence.rs::test_information_metrics
    x, y_1, _ = _load_series()

    mi = codependence.get_mutual_info
    assert mi(x, y_1) == pytest.approx(REF["mutual_info_x_y1"], abs=TOL)
    assert mi(x, y_1, normalize=True) == pytest.approx(REF["mutual_info_normalised_x_y1"], abs=TOL)
    assert mi(x, y_1, n_bins=10) == pytest.approx(REF["mutual_info_x_y1_10_bins"], abs=TOL)

    voi = codependence.variation_of_information_score
    assert voi(x, y_1) == pytest.approx(REF["variation_of_information_x_y1"], abs=TOL)
    assert voi(x, y_1, normalize=True) == pytest.approx(
        REF["variation_of_information_normalised_x_y1"], abs=TOL
    )
    assert voi(x, y_1, n_bins=10) == pytest.approx(
        REF["variation_of_information_x_y1_10_bins"], abs=TOL
    )


def test_optimal_number_of_bins_match_reference():
    # Mirrors crates/openquant/tests/codependence.rs::test_number_of_bins
    x, y_1, _ = _load_series()

    assert codependence.get_optimal_number_of_bins(len(x)) == REF["optimal_bins_univariate"]
    assert (
        codependence.get_optimal_number_of_bins(len(x), _corrcoef(x, y_1))
        == REF["optimal_bins_bivariate_x_y1"]
    )


def test_codependence_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="different lengths"):
        codependence.angular_distance([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="different lengths"):
        codependence.get_mutual_info([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="too short"):
        codependence.distance_correlation([], [])
    with pytest.raises(ValueError, match="too short"):
        codependence.get_optimal_number_of_bins(0)
