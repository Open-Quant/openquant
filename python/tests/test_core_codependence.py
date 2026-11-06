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


@pytest.mark.parametrize("scale", [1.7, -1.7])
@pytest.mark.parametrize(
    "distance_name",
    ["angular_distance", "absolute_angular_distance", "squared_angular_distance"],
)
def test_angular_distances_at_perfect_correlation_boundaries(distance_name, scale):
    # Decimal returns that round rho just outside [-1, 1] before the square root.
    x = [-0.019515858338768028, 0.018977241163271315, 0.005709923350367178]
    y = [v * scale for v in x]
    expected = 1.0 if distance_name == "angular_distance" and scale < 0.0 else 0.0
    assert getattr(codependence, distance_name)(x, y) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize(
    "function_name",
    [
        "angular_distance",
        "absolute_angular_distance",
        "squared_angular_distance",
        "get_mutual_info",
        "variation_of_information_score",
    ],
)
def test_decimal_constant_series_raise_zero_variance(function_name, swap):
    # The rounded arithmetic mean must not manufacture variance for repeated values.
    x, y = [0.1] * 10, [i / 100 for i in range(10)]
    if swap:
        x, y = y, x
    with pytest.raises(ValueError, match="zero variance"):
        getattr(codependence, function_name)(x, y)


def test_constant_series_keep_explicit_histogram_contract():
    constant = [0.1] * 10
    variable = [i / 100 for i in range(10)]
    for x, y in [(constant, variable), (variable, constant)]:
        assert codependence.get_mutual_info(x, y, n_bins=2) == pytest.approx(0.0, abs=TOL)
        assert codependence.variation_of_information_score(
            x, y, n_bins=2, normalize=True
        ) == pytest.approx(1.0, abs=TOL)
