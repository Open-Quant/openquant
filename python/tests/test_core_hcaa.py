import csv
import json

import pytest
from _core_fixtures import FIXTURES
from openquant import hcaa

COV_2 = [[0.04, 0.006], [0.006, 0.09]]


def _load_prices_and_names():
    with (FIXTURES / "portfolio_optimization" / "stock_prices.csv").open("r", newline="") as f:
        rows = list(csv.reader(f))
    return [[float(x) for x in row[1:]] for row in rows[1:]], rows[0][1:]


def test_hcaa_two_assets_closed_form_per_metric():
    # One bisection between two single-asset clusters, so each metric has a closed form:
    # alpha = 1 - risk_0 / (risk_0 + risk_1), with risk = variance or standard deviation.
    names = ["a", "b"]

    weights, order = hcaa.allocate_hcaa(
        names, covariance_matrix=COV_2, allocation_metric="minimum_variance"
    )
    assert order == [0, 1]
    assert weights == pytest.approx([1 - 0.04 / 0.13, 0.04 / 0.13], abs=1e-12)

    weights, _ = hcaa.allocate_hcaa(
        names, covariance_matrix=COV_2, allocation_metric="minimum_standard_deviation"
    )
    assert weights == pytest.approx([1 - 0.2 / 0.5, 0.2 / 0.5], abs=1e-12)

    weights, _ = hcaa.allocate_hcaa(names, covariance_matrix=COV_2)
    assert weights == pytest.approx([0.5, 0.5], abs=1e-12)


@pytest.mark.parametrize(
    "metric, expected_returns",
    [
        ("equal_weighting", "mean"),
        ("minimum_variance", "mean"),
        ("minimum_standard_deviation", "mean"),
        ("sharpe_ratio", "mean"),
        ("sharpe_ratio", "exponential"),
        ("expected_shortfall", "mean"),
        ("conditional_drawdown_risk", "mean"),
    ],
)
def test_hcaa_on_price_fixture(metric, expected_returns):
    # Mirrors crates/openquant/tests/hcaa.rs::test_hcaa_equal_weight, test_hcaa_min_variance,
    # test_hcaa_min_standard_deviation, test_hcaa_sharpe_ratio_mean,
    # test_hcaa_sharpe_ratio_exponential, test_hcaa_expected_shortfall and
    # test_hcaa_conditional_drawdown_risk. test_quasi_diagonalization is deliberately NOT
    # mirrored: see test_hcaa_ordering_depends_on_the_data below.
    prices, names = _load_prices_and_names()
    weights, order = hcaa.allocate_hcaa(
        names,
        asset_prices=prices,
        allocation_metric=metric,
        optimal_num_clusters=5,
        calculate_expected_returns=expected_returns,
    )

    assert len(weights) == len(names)
    assert all(w >= 0.0 for w in weights)
    assert abs(sum(weights) - 1.0) < 1e-6
    assert sorted(order) == list(range(len(names)))


def test_hcaa_rejects_invalid_inputs():
    # Mirrors crates/openquant/tests/hcaa.rs::test_all_inputs_none,
    # test_value_error_for_allocation_metric, test_value_error_for_unknown_returns and
    # test_value_error_for_sharpe_ratio_without_prices_or_expected
    prices, names = _load_prices_and_names()
    with pytest.raises(ValueError, match="no data"):
        hcaa.allocate_hcaa(names)
    with pytest.raises(ValueError, match="unknown allocation metric"):
        hcaa.allocate_hcaa(names, asset_prices=prices, allocation_metric="random_metric")
    with pytest.raises(ValueError, match="unknown returns method"):
        hcaa.allocate_hcaa(
            names,
            asset_prices=prices,
            allocation_metric="sharpe_ratio",
            calculate_expected_returns="unknown_returns",
        )
    with pytest.raises(ValueError, match="needs expected returns"):
        hcaa.allocate_hcaa(["a", "b"], covariance_matrix=COV_2, allocation_metric="sharpe_ratio")


def test_hcaa_reads_asset_returns_as_rows_of_observations():
    n_obs = 40
    a = [0.001 if t % 2 == 0 else -0.001 for t in range(n_obs)]
    b = [0.02 if t % 4 < 2 else -0.02 for t in range(n_obs)]

    def sample_variance(values):
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / (len(values) - 1)

    expected_a = 1 - sample_variance(a) / (sample_variance(a) + sample_variance(b))

    weights, _ = hcaa.allocate_hcaa(
        ["a", "b"],
        asset_returns=[[a[t], b[t]] for t in range(n_obs)],
        allocation_metric="minimum_variance",
    )

    assert weights == pytest.approx([expected_a, 1 - expected_a], abs=1e-9)


def test_hcaa_ordering_depends_on_the_data():
    n = 23
    cov = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    cov[0][12] = cov[12][0] = 0.99

    _, order = hcaa.allocate_hcaa([str(i) for i in range(n)], covariance_matrix=cov)

    assert abs(order.index(0) - order.index(12)) == 1


@pytest.mark.parametrize("n", [22, 24])
def test_hcaa_ordering_places_correlated_pair_together(n):
    # Control for the 23-asset case above: uncorrelated assets except 0 and 12, which
    # correlate at 0.99, so quasi-diagonalisation must make them neighbours.
    cov = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    cov[0][12] = cov[12][0] = 0.99

    _, order = hcaa.allocate_hcaa([str(i) for i in range(n)], covariance_matrix=cov)

    assert abs(order.index(0) - order.index(12)) == 1


@pytest.mark.parametrize("distance", ["correlation", "distance_of_distances"])
@pytest.mark.parametrize(
    "metric", ["minimum_variance", "minimum_standard_deviation", "equal_weighting"]
)
@pytest.mark.parametrize("k", [None, 2, 4])
def test_hcaa_distance_matches_independent_reference(distance, metric, k):
    # tests/fixtures/hcaa/generate.py: scipy single linkage on the pairwise distances, or on the
    # square distance matrix as AFML Snippet 16.4 passes it, then the tree walk in numpy.
    reference = json.loads((FIXTURES / "hcaa" / "reference.json").read_text())
    want = reference["stock_prices"][distance]
    prices, names = _load_prices_and_names()

    weights, order = hcaa.allocate_hcaa(
        names,
        asset_prices=prices,
        allocation_metric=metric,
        optimal_num_clusters=k,
        distance=distance,
    )

    assert order == want["order"]
    assert weights == pytest.approx(
        want["weights"][metric]["none" if k is None else str(k)], abs=1e-10
    )


def test_hcaa_distance_default_and_validation():
    prices, names = _load_prices_and_names()
    kwargs = {"asset_prices": prices, "allocation_metric": "minimum_variance"}
    default = hcaa.allocate_hcaa(names, **kwargs)
    # The default is the pairwise tree (mlfinlab's), unlike HRP's default.
    assert default == hcaa.allocate_hcaa(names, **kwargs, distance="correlation")
    assert default != hcaa.allocate_hcaa(names, **kwargs, distance="distance_of_distances")
    assert default == hcaa.allocate_hcaa(names, **kwargs, distance="Correlation")
    with pytest.raises(ValueError, match="unknown distance"):
        hcaa.allocate_hcaa(names, **kwargs, distance="euclidean")
