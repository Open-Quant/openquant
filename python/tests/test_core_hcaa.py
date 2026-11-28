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


def test_hcaa_sharpe_ratio_estimates_expected_returns_from_asset_returns():
    # #185 item 13: returns alone are enough; same answer as from the prices behind them.
    prices, names = _load_prices_and_names()
    returns = [
        [prices[t][j] / prices[t - 1][j] - 1.0 for j in range(len(names))]
        for t in range(1, len(prices))
    ]
    from_returns, _ = hcaa.allocate_hcaa(
        names, asset_returns=returns, allocation_metric="sharpe_ratio", optimal_num_clusters=5
    )
    from_prices, _ = hcaa.allocate_hcaa(
        names, asset_prices=prices, allocation_metric="sharpe_ratio", optimal_num_clusters=5
    )
    assert from_returns == pytest.approx(from_prices, abs=1e-12)


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
@pytest.mark.parametrize("linkage", ["single", "complete", "average", "ward"])
@pytest.mark.parametrize(
    "metric", ["minimum_variance", "minimum_standard_deviation", "equal_weighting"]
)
@pytest.mark.parametrize("k", [None, 2, 4])
def test_hcaa_tree_matches_independent_reference(distance, linkage, metric, k):
    # tests/fixtures/hcaa/generate.py: scipy's linkage(method=linkage) on the pairwise
    # distances, or on the square distance matrix as AFML Snippet 16.4 passes it, then the tree
    # walk in numpy.
    reference = json.loads((FIXTURES / "hcaa" / "reference.json").read_text())
    want = reference["stock_prices"][distance][linkage]
    prices, names = _load_prices_and_names()

    weights, order = hcaa.allocate_hcaa(
        names,
        asset_prices=prices,
        allocation_metric=metric,
        optimal_num_clusters=k,
        distance=distance,
        linkage=linkage,
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


def test_hcaa_linkage_default_and_validation():
    prices, names = _load_prices_and_names()
    kwargs = {"asset_prices": prices, "allocation_metric": "minimum_variance"}
    default = hcaa.allocate_hcaa(names, **kwargs)
    # The default is Ward, as in mlfinlab's and R HierPortfolios' HCAA.
    assert default == hcaa.allocate_hcaa(names, **kwargs, linkage="ward")
    assert default == hcaa.allocate_hcaa(names, **kwargs, linkage="Ward")
    for other in ("single", "complete", "average"):
        assert default != hcaa.allocate_hcaa(names, **kwargs, linkage=other)
    with pytest.raises(ValueError, match="unknown linkage"):
        hcaa.allocate_hcaa(names, **kwargs, linkage="centroid")


COV_THREE = [[1.0, 0.1, 0.8], [0.1, 1.0, 0.2], [0.8, 0.2, 1.0]]


def test_hcaa_finite_covariance_rational_control():
    # The first cluster is {a, c}; its equal-weight variance is 9/10,
    # giving it 10/19 of the portfolio, split equally between the two assets.
    weights, _ = hcaa.allocate_hcaa(
        ["a", "b", "c"], covariance_matrix=COV_THREE, allocation_metric="minimum_variance"
    )
    assert weights == pytest.approx([5 / 19, 9 / 19, 5 / 19], abs=1e-12)


@pytest.mark.parametrize("distance", ["correlation", "distance_of_distances"])
@pytest.mark.parametrize("metric", ["equal_weighting", "minimum_variance"])
@pytest.mark.parametrize(
    "bad_indices, bad_value",
    [
        ([(0, 1), (1, 0)], float("nan")),
        ([(0, 0)], float("nan")),
        ([(1, 1)], float("inf")),
        ([(1, 0)], float("-inf")),
    ],
)
def test_hcaa_rejects_nonfinite_covariance(distance, metric, bad_indices, bad_value):
    covariance = [row.copy() for row in COV_THREE]
    for row, col in bad_indices:
        covariance[row][col] = bad_value
    with pytest.raises(ValueError, match="covariance.*finite"):
        hcaa.allocate_hcaa(
            ["a", "b", "c"],
            covariance_matrix=covariance,
            allocation_metric=metric,
            distance=distance,
        )


def test_hcaa_rejects_overflowed_estimated_covariance():
    with pytest.raises(ValueError, match="covariance.*finite"):
        hcaa.allocate_hcaa(
            ["a", "b"],
            asset_returns=[[1e200, -1e200], [-1e200, 1e200]],
            allocation_metric="minimum_variance",
        )


@pytest.mark.parametrize(
    "metric", ["expected_shortfall", "conditional_drawdown_risk", "sharpe_ratio"]
)
def test_hcaa_rejects_nonfinite_history_even_with_explicit_covariance(metric):
    with pytest.raises(ValueError, match="asset_returns.*finite"):
        hcaa.allocate_hcaa(
            ["a", "b"],
            asset_returns=[[0.01, float("nan")], [-0.01, 0.02]],
            covariance_matrix=COV_2,
            allocation_metric=metric,
        )


def test_hcaa_rejects_nonfinite_expected_returns_only_when_consumed():
    expected = [0.1, float("inf")]
    with pytest.raises(ValueError, match="expected_asset_returns.*finite"):
        hcaa.allocate_hcaa(
            ["a", "b"],
            covariance_matrix=COV_2,
            expected_asset_returns=expected,
            allocation_metric="sharpe_ratio",
        )
    weights, _ = hcaa.allocate_hcaa(
        ["a", "b"],
        covariance_matrix=COV_2,
        expected_asset_returns=expected,
        allocation_metric="equal_weighting",
    )
    assert weights == [0.5, 0.5]


def test_hcaa_rejects_nonfinite_sampled_prices_when_consumed():
    # A positive infinity used as denominator produces a finite -1 return;
    # rejecting bad prices themselves is needed, not just checking the quotient.
    with pytest.raises(ValueError, match="asset_prices.*finite"):
        hcaa.allocate_hcaa(
            ["a", "b"],
            asset_prices=[[float("inf"), 100.0], [105.0, 95.0]],
            covariance_matrix=COV_2,
            allocation_metric="expected_shortfall",
        )


def test_hcaa_covariance_based_metric_still_ignores_unused_history_values():
    weights, _ = hcaa.allocate_hcaa(
        ["a", "b"],
        asset_returns=[[float("nan"), float("inf")]],
        covariance_matrix=COV_2,
        allocation_metric="equal_weighting",
    )
    assert weights == [0.5, 0.5]


@pytest.mark.parametrize("metric", ["minimum_variance", "sharpe_ratio"])
@pytest.mark.parametrize(
    "variances, expected_left",
    [
        ([1e308, 1e308], 1 / 2),
        ([1e308, 9e307], 9 / 19),
        ([1e-320, 3e-320], 3 / 4),
        ([1e-310, 2e-310], 2 / 3),
    ],
)
def test_hcaa_minimum_variance_split_stays_scale_invariant(metric, variances, expected_left):
    # For two independent singletons with positive variances, 1/V weights
    # reduce to right_variance / (left_variance + right_variance).
    weights, _ = hcaa.allocate_hcaa(
        ["a", "b"],
        covariance_matrix=[[variances[0], 0.0], [0.0, variances[1]]],
        allocation_metric=metric,
        expected_asset_returns=[-1.0, -1.0],  # Sharpe must use its variance fallback.
    )
    assert weights == pytest.approx([expected_left, 1 - expected_left], abs=1e-12)
