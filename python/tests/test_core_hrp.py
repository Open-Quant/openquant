import csv

import pytest

from _core_fixtures import FIXTURES

from openquant import hrp

# Two tight pairs: (0, 1) correlate at 0.9 and (2, 3) at 0.8, the pairs at 0.1.
_VOLS = [0.1, 0.2, 0.3, 0.4]
_CORR = [
    [1.0, 0.9, 0.1, 0.1],
    [0.9, 1.0, 0.1, 0.1],
    [0.1, 0.1, 1.0, 0.8],
    [0.1, 0.1, 0.8, 1.0],
]
COV_4 = [[_CORR[i][j] * _VOLS[i] * _VOLS[j] for j in range(4)] for i in range(4)]


def _load_prices_and_names():
    with (FIXTURES / "portfolio_optimization" / "stock_prices.csv").open("r", newline="") as f:
        rows = list(csv.reader(f))
    return [[float(x) for x in row[1:]] for row in rows[1:]], rows[0][1:]


def _cluster_variance(cov, members):
    inverse = [1.0 / cov[i][i] for i in members]
    weights = [v / sum(inverse) for v in inverse]
    return sum(
        weights[a] * weights[b] * cov[i][j]
        for a, i in enumerate(members)
        for b, j in enumerate(members)
    )


def test_hrp_two_assets_is_inverse_variance_split():
    # AFML 16.4.3 with one bisection: alpha = 1 - V_0 / (V_0 + V_1).
    cov = [[0.04, 0.006], [0.006, 0.09]]
    weights, order = hrp.allocate_hrp(["a", "b"], covariance_matrix=cov)

    assert order == [0, 1]
    assert weights == pytest.approx([1 - 0.04 / 0.13, 0.04 / 0.13], abs=1e-12)


def test_hrp_four_assets_recursive_bisection():
    # Recursive bisection worked from COV_4: split {0,1} | {2,3} on inverse-variance
    # cluster variances, then split each pair on the single-asset variances.
    left = _cluster_variance(COV_4, [0, 1])
    right = _cluster_variance(COV_4, [2, 3])
    alpha = 1 - left / (left + right)
    expected = [
        alpha * (1 - 0.01 / 0.05),
        alpha * (0.01 / 0.05),
        (1 - alpha) * (1 - 0.09 / 0.25),
        (1 - alpha) * (0.09 / 0.25),
    ]

    weights, order = hrp.allocate_hrp(list("abcd"), covariance_matrix=COV_4)

    assert order == [0, 1, 2, 3]
    assert weights == pytest.approx(expected, abs=1e-12)
    assert expected[0] == pytest.approx(0.70477245, abs=1e-8)


def test_hrp_on_price_fixture():
    # Mirrors crates/openquant/tests/hrp.rs::test_hrp and test_hrp_with_shrinkage.
    # test_quasi_diagonalization is deliberately NOT mirrored: see
    # test_hrp_ordering_depends_on_the_data below.
    prices, names = _load_prices_and_names()
    for use_shrinkage in (False, True):
        weights, order = hrp.allocate_hrp(
            names, asset_prices=prices, use_shrinkage=use_shrinkage
        )
        assert len(weights) == len(names)
        assert all(w >= 0.0 for w in weights)
        assert abs(sum(weights) - 1.0) < 1e-6
        assert sorted(order) == list(range(len(names)))


def test_hrp_rejects_invalid_inputs():
    # Mirrors crates/openquant/tests/hrp.rs::test_all_inputs_none and
    # test_value_error_for_incorrect_dimensions
    prices, names = _load_prices_and_names()
    with pytest.raises(ValueError, match="NoData"):
        hrp.allocate_hrp(names)
    with pytest.raises(ValueError, match="DimensionMismatch"):
        hrp.allocate_hrp(names[:-1], asset_prices=prices)
    with pytest.raises(ValueError, match="rectangular"):
        hrp.allocate_hrp(["a", "b"], asset_returns=[[0.1, 0.2], [0.1]])


def test_hrp_reads_asset_returns_as_rows_of_observations():
    # Asset `a` is 20x less volatile than `b`, so it must take nearly all the weight.
    n_obs = 40
    a = [0.001 if t % 2 == 0 else -0.001 for t in range(n_obs)]
    b = [0.02 if t % 4 < 2 else -0.02 for t in range(n_obs)]

    def sample_variance(values):
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / (len(values) - 1)

    expected_a = 1 - sample_variance(a) / (sample_variance(a) + sample_variance(b))

    weights, _ = hrp.allocate_hrp(["a", "b"], asset_returns=[[a[t], b[t]] for t in range(n_obs)])

    assert weights == pytest.approx([expected_a, 1 - expected_a], abs=1e-9)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FINDING: openquant::hrp overwrites ordered_indices with a hard-coded list whenever "
        "there are exactly 23 assets (the size of the stock_prices fixture), so "
        "crates/openquant/tests/hrp.rs::test_quasi_diagonalization cannot fail"
    ),
)
def test_hrp_ordering_depends_on_the_data():
    n = 23
    cov = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    cov[0][12] = cov[12][0] = 0.99

    _, order = hrp.allocate_hrp([str(i) for i in range(n)], covariance_matrix=cov)

    assert abs(order.index(0) - order.index(12)) == 1


@pytest.mark.parametrize("n", [22, 24])
def test_hrp_ordering_places_correlated_pair_together(n):
    # Control for the 23-asset xfail above: uncorrelated assets except 0 and 12, which
    # correlate at 0.99, so quasi-diagonalisation must make them neighbours.
    cov = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    cov[0][12] = cov[12][0] = 0.99

    _, order = hrp.allocate_hrp([str(i) for i in range(n)], covariance_matrix=cov)

    assert abs(order.index(0) - order.index(12)) == 1
