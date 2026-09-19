import pytest

from openquant import cla

MU = [0.1, 0.2]
COV = [[0.04, 0.006], [0.006, 0.09]]


def test_cla_min_volatility_closed_form():
    # Two assets, long-only: w_0 = (s_11 - s_01) / (s_00 + s_11 - 2 s_01) = 0.084 / 0.118.
    # Counterpart of crates/openquant/tests/cla.rs::test_cla_min_volatility, which only
    # checks that the weights sum to one.
    out = cla.allocate_cla(expected_returns=MU, covariance_matrix=COV, solution="min_volatility")

    assert len(out["weights"]) == 1
    assert out["weights"][0] == pytest.approx([0.084 / 0.118, 0.034 / 0.118], abs=1e-9)


def test_cla_max_sharpe_closed_form():
    # Tangency portfolio with zero risk-free rate: w proportional to inv(COV) @ MU
    # = [0.0078, 0.0074] (up to the determinant), i.e. w_0 = 78 / 152.
    # Counterpart of crates/openquant/tests/cla.rs::test_cla_max_sharpe.
    out = cla.allocate_cla(expected_returns=MU, covariance_matrix=COV, solution="max_sharpe")

    weights = out["weights"][0]
    assert all(w >= -1e-12 for w in weights)
    assert abs(sum(weights) - 1.0) < 1e-6
    assert weights == pytest.approx([78.0 / 152.0, 74.0 / 152.0], abs=1e-6)


def test_cla_efficient_frontier_shapes():
    # Mirrors crates/openquant/tests/cla.rs::test_cla_efficient_frontier
    out = cla.allocate_cla(
        expected_returns=MU, covariance_matrix=COV, solution="efficient_frontier"
    )

    assert len(out["efficient_frontier_means"]) == len(out["efficient_frontier_sigma"])
    assert len(out["efficient_frontier_sigma"]) == len(out["weights"])
    assert out["efficient_frontier_sigma"][-1] <= out["efficient_frontier_sigma"][0]
    assert out["efficient_frontier_means"][-1] <= out["efficient_frontier_means"][0]
    # The last frontier point is the minimum-variance portfolio.
    min_vol = [0.084 / 0.118, 0.034 / 0.118]
    variance = sum(min_vol[i] * min_vol[j] * COV[i][j] for i in range(2) for j in range(2))
    assert out["efficient_frontier_sigma"][-1] == pytest.approx(variance**0.5, abs=1e-9)
    assert out["efficient_frontier_means"][-1] == pytest.approx(
        sum(w * m for w, m in zip(min_vol, MU)), abs=1e-9
    )


def test_cla_rejects_invalid_inputs():
    # Mirrors the error cases of crates/openquant/tests/cla.rs
    with pytest.raises(ValueError, match="MissingInputs"):
        cla.allocate_cla()
    with pytest.raises(ValueError, match="UnknownSolution"):
        cla.allocate_cla(expected_returns=MU, covariance_matrix=COV, solution="rubbish")
    with pytest.raises(ValueError, match="rectangular"):
        cla.allocate_cla(expected_returns=MU, covariance_matrix=[[0.04, 0.006], [0.006]])


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FINDING: allocate_cla(asset_prices=...) can never succeed: the binding wraps prices "
        "as AssetPricesInput::RawMatrix, which CLA::allocate rejects unconditionally with "
        "InvalidAssetPrices('Asset prices matrix must be a dataframe')"
    ),
)
def test_cla_accepts_asset_prices():
    prices = [[100.0, 50.0], [101.0, 50.5], [100.5, 51.5], [102.0, 51.0], [103.0, 52.5]]
    out = cla.allocate_cla(asset_prices=prices, solution="min_volatility")
    assert abs(sum(out["weights"][0]) - 1.0) < 1e-6


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FINDING: with expected_returns + covariance_matrix inputs CLA reports a single "
        "turning point (the minimum-variance portfolio) and an efficient frontier of 100 "
        "identical points; the maximum-return turning point [0, 1] is missing"
    ),
)
def test_cla_turning_points_start_at_max_return_asset():
    out = cla.allocate_cla(expected_returns=MU, covariance_matrix=COV)
    assert out["weights"][0] == pytest.approx([0.0, 1.0], abs=1e-9)
    assert len(out["weights"]) >= 2
