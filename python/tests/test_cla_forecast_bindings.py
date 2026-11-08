import pytest
from openquant import cla


@pytest.mark.parametrize("mu", [1e-8, 0.1])
def test_equal_forecasts_preserve_tangency_weights(mu):
    result = cla.allocate_cla(
        expected_returns=[mu, mu],
        covariance_matrix=[[4.0, 0.0], [0.0, 1.0]],
        solution="max_sharpe",
    )
    assert result["weights"][0] == pytest.approx([0.2, 0.8], abs=1e-9)


@pytest.mark.parametrize("means", [[0.1], [1e-8, 1e-8], [1e-8 + 5e-13, 1e-8]])
def test_frontier_reports_unmodified_forecasts(means):
    covariance = [[float(i == j) for j in range(len(means))] for i in range(len(means))]
    result = cla.allocate_cla(
        expected_returns=means, covariance_matrix=covariance, solution="efficient_frontier"
    )
    for weights, reported in zip(result["weights"], result["efficient_frontier_means"]):
        assert reported == pytest.approx(
            sum(w * m for w, m in zip(weights, means)), rel=1e-13, abs=0
        )


def test_near_equal_forecasts_preserve_maximum_return_order():
    result = cla.allocate_cla(
        expected_returns=[1e-8 + 5e-13, 1e-8],
        covariance_matrix=[[1.0, 0.0], [0.0, 1.0]],
        solution="cla_turning_points",
    )
    assert result["weights"][0] == pytest.approx([1.0, 0.0], abs=1e-9)
