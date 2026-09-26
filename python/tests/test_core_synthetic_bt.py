import math

import pytest
from openquant import synthetic_bt

# StabilityCriteria::default() in crates/openquant/src/synthetic_backtesting.rs, passed
# explicitly because the Rust tests use it and the binding's own defaults differ.
DEFAULT_CRITERIA = dict(
    random_walk_phi_threshold=0.97,
    min_peak_margin=0.20,
    min_surface_std=0.10,
    min_best_sharpe=0.30,
)


def _ou_paths(phi, intercept, equilibrium, sigma, initial_price, n_paths, horizon, seed):
    return synthetic_bt.generate_ou_paths(
        phi, intercept, equilibrium, sigma, 0.0, True, initial_price, n_paths, horizon, seed
    )


def test_generate_ou_paths_is_seeded_and_reproducible():
    # Mirrors crates/openquant/tests/synthetic_backtesting.rs::
    # test_generate_ou_paths_is_seeded_and_reproducible
    args = (0.85, 15.0, 100.0, 1.25, 0.9, True, 98.0, 16, 64)
    p1 = synthetic_bt.generate_ou_paths(*args, 42)
    p2 = synthetic_bt.generate_ou_paths(*args, 42)
    p3 = synthetic_bt.generate_ou_paths(*args, 43)

    assert p1 == p2
    assert p1 != p3
    assert len(p1) == 16
    assert all(len(path) == 64 and path[0] == 98.0 for path in p1)


def test_generate_ou_paths_without_noise_is_the_ar1_recursion():
    # With sigma = 0 the path is x_t = intercept + phi * x_{t-1}, starting at initial_price.
    paths = _ou_paths(0.5, 1.0, 2.0, 0.0, 0.0, 1, 5, 1)
    assert paths == [pytest.approx([0.0, 1.0, 1.5, 1.75, 1.875], abs=1e-12)]


def test_calibration_recovers_ou_phi_reasonably():
    # Mirrors crates/openquant/tests/synthetic_backtesting.rs::
    # test_calibration_recovers_ou_phi_reasonably
    path = _ou_paths(0.82, 18.0, 100.0, 0.8, 100.0, 1, 1200, 7)[0]
    fit = synthetic_bt.calibrate_ou_params(path)

    assert abs(fit["phi"] - 0.82) < 0.08
    assert fit["sigma"] > 0.0
    assert fit["stationary"] is True


def test_evaluate_rule_on_paths_hand_worked():
    # Path 1 takes profit at 1.2, path 2 stops out at -1.0, path 3 exits at the
    # max holding step with 0.2: mean 0.4/3, sample std of [1.2, -1.0, 0.2], 2 of 3 win.
    paths = [[0.0, 0.5, 1.2], [0.0, -0.6, -1.0], [0.0, 0.1, 0.2]]
    returns = [1.2, -1.0, 0.2]
    mean = sum(returns) / 3
    std = math.sqrt(sum((r - mean) ** 2 for r in returns) / 2)

    point = synthetic_bt.evaluate_rule_on_paths(paths, 1.0, 1.0, 2, 1.0)

    assert point["mean_return"] == pytest.approx(mean, abs=1e-12)
    assert point["std_return"] == pytest.approx(std, abs=1e-12)
    assert point["sharpe"] == pytest.approx(mean / std, abs=1e-12)
    assert point["win_rate"] == pytest.approx(2 / 3, abs=1e-12)
    assert point["avg_holding_steps"] == pytest.approx(2.0)


def test_regime_contrast_mean_reverting_vs_random_walk_like():
    # Mirrors crates/openquant/tests/synthetic_backtesting.rs::
    # test_regime_contrast_mean_reverting_vs_random_walk_like
    pt_grid = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    sl_grid = [0.5, 1.0, 2.0, 3.0, 4.0]
    criteria = tuple(DEFAULT_CRITERIA.values())

    mr_paths = _ou_paths(0.65, 0.35, 1.0, 1.0, 0.0, 4000, 128, 11)
    rw_paths = _ou_paths(0.995, 0.0, 0.0, 1.0, 0.0, 4000, 128, 11)
    mr = synthetic_bt.search_optimal_trading_rule(
        0.65, 0.35, 1.0, 1.0, 0.0, True, mr_paths, pt_grid, sl_grid, 64, 1.0, *criteria
    )
    rw = synthetic_bt.search_optimal_trading_rule(
        0.995, 0.0, 0.0, 1.0, 0.0, True, rw_paths, pt_grid, sl_grid, 64, 1.0, *criteria
    )

    assert mr["best_point"]["sharpe"] > rw["best_point"]["sharpe"] + 0.1
    assert mr["diagnostics"]["no_stable_optimum"] is False
    assert rw["diagnostics"]["no_stable_optimum"] is True


def test_detect_no_stable_optimum_for_flat_surface():
    # Mirrors crates/openquant/tests/synthetic_backtesting.rs::
    # test_detect_no_stable_optimum_for_flat_surface
    flat = [
        (1.0, 1.0, 0.04, 0.01, 0.25, 0.50, 10.0),
        (1.0, 2.0, 0.05, 0.01, 0.25, 0.50, 10.0),
        (2.0, 1.0, 0.03, 0.01, 0.25, 0.50, 10.0),
    ]
    diag = synthetic_bt.detect_no_stable_optimum(flat, 0.99, *DEFAULT_CRITERIA.values())

    assert diag["no_stable_optimum"] is True
    assert "no stable optimum" in diag["reason"]
    assert diag["best_sharpe"] == pytest.approx(0.05)
    assert diag["median_sharpe"] == pytest.approx(0.04)
    assert diag["peak_margin"] == pytest.approx(0.01)


def test_run_synthetic_otr_workflow_end_to_end():
    # Mirrors crates/openquant/tests/synthetic_backtesting.rs::
    # test_run_synthetic_otr_workflow_end_to_end
    historical = _ou_paths(0.75, 0.5, 2.0, 1.0, 0.0, 1, 800, 9)[0]
    result = synthetic_bt.run_synthetic_otr_workflow(
        historical,
        initial_price=0.0,
        n_paths=1500,
        horizon=96,
        seed=77,
        profit_taking_grid=[0.5, 1.0, 2.0, 3.0],
        stop_loss_grid=[0.5, 1.0, 2.0, 3.0],
        max_holding_steps=64,
        annualization_factor=1.0,
        **DEFAULT_CRITERIA,
    )

    assert len(result["response_surface"]) == 16
    assert math.isfinite(result["best_point"]["sharpe"])
    assert result["params"]["sigma"] > 0.0
    # The calibration step should recover the generating process (phi 0.75, sigma 1.0).
    assert abs(result["params"]["phi"] - 0.75) < 0.08
    assert abs(result["params"]["sigma"] - 1.0) < 0.08


def test_synthetic_bt_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="at least 3 observations"):
        synthetic_bt.calibrate_ou_params([1.0, 2.0])
    with pytest.raises(ValueError, match="paths cannot be empty"):
        synthetic_bt.evaluate_rule_on_paths([], 1.0, 1.0, 2, 1.0)
    with pytest.raises(ValueError, match="must be > 0"):
        synthetic_bt.evaluate_rule_on_paths([[0.0, 1.0]], -1.0, 1.0, 2, 1.0)
    with pytest.raises(ValueError, match="response_surface cannot be empty"):
        synthetic_bt.detect_no_stable_optimum([], 0.99, *DEFAULT_CRITERIA.values())


def test_run_synthetic_otr_workflow_documented_default_call():
    # Issue #194: the default stop-loss grid was negative, so calling the workflow without
    # grids always raised ValueError. Both default grids are now 0.25, 0.5, ..., 5.0.
    historical = _ou_paths(0.75, 0.5, 2.0, 1.0, 0.0, 1, 300, 9)[0]
    result = synthetic_bt.run_synthetic_otr_workflow(historical)

    widths = [0.25 * i for i in range(1, 21)]
    surface = result["response_surface"]
    assert len(surface) == 400
    assert sorted({p["stop_loss"] for p in surface}) == pytest.approx(widths)
    assert sorted({p["profit_taking"] for p in surface}) == pytest.approx(widths)
