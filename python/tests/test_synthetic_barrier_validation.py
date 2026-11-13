"""Nonfinite OTR widths fail at both direct entry points with useful names."""

import math

import pytest
from openquant import synthetic_bt


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("name", ["profit_taking", "stop_loss"])
def test_nonfinite_rule_barrier_raises_value_error(value, name):
    widths = {"profit_taking": 1.0, "stop_loss": 1.0, name: value}
    with pytest.raises(ValueError, match=f"{name} must be finite and > 0"):
        synthetic_bt.evaluate_rule_on_paths(
            [[0.0, 1.0, -1.0]], **widths, max_holding_steps=2, annualization_factor=1.0
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("name", ["profit_taking", "stop_loss"])
def test_nonfinite_search_barrier_raises_value_error(value, name):
    grids = {"profit_taking_grid": [1.0], "stop_loss_grid": [1.0], f"{name}_grid": [1.0, value]}
    with pytest.raises(ValueError, match=f"{name} must be finite and > 0"):
        synthetic_bt.search_optimal_trading_rule(
            phi=0.9,
            intercept=0.0,
            equilibrium=0.0,
            sigma=1.0,
            r_squared=0.0,
            stationary=True,
            paths=[[0.0, 1.0, -1.0]],
            **grids,
            max_holding_steps=2,
            annualization_factor=1.0,
            random_walk_phi_threshold=0.97,
            min_peak_margin=0.2,
            min_surface_std=0.1,
            min_best_sharpe=0.3,
        )
