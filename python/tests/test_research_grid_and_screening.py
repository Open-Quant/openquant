from __future__ import annotations

import openquant
import polars as pl
import pytest


def test_run_flywheel_grid_returns_sorted_leaderboard():
    ds = openquant.research.make_synthetic_futures_dataset(n_bars=160, seed=21)
    grid = openquant.research.run_flywheel_grid(
        ds,
        configs=[
            {"step_size": 0.05},
            {"step_size": 0.10},
            {"step_size": 0.20, "commission_bps": 3.0},
        ],
        run_names=["a", "b", "c"],
    )

    leaderboard: pl.DataFrame = grid["leaderboard"]
    assert leaderboard.height == 3
    assert set(leaderboard["run_name"].to_list()) == {"a", "b", "c"}
    assert "net_sharpe" in leaderboard.columns
    assert "promote_candidate" in leaderboard.columns
    assert len(grid["runs"]) == 3


def test_net_sharpe_deducts_costs():
    # Issue #194: net_sharpe was the Sharpe ratio of the gross returns. Each bar is now
    # charged abs(position change into it) * cost_per_turn.
    ds = openquant.research.make_synthetic_futures_dataset(n_bars=160, seed=21)
    free = {"commission_bps": 0.0, "spread_bps": 0.0, "slippage_vol_mult": 0.0}
    gross = openquant.research.run_flywheel_iteration(ds, config=free)
    costly = openquant.research.run_flywheel_iteration(ds, config={"commission_bps": 50.0})

    bt = costly["frames"]["backtest"]
    r, pos = bt["returns"].to_list(), bt["position"].to_list()
    costs = costly["costs"]
    assert costs["turnover"] > 0
    charges = [0.0] + [abs(b - a) * costs["cost_per_turn"] for a, b in zip(pos, pos[1:])]
    assert sum(charges) == pytest.approx(costs["estimated_total_cost"], rel=1e-12)
    net = pl.Series([x - c for x, c in zip(r, charges)])
    want = net.mean() / net.std() * (252.0 * 390.0 / len(r)) ** 0.5
    assert costs["net_sharpe"] == pytest.approx(want, rel=1e-12)
    assert costly["summary"]["net_sharpe"][0] == costs["net_sharpe"]

    # With no costs it is the gross Sharpe ratio; costs lower it.
    g = bt["returns"]
    assert gross["costs"]["net_sharpe"] == pytest.approx(
        g.mean() / g.std() * (252.0 * 390.0 / len(r)) ** 0.5, rel=1e-12
    )
    assert costs["net_sharpe"] < gross["costs"]["net_sharpe"]


def test_feature_screen_report_flags_missing_constant_and_correlation():
    frame = pl.DataFrame(
        {
            "f0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f1": [1.01, 2.01, 3.01, 4.01, 5.01, 6.01],  # highly correlated with f0
            "f2": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],  # constant
            "f3": [1.0, None, 2.0, None, 3.0, None],  # low coverage
        }
    )
    out = openquant.feature_diagnostics.feature_screen_report(
        frame,
        min_coverage=0.8,
        max_corr=0.95,
    )
    assert "table" in out
    assert set(out["table"]["feature"].to_list()) == {"f0", "f1", "f2", "f3"}
    assert len(out["selected_features"]) >= 1
    assert "f2" in out["rejected_features"]
    assert "f3" in out["rejected_features"]
