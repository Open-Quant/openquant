from __future__ import annotations

import dataclasses

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


def test_flywheel_annualises_one_minute_bars_with_98280_bars_a_year():
    # #205: the synthetic bars are one minute apart; the loop used to annualise them with 252.
    ds = openquant.research.make_synthetic_futures_dataset(n_bars=160, seed=21)
    assert ds.periods_per_year == 98_280.0
    assert openquant.pipeline.infer_periods_per_year(ds.timestamps) == 98_280.0

    minute = openquant.research.run_flywheel_iteration(ds)
    daily = openquant.research.run_flywheel_iteration(ds, config={"periods_per_year": 252.0})
    root = 390.0**0.5

    costs = minute["costs"]
    assert costs["periods_per_year"] == 98_280.0
    assert daily["costs"]["periods_per_year"] == 252.0
    assert minute["risk"]["periods_per_year"] == 98_280.0
    assert costs["realized_vol"] == pytest.approx(costs["bar_vol"] * 98_280.0**0.5)
    assert costs["realized_vol"] == pytest.approx(daily["costs"]["realized_vol"] * root)
    assert costs["net_sharpe"] == pytest.approx(daily["costs"]["net_sharpe"] * root)
    assert minute["risk"]["realized_sharpe"] == pytest.approx(
        daily["risk"]["realized_sharpe"] * root
    )
    # The cost estimate scales with the per-bar volatility, not with the annualisation.
    assert costs["cost_per_turn"] == pytest.approx(daily["costs"]["cost_per_turn"])
    assert costs["estimated_total_cost"] == pytest.approx(daily["costs"]["estimated_total_cost"])


def test_flywheel_derives_periods_per_year_from_timestamps_when_unset():
    ds = openquant.research.make_synthetic_futures_dataset(n_bars=64, seed=3)
    hourly = [f"2024-01-{2 + i // 7:02d} {9 + i % 7:02d}:30:00" for i in range(64)]
    ds = dataclasses.replace(ds, timestamps=hourly, periods_per_year=None)
    out = openquant.research.run_flywheel_iteration(ds)
    assert out["costs"]["periods_per_year"] == pytest.approx(252.0 * 390.0 / 60.0)


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
