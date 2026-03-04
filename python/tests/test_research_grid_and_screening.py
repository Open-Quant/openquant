from __future__ import annotations

import openquant
import polars as pl


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
