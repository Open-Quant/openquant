from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import polars as pl

import openquant


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_runner_module():
    runner_path = REPO_ROOT / "experiments" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("openquant_experiment_runner", runner_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_notebook_script_parity_pipeline_key_metrics():
    ds = openquant.research.make_synthetic_futures_dataset(n_bars=192, seed=7)

    script_out = openquant.research.run_flywheel_iteration(ds)
    notebook_like = openquant.pipeline.run_mid_frequency_pipeline_frames(
        timestamps=ds.timestamps,
        close=ds.close,
        model_probabilities=ds.model_probabilities,
        model_sides=ds.model_sides,
        asset_prices=ds.asset_prices,
        asset_names=ds.asset_names,
    )

    assert script_out["frames"]["events"].height == notebook_like["frames"]["events"].height
    assert script_out["portfolio"]["portfolio_sharpe"] == notebook_like["portfolio"]["portfolio_sharpe"]
    assert script_out["risk"]["realized_sharpe"] == notebook_like["risk"]["realized_sharpe"]


def test_experiment_runner_outputs_expected_artifacts():
    runner = _load_runner_module()
    cfg = REPO_ROOT / "experiments" / "configs" / "futures_oil_baseline.toml"

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        run_dir = runner.run(cfg, out_dir)

        assert (run_dir / "run_manifest.json").exists()
        assert (run_dir / "decision.md").exists()
        assert (run_dir / "metrics.parquet").exists()
        assert (run_dir / "events.parquet").exists()
        assert (run_dir / "signals.parquet").exists()
        assert (run_dir / "weights.parquet").exists()
        assert (run_dir / "backtest.parquet").exists()

        metrics = pl.read_parquet(run_dir / "metrics.parquet")
        assert metrics.height == 1
        assert "net_sharpe" in metrics.columns

        equity_svg = (run_dir / "equity_curve.svg").read_text(encoding="utf-8")
        drawdown_svg = (run_dir / "drawdown.svg").read_text(encoding="utf-8")
        assert equity_svg.startswith("<svg")
        assert "Equity Curve" in equity_svg
        assert drawdown_svg.startswith("<svg")
        assert "Drawdown" in drawdown_svg

        # One point per backtest row, and the same run writes the same bytes.
        n_rows = pl.read_parquet(run_dir / "backtest.parquet").height
        assert f"n={n_rows}<" in equity_svg
        second = runner.run(cfg, out_dir / "again")
        assert (second / "equity_curve.svg").read_text(encoding="utf-8") == equity_svg
        assert (second / "drawdown.svg").read_text(encoding="utf-8") == drawdown_svg


def test_drawdown_from_equity():
    runner = _load_runner_module()
    assert runner._drawdown_from_equity([1.0, 1.2, 0.9, 1.2, 1.5]) == [
        0.0,
        0.0,
        0.9 / 1.2 - 1.0,
        0.0,
        0.0,
    ]
    assert runner._drawdown_from_equity([]) == []


def test_experiment_runner_grid_outputs_leaderboard_and_subruns():
    runner = _load_runner_module()
    cfg = REPO_ROOT / "experiments" / "configs" / "futures_oil_baseline.toml"
    grid_cfg = REPO_ROOT / "experiments" / "configs" / "futures_oil_grid.toml"

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        run_dir = runner.run_grid(cfg, grid_cfg, out_dir)

        assert (run_dir / "run_manifest.json").exists()
        assert (run_dir / "leaderboard.parquet").exists()

        leaderboard = pl.read_parquet(run_dir / "leaderboard.parquet")
        assert leaderboard.height >= 3
        assert "run_name" in leaderboard.columns
        assert "net_sharpe" in leaderboard.columns

        per_run_dirs = [p for p in run_dir.iterdir() if p.is_dir()]
        assert len(per_run_dirs) >= 3
        for p in per_run_dirs:
            assert (p / "metrics.parquet").exists()
            assert (p / "run_manifest.json").exists()
            assert (p / "equity_curve.svg").exists()
            assert (p / "drawdown.svg").exists()
