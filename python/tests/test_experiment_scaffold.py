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
