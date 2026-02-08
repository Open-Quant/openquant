import polars as pl
import pytest

import openquant


def _toy_pipeline_input():
    timestamps = [
        "2024-01-01 09:30:00",
        "2024-01-01 09:31:00",
        "2024-01-01 09:32:00",
        "2024-01-01 09:33:00",
        "2024-01-01 09:34:00",
        "2024-01-01 09:35:00",
        "2024-01-01 09:36:00",
        "2024-01-01 09:37:00",
    ]
    close = [100.0, 100.2, 99.9, 100.4, 100.0, 100.7, 100.4, 100.9]
    probabilities = [0.55, 0.60, 0.52, 0.48, 0.61, 0.58, 0.63, 0.57]
    sides = [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0]
    asset_prices = [
        [100.0, 100.0, 100.0],
        [100.2, 99.8, 100.1],
        [100.3, 99.9, 100.2],
        [100.4, 100.2, 100.1],
        [100.3, 100.0, 100.3],
        [100.6, 100.1, 100.4],
        [100.7, 100.3, 100.5],
        [100.8, 100.4, 100.6],
    ]
    asset_names = ["A", "B", "C"]
    return timestamps, close, probabilities, sides, asset_prices, asset_names


def test_pipeline_run_contract():
    timestamps, close, probabilities, sides, asset_prices, asset_names = _toy_pipeline_input()

    out = openquant.pipeline.run_mid_frequency_pipeline(
        timestamps=timestamps,
        close=close,
        model_probabilities=probabilities,
        model_sides=sides,
        asset_prices=asset_prices,
        asset_names=asset_names,
        cusum_threshold=0.0005,
    )

    assert set(out.keys()) == {"events", "signals", "portfolio", "risk", "backtest", "leakage_checks"}
    assert len(out["signals"]["values"]) == len(timestamps)
    assert len(out["backtest"]["equity_curve"]) == len(timestamps)
    assert len(out["backtest"]["strategy_returns"]) == len(timestamps) - 1
    assert len(out["portfolio"]["weights"]) == 3
    assert sum(out["portfolio"]["weights"]) == pytest.approx(1.0, abs=1e-6)
    assert out["leakage_checks"]["inputs_aligned"] is True
    assert out["leakage_checks"]["has_forward_look_bias"] is False


def test_pipeline_run_frames():
    timestamps, close, probabilities, sides, asset_prices, asset_names = _toy_pipeline_input()
    out = openquant.pipeline.run_mid_frequency_pipeline_frames(
        timestamps=timestamps,
        close=close,
        model_probabilities=probabilities,
        model_sides=sides,
        asset_prices=asset_prices,
        asset_names=asset_names,
        cusum_threshold=0.0005,
    )
    frames = out["frames"]
    assert isinstance(frames["signals"], pl.DataFrame)
    assert isinstance(frames["events"], pl.DataFrame)
    assert isinstance(frames["backtest"], pl.DataFrame)
    assert isinstance(frames["weights"], pl.DataFrame)
    assert frames["signals"].height == len(timestamps)
    assert frames["backtest"].height == len(timestamps)
    assert frames["weights"]["weight"].sum() == pytest.approx(1.0, abs=1e-6)


def test_pipeline_summary_frame():
    timestamps, close, probabilities, sides, asset_prices, asset_names = _toy_pipeline_input()
    out = openquant.pipeline.run_mid_frequency_pipeline(
        timestamps=timestamps,
        close=close,
        model_probabilities=probabilities,
        model_sides=sides,
        asset_prices=asset_prices,
        asset_names=asset_names,
        cusum_threshold=0.0005,
    )
    summary = openquant.pipeline.summarize_pipeline(out)
    assert isinstance(summary, pl.DataFrame)
    assert summary.height == 1
    assert "portfolio_sharpe" in summary.columns
