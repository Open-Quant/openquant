import openquant
import polars as pl
import pytest


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

    assert set(out.keys()) == {
        "events",
        "signals",
        "portfolio",
        "risk",
        "backtest",
        "leakage_checks",
    }
    assert len(out["signals"]["values"]) == len(timestamps)
    assert len(out["backtest"]["equity_curve"]) == len(timestamps)
    assert len(out["backtest"]["strategy_returns"]) == len(timestamps) - 1
    assert len(out["portfolio"]["weights"]) == 3
    assert sum(out["portfolio"]["weights"]) == pytest.approx(1.0, abs=1e-6)
    assert out["leakage_checks"]["inputs_aligned"] is True
    assert out["leakage_checks"]["timestamps_increasing"] is True
    assert out["leakage_checks"]["has_forward_look_bias"] is False


def test_pipeline_reports_unordered_timestamps():
    # #185 item 11: the ordering check is computed, not a constant.
    timestamps, close, probabilities, sides, asset_prices, asset_names = _toy_pipeline_input()
    timestamps[2], timestamps[3] = timestamps[3], timestamps[2]
    out = openquant.pipeline.run_mid_frequency_pipeline(
        timestamps=timestamps,
        close=close,
        model_probabilities=probabilities,
        model_sides=sides,
        asset_prices=asset_prices,
        asset_names=asset_names,
        cusum_threshold=0.0005,
    )
    assert out["leakage_checks"]["timestamps_increasing"] is False
    assert openquant.pipeline.summarize_pipeline(out)["timestamps_increasing"][0] is False


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


def test_infer_periods_per_year_from_bar_spacing():
    # #205: one-minute bars on a 390-minute session, 252 sessions a year.
    minute = [f"2024-01-02 09:{m:02d}:00" for m in range(30, 60)]
    assert openquant.pipeline.infer_periods_per_year(minute) == 98_280.0
    assert openquant.pipeline.MINUTE_BARS_PER_YEAR == 98_280.0
    daily = [f"2024-01-{d:02d} 16:00:00" for d in (2, 3, 4, 5, 8, 9, 10, 11, 12, 15)]
    assert openquant.pipeline.infer_periods_per_year(daily) == 252.0
    assert openquant.pipeline.infer_periods_per_year(minute[:1]) is None


def test_pipeline_annualises_with_the_bar_frequency():
    # #205: the toy input is one-minute bars, so the default derives 98,280 bars a year and
    # every annualised figure is sqrt(390) (returns: 390) times the daily-bar convention.
    timestamps, close, probabilities, sides, asset_prices, asset_names = _toy_pipeline_input()
    kwargs = dict(
        timestamps=timestamps,
        close=close,
        model_probabilities=probabilities,
        model_sides=sides,
        asset_prices=asset_prices,
        asset_names=asset_names,
        cusum_threshold=0.0005,
    )
    inferred = openquant.pipeline.run_mid_frequency_pipeline(**kwargs)
    daily = openquant.pipeline.run_mid_frequency_pipeline(**kwargs, periods_per_year=252.0)
    core = openquant._core.pipeline.run_mid_frequency_pipeline(
        timestamps, close, probabilities, asset_prices, sides, asset_names, 0.0005
    )

    assert inferred["risk"]["periods_per_year"] == 98_280.0
    assert daily["risk"]["periods_per_year"] == 252.0
    assert core["risk"]["realized_sharpe"] == pytest.approx(daily["risk"]["realized_sharpe"])
    root = 390.0**0.5
    assert inferred["risk"]["realized_sharpe"] == pytest.approx(
        daily["risk"]["realized_sharpe"] * root
    )
    assert inferred["portfolio"]["portfolio_sharpe"] == pytest.approx(
        daily["portfolio"]["portfolio_sharpe"] * root
    )
    assert inferred["portfolio"]["portfolio_risk"] == pytest.approx(
        daily["portfolio"]["portfolio_risk"] * root
    )
    assert inferred["portfolio"]["portfolio_return"] == pytest.approx(
        daily["portfolio"]["portfolio_return"] * 390.0
    )
    assert inferred["portfolio"]["weights"] == pytest.approx(daily["portfolio"]["weights"])
    # Tail-risk figures are per bar and do not depend on the factor.
    assert inferred["risk"]["value_at_risk"] == daily["risk"]["value_at_risk"]
