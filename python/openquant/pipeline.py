from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

from . import _core, adapters


def run_mid_frequency_pipeline(
    timestamps: Sequence[str],
    close: Sequence[float],
    model_probabilities: Sequence[float],
    asset_prices: Sequence[Sequence[float]],
    model_sides: Sequence[float] | None = None,
    asset_names: Sequence[str] | None = None,
    cusum_threshold: float = 0.001,
    num_classes: int = 2,
    step_size: float = 0.1,
    risk_free_rate: float = 0.0,
    confidence_level: float = 0.05,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Run an end-to-end AFML-style research pipeline.

    Returns nested dictionaries with stage outputs:
    events, signals, portfolio, risk, backtest, leakage_checks.

    ``risk_free_rate`` is an annual rate for both the portfolio stage and
    ``realized_sharpe``. ``periods_per_year`` is the number of bars a year of ``close`` and
    of the rows of ``asset_prices`` (252 for daily bars; about ``252 * 390`` for one-minute
    bars); it annualises ``realized_sharpe`` and the portfolio's return, risk and Sharpe
    ratio.

    ``leakage_checks["timestamps_increasing"]`` and ``["event_indices_sorted"]`` are computed
    from the data. ``["inputs_aligned"]`` (always True) and ``["has_forward_look_bias"]``
    (always False) are deprecated constants: mismatched lengths raise instead, and the
    pipeline does not detect look-ahead in ``model_probabilities``.
    """
    return _core.pipeline.run_mid_frequency_pipeline(
        list(timestamps),
        list(close),
        list(model_probabilities),
        [list(row) for row in asset_prices],
        list(model_sides) if model_sides is not None else None,
        list(asset_names) if asset_names is not None else None,
        cusum_threshold,
        num_classes,
        step_size,
        risk_free_rate,
        confidence_level,
        periods_per_year,
    )


def run_mid_frequency_pipeline_frames(
    timestamps: Sequence[str],
    close: Sequence[float],
    model_probabilities: Sequence[float],
    asset_prices: Sequence[Sequence[float]],
    model_sides: Sequence[float] | None = None,
    asset_names: Sequence[str] | None = None,
    cusum_threshold: float = 0.001,
    num_classes: int = 2,
    step_size: float = 0.1,
    risk_free_rate: float = 0.0,
    confidence_level: float = 0.05,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Run the pipeline and enrich output with polars DataFrames."""
    out = run_mid_frequency_pipeline(
        timestamps=timestamps,
        close=close,
        model_probabilities=model_probabilities,
        asset_prices=asset_prices,
        model_sides=model_sides,
        asset_names=asset_names,
        cusum_threshold=cusum_threshold,
        num_classes=num_classes,
        step_size=step_size,
        risk_free_rate=risk_free_rate,
        confidence_level=confidence_level,
        periods_per_year=periods_per_year,
    )

    signals = out["signals"]
    backtest = out["backtest"]
    events = out["events"]
    portfolio = out["portfolio"]

    signal_frame = adapters.to_polars_signal_frame(
        signals["timestamps"],
        signals["values"],
    )
    event_frame = adapters.to_polars_event_frame(
        starts=events["timestamps"],
        ends=events["timestamps"],
        probs=events["probabilities"],
        sides=events["sides"],
    )
    backtest_frame = adapters.to_polars_backtest_frame(
        timestamps=backtest["timestamps"],
        equity_curve=backtest["equity_curve"],
        returns=[0.0] + list(backtest["strategy_returns"]),
        positions=signals["values"],
    )
    weights_frame = adapters.to_polars_weights_frame(
        asset_names=portfolio["asset_names"],
        weights=portfolio["weights"],
    )

    out["frames"] = {
        "signals": signal_frame,
        "events": event_frame,
        "backtest": backtest_frame,
        "weights": weights_frame,
    }
    return out


def summarize_pipeline(out: dict[str, Any]) -> pl.DataFrame:
    """Tabular summary for quick notebook inspection."""
    risk = out["risk"]
    portfolio = out["portfolio"]
    leakage = out["leakage_checks"]
    return pl.DataFrame(
        {
            "portfolio_sharpe": [portfolio["portfolio_sharpe"]],
            "portfolio_return": [portfolio["portfolio_return"]],
            "portfolio_risk": [portfolio["portfolio_risk"]],
            "realized_sharpe": [risk["realized_sharpe"]],
            "value_at_risk": [risk["value_at_risk"]],
            "expected_shortfall": [risk["expected_shortfall"]],
            "conditional_drawdown_risk": [risk["conditional_drawdown_risk"]],
            "inputs_aligned": [leakage["inputs_aligned"]],
            "timestamps_increasing": [leakage["timestamps_increasing"]],
            "event_indices_sorted": [leakage["event_indices_sorted"]],
            "has_forward_look_bias": [leakage["has_forward_look_bias"]],
        }
    )
