from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

from . import _core, adapters

#: Trading days (sessions) a year: the annualisation factor for daily bars.
TRADING_DAYS_PER_YEAR = 252.0
#: Minutes in one trading session (a 6.5-hour US equity session, 09:30 to 16:00).
SESSION_MINUTES = 390.0
#: One-minute bars a year, ``TRADING_DAYS_PER_YEAR * SESSION_MINUTES`` = 98,280.
MINUTE_BARS_PER_YEAR = TRADING_DAYS_PER_YEAR * SESSION_MINUTES


def infer_periods_per_year(timestamps: Sequence[str]) -> float | None:
    """Bars a year implied by the spacing of ``timestamps``.

    Thin wrapper over ``openquant._core.pipeline.infer_periods_per_year``: the median gap
    between consecutive strictly increasing timestamps is mapped with the convention of
    252 sessions of 390 minutes a year. Intraday gaps give ``252 * 390 / gap_minutes``
    (one-minute bars: 98,280); gaps from 20 hours to under 4 days give 252 (daily bars);
    longer gaps give ``365.25 / gap_days``.

    Parameters
    ----------
    timestamps : Sequence[str]
        Bar timestamps as ``"%Y-%m-%d %H:%M:%S"`` strings, oldest first.

    Returns
    -------
    float or None
        Bars a year, or None when fewer than two timestamps strictly increase.
    """
    return _core.pipeline.infer_periods_per_year(list(timestamps))


def _resolve_periods_per_year(timestamps: Sequence[str], periods_per_year: float | None) -> float:
    if periods_per_year is not None:
        return float(periods_per_year)
    inferred = infer_periods_per_year(timestamps)
    return TRADING_DAYS_PER_YEAR if inferred is None else inferred


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
    periods_per_year: float | None = None,
) -> dict[str, Any]:
    """Run an end-to-end AFML-style research pipeline.

    Returns nested dictionaries with stage outputs:
    events, signals, portfolio, risk, backtest, leakage_checks.

    ``risk_free_rate`` is an annual rate for both the portfolio stage and
    ``realized_sharpe``. ``periods_per_year`` is the number of bars a year of ``close`` and
    of the rows of ``asset_prices``; it annualises ``realized_sharpe`` and the portfolio's
    return, risk and Sharpe ratio. With the default None it is derived from the spacing of
    ``timestamps`` by `infer_periods_per_year` (252 for daily bars, 98,280 =
    ``252 * 390`` for one-minute bars on a 6.5-hour session; 252 when there is no gap to
    measure). Pass a number to override it, e.g. for bars that trade around the clock. The
    value used is returned as ``out["risk"]["periods_per_year"]``.

    Unlike this wrapper, ``openquant._core.pipeline.run_mid_frequency_pipeline`` (and the
    Rust ``ResearchPipelineConfig``) default to 252 whatever the bar spacing.

    ``leakage_checks["timestamps_increasing"]`` and ``["event_indices_sorted"]`` are computed
    from the data. ``["inputs_aligned"]`` (always True) and ``["has_forward_look_bias"]``
    (always False) are deprecated constants: mismatched lengths raise instead, and the
    pipeline does not detect look-ahead in ``model_probabilities``.
    """
    timestamps = list(timestamps)
    ppy = _resolve_periods_per_year(timestamps, periods_per_year)
    out = _core.pipeline.run_mid_frequency_pipeline(
        timestamps,
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
        ppy,
    )
    out["risk"]["periods_per_year"] = ppy
    return out


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
    periods_per_year: float | None = None,
) -> dict[str, Any]:
    """Run the pipeline and enrich output with polars DataFrames.

    Takes the same parameters as `run_mid_frequency_pipeline`, including the default
    ``periods_per_year=None`` that derives the annualisation factor from ``timestamps``.
    """
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
