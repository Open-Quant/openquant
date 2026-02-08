from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import sin
import random
from typing import Any

import polars as pl

from . import pipeline


@dataclass(frozen=True)
class ResearchDataset:
    timestamps: list[str]
    close: list[float]
    model_probabilities: list[float]
    model_sides: list[float]
    asset_prices: list[list[float]]
    asset_names: list[str]


def make_synthetic_futures_dataset(
    n_bars: int = 192,
    seed: int = 7,
    asset_names: list[str] | None = None,
) -> ResearchDataset:
    """Build a deterministic synthetic multi-asset futures dataset.

    The first asset is treated as the primary traded instrument (e.g., crude oil),
    while the other assets provide cross-asset context for allocation/risk.
    """
    if n_bars < 32:
        raise ValueError("n_bars must be >= 32")
    rng = random.Random(seed)
    asset_names = asset_names or ["CL", "NG", "RB", "GC"]
    n_assets = len(asset_names)
    if n_assets < 2:
        raise ValueError("asset_names must contain at least 2 assets")

    start = datetime(2024, 1, 1, 9, 30, 0)
    timestamps = [(start + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(n_bars)]

    base = 80.0
    close: list[float] = []
    for i in range(n_bars):
        seasonal = 0.45 * sin(i / 9.0) + 0.25 * sin(i / 17.0)
        drift = 0.006 * i
        noise = rng.uniform(-0.10, 0.10)
        price = base + drift + seasonal + noise
        close.append(max(price, 1.0))

    model_probabilities: list[float] = []
    model_sides: list[float] = []
    for i in range(n_bars):
        edge = 0.53 + 0.08 * sin(i / 13.0) + rng.uniform(-0.025, 0.025)
        p = min(max(edge, 0.05), 0.95)
        model_probabilities.append(p)
        model_sides.append(1.0 if sin(i / 11.0) >= 0.0 else -1.0)

    asset_prices: list[list[float]] = []
    for i in range(n_bars):
        row: list[float] = []
        for j in range(n_assets):
            lag = max(i - (j + 1), 0)
            spread = 0.45 + 0.07 * j
            px = close[lag] * (1.0 + 0.0015 * j) + spread * sin((i + 3 * j) / (9.5 + j))
            px += rng.uniform(-0.08, 0.08)
            row.append(max(px, 1.0))
        asset_prices.append(row)

    return ResearchDataset(
        timestamps=timestamps,
        close=close,
        model_probabilities=model_probabilities,
        model_sides=model_sides,
        asset_prices=asset_prices,
        asset_names=asset_names,
    )


def run_flywheel_iteration(
    dataset: ResearchDataset,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = {
        "cusum_threshold": 0.001,
        "num_classes": 2,
        "step_size": 0.1,
        "risk_free_rate": 0.0,
        "confidence_level": 0.05,
        "commission_bps": 1.5,
        "spread_bps": 2.0,
        "slippage_vol_mult": 8.0,
        "min_net_sharpe": 0.30,
        "min_realized_sharpe": 0.25,
    }
    if config:
        cfg.update(config)

    out = pipeline.run_mid_frequency_pipeline_frames(
        timestamps=dataset.timestamps,
        close=dataset.close,
        model_probabilities=dataset.model_probabilities,
        model_sides=dataset.model_sides,
        asset_prices=dataset.asset_prices,
        asset_names=dataset.asset_names,
        cusum_threshold=float(cfg["cusum_threshold"]),
        num_classes=int(cfg["num_classes"]),
        step_size=float(cfg["step_size"]),
        risk_free_rate=float(cfg["risk_free_rate"]),
        confidence_level=float(cfg["confidence_level"]),
    )

    backtest = out["frames"]["backtest"]
    strategy_returns = backtest["returns"].to_list()
    positions = backtest["position"].to_list()

    turnover = _turnover(positions)
    realized_vol = _annualized_vol(strategy_returns)
    cost_per_turn = (
        float(cfg["commission_bps"]) * 1e-4
        + float(cfg["spread_bps"]) * 1e-4
        + float(cfg["slippage_vol_mult"]) * realized_vol * 1e-3
    )
    total_cost = turnover * cost_per_turn
    gross_total_return = backtest["equity"][-1] - 1.0
    net_total_return = gross_total_return - total_cost

    bars = len(strategy_returns)
    annualizer = (252.0 * 390.0 / max(bars, 1)) ** 0.5
    mean_r = sum(strategy_returns) / max(bars, 1)
    std_r = _std(strategy_returns)
    net_sharpe = (mean_r / std_r) * annualizer if std_r > 0 else 0.0

    promotion = {
        "passed_realized_sharpe": out["risk"]["realized_sharpe"] >= float(cfg["min_realized_sharpe"]),
        "passed_net_sharpe": net_sharpe >= float(cfg["min_net_sharpe"]),
        "passed_alignment_guard": bool(out["leakage_checks"]["inputs_aligned"]),
        "passed_event_order_guard": bool(out["leakage_checks"]["event_indices_sorted"]),
    }
    promotion["promote_candidate"] = bool(all(promotion.values()))

    summary = pipeline.summarize_pipeline(out).with_columns(
        pl.lit(turnover).alias("turnover"),
        pl.lit(realized_vol).alias("realized_vol"),
        pl.lit(total_cost).alias("estimated_cost"),
        pl.lit(gross_total_return).alias("gross_total_return"),
        pl.lit(net_total_return).alias("net_total_return"),
        pl.lit(net_sharpe).alias("net_sharpe"),
    )

    out["costs"] = {
        "turnover": turnover,
        "realized_vol": realized_vol,
        "cost_per_turn": cost_per_turn,
        "estimated_total_cost": total_cost,
        "gross_total_return": gross_total_return,
        "net_total_return": net_total_return,
        "net_sharpe": net_sharpe,
    }
    out["promotion"] = promotion
    out["summary"] = summary
    return out


def _turnover(positions: list[float]) -> float:
    if len(positions) <= 1:
        return 0.0
    return sum(abs(positions[i] - positions[i - 1]) for i in range(1, len(positions)))


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_v = sum(values) / len(values)
    var = sum((v - mean_v) ** 2 for v in values) / (len(values) - 1)
    return var**0.5


def _annualized_vol(values: list[float]) -> float:
    std = _std(values)
    return std * (252.0**0.5)
