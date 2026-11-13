"""Synthetic finite-horizon acquisition timing; no optimizer or market calibration."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import numbers
import platform
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

SCHEDULES = ("front", "uniform", "back")


def _real(value, name, *, positive=False, upper=None):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, not bool or text")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite binary64") from exc
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    if upper is not None and value > upper:
        raise ValueError(f"{name} must be at most {upper}")
    return value


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _display(value, name, limits):
    try:
        result = float(value)
    except OverflowError:
        result = math.inf
    if not math.isfinite(result):
        limits[name] = "above_float64_range"
        return None
    if result == 0 and value != 0:
        limits[name] = "below_float64_range"
        return None
    return result


def evaluate_schedule(*, target, horizon, alpha, decay, spread, impact, latency, schedule):
    """Buy target shares before final mark; exact arithmetic on binary64 inputs.

    Slot t: buy q[t], then earn x[t]*alpha*decay**t during [t,t+1].
    Costs are spread*q[t] + impact*q[t]**2. Latency removes leading slots;
    remaining orders are compressed, not dropped. Terminal shares are not sold.
    """
    horizon = _integer(horizon, "horizon", 1)
    latency = _integer(latency, "latency", 0)
    if latency >= horizon:
        raise ValueError("latency leaves no execution slots; terminal completion infeasible")
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    q_target = Fraction(_real(target, "target", positive=True))
    a = Fraction(_real(alpha, "alpha"))
    rho = Fraction(_real(decay, "decay", upper=1))
    s = Fraction(_real(spread, "spread"))
    eta = Fraction(_real(impact, "impact"))
    trades = [Fraction(0) for _ in range(horizon)]
    if schedule == "front":
        trades[latency] = q_target
    elif schedule == "back":
        trades[-1] = q_target
    else:
        trades[latency:] = [q_target / (horizon - latency)] * (horizon - latency)
    inventory, gross = Fraction(0), Fraction(0)
    holdings, increments = [], []
    for t, trade in enumerate(trades):
        inventory += trade
        increment = a * rho**t
        holdings.append(inventory)
        increments.append(increment)
        gross += inventory * increment
    spread_cost = s * sum(trades)
    impact_cost = eta * sum(q * q for q in trades)
    benchmark = q_target * sum(increments)
    values = {
        "gross_pnl": gross,
        "spread_cost": spread_cost,
        "impact_cost": impact_cost,
        "total_cost": spread_cost + impact_cost,
        "net_pnl": gross - spread_cost - impact_cost,
        "immediate_costless_gross_pnl": benchmark,
        "opportunity_cost": benchmark - gross,
        "implementation_shortfall": benchmark - gross + spread_cost + impact_cost,
        "terminal_inventory": inventory,
    }
    limits = {}
    row = {key: _display(value, key, limits) for key, value in values.items()}
    exact = {key: str(value) for key, value in values.items()}
    exact.update(
        trades=[str(q) for q in trades],
        holdings=[str(x) for x in holdings],
        alpha_increments=[str(x) for x in increments],
    )
    row.update(
        schedule=schedule,
        latency=latency,
        exact=exact,
        representation_limits=limits,
        terminal_inventory_liquidated=False,
        completed=inventory == q_target,
        positive_net=values["net_pnl"] > 0,
    )
    return row


def _controls():
    base = dict(
        target=2,
        horizon=3,
        alpha=1,
        decay=0.5,
        spread=0.25,
        impact=0.5,
        latency=0,
        schedule="uniform",
    )
    uniform = evaluate_schedule(**base)
    front = evaluate_schedule(**{**base, "schedule": "front"})
    delayed = evaluate_schedule(**{**base, "schedule": "front", "latency": 2})
    controls = {
        "uniform_hand_net": {
            "observed": uniform["exact"]["net_pnl"],
            "expected": "2/3",
            "passed": uniform["exact"]["net_pnl"] == "2/3",
        },
        "spread_schedule_invariant": {"passed": front["spread_cost"] == uniform["spread_cost"]},
        "smoothing_reduces_quadratic_cost": {
            "passed": uniform["impact_cost"] < front["impact_cost"]
        },
        "latency_keeps_completion": {"passed": delayed["completed"]},
    }
    negative = {
        "restart_alpha_at_execution": {
            "wrong_gross_pnl": "2",
            "correct_gross_pnl": delayed["exact"]["gross_pnl"],
            "detected": delayed["exact"]["gross_pnl"] != "2",
        },
        "drop_delayed_orders": {
            "wrong_terminal_inventory": "2/3",
            "required_terminal_inventory": "2",
            "detected": delayed["exact"]["terminal_inventory"] != "2/3",
        },
        "credit_pretrade_inventory": {
            "wrong_uniform_gross_pnl": "2/3",
            "correct_gross_pnl": uniform["exact"]["gross_pnl"],
            "detected": uniform["exact"]["gross_pnl"] != "2/3",
        },
    }
    return controls, negative


def run_lab(
    *,
    target=2,
    horizon=4,
    alpha=1,
    decays=(0, 0.5, 1),
    spread=0.25,
    impacts=(0, 0.5),
    latencies=(0, 1, 3),
    schedules=SCHEDULES,
):
    """Retain the entire prespecified grid, with no fitted or selected policy."""
    start = time.perf_counter()
    target = _real(target, "target", positive=True)
    horizon = _integer(horizon, "horizon", 1)
    alpha, spread = _real(alpha, "alpha"), _real(spread, "spread")
    decays = [_real(x, "decay", upper=1) for x in decays]
    impacts = [_real(x, "impact") for x in impacts]
    latencies = [_integer(x, "latency", 0) for x in latencies]
    schedules = list(schedules)
    if not all((decays, impacts, latencies, schedules)):
        raise ValueError("all grids must be nonempty")
    config = dict(
        target=target,
        horizon=horizon,
        alpha=alpha,
        decays=decays,
        spread=spread,
        impacts=impacts,
        latencies=latencies,
        schedules=schedules,
    )
    cells = []
    for decay, impact, latency, schedule in itertools.product(
        decays, impacts, latencies, schedules
    ):
        row = evaluate_schedule(
            target=target,
            horizon=horizon,
            alpha=alpha,
            decay=decay,
            spread=spread,
            impact=impact,
            latency=latency,
            schedule=schedule,
        )
        row.update(decay=decay, impact=impact)
        cells.append(row)
    controls, negative = _controls()
    source = Path(__file__).resolve()
    try:
        git_sha = subprocess.check_output(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unavailable"
    return {
        "lab": "execution_timing",
        "schema_version": 1,
        "synthetic_only": True,
        "config": config,
        "cells": cells,
        "controls": controls,
        "negative_controls": negative,
        "limitations": [
            "Synthetic acquisition accounting, not an optimizer or full Almgren-Chriss model.",
            "All alpha, spread and impact inputs are assumptions, not empirical forecasts.",
            "Terminal inventory is marked, not liquidated; no financing or risk model.",
            "Linear mid-price increments and temporary quadratic impact only; no permanent impact.",
            "Exact values refer to binary64-converted inputs; bare numbers cannot verify units.",
            "No uncertainty, performance, capacity, calibrated fill or trading recommendation.",
            "Trusted local configurations only; large horizons and grids may require large resources.",
        ],
        "provenance": {
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "git_sha": git_sha,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "performance": {"wall_seconds": time.perf_counter() - start},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.out.exists() or args.out.is_symlink():
            parser.error("output already exists; refusing to overwrite")
        payload = json.dumps(run_lab(), indent=2, allow_nan=False) + "\n"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            stream.write(payload)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(args.out)


if __name__ == "__main__":
    main()
