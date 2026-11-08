"""Offline fixed-dollar-schedule scaling, not calibrated capacity or an optimizer."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
import time
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from fractions import Fraction
from pathlib import Path

import numpy as np

# Explicit context isolates results from callers' Decimal precision and rounding.
_DECIMAL_CONTEXT = Context(
    prec=80,
    rounding=ROUND_HALF_EVEN,
    Emin=-999999,
    Emax=999999,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[InvalidOperation, DivisionByZero, Overflow],
)


def _scalar(value, name):
    if (
        isinstance(value, (bool, np.bool_, str, bytes))
        or not np.isscalar(value)
        or np.iscomplexobj(value)
    ):
        raise TypeError(f"{name} must be a real numeric scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be representable as a finite binary64 number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _array(values, name, *, shape=None, nonnegative=False, positive=False):
    array = np.asarray(values)
    if array.ndim not in (1, 2) or not array.size:
        raise ValueError(f"{name} must be a nonempty vector or matrix")
    if array.dtype.kind not in "iuf" or any(
        isinstance(x, (bool, np.bool_)) for x in np.asarray(values, dtype=object).flat
    ):
        raise TypeError(f"{name} must contain real numeric values, not booleans or strings")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite binary64 values")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have exactly the trade schedule's shape")
    if nonnegative and np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} must be strictly positive; unknown volume is unavailable")
    return array


def _schedule(trade_fractions, spread, impact, volatility, dollar_volume, participation):
    d = _array(trade_fractions, "trade_fractions")
    arrays = [d]
    for name, values in (("spread", spread), ("impact", impact), ("volatility", volatility)):
        arrays.append(_array(values, name, shape=d.shape, nonnegative=True))
    arrays.append(_array(dollar_volume, "dollar_volume", shape=d.shape, positive=True))
    if np.isscalar(participation):
        p = np.full(d.shape, _scalar(participation, "participation"))
    else:
        p = _array(participation, "participation", shape=d.shape)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("participation must be in [0,1]")
    arrays.append(p)
    return arrays


def _decimal(value):
    return Decimal.from_float(float(value))


def _represented(value):
    """Null and a status for an unrepresentable nonzero number, never NaN/Infinity."""
    try:
        result = float(value)
    except OverflowError:
        return None, "above_float64_range"
    if not math.isfinite(result):
        return None, "above_float64_range"
    if result == 0 and value != 0:
        return None, "below_float64_range"
    return result, "finite"


def _fraction_decimal(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def _rational_sqrt(value):
    numerator, denominator = math.isqrt(value.numerator), math.isqrt(value.denominator)
    if numerator * numerator == value.numerator and denominator * denominator == value.denominator:
        return Fraction(numerator, denominator)
    return None


def _threshold(value):
    scale, status = _represented(value)
    return {"status": status, "scale": scale}


def evaluate_schedule(
    *,
    scale,
    trade_fractions,
    spread,
    impact,
    volatility,
    dollar_volume,
    gross_return,
    participation,
):
    """Evaluate additive expected PnL for q=A*d under a fixed 3/2-power cost model.

    Arrays have the same nonempty one- or two-dimensional shape; participation
    alone may be a scalar. Spread and volatility are fractions, not bps/percent.
    Both scale and volume are dollars in the same currency and horizon convention.
    No position, financing, solvency, optimization, or empirical calibration is implied.
    Numeric validation cannot detect shares substituted for dollar volumes.
    """
    scale, gross_return = _scalar(scale, "scale"), _scalar(gross_return, "gross_return")
    if scale <= 0:
        raise ValueError("scale must be strictly positive")
    arrays = _schedule(trade_fractions, spread, impact, volatility, dollar_volume, participation)
    with localcontext(_DECIMAL_CONTEXT):
        a_fraction, g_fraction = Fraction(scale), Fraction(gross_return)
        a = _decimal(scale)
        linear_fraction, impact_coefficient = Fraction(0), Decimal(0)
        exact_impact, exact_normalized_impact = Fraction(0), Fraction(0)
        caps = []
        for raw in zip(*(x.flat for x in arrays), strict=True):
            d, s, eta, sigma, volume, p = (Fraction(float(x)) for x in raw)
            absolute_d = abs(d)
            if absolute_d == 0:
                continue
            linear_fraction += s * absolute_d
            caps.append(p * volume / absolute_d)
            multiplier = eta * sigma * absolute_d
            if multiplier == 0:
                continue
            radicand = absolute_d / volume
            root = _rational_sqrt(radicand)
            normalized_root = _rational_sqrt(a_fraction * radicand)
            if root is None:
                exact_impact = None
            elif exact_impact is not None:
                exact_impact += multiplier * root
            if normalized_root is None:
                exact_normalized_impact = None
            elif exact_normalized_impact is not None:
                exact_normalized_impact += multiplier * normalized_root
            impact_coefficient += _fraction_decimal(multiplier) * _fraction_decimal(radicand).sqrt()

        # Preserve exact rational cancellations when all radicals permit it.
        # Otherwise retain a numerical approximation, never a certified sign.
        margin_fraction = g_fraction - linear_fraction
        linear = _fraction_decimal(linear_fraction)
        margin = _fraction_decimal(margin_fraction)
        if exact_impact is not None:
            impact_coefficient = _fraction_decimal(exact_impact)
        normalized_impact = a.sqrt() * impact_coefficient
        net = margin - normalized_impact
        net_evaluation = "numerical_approximation"
        unresolved = False
        if exact_normalized_impact is not None:
            net = margin_fraction - exact_normalized_impact
            normalized_cost = linear_fraction + exact_normalized_impact
            net_evaluation = "exact_rational"
        else:
            normalized_cost = linear + normalized_impact
            # A conservative warning region, not a certified error bound.
            # Ten of the eighty working digits are reserved as guard digits.
            guard = Decimal("1e-70") * max(abs(margin), abs(normalized_impact))
            unresolved = abs(net) <= guard
            if unresolved:
                net_evaluation = "near_cancellation_unresolved"

        if margin_fraction <= 0:
            break_even = {"status": "no_positive_net_scale", "scale": None}
        elif impact_coefficient == 0:
            break_even = {"status": "no_finite_crossing", "scale": None}
        elif exact_impact is not None:
            break_even = _threshold((margin_fraction / exact_impact) ** 2)
        else:
            break_even = _threshold((margin / impact_coefficient) ** 2)
        cap = min(caps) if caps else None
        cap_output = (
            _threshold(cap) if cap is not None else {"status": "no_trade_limit", "scale": None}
        )
        result = {
            "scale": scale,
            "gross_return": gross_return,
            "break_even": break_even,
            "break_even_evaluation": (
                "exact_rational"
                if margin_fraction <= 0 or exact_impact is not None
                else "numerical_approximation"
            ),
            "participation_cap": cap_output,
            "participation_cap_exact": str(cap) if cap is not None else None,
            "participation_boundary_rounded": (
                cap is not None and cap_output["scale"] == scale and a_fraction != cap
            ),
            "within_participation_cap": cap is None or a_fraction <= cap,
            "positive_net": None if unresolved else net > 0,
            "net_evaluation": net_evaluation,
            "representation_limits": {},
            "precision_limits": (
                ["net_expected_pnl_per_scale", "net_expected_pnl_dollars", "positive_net"]
                if unresolved
                else []
            ),
        }
        values = {
            "linear_coefficient": linear_fraction,
            "impact_coefficient": exact_impact if exact_impact is not None else impact_coefficient,
            "linear_cost_dollars": a_fraction * linear_fraction,
            "impact_cost_dollars": (
                a_fraction * exact_normalized_impact
                if exact_normalized_impact is not None
                else a * normalized_impact
            ),
            "total_cost_dollars": (
                a_fraction * normalized_cost
                if exact_normalized_impact is not None
                else a * normalized_cost
            ),
            "normalized_cost": normalized_cost,
            "net_expected_pnl_per_scale": net,
            "net_expected_pnl_dollars": (
                a_fraction * net if exact_normalized_impact is not None else a * net
            ),
        }
        for name, value in values.items():
            if name in result["precision_limits"]:
                result[name] = None
                continue
            result[name], status = _represented(value)
            if status != "finite":
                result["representation_limits"][name] = status
        return result


def _grid(values, name, *, positive=False, nonnegative=False):
    array = _array(values, name, positive=positive, nonnegative=nonnegative)
    if array.ndim != 1 or len(set(array)) != len(array):
        raise ValueError(f"{name} must be a one-dimensional grid of distinct values")
    return array.tolist()


def _controls():
    base = dict(
        scale=100,
        trade_fractions=[0.25],
        spread=[0.01],
        impact=[0.5],
        volatility=[0.2],
        dollar_volume=[100],
        gross_return=0.1,
        participation=0.1,
    )
    a = evaluate_schedule(**base)
    doubled = evaluate_schedule(**{**base, "scale": 200})
    together = evaluate_schedule(**{**base, "scale": 700, "dollar_volume": [700]})
    linear = evaluate_schedule(**{**base, "impact": [0]})
    linear_large = evaluate_schedule(**{**base, "impact": [0], "scale": 1e12})
    zero_gross = evaluate_schedule(**{**base, "gross_return": 0})

    def close(observed, expected):
        return {
            "observed": observed,
            "expected": expected,
            "passed": math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-15),
        }

    controls = {
        "linear_doubling": close(doubled["linear_cost_dollars"] / a["linear_cost_dollars"], 2),
        "impact_doubling": close(doubled["impact_cost_dollars"] / a["impact_cost_dollars"], 2**1.5),
        "joint_scale_volume": close(together["normalized_cost"], a["normalized_cost"]),
        "zero_gross_no_positive_net": {
            "observed": zero_gross["net_expected_pnl_per_scale"],
            "passed": not zero_gross["positive_net"]
            and zero_gross["net_expected_pnl_per_scale"] <= 0,
        },
        "linear_only_no_finite_crossing": {
            "observed": linear["break_even"]["status"],
            "passed": linear["break_even"]["status"] == "no_finite_crossing"
            and linear["net_expected_pnl_per_scale"] == linear_large["net_expected_pnl_per_scale"],
        },
    }
    wrong_units = evaluate_schedule(**{**base, "dollar_volume": [2]})
    negative_controls = {
        "shares_as_dollars": {
            "description": "Deliberately substitute 2 shares for $100 volume at $50/share.",
            "detected_by_input_validation": False,
            "correct_impact_dollars": a["impact_cost_dollars"],
            "wrong_impact_dollars": wrong_units["impact_cost_dollars"],
            "limitation": "Bare numbers cannot validate their economic units.",
        },
        "omitted_scale_normalization": {
            "description": "Deliberately label total dollar cost as cost per dollar of scale.",
            "correct_value": a["normalized_cost"],
            "wrong_value": a["total_cost_dollars"],
        },
    }
    return controls, negative_controls


def run_lab(
    *,
    scales=(1e3, 1e4, 1e5, 1e6, 1e7),
    volume_scales=(0.1, 1, 10),
    impact_coefficients=(0, 0.1, 0.5),
    gross_returns=(0, 0.001),
    trade_fractions=(0.1, -0.1),
    spread=(0.0001, 0.0001),
    volatility=(0.02, 0.02),
    dollar_volume=(1e6, 4e6),
    participation=0.01,
):
    """Retain every deterministic grid cell; no tuning, simulations or selection."""
    start = time.perf_counter()
    scales = _grid(scales, "scales", positive=True)
    volume_scales = _grid(volume_scales, "volume_scales", positive=True)
    impact_coefficients = _grid(impact_coefficients, "impact_coefficients", nonnegative=True)
    gross_returns = _grid(gross_returns, "gross_returns")
    d = _array(trade_fractions, "trade_fractions")
    d, s, _, sigma, volume, p = _schedule(
        d, spread, np.zeros(d.shape), volatility, dollar_volume, participation
    )
    config = {
        "scales": scales,
        "volume_scales": volume_scales,
        "impact_coefficients": impact_coefficients,
        "gross_returns": gross_returns,
        "trade_fractions": d.tolist(),
        "spread": s.tolist(),
        "volatility": sigma.tolist(),
        "dollar_volume": volume.tolist(),
        "participation": p.tolist(),
    }
    scaled_volumes = {}
    for multiplier in volume_scales:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scaled_volumes[multiplier] = volume * multiplier
        if not np.all(np.isfinite(scaled_volumes[multiplier])) or np.any(
            scaled_volumes[multiplier] <= 0
        ):
            raise ValueError("scaled dollar volumes must remain positive finite binary64 inputs")
    cells = []
    for scale, multiplier, eta, gross in itertools.product(
        scales, volume_scales, impact_coefficients, gross_returns
    ):
        row = evaluate_schedule(
            scale=scale,
            trade_fractions=d,
            spread=s,
            impact=np.full(d.shape, eta),
            volatility=sigma,
            dollar_volume=scaled_volumes[multiplier],
            gross_return=gross,
            participation=p,
        )
        row.update(volume_scale=multiplier, impact_coefficient_input=eta)
        cells.append(row)
    controls, negative_controls = _controls()
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
        "lab": "capacity_scaling",
        "schema_version": 1,
        "synthetic_only": True,
        "config": config,
        "cells": cells,
        "controls": controls,
        "negative_controls": negative_controls,
        "limitations": [
            "Synthetic fixed additive execution schedule, not an optimizer or self-financing ledger.",
            "Expected gross PnL and costs are supplied assumptions, not calibrated market estimates.",
            "Zero-net scale and participation feasibility are separate definitions, not trading advice.",
            "No Monte Carlo sampling, uncertainty estimate, capital recommendation or winner selection.",
            "Numeric validation cannot identify shares versus dollars or inconsistent currency/horizon.",
            "General radical calculations use 80 digits, not certified intervals; near-cancellation net is unresolved.",
            "Exact rational cases and participation comparisons preserve binary64 input values without tolerances.",
            "Unrepresentable numeric outputs are null with explicit range status, not zero or infinity.",
        ],
        "provenance": {
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "git_sha": git_sha,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "decimal_precision": _DECIMAL_CONTEXT.prec,
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
        report = run_lab()
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also closes the race after the initial exists check.
        with args.out.open("x") as stream:
            stream.write(serialized)
    except (TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    print(args.out)


if __name__ == "__main__":
    main()
