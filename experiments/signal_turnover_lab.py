"""Stationary signal persistence and fixed-speed turnover; synthetic additive PnL only."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

COST_MODELS = ("proportional", "quadratic")
METRICS = (
    "signal_second_moment",
    "position_second_moment",
    "position_signal_product",
    "squared_turnover",
    "absolute_turnover",
    "gross_pnl",
    "cost",
    "net_pnl",
    "forecast_mse",
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
        raise ValueError(f"{name} must be a finite binary64 scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _vector(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or not array.size:
        raise ValueError(f"{name} must be a nonempty vector")
    if array.dtype.kind not in "iuf" or any(
        isinstance(x, (bool, np.bool_)) for x in np.asarray(values, dtype=object).flat
    ):
        raise TypeError(f"{name} must contain real numbers, not booleans or strings")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite binary64 values")
    return array


def _represented(value, name):
    """Reject nonrepresentable analytic quantities instead of inventing an outcome."""
    try:
        result = float(value)
    except OverflowError as exc:
        raise ArithmeticError(f"{name} is outside finite binary64 range") from exc
    if not math.isfinite(result) or (result == 0 and value != 0):
        raise ArithmeticError(f"{name} is outside nonzero finite binary64 range")
    return result


def _model(rho, eta, k, beta, sigma):
    rho, eta, k, beta, sigma = (
        _scalar(value, name)
        for value, name in zip(
            (rho, eta, k, beta, sigma), ("rho", "eta", "k", "beta", "sigma"), strict=True
        )
    )
    if not -1 < rho < 1:
        raise ValueError("rho must be strictly between -1 and 1")
    if not 0 <= eta <= 1:
        raise ValueError("eta must be in [0,1]")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    return rho, eta, k, beta, sigma


def stationary_moments(*, rho, eta, k, beta, sigma):
    """Original stationary Gaussian identities, not a production policy objective.

    Inputs denote their binary64 values. Rational intermediate arithmetic avoids
    near-boundary subtraction and overflow. Unrepresentable nonzero output
    moments raise ArithmeticError. Eta zero specifies the cash state x=0.
    """
    rho, eta, k, beta, sigma = _model(rho, eta, k, beta, sigma)
    r, e, scale, b, noise = map(Fraction, (rho, eta, k, beta, sigma))
    a, d = 1 - e, e + (1 - e) * (1 - r)
    covariance = e * scale / d
    variance = e * scale**2 * (1 + a * r) / ((2 - e) * d)
    conditional = e * scale**2 * a**2 * (1 - r) * (1 + r) / ((2 - e) * d**2)
    delta_variance = 2 * e**2 * scale**2 * (1 - r) / ((2 - e) * d)
    result = {
        "signal_second_moment": 1.0,
        "position_signal_product": _represented(covariance, "position_signal_product"),
        "position_second_moment": _represented(variance, "position_second_moment"),
        "conditional_position_variance": _represented(conditional, "conditional_position_variance"),
        "squared_turnover": _represented(delta_variance, "squared_turnover"),
        "gross_pnl": _represented(b * covariance, "gross_pnl"),
        "forecast_mse": _represented(noise**2, "forecast_mse"),
        # Historical field name: this is signed Corr(signal, future return),
        # not Corr(beta*signal, future return). Preserve nonzero range failures.
        "forecast_correlation": (
            _represented(b / Fraction(math.hypot(beta, sigma)), "forecast_correlation")
            if beta or sigma
            else None
        ),
    }
    result["absolute_turnover"] = math.sqrt(2 / math.pi) * math.sqrt(result["squared_turnover"])
    return result


def _cost(rate, model):
    rate = _scalar(rate, "cost_rate")
    if rate < 0:
        raise ValueError("cost_rate must be nonnegative")
    if model not in COST_MODELS:
        raise ValueError("unknown cost_model")
    return rate


def _finite(*arrays):
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ArithmeticError("simulation arithmetic produced nonfinite binary64 output")


def _state_paths(z, initial_signal, residual, rho, eta, k, moments):
    """Array rows are independent replications, columns are chronological dates."""
    if eta and k and 1 - eta == 1:
        raise ArithmeticError("adjustment is too small for the binary64 recurrence")
    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        try:
            initial_x = (
                moments["position_signal_product"] * initial_signal
                + math.sqrt(moments["conditional_position_variance"]) * residual
            )
            signal = np.empty(z.shape)
            position, trade = np.empty(z.shape), np.empty(z.shape)
            previous_s, previous_x = initial_signal, initial_x
            scale = math.sqrt((1 - rho) * (1 + rho))
            for t in range(z.shape[1]):
                s = rho * previous_s + scale * z[:, t]
                x = (1 - eta) * previous_x + eta * k * s
                signal[:, t], position[:, t], trade[:, t] = s, x, x - previous_x
                previous_s, previous_x = s, x
            _finite(signal, position, trade, initial_x)
        except FloatingPointError as exc:
            raise ArithmeticError("state arithmetic exceeds finite binary64 range") from exc
    return signal, position, trade, initial_x


def _path_metrics(signal, position, trade, epsilon, beta, sigma, rate, model):
    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        try:
            noise = sigma * epsilon
            returns = beta * signal + noise
            gross = position * returns
            squared_trade = trade**2
            costs = rate * np.abs(trade) if model == "proportional" else rate * (squared_trade / 2)
            dated = {
                "signal_second_moment": signal**2,
                "position_second_moment": position**2,
                "position_signal_product": position * signal,
                "squared_turnover": squared_trade,
                "absolute_turnover": np.abs(trade),
                "gross_pnl": gross,
                "cost": costs,
                "net_pnl": gross - costs,
                "forecast_mse": noise**2,
            }
            metrics = {key: np.mean(value, axis=1) for key, value in dated.items()}
            _finite(returns, *dated.values(), *metrics.values())
        except FloatingPointError as exc:
            raise ArithmeticError("PnL arithmetic exceeds finite binary64 range") from exc
    return metrics, returns, dated


def simulate(
    signal_innovations,
    return_innovations,
    *,
    initial_signal,
    initial_residual,
    rho,
    eta,
    k,
    beta,
    sigma,
    cost_rate,
    cost_model,
    keep_records=False,
):
    """One primitive-driven path initialized jointly stationary, without burn-in.

    Observe s_t, choose x_t, debit change from x_(t-1), then earn x_t*r_(t+1).
    There is no charge for establishing the pre-existing stationary position,
    and no terminal liquidation. Epsilon innovations never affect positions.
    """
    rho, eta, k, beta, sigma = _model(rho, eta, k, beta, sigma)
    rate = _cost(cost_rate, cost_model)
    z, epsilon = (
        _vector(signal_innovations, "signal_innovations"),
        _vector(return_innovations, "return_innovations"),
    )
    if z.shape != epsilon.shape:
        raise ValueError("innovation vectors must have equal length")
    initial_signal = _scalar(initial_signal, "initial_signal")
    residual = _scalar(initial_residual, "initial_residual")
    moments = stationary_moments(rho=rho, eta=eta, k=k, beta=beta, sigma=sigma)
    signal, position, trade, initial_x = _state_paths(
        z[None, :], np.array([initial_signal]), np.array([residual]), rho, eta, k, moments
    )
    metrics, returns, dated = _path_metrics(
        signal, position, trade, epsilon[None, :], beta, sigma, rate, cost_model
    )
    result = {
        "initial_position": float(initial_x[0]),
        "metrics": {key: float(value[0]) for key, value in metrics.items()},
    }
    if keep_records:
        result["records"] = [
            {
                "period": t,
                "signal": float(signal[0, t]),
                "position": float(position[0, t]),
                "trade": float(trade[0, t]),
                "next_return": float(returns[0, t]),
                "gross_pnl": float(dated["gross_pnl"][0, t]),
                "cost": float(dated["cost"][0, t]),
                "net_pnl": float(dated["net_pnl"][0, t]),
            }
            for t in range(len(z))
        ]
    return result


def _grid(values, name):
    array = _vector(values, name).tolist()
    if len(set(array)) != len(array):
        raise ValueError(f"{name} must not contain duplicates")
    return array


def _analytic(moments, rate, model):
    result = {key: moments[key] for key in METRICS if key not in ("cost", "net_pnl")}
    c = Fraction(rate)
    cost = (
        c * Fraction(moments["absolute_turnover"])
        if model == "proportional"
        else c * Fraction(moments["squared_turnover"]) / 2
    )
    result["cost"] = _represented(cost, "cost")
    result["net_pnl"] = _represented(Fraction(moments["gross_pnl"]) - cost, "net_pnl")
    return result


def _summary(values, analytic):
    try:
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            mean = float(np.mean(values))
            se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
            deviation = mean - analytic
            _finite(mean, se, deviation)
    except FloatingPointError as exc:
        raise ArithmeticError("summary arithmetic exceeds finite binary64 range") from exc
    return {"mean": mean, "mc_se": se, "analytic": analytic, "deviation": deviation}


def run_lab(
    *,
    seed=2026110823,
    repetitions=256,
    horizon=512,
    rhos=(-0.5, 0, 0.5, 0.9),
    etas=(0, 0.1, 0.5, 1),
    betas=(0, 0.001),
    cost_rates=(0, 0.0001, 0.001),
    cost_models=COST_MODELS,
    k=1,
    sigma=0.02,
):
    """Retain every fixed grid cell and whole-path output, without selection."""
    start = time.perf_counter()
    seed, repetitions, horizon = (
        _integer(value, name, minimum)
        for value, name, minimum in (
            (seed, "seed", 0),
            (repetitions, "repetitions", 2),
            (horizon, "horizon", 1),
        )
    )
    rhos, etas, betas, rates = (
        _grid(values, name)
        for values, name in (
            (rhos, "rhos"),
            (etas, "etas"),
            (betas, "betas"),
            (cost_rates, "cost_rates"),
        )
    )
    if (
        not isinstance(cost_models, (list, tuple))
        or not cost_models
        or any(x not in COST_MODELS for x in cost_models)
        or len(set(cost_models)) != len(cost_models)
    ):
        raise ValueError("cost_models must be a nonempty unique list of named models")
    k, sigma = _scalar(k, "k"), _scalar(sigma, "sigma")
    cells_count = len(rhos) * len(etas) * len(betas) * len(rates) * len(cost_models)
    if (
        horizon > 10000
        or repetitions > 10000
        or repetitions * horizon * cells_count > 30_000_000
        or cells_count > 2048
    ):
        raise ValueError("study budget exceeds explicit offline limit")
    # Validate all analytic inputs before allocating or scoring any outcomes.
    moments_by_cell = {}
    for rho, eta, beta in itertools.product(rhos, etas, betas):
        moments = stationary_moments(rho=rho, eta=eta, k=k, beta=beta, sigma=sigma)
        moments_by_cell[(rho, eta, beta)] = moments
        for rate, model in itertools.product(rates, cost_models):
            _analytic(moments, _cost(rate, model), model)
    config = dict(
        seed=seed,
        repetitions=repetitions,
        horizon=horizon,
        rhos=rhos,
        etas=etas,
        betas=betas,
        cost_rates=rates,
        cost_models=list(cost_models),
        k=k,
        sigma=sigma,
    )
    initial = np.empty((repetitions, 2))
    innovations = np.empty((repetitions, horizon, 2))
    primitive_paths = []
    for rep in range(repetitions):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, rep])))
        initial[rep] = rng.standard_normal(2)
        innovations[rep] = rng.standard_normal((horizon, 2))
        raw = np.concatenate([initial[rep], innovations[rep].reshape(-1)]).astype("<f8").tobytes()
        primitive_paths.append({"replication": rep, "sha256": hashlib.sha256(raw).hexdigest()})
    cells = []
    for rho, eta in itertools.product(rhos, etas):
        reference = moments_by_cell[(rho, eta, betas[0])]
        signal, position, trade, _ = _state_paths(
            innovations[:, :, 0], initial[:, 0], initial[:, 1], rho, eta, k, reference
        )
        for beta, rate, model in itertools.product(betas, rates, cost_models):
            metrics, _, _ = _path_metrics(
                signal, position, trade, innovations[:, :, 1], beta, sigma, rate, model
            )
            moments = moments_by_cell[(rho, eta, beta)]
            analytic = _analytic(moments, rate, model)
            cells.append(
                {
                    "rho": rho,
                    "eta": eta,
                    "beta": beta,
                    "cost_rate": rate,
                    "cost_model": model,
                    "replication_ids": list(range(repetitions)),
                    "per_replication": {key: value.tolist() for key, value in metrics.items()},
                    "summary": {
                        key: _summary(value, analytic[key]) for key, value in metrics.items()
                    },
                    "stationary_reference": moments,
                }
            )
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
        "lab": "signal_turnover",
        "schema_version": 1,
        "synthetic_only": True,
        "config": config,
        "primitive_paths": primitive_paths,
        "cells": cells,
        "sampling": {
            "independent_unit": "whole replication path",
            "coupling": "common initial normals and chronological innovations across all grid cells",
            "initialization": "joint stationary Gaussian; cash eta=0; no burn-in",
            "standard_error": "pointwise sample SD of path means / sqrt(repetitions), ddof=1",
            "random_generator": "PCG64(SeedSequence([seed, replication]))",
            "draw_order": "two initial normals, then horizon-by-2 signal and return normals in C order",
        },
        "limitations": [
            "Synthetic additive unit-notional PnL, not compounded wealth or investment advice.",
            "Unbounded positions and Gaussian returns; no financing, margin, solvency or capital feasibility.",
            "Pre-existing stationary inventory has no inception charge; no terminal liquidation.",
            "Fixed adjustment speeds, not optimized policies or risk-adjusted utility comparisons.",
            "Proportional and quadratic coefficients have different units; equal numeric rates need not be economically matched.",
            "Original elementary stationary moments; no theorem attributed to unread Paleologo chapters.",
            "Pointwise Monte Carlo SE uses independent paths, not serially dependent dates; no simultaneous coverage claim.",
            "Zero beta is a negative control in expectation; finite-sample gross or net may be positive.",
            "No significance pass threshold, selected winner, fitted parameters or live calibration.",
            "Arithmetic outside supported binary64 output range fails explicitly; no clipped economic results.",
            "Exact replay requires matching source, configuration, NumPy and floating-point runtime.",
        ],
        "provenance": {
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "git_sha": git_sha,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "rng": "PCG64",
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
        with args.out.open("x") as stream:
            stream.write(serialized)
    except (TypeError, ValueError, ArithmeticError, OSError) as exc:
        parser.error(str(exc))
    print(args.out)


if __name__ == "__main__":
    main()
