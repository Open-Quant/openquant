"""Offline rolling risk-model comparison, not a strategy or a trading recommendation.

Existing OpenQuant allocators are evaluated without selecting a winner. See
RISK_MODEL_LAB.md for covariance, chronology, drift, cost and uncertainty conventions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

FAMILIES = ("diagonal", "correlated", "regime_shift")
METHODS = (
    "equal_weight",
    "inverse_variance",
    "min_vol_sample",
    "min_vol_shrunk",
    "hrp_sample",
    "hrp_shrunk",
)
METRICS = (
    "realized_variance",
    "mean_conditional_variance",
    "total_traded_fraction",
    "mean_target_weight_change_l1",
    "mean_concentration",
    "gross_terminal_return",
    "net_terminal_return",
)


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _returns(values: Any) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError("returns must be real")
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("returns need at least two observations and two assets")
    if not np.all(np.isfinite(values)) or np.any(values <= -1):
        raise ValueError("simple returns must be finite and greater than -1")
    return values


def _weights(values: Any, n_assets: int) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError("weights must be real")
    weights = np.asarray(values, dtype=float)
    if (
        weights.shape != (n_assets,)
        or not np.all(np.isfinite(weights))
        or np.any(weights < -1e-10)
        or not math.isclose(float(weights.sum()), 1, abs_tol=1e-8, rel_tol=0)
    ):
        raise ValueError("weights must be finite, long-only and fully invested")
    # Only numerical solver residue is clipped; no leverage or hidden fallback.
    weights = np.maximum(weights, 0)
    return weights / weights.sum()


def estimate_covariances(training_returns: Any) -> dict[str, np.ndarray]:
    """Sample covariance and the HRP API's fixed 10% diagonal shrink convention."""
    values = _returns(training_returns)
    with np.errstate(over="ignore", invalid="ignore"):
        sample = np.cov(values, rowvar=False, ddof=1)
    if not np.all(np.isfinite(sample)) or np.any(np.diag(sample) <= 0):
        raise ValueError("training covariance must have finite entries and positive variances")
    return {"sample": sample, "diagonal_shrink_10": 0.9 * sample + 0.1 * np.diag(np.diag(sample))}


def covariance_diagnostics(covariance: np.ndarray) -> dict[str, Any]:
    """Numerical conditioning; null denotes singularity at relative float64 tolerance."""
    eigenvalues = np.linalg.eigvalsh(covariance)
    largest = float(eigenvalues[-1])
    tolerance = largest * len(eigenvalues) * np.finfo(float).eps
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    return {
        "numerical_rank": rank,
        "condition_number": largest / float(eigenvalues[0]) if rank == len(eigenvalues) else None,
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": largest,
    }


def allocate(method: str, covariances: dict[str, np.ndarray]) -> np.ndarray:
    """Call existing allocators with explicit long-only bounds and zero expected returns."""
    from openquant import hrp, portfolio

    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    covariance = covariances["diagonal_shrink_10" if method.endswith("shrunk") else "sample"]
    n_assets = len(covariance)
    if method == "equal_weight":
        weights = np.full(n_assets, 1 / n_assets)
    elif method.startswith("hrp"):
        weights, _ = hrp.allocate_hrp(
            [str(i) for i in range(n_assets)],
            covariance_matrix=covariance.tolist(),
            distance="distance_of_distances",
        )
    else:
        weights, _, _, _ = portfolio.allocate_from_inputs(
            [0.0] * n_assets,
            covariance.tolist(),
            "inverse_variance" if method == "inverse_variance" else "min_volatility",
            tuple_bounds=(0.0, 1.0),
        )
    return _weights(weights, n_assets)


def hold_block(returns: Any, weights: Any, *, cost_fraction: float = 0.0) -> dict[str, Any]:
    """Buy-and-hold inside a block; debit the specified toy cost before its first return."""
    values = np.asarray(returns)
    # A final block may contain one observation, unlike a covariance fit.
    if values.ndim != 2 or len(values) < 1:
        raise ValueError("holdout block must be a nonempty matrix")
    values = _returns(np.concatenate([values, values], axis=0))[: len(values)]
    current = _weights(weights, values.shape[1])
    if not math.isfinite(cost_fraction) or not 0 <= cost_fraction < 1:
        raise ValueError("cost_fraction must be finite and in [0, 1)")
    gross, net, starts = [], [], []
    for index, row in enumerate(values):
        starts.append(current.tolist())
        period_return = float(current @ row)
        gross.append(period_return)
        net.append((1 - cost_fraction) * (1 + period_return) - 1 if index == 0 else period_return)
        current = current * (1 + row) / (1 + period_return)
    return {
        "gross_returns": gross,
        "net_returns": net,
        "starting_weights": starts,
        "ending_weights": current.tolist(),
    }


def _truth(values: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if values is None:
        return None
    if np.iscomplexobj(values):
        raise ValueError("true covariances must be real")
    truth = np.asarray(values, dtype=float)
    if truth.shape != (shape[0], shape[1], shape[1]) or not np.all(np.isfinite(truth)):
        raise ValueError("true covariances need one finite square matrix per observation")
    if not np.allclose(truth, truth.swapaxes(1, 2), rtol=1e-10, atol=1e-14):
        raise ValueError("true covariances must be symmetric")
    if np.any(np.linalg.eigvalsh(truth) < -1e-14):
        raise ValueError("true covariances must be positive semidefinite")
    return truth


def evaluate_path(
    returns: Any,
    *,
    train_size: int = 126,
    rebalance_every: int = 21,
    cost_bps: float = 10.0,
    true_covariances: Any = None,
) -> dict[str, Any]:
    """Rolling training-only allocation with disjoint, exhaustive chronological holdout blocks."""
    values = _returns(returns)
    train_size = _integer(train_size, "train_size", 2)
    rebalance_every = _integer(rebalance_every, "rebalance_every", 1)
    cost_bps = float(cost_bps)
    if not math.isfinite(cost_bps) or not 0 <= cost_bps < 5000:
        raise ValueError("cost_bps must be finite and in [0, 5000)")
    if len(values) - train_size < 2:
        raise ValueError("need at least two holdout observations")
    truth = _truth(true_covariances, values.shape)
    result: dict[str, Any] = {}
    for method in METHODS:
        records, gross, net, conditional = [], [], [], []
        previous_end = np.zeros(values.shape[1])  # Initial investment from cash.
        previous_target = None
        try:
            for start in range(train_size, len(values), rebalance_every):
                end = min(start + rebalance_every, len(values))
                covariances = estimate_covariances(values[start - train_size : start])
                weights = allocate(method, covariances)
                covariance = covariances[
                    "diagonal_shrink_10" if method.endswith("shrunk") else "sample"
                ]
                traded = float(np.abs(weights - previous_end).sum())
                cost = cost_bps * 1e-4 * traded
                held = hold_block(values[start:end], weights, cost_fraction=cost)
                record = {
                    "train_range": [start - train_size, start],
                    "holdout_range": [start, end],
                    "weights": weights.tolist(),
                    "traded_fraction": traded,
                    "cost_fraction": cost,
                    "target_weight_change_l1": None
                    if previous_target is None
                    else float(np.abs(weights - previous_target).sum()),
                    "concentration": float(weights @ weights),
                    "forecast_variance": float(weights @ covariance @ weights),
                    "covariance": covariance_diagnostics(covariance),
                }
                if truth is not None:
                    record["covariance_frobenius_error_next_period"] = float(
                        np.linalg.norm(covariance - truth[start])
                    )
                    starts = np.asarray(held["starting_weights"])
                    conditional.extend(
                        np.einsum("ti,tij,tj->t", starts, truth[start:end], starts).tolist()
                    )
                records.append(record)
                gross.extend(held["gross_returns"])
                net.extend(held["net_returns"])
                previous_end, previous_target = np.asarray(held["ending_weights"]), weights
        except (ValueError, np.linalg.LinAlgError) as exc:
            result[method] = {
                "status": "unavailable",
                "reason": str(exc),
                "completed_blocks": len(records),
            }
            continue
        changes = [r["target_weight_change_l1"] for r in records[1:]]
        result[method] = {
            "status": "available",
            "realized_variance": float(np.var(gross, ddof=1)),
            "mean_conditional_variance": float(np.mean(conditional)) if conditional else None,
            "total_traded_fraction": sum(r["traded_fraction"] for r in records),
            "mean_target_weight_change_l1": float(np.mean(changes)) if changes else None,
            "mean_concentration": float(np.mean([r["concentration"] for r in records])),
            "gross_terminal_return": float(np.prod(1 + np.asarray(gross)) - 1),
            "net_terminal_return": float(np.prod(1 + np.asarray(net)) - 1),
            "gross_returns": gross,
            "net_returns": net,
            "rebalances": records,
        }
    # Never serialize nonfinite results as though they were usable measurements.
    json.dumps(result, allow_nan=False)
    return result


def simulate_family(
    family: str, n_observations: int, n_assets: int, *, seed: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-mean Gaussian controls, with analytic per-observation covariance."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    n_observations = _integer(n_observations, "n_observations", 2)
    n_assets = _integer(n_assets, "n_assets", 2)
    rng = np.random.default_rng(seed)
    scales = np.linspace(0.005, 0.02, n_assets)
    correlation = np.eye(n_assets) if family == "diagonal" else 0.25 * np.eye(n_assets) + 0.75
    first = correlation * np.outer(scales, scales)
    second = (0.8 * np.eye(n_assets) + 0.2) * np.outer(scales[::-1], scales[::-1])
    truth = np.repeat(first[None, :, :], n_observations, axis=0)
    if family == "regime_shift":
        truth[n_observations // 2 :] = second
    innovations = rng.standard_normal((n_observations, n_assets))
    values = np.einsum("tij,tj->ti", np.linalg.cholesky(truth), innovations)
    _returns(values)  # Fail rather than clip or resample an impossible simple return.
    return values, truth


def _summary(values: list[float | None]) -> dict[str, Any]:
    available = [v for v in values if v is not None]
    return {
        "n_available": len(available),
        "n_unavailable": len(values) - len(available),
        "mean": float(np.mean(available)) if available else None,
        "monte_carlo_standard_error": float(np.std(available, ddof=1) / math.sqrt(len(available)))
        if len(available) > 1
        else None,
    }


def run_lab(
    *,
    repetitions: int = 50,
    train_size: int = 126,
    holdout_size: int = 252,
    n_assets: int = 8,
    rebalance_every: int = 21,
    cost_bps: float = 10.0,
    seed: int = 7,
    families: Any = FAMILIES,
) -> dict[str, Any]:
    repetitions = _integer(repetitions, "repetitions", 1)
    train_size = _integer(train_size, "train_size", 2)
    holdout_size = _integer(holdout_size, "holdout_size", 2)
    n_assets = _integer(n_assets, "n_assets", 2)
    seed = _integer(seed, "seed", 0)
    families = list(families)
    if (
        not families
        or len(set(families)) != len(families)
        or any(f not in FAMILIES for f in families)
    ):
        raise ValueError("families must be nonempty, known, and unique")
    scenarios = []
    for family in families:
        records, digest = [], hashlib.sha256()
        for repetition in range(repetitions):
            stream = np.random.SeedSequence([seed, FAMILIES.index(family), repetition])
            values, truth = simulate_family(
                family, train_size + holdout_size, n_assets, seed=stream
            )
            for array in (values, truth):
                digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
                digest.update(np.asarray(array, dtype="<f8").tobytes())
            records.append(
                {
                    "repetition": repetition,
                    "methods": evaluate_path(
                        values,
                        train_size=train_size,
                        rebalance_every=rebalance_every,
                        cost_bps=cost_bps,
                        true_covariances=truth,
                    ),
                }
            )
        summary, paired = {}, {}
        for method in METHODS:
            summary[method], paired[method] = {}, {}
            for metric in METRICS:
                scores, differences = [], []
                for record in records:
                    score = record["methods"][method].get(metric)
                    baseline = record["methods"]["equal_weight"].get(metric)
                    scores.append(score)
                    differences.append(
                        None if score is None or baseline is None else score - baseline
                    )
                summary[method][metric] = _summary(scores)
                paired[method][metric] = _summary(differences)
        scenarios.append(
            {
                "family": family,
                "dataset_sha256": digest.hexdigest(),
                "summary": summary,
                "paired_vs_equal_weight": paired,
                "replications": records,
            }
        )
    return {
        "schema_version": 1,
        "synthetic_only": True,
        "variance_units": "per_observation",
        "config": {
            "repetitions": repetitions,
            "train_size": train_size,
            "holdout_size": holdout_size,
            "n_assets": n_assets,
            "rebalance_every": rebalance_every,
            "cost_bps": float(cost_bps),
            "seed": seed,
            "families": families,
        },
        "method_order": list(METHODS),
        "generator": {
            "mean": 0.0,
            "distribution": "Gaussian",
            "asset_std_range": [0.005, 0.02],
            "correlated_rho": 0.75,
            "regime_shift_fraction": 0.5,
            "shifted_rho": 0.2,
            "shifted_std_order": "reversed",
        },
        "caveats": [
            "Synthetic controls, not calibrated market data or a replication of published performance.",
            "Fixed methods and shrinkage; no winner selection or promotion decision.",
            "Rolling fits may use prior holdout observations once they are historical; never current or future rows.",
            "Full L1 traded fraction includes initial purchase from cash; toy proportional costs ignore market impact, taxes, and financing.",
            "Gaussian returns are unbounded; draws at or below -100% fail rather than clip or resample.",
            "Monte Carlo standard errors use independent paths, not overlapping windows; unavailable paths are explicitly counted.",
            "No annualization, expected-return forecasts, live trading, or investment guarantee.",
        ],
        "scenarios": scenarios,
    }


def _provenance() -> dict[str, str]:
    from openquant import _core

    source = Path(__file__).resolve()
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "unavailable"
    return {
        "git_sha": sha,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "native_extension_sha256": hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        ("repetitions", 50),
        ("train-size", 126),
        ("holdout-size", 252),
        ("assets", 8),
        ("rebalance-every", 21),
        ("seed", 7),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--cost-bps", type=float, default=10)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    try:
        report = run_lab(
            repetitions=args.repetitions,
            train_size=args.train_size,
            holdout_size=args.holdout_size,
            n_assets=args.assets,
            rebalance_every=args.rebalance_every,
            cost_bps=args.cost_bps,
            seed=args.seed,
            families=args.families,
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    report["performance"] = {"wall_seconds": time.perf_counter() - start}
    report["provenance"] = _provenance()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
