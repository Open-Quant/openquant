"""Bounded-IID finite-family mean bounds, not Paleologo RAS or a trading metric.

Original synthetic experiment using the empirical Rademacher theorem documented
in BOUNDED_FINITE_FAMILY_LAB.md. Fixed payoff functions take values in [0, 1].
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

FAMILIES = ("iid_null", "duplicated_null", "correlated_null", "known_signal", "repeated_row")


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _probability(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or np.iscomplexobj(value)
        or not isinstance(value, (int, float, np.number))
    ):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return value


def _matrix(values: Any) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("payoffs must be a nonempty observations-by-candidates matrix")
    if values.dtype.kind not in "iuf":
        raise TypeError("payoffs must be real numeric values")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("payoffs must be finite and within the known range [0, 1]")
    return values


def antithetic_scores(values: Any, signs: Any) -> np.ndarray:
    """One paired maximum score per independent sign vector; each lies in [0, .5].

    For a_j=mean_i(eps_i X_ij), Z=(max_j a_j-min_j a_j)/2.
    Pairing eps and -eps preserves E[max_j a_j], without an absolute supremum.
    """
    values = _matrix(values)
    signs = np.asarray(signs)
    if signs.ndim != 2 or signs.shape[1] != len(values) or signs.shape[0] < 1:
        raise ValueError("signs must have shape (draws >= 1, observations)")
    if signs.dtype.kind not in "iuf" or not np.all((signs == -1) | (signs == 1)):
        raise ValueError("sign entries must be exactly -1 or +1")
    scores = signs @ values / len(values)
    return (scores.max(axis=1) - scores.min(axis=1)) / 2


def exact_complexity(values: Any) -> float:
    """Enumerate signs for at most 16 observations as an independent tiny-case oracle."""
    values = _matrix(values)
    n = len(values)
    if n > 16:
        raise ValueError("exact enumeration is restricted to at most 16 observations")
    indices = np.arange(2**n, dtype=np.uint32)[:, None]
    bits = (indices >> np.arange(n, dtype=np.uint32)) & 1
    signs = bits.astype(np.int8) * 2 - 1
    return float(antithetic_scores(values, signs).mean())


def estimate_complexity(values: Any, *, draws: int, eta: float, seed: Any) -> dict[str, Any]:
    """Upper-bound conditional sign expectation, with Monte Carlo failure at most eta."""
    values = _matrix(values)
    draws = _integer(draws, "draws", 2)
    eta = _probability(eta, "eta")
    # The empirical class may collapse even when population functions differ.
    if np.all(values == values[:, :1]):
        return {
            "method": "exact_identical_columns",
            "draws_used": 0,
            "estimate": 0.0,
            "conditional_mcse": 0.0,
            "mc_upper_margin": 0.0,
            "upper": 0.0,
        }
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, draws, 64):
        signs = (
            rng.integers(0, 2, size=(min(64, draws - start), len(values)), dtype=np.int8) * 2 - 1
        )
        samples.extend(antithetic_scores(values, signs).tolist())
    estimate = float(np.mean(samples))
    # Range(Z) <= 1/2: Hoeffding gives exp(-8 B u^2).
    margin = math.sqrt(-math.log(eta) / (8 * draws))
    return {
        "method": "antithetic_monte_carlo",
        "draws_used": draws,
        "estimate": estimate,
        "conditional_mcse": float(np.std(samples, ddof=1) / math.sqrt(draws)),
        "mc_upper_margin": margin,
        "upper": min(0.5, estimate + margin),
    }


def evaluate_matrix(
    payoffs: Any,
    true_means: Any,
    *,
    draws: int = 512,
    total_failure: float = 0.05,
    seed: Any = 7,
) -> dict[str, Any]:
    """Compute two simultaneous lower bounds and compare with known simulation means.

    This routine checks range/shape, not independence or whether the family was fixed.
    The caller must establish those assumptions externally before using a guarantee.
    """
    values = _matrix(payoffs)
    truth = np.asarray(true_means)
    if truth.shape != (values.shape[1],) or truth.dtype.kind not in "iuf":
        raise ValueError("true_means must be one real number per candidate")
    truth = np.asarray(truth, dtype=np.float64)
    if not np.isfinite(truth).all() or np.any(truth < 0) or np.any(truth > 1):
        raise ValueError("true_means must lie in [0, 1]")
    alpha = _probability(total_failure, "total_failure")
    delta = alpha / 2
    eta = alpha - delta
    if delta <= 0 or eta <= 0:
        raise ValueError("total_failure is too small to split representably")
    n, k = values.shape
    complexity = estimate_complexity(values, draws=draws, eta=eta, seed=seed)
    sample_margin = 3 * math.sqrt((math.log(2) - math.log(delta)) / (2 * n))
    rad_width = 2 * complexity["upper"] + sample_margin
    # One-sided Hoeffding + union over the complete fixed candidate family.
    union_width = math.sqrt((math.log(k) - math.log(alpha)) / (2 * n))
    means = values.mean(axis=0)
    winner = int(np.argmax(means))
    bounds = {}
    for name, width in (("rademacher", rad_width), ("finite_union", union_width)):
        lower = means - width
        bounds[name] = {
            "width": width,
            "lower": lower.tolist(),
            "simultaneous_coverage": bool(np.all(lower <= truth)),
            "selected_coverage": bool(lower[winner] <= truth[winner]),
            "selected_lower": float(lower[winner]),
            "selected_lower_above_null_half": bool(lower[winner] > 0.5),
            "all_lower_at_or_below_zero": bool(np.all(lower <= 0)),
        }
    return {
        "empirical_means": means.tolist(),
        "true_means": truth.tolist(),
        "selected_candidate": winner,
        "selected_true_mean": float(truth[winner]),
        "total_failure": alpha,
        "sample_failure_delta": delta,
        "mc_failure_eta": eta,
        "rademacher_sample_margin": sample_margin,
        "complexity": complexity,
        "bounds": bounds,
    }


def simulate_family(family: str, n: int, k: int, *, seed: Any) -> tuple[np.ndarray, np.ndarray]:
    """Fixed coordinate payoff functions on synthetic vectors; independent rows except one control."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    n, k = _integer(n, "n", 1), _integer(k, "k", 1)
    rng = np.random.default_rng(seed)
    truth = np.full(k, 0.5)
    if family == "duplicated_null":
        independent = rng.binomial(1, 0.5, size=(n, (k + 1) // 2))
        values = np.repeat(independent, 2, axis=1)[:, :k]
    elif family == "correlated_null":
        values = rng.binomial(1, 0.5, size=(n, k))
        common = rng.binomial(1, 0.5, size=(n, 1))
        shared_rows = rng.random((n, 1)) < 0.75
        values = np.where(shared_rows, common, values)
    elif family == "repeated_row":
        values = np.repeat(rng.binomial(1, 0.5, size=(1, k)), n, axis=0)
    else:
        if family == "known_signal":
            truth[0] = 0.8
        values = rng.binomial(1, truth, size=(n, k))
    return values.astype(float), truth


def _mean_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "monte_carlo_standard_error": float(np.std(values, ddof=1) / math.sqrt(len(values))),
    }


def _event_summary(values: list[bool]) -> dict[str, Any]:
    # Wilson interval across independent simulation replications, not strategy confidence.
    n, hits, z = len(values), sum(values), 1.959963984540054
    p = hits / n
    divisor = 1 + z * z / n
    center = (p + z * z / (2 * n)) / divisor
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / divisor
    return {
        "count": hits,
        "total": n,
        "fraction": p,
        "wilson_95_interval": [
            0.0 if hits == 0 else max(0.0, center - radius),
            1.0 if hits == n else min(1.0, center + radius),
        ],
    }


def run_lab(
    *,
    repetitions: int = 200,
    sample_size: int = 256,
    n_candidates: int = 8,
    draws: int = 512,
    total_failure: float = 0.05,
    seed: int = 7,
    families: tuple[str, ...] = FAMILIES,
) -> dict[str, Any]:
    repetitions = _integer(repetitions, "repetitions", 2)
    sample_size = _integer(sample_size, "sample_size", 1)
    n_candidates = _integer(n_candidates, "n_candidates", 1)
    draws = _integer(draws, "draws", 2)
    total_failure = _probability(total_failure, "total_failure")
    seed = _integer(seed, "seed", 0)
    if (
        not families
        or len(set(families)) != len(families)
        or any(f not in FAMILIES for f in families)
    ):
        raise ValueError("families must be a nonempty unique selection of known families")
    start = time.perf_counter()
    reports = {}
    for family in families:
        family_id = FAMILIES.index(family)
        records, digest = [], hashlib.sha256()
        for replication in range(repetitions):
            data_seed = np.random.SeedSequence([seed, family_id, replication, 0, 6501])
            mc_seed = np.random.SeedSequence([seed, family_id, replication, 1, 6501])
            values, truth = simulate_family(family, sample_size, n_candidates, seed=data_seed)
            digest.update(np.asarray(values.shape, dtype="<u8").tobytes())
            digest.update(np.asarray(values, dtype="<f8").tobytes())
            record = evaluate_matrix(
                values, truth, draws=draws, total_failure=total_failure, seed=mc_seed
            )
            record["replication"] = replication
            records.append(record)
        summary = {}
        for method in ("rademacher", "finite_union"):
            summary[method] = {
                "width": _mean_summary([r["bounds"][method]["width"] for r in records])
            }
            for event in (
                "simultaneous_coverage",
                "selected_coverage",
                "selected_lower_above_null_half",
                "all_lower_at_or_below_zero",
            ):
                summary[method][event] = _event_summary(
                    [r["bounds"][method][event] for r in records]
                )
        summary["rademacher_minus_union_width"] = _mean_summary(
            [
                r["bounds"]["rademacher"]["width"] - r["bounds"]["finite_union"]["width"]
                for r in records
            ]
        )
        reports[family] = {
            "iid_rows": family != "repeated_row",
            "fixed_family": True,
            "input_sha256": digest.hexdigest(),
            "records": records,
            "summary": summary,
        }
    root = Path(__file__).resolve().parents[1]
    try:
        base = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        base = None
    return {
        "lab": "bounded_finite_family",
        "schema_version": 1,
        "synthetic_only": True,
        "config": {
            "repetitions": repetitions,
            "sample_size": sample_size,
            "n_candidates": n_candidates,
            "draws": draws,
            "total_failure": total_failure,
            "seed": seed,
            "families": list(families),
        },
        "provenance": {
            "git_base": base,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "performance": {"wall_seconds": time.perf_counter() - start},
        "families": reports,
        "caveats": [
            "Not Paleologo RAS: exact book/slides equations and assumptions remain unverified.",
            "Bounds require a fixed family, known [0,1] support and iid observation rows.",
            "Cross-candidate dependence is allowed; serial dependence is not covered.",
            "Repeated-row control deliberately violates iid; no nominal coverage applies.",
            "Mean payoff bounds are not Sharpe bounds or investment recommendations.",
            "Observed simulation coverage does not prove the theorem or real-data assumptions.",
            "The two methods have separate same-budget guarantees, not joint 95% coverage.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    for name, default in (
        ("repetitions", 200),
        ("sample-size", 256),
        ("candidates", 8),
        ("draws", 512),
        ("seed", 7),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--total-failure", type=float, default=0.05)
    args = parser.parse_args()
    try:
        report = run_lab(
            repetitions=args.repetitions,
            sample_size=args.sample_size,
            n_candidates=args.candidates,
            draws=args.draws,
            total_failure=args.total_failure,
            seed=args.seed,
            families=tuple(args.families),
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
