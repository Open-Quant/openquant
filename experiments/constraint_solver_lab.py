"""Singular simplex ambiguity and a verified feasible-unconstrained fast path."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import covariance_robustness_lab as robustness  # noqa: E402

CASES = {
    "symmetric_null_assets": ([[1, -1, 0, 0]], [1, 1, 4, 16]),
    "asymmetric_factor": ([[2, -1, 0, 0]], [1, 1, 4, 16]),
    "two_factors": ([[1, -1, 0, 0, 1, -1], [0, 0, 1, -1, 1, -1]], [1, 2, 4, 8, 16, 32]),
}


def fast_long_only(covariance, unconstrained=None):
    """Reuse a qualified feasible SPD inverse solution exactly; else strict SLSQP."""
    a = robustness._matrix(covariance, "covariance", square=True)
    scale = float(np.max(np.abs(a)))
    qualified = False
    if scale > 0:
        normalized = a / scale
        try:
            np.linalg.cholesky(normalized)
            qualified = np.linalg.cond(normalized) <= 1e12
        except np.linalg.LinAlgError:
            pass
    if unconstrained is not None and not isinstance(unconstrained, dict):
        raise TypeError("unconstrained candidate must be a result mapping")
    candidate = unconstrained if unconstrained is not None else robustness.inverse_gmv(a)
    if qualified and candidate.get("available"):
        w = np.asarray(candidate.get("weights"))
        if w.shape == (len(a),) and w.dtype.kind in "iuf" and np.isfinite(w).all() and w.min() >= 0:
            try:
                diagnostic = robustness.kkt_diagnostics(a, w)
                residual = max(v for k, v in diagnostic.items() if k.endswith("error"))
                gradient = normalized @ np.asarray(w, dtype=float)
                full_stationarity = float(np.max(np.abs(gradient - float(w @ gradient))))
                diagnostic["unconstrained_stationarity_error"] = full_stationarity
                if (
                    residual <= robustness.KKT_TOLERANCE
                    and full_stationarity <= robustness.KKT_TOLERANCE
                ):
                    return {
                        "available": True,
                        "reason": None,
                        "weights": w.tolist(),
                        "used_fast_path": True,
                        "kkt": diagnostic,
                    }
            except (ValueError, ArithmeticError):
                pass
    return {**robustness.long_only_gmv(a), "used_fast_path": False}


def equality_system(factors):
    factors = np.asarray(factors)
    if (
        factors.ndim != 2
        or min(factors.shape) < 1
        or factors.shape[1] < 2
        or factors.dtype.kind not in "iuf"
    ):
        raise ValueError("factors must be real nonempty rows with at least two columns")
    factors = np.asarray(factors, dtype=float)
    if not np.isfinite(factors).all():
        raise ValueError("factors must be finite")
    scale = np.max(np.abs(factors), axis=1)
    if np.any(scale == 0):
        raise ValueError("factor rows must be nonzero")
    a = np.vstack([np.ones(factors.shape[1]), factors / scale[:, None]])
    b = np.r_[1.0, np.zeros(len(factors))]
    if np.linalg.matrix_rank(a) != len(a):
        raise ValueError("equality rows must be independent")
    return a, b


def zero_risk_vertices(factors):
    """Enumerate basic feasible solutions of 1'w=1, Fw=0, w>=0 (small p)."""
    a, b = equality_system(factors)
    p = a.shape[1]
    if p > 8:
        raise ValueError("exhaustive reference limited to eight coordinates")
    vertices = []
    for size in range(1, len(a) + 1):
        for support in itertools.combinations(range(p), size):
            cols = a[:, support]
            if np.linalg.matrix_rank(cols) < size:
                continue
            active = np.linalg.lstsq(cols, b, rcond=None)[0]
            if np.max(np.abs(cols @ active - b)) > 1e-10 or np.min(active) <= 1e-10:
                continue
            w = np.zeros(p)
            w[list(support)] = active
            vertices.append(w)
    if not vertices:
        raise ValueError("no feasible zero-risk simplex vertex")
    return np.asarray(vertices)


def minimum_on_zero_set(factors, objective):
    """SPD quadratic minimum over all faces; no future objective enters training fits."""
    a, b = equality_system(factors)
    q = robustness._matrix(objective, "objective", square=True)
    p = a.shape[1]
    if q.shape != (p, p) or p > 8:
        raise ValueError("incompatible or unsupported objective")
    q = q / np.max(np.abs(q))
    np.linalg.cholesky(q)
    choices = []
    for size in range(1, p + 1):
        for support in itertools.combinations(range(p), size):
            cols = a[:, support]
            sub = q[np.ix_(support, support)]
            mapped = np.linalg.solve(sub, cols.T)
            multiplier = np.linalg.lstsq(cols @ mapped, b, rcond=None)[0]
            active = mapped @ multiplier
            if np.max(np.abs(cols @ active - b)) > 1e-9 or np.min(active) < -1e-10:
                continue
            w = np.zeros(p)
            w[list(support)] = active
            choices.append((float(w @ q @ w), w))
    if not choices:
        raise ValueError("no feasible face minimizer")
    return min(choices, key=lambda row: row[0])[1]


def primary_from_start(covariance, start):
    a = robustness._matrix(covariance, "covariance", square=True)
    start = np.asarray(start)
    if start.dtype.kind not in "iuf":
        raise ValueError("start must be real")
    start = np.asarray(start, dtype=float)
    if (
        start.shape != (len(a),)
        or not np.isfinite(start).all()
        or start.min() < 0
        or abs(start.sum() - 1) > 1e-10
    ):
        raise ValueError("start must be a feasible simplex vector")
    scale = float(np.max(np.abs(a)))
    normalized = a / scale if scale else a
    if np.linalg.eigvalsh(normalized).min() < -64 * len(a) * np.finfo(float).eps:
        raise ValueError("primary covariance must be PSD")
    try:
        result = minimize(
            lambda w: float(w @ normalized @ w),
            start,
            jac=lambda w: 2 * normalized @ w,
            bounds=[(0.0, 1.0)] * len(a),
            constraints={
                "type": "eq",
                "fun": lambda w: float(w.sum() - 1),
                "jac": lambda w: np.ones(len(a)),
            },
            method="SLSQP",
            options={"ftol": 1e-14, "maxiter": 500},
        )
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        return {
            "available": False,
            "weights": None,
            "reason": "solver exception",
            "message": str(exc),
        }

    try:
        diagnostics = robustness.kkt_diagnostics(a, result.x)
    except (ValueError, ArithmeticError):
        return {"available": False, "weights": None, "reason": "invalid solver output"}
    available = (
        bool(result.success)
        and max(v for k, v in diagnostics.items() if k.endswith("error")) <= 1e-7
    )
    return {
        "available": available,
        "weights": result.x.tolist() if available else None,
        "reason": None if available else "solver or KKT failure",
        "kkt": diagnostics,
        "solver_status": int(result.status),
    }


def ambiguity_case(name):
    factors, diagonal = CASES[name]
    factors = np.array(factors, float)
    future = np.diag(diagonal)
    cov = factors.T @ factors
    p = len(future)
    vertices = zero_risk_vertices(factors)
    minimum_norm = minimum_on_zero_set(factors, np.eye(p))
    future_minimum = minimum_on_zero_set(factors, future)

    def scored(w):
        return {
            "weights": w.tolist(),
            "training_variance": float(w @ cov @ w),
            "future_variance": float(w @ future @ w),
            "squared_norm": float(w @ w),
            "zero_exposure_residual": float(np.max(np.abs(factors @ w))),
            "budget_error": abs(float(w.sum() - 1)),
        }

    starts = [np.full(p, 1 / p), *np.eye(p)]
    solutions = []
    for start in starts:
        fit = primary_from_start(cov, start)
        solutions.append(
            {
                "start": start.tolist(),
                "fit": fit,
                "score": scored(np.array(fit["weights"])) if fit["available"] else None,
            }
        )
    ridge = []
    for strength in [1e-8, 1e-4, 0.01, 0.1]:
        changed = cov + strength * np.max(np.abs(cov)) * np.eye(p)
        fit = fast_long_only(changed)
        ridge.append(
            {
                "strength": strength,
                "fit": fit,
                "original_objective_score": scored(np.array(fit["weights"]))
                if fit["available"]
                else None,
            }
        )
    return {
        "case": name,
        "factors": factors.tolist(),
        "future_covariance": future.tolist(),
        "zero_risk_vertices": [scored(w) for w in vertices],
        "minimum_norm_tie_break": scored(minimum_norm),
        "future_minimum_on_zero_set_diagnostic_only": scored(future_minimum),
        "maximum_future_variance_on_zero_set": max(float(w @ future @ w) for w in vertices),
        "multistart_primary": solutions,
        "ridge_changes_primary_objective": ridge,
    }


def run_lab(*, benchmark_repetitions=100, seed=419):
    start = time.perf_counter()
    reps = robustness._integer(benchmark_repetitions, "benchmark_repetitions", 1)
    seed = robustness._integer(seed, "seed", 0)
    cases = [ambiguity_case(name) for name in CASES]
    records = []
    timings = {"strict_seconds": 0.0, "fast_seconds": 0.0}
    for t in [4, 12, 68]:
        for di, distribution in enumerate(robustness.DISTRIBUTIONS):
            for ri, regime in enumerate(robustness.REGIMES):
                cov, future = robustness.covariances(regime, 8)
                for rep in range(reps):
                    rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [seed, 8, t, di, 1 if regime == "diagonal_reversal" else ri, rep]
                        )
                    )
                    x = robustness.draw_training(rng, distribution, t, cov)
                    centered = x - x.mean(axis=0)
                    sample = centered.T @ centered / (t - 1)
                    inverse = robustness.inverse_gmv(sample) if t > 8 else {"available": False}
                    fits = {}
                    for name in ["strict", "fast"] if rep % 2 == 0 else ["fast", "strict"]:
                        begin = time.perf_counter()
                        fits[name] = (
                            robustness.long_only_gmv(sample)
                            if name == "strict"
                            else fast_long_only(sample, inverse)
                        )
                        timings[name + "_seconds"] += time.perf_counter() - begin
                    row = {
                        "train_size": t,
                        "distribution": distribution,
                        "regime": regime,
                        "replication": rep,
                        "training_sha256": robustness._digest(x),
                        "used_fast_path": fits["fast"]["used_fast_path"],
                        "strict_available": fits["strict"]["available"],
                        "fast_available": fits["fast"]["available"],
                    }
                    for name, fit in fits.items():
                        row[name + "_weights"] = fit["weights"]
                        row[name + "_future_variance"] = (
                            float(np.array(fit["weights"]) @ future @ np.array(fit["weights"]))
                            if fit["available"]
                            else None
                        )
                    if fits["fast"]["used_fast_path"]:
                        assert fits["fast"]["weights"] == inverse["weights"]
                    records.append(row)
    return {
        "lab": "constraint_solver",
        "schema_version": 1,
        "synthetic_only": True,
        "config": {"benchmark_repetitions": reps, "seed": seed},
        "cases": cases,
        "benchmark_records": records,
        "provenance": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependency_sha256": {
                "covariance_robustness_lab.py": hashlib.sha256(
                    Path(robustness.__file__).read_bytes()
                ).hexdigest()
            },
        },
        "performance": {"wall_seconds": time.perf_counter() - start, **timings},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-repetitions", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run_lab(benchmark_repetitions=args.benchmark_repetitions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
