"""Training-only covariance fits under heavy tails, shifts and constraints.

Standalone mathematical solvers, not native allocators or investment backtests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize

DISTRIBUTIONS = ("gaussian", "elliptical_t5", "independent_t5")
REGIMES = ("isotropic", "diagonal", "hedge_blocks", "diagonal_reversal")
METHODS = ("sample_gmv", "equal_weight", "weight_blend_50", "long_only", "diagonal_shrink_50")
KKT_TOLERANCE = 1e-7


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _matrix(value, name, square=False):
    a = np.asarray(value)
    if a.ndim != 2 or a.dtype.kind not in "iuf" or min(a.shape) < 2:
        raise ValueError(f"{name} must be a real matrix with dimensions >=2")
    a = np.asarray(a, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite")
    if square and (a.shape[0] != a.shape[1] or not np.array_equal(a, a.T)):
        raise ValueError(f"{name} must be square and symmetric")
    return a


def _digest(a):
    a = np.ascontiguousarray(a, dtype="<f8")
    return hashlib.sha256(str(a.shape).encode() + b":<f8:C:" + a.tobytes()).hexdigest()


def covariances(regime, n_assets):
    p = _integer(n_assets, "n_assets", 2)
    if p % 2:
        raise ValueError("n_assets must be even for the paired-block control")
    if regime not in REGIMES:
        raise ValueError("unknown regime")
    if regime == "isotropic":
        train = np.eye(p)
    elif regime == "hedge_blocks":
        train = np.kron(np.eye(p // 2), [[1.0, 1.5], [1.5, 4.0]])
    else:
        train = np.diag(np.geomspace(1.0, 16.0, p))
    future = np.diag(np.diag(train)[::-1]) if regime == "diagonal_reversal" else train.copy()
    return train, future


def draw_training(rng, distribution, train_size, covariance):
    t = _integer(train_size, "train_size", 2)
    a = _matrix(covariance, "covariance", square=True)
    if distribution not in DISTRIBUTIONS:
        raise ValueError("unknown distribution")
    z = rng.standard_normal((t, len(a)))
    if distribution != "gaussian":
        shape = (t, 1) if distribution == "elliptical_t5" else z.shape
        # E[1/chi2_5] = 1/3, hence each latent component has variance one.
        z *= np.sqrt(3.0 / rng.chisquare(5, size=shape))
    return z @ np.linalg.cholesky(a).T


def _unavailable(reason, **details):
    return {"available": False, "reason": reason, "weights": None, **details}


def inverse_gmv(covariance):
    a = _matrix(covariance, "covariance", square=True)
    scale = float(np.max(np.abs(a)))
    if scale == 0:
        return _unavailable("zero covariance")
    a = a / scale
    try:
        np.linalg.cholesky(a)
        if np.linalg.cond(a) > 1e12:
            return _unavailable("condition number exceeds fixed 1e12 threshold")
        direction = np.linalg.solve(a, np.ones(len(a)))
        denominator = float(direction.sum())
        if not np.isfinite(denominator) or denominator <= 0:
            return _unavailable("invalid inverse normalizer")
        w = direction / denominator
    except np.linalg.LinAlgError:
        return _unavailable("positive-definite inverse unavailable")
    if not np.isfinite(w).all() or abs(w.sum() - 1) > 1e-8:
        return _unavailable("invalid inverse weights")
    return {"available": True, "reason": None, "weights": w.tolist()}


def kkt_diagnostics(covariance, weights):
    """Normalized convex QP certificate: S w - gamma >=0, w*(S w-gamma)=0."""
    a = _matrix(covariance, "covariance", square=True)
    w = np.asarray(weights)
    if w.dtype.kind not in "iuf":
        raise ValueError("weights must be real")
    w = np.asarray(w, dtype=float)
    if w.shape != (len(a),) or not np.isfinite(w).all():
        raise ValueError("invalid weights")
    scale = float(np.max(np.abs(a)))
    a = a / scale if scale > 0 else a
    with np.errstate(over="raise", invalid="raise"):
        gradient = a @ w
        gamma = float(w @ gradient)
        reduced = gradient - gamma
    return {
        "budget_error": abs(float(w.sum()) - 1),
        "negative_weight_error": max(0.0, -float(w.min())),
        "dual_error": max(0.0, -float(reduced.min())),
        "complementarity_error": float(np.max(np.abs(w * reduced))),
        "objective_normalized": gamma,
    }


def long_only_gmv(covariance):
    a = _matrix(covariance, "covariance", square=True)
    scale = float(np.max(np.abs(a)))
    normalized = a / scale if scale > 0 else a.copy()
    if np.linalg.eigvalsh(normalized).min() < -64 * len(a) * np.finfo(float).eps:
        raise ValueError("long-only objective must be positive semidefinite")
    p = len(a)
    try:
        result = minimize(
            lambda w: float(w @ normalized @ w),
            np.full(p, 1 / p),
            jac=lambda w: 2 * normalized @ w,
            bounds=[(0.0, 1.0)] * p,
            constraints={
                "type": "eq",
                "fun": lambda w: float(w.sum() - 1),
                "jac": lambda w: np.ones(p),
            },
            method="SLSQP",
            options={"ftol": 1e-14, "maxiter": 500},
        )
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        return _unavailable("solver raised a numerical exception", solver_message=str(exc))
    details = {
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "iterations": int(result.nit),
    }
    try:
        diagnostics = kkt_diagnostics(a, result.x)
    except (ValueError, ArithmeticError):
        return _unavailable("invalid solver output", kkt=None, **details)
    details["kkt"] = diagnostics
    residual = max(v for k, v in diagnostics.items() if k.endswith("error"))
    if not result.success or residual > KKT_TOLERANCE:
        return _unavailable("solver failure or KKT residual exceeds tolerance", **details)
    # Preserve solver output: no clipping, projection, pseudoinverse or fallback.
    return {"available": True, "reason": None, "weights": np.asarray(result.x).tolist(), **details}


def fit_models(training, *, timings=None):
    x = _matrix(training, "training")
    t, p = x.shape
    start = time.perf_counter()
    with np.errstate(over="raise", invalid="raise"):
        try:
            centered = x - x.mean(axis=0)
            sample = centered.T @ centered / (t - 1)
        except FloatingPointError as exc:
            raise ValueError("sample covariance is not representable") from exc
    if timings is not None:
        timings["covariance"] = timings.get("covariance", 0.0) + time.perf_counter() - start

    def timed(name, operation):
        begin = time.perf_counter()
        result = operation()
        if timings is not None:
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - begin
        return result

    if not np.isfinite(sample).all():
        raise ValueError("sample covariance is not representable")
    equal = np.full(p, 1 / p)
    sample_fit = (
        timed("sample_gmv", lambda: inverse_gmv(sample))
        if t > p
        else _unavailable("T<=p: centered sample inverse unavailable")
    )
    blend = (
        {
            "available": True,
            "reason": None,
            "weights": (0.5 * np.array(sample_fit["weights"]) + 0.5 * equal).tolist(),
        }
        if sample_fit["available"]
        else _unavailable("sample GMV component unavailable")
    )
    shrink = 0.5 * sample + 0.5 * np.diag(np.diag(sample))
    fits = {
        "sample_gmv": sample_fit,
        "equal_weight": {"available": True, "reason": None, "weights": equal.tolist()},
        "weight_blend_50": blend,
        "long_only": timed("long_only", lambda: long_only_gmv(sample)),
        "diagonal_shrink_50": timed("diagonal_shrink_50", lambda: inverse_gmv(shrink)),
    }
    return {"sample_covariance": sample, "diagonal_shrink_covariance": shrink, "methods": fits}


def evaluate(training, future_covariance, *, timings=None):
    fitted = fit_models(training, timings=timings)
    future = _matrix(future_covariance, "future covariance", square=True)
    if future.shape != fitted["sample_covariance"].shape:
        raise ValueError("future covariance dimension mismatch")
    np.linalg.cholesky(future)
    for name, row in fitted["methods"].items():
        if not row["available"]:
            continue
        w = np.array(row["weights"])
        objective = (
            fitted["diagonal_shrink_covariance"]
            if name == "diagonal_shrink_50"
            else fitted["sample_covariance"]
        )
        row.update(
            {
                "future_variance": float(w @ future @ w),
                "sample_variance": float(w @ fitted["sample_covariance"] @ w),
                "fitted_objective": float(w @ objective @ w),
                "gross_leverage": float(np.abs(w).sum()),
                "squared_weight_norm": float(w @ w),
                "max_abs_weight": float(np.abs(w).max()),
                "negative_weight_count": int((w < -1e-10).sum()),
            }
        )
        metrics = (
            "future_variance",
            "sample_variance",
            "fitted_objective",
            "gross_leverage",
            "squared_weight_norm",
            "max_abs_weight",
        )
        if not all(np.isfinite(row[key]) for key in metrics):
            fitted["methods"][name] = _unavailable("nonfinite scored metric")
    return fitted["methods"]


def population_oracle(regime, n_assets):
    """Analytical future-covariance oracle for the four specified families only."""
    _, a = covariances(regime, n_assets)
    unconstrained = np.array(inverse_gmv(a)["weights"])
    long_only = np.zeros(n_assets) if regime == "hedge_blocks" else unconstrained.copy()
    if regime == "hedge_blocks":
        long_only[::2] = 2 / n_assets
    return {
        "unrestricted_weights": unconstrained.tolist(),
        "long_only_weights": long_only.tolist(),
        "unrestricted_variance": float(unconstrained @ a @ unconstrained),
        "long_only_variance": float(long_only @ a @ long_only),
        "equal_weight_variance": float(
            np.full(n_assets, 1 / n_assets) @ a @ np.full(n_assets, 1 / n_assets)
        ),
    }


def _summary(values):
    if not values:
        return {"available": 0, "descriptive_mean": None, "mean_mcse": None, "quantiles": None}
    a = np.array(values)
    return {
        "available": len(values),
        "descriptive_mean": float(a.mean()),
        "mean_mcse": None,
        "quantiles": {str(q): float(np.quantile(a, q)) for q in (0.5, 0.9, 0.99)},
    }


def _bernoulli(values):
    n = len(values)
    if not n:
        return {"available": 0, "rate": None, "mcse": None, "wilson_95": None}
    p = float(np.mean(values))
    z = 1.959963984540054
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {
        "available": n,
        "rate": p,
        "mcse": float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else None,
        "wilson_95": [float(center - radius), float(center + radius)],
    }


def _provenance():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_sha256": {},
    }


def run_lab(*, repetitions=1000, seed=419, n_assets=8, train_size=12):
    started = time.perf_counter()
    timings = {}
    repetitions = _integer(repetitions, "repetitions", 1)
    seed = _integer(seed, "seed", 0)
    p = _integer(n_assets, "n_assets", 2)
    t = _integer(train_size, "train_size", 2)
    covariances("isotropic", p)
    report = {
        "schema_version": 1,
        "lab": "covariance_robustness",
        "synthetic_only": True,
        "config": {"repetitions": repetitions, "seed": seed, "n_assets": p, "train_size": t},
        "provenance": _provenance(),
        "scenarios": [],
    }
    for di, distribution in enumerate(DISTRIBUTIONS):
        for ri, regime in enumerate(REGIMES):
            train, future = covariances(regime, p)
            records = []
            for rep in range(repetitions):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [seed, p, t, di, 1 if regime == "diagonal_reversal" else ri, rep]
                    )
                )
                x = draw_training(rng, distribution, t, train)
                records.append(
                    {
                        "replication": rep,
                        "training_sha256": _digest(x),
                        "methods": evaluate(x, future, timings=timings),
                    }
                )
            summaries = {}
            equal_risk = population_oracle(regime, p)["equal_weight_variance"]
            for method in METHODS:
                available = [
                    r["methods"][method] for r in records if r["methods"][method]["available"]
                ]
                summaries[method] = {
                    "unavailable": repetitions - len(available),
                    "total_repetitions": repetitions,
                    "conditioning": "successful fits only; availability may select samples",
                    "failure_reasons": {
                        reason: sum(r["methods"][method]["reason"] == reason for r in records)
                        for reason in sorted(
                            {
                                r["methods"][method]["reason"]
                                for r in records
                                if not r["methods"][method]["available"]
                            }
                        )
                    },
                    **{
                        key: _summary([r[key] for r in available])
                        for key in (
                            "future_variance",
                            "sample_variance",
                            "fitted_objective",
                            "gross_leverage",
                            "squared_weight_norm",
                        )
                    },
                    "future_risk_below_equal_weight": _bernoulli(
                        [r["future_variance"] < equal_risk - 1e-12 * equal_risk for r in available]
                    ),
                }
            paired = {}
            for left, right in (
                ("long_only", "sample_gmv"),
                ("diagonal_shrink_50", "weight_blend_50"),
                ("long_only", "diagonal_shrink_50"),
            ):
                both = [
                    (r["methods"][left], r["methods"][right])
                    for r in records
                    if r["methods"][left]["available"] and r["methods"][right]["available"]
                ]
                differences = [a["future_variance"] - b["future_variance"] for a, b in both]
                paired[left + "_minus_" + right] = {
                    "total_repetitions": repetitions,
                    "unavailable_pairs": repetitions - len(both),
                    "conditioning": "both fits available; intervals are pointwise, not simultaneous",
                    "future_variance_difference": _summary(differences),
                    "left_lower": _bernoulli([d < -1e-12 * equal_risk for d in differences]),
                }
            report["scenarios"].append(
                {
                    "distribution": distribution,
                    "regime": regime,
                    "train_covariance_sha256": _digest(train),
                    "future_covariance_sha256": _digest(future),
                    "train_covariance": train.tolist(),
                    "future_covariance": future.tolist(),
                    "population_oracle": population_oracle(regime, p),
                    "records": records,
                    "summary": summaries,
                    "paired": paired,
                }
            )
    report["performance"] = {"wall_seconds": time.perf_counter() - started, "fit_seconds": timings}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=419)
    parser.add_argument("--n-assets", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run_lab(
        repetitions=args.repetitions,
        seed=args.seed,
        n_assets=args.n_assets,
        train_size=args.train_size,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
