"""Gaussian covariance-estimation risk with exact GMV reference laws.

Unrestricted fully invested weights; no native allocator, execution or trading.
See COVARIANCE_ESTIMATION_LAB.md for the independent regression derivation.
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

import numpy as np

try:
    import scipy
    from scipy.stats import chi2, f
except ImportError:
    scipy = None
    chi2 = f = None

FAMILIES = ("isotropic", "heterogeneous_diagonal")
SHRINKAGES = (0.0, 0.25, 0.5, 0.75, 1.0)
QUANTILES = (0.5, 0.9, 0.99)


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _shrinkage(value):
    if (
        isinstance(value, (bool, np.bool_, str, bytes))
        or np.iscomplexobj(value)
        or not np.isscalar(value)
    ):
        raise TypeError("shrinkage must be a real scalar")
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError("shrinkage must be in [0,1]")
    return value


def true_covariance(family, n_assets):
    p = _integer(n_assets, "n_assets", 2)
    if family not in FAMILIES:
        raise ValueError("unknown covariance family")
    return np.eye(p) if family == "isotropic" else np.diag(np.geomspace(1.0, 16.0, p))


def population_reference(covariance):
    """Unrestricted GMV and equal-weight population risks, diagnostic truth only."""
    a = np.asarray(covariance)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) < 2 or a.dtype.kind not in "iuf":
        raise ValueError("covariance must be a real square matrix of dimension >=2")
    a = np.asarray(a, dtype=float)
    if not np.isfinite(a).all() or not np.array_equal(a, a.T):
        raise ValueError("covariance must be finite and symmetric")
    try:
        scale = float(np.max(np.abs(a)))
        if scale == 0:
            raise ValueError("covariance must be positive definite")
        normalized = a / scale
        np.linalg.cholesky(normalized)
        direction = np.linalg.solve(normalized, np.ones(len(a)))
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    denominator = float(direction.sum())
    if not np.isfinite(direction).all() or not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("population weights are not numerically representable")
    weights = direction / denominator
    equal = np.full(len(a), 1 / len(a))
    oracle_variance = float(weights @ a @ weights)
    equal_variance = float(equal @ a @ equal)
    if not (
        math.isfinite(oracle_variance)
        and oracle_variance > 0
        and math.isfinite(equal_variance)
        and equal_variance > 0
    ):
        raise ValueError("population risks are not numerically representable")
    return {
        "oracle_weights": weights,
        "equal_weights": equal,
        "oracle_variance": oracle_variance,
        "equal_variance": equal_variance,
    }


def exact_reference(train_size, n_assets, covariance, shrinkage):
    """Moment existence and optional F/chi-square quantiles, not fitted thresholds."""
    t = _integer(train_size, "train_size", 2)
    p = _integer(n_assets, "n_assets", 2)
    lam = _shrinkage(shrinkage)
    if np.shape(covariance) != (p, p):
        raise ValueError("covariance dimension must match n_assets")
    population = population_reference(covariance)
    oracle, equal = population["oracle_variance"], population["equal_variance"]
    available = lam == 1 or t > p
    mean_exists = lam == 1 or t > p + 1
    variance_exists = lam == 1 or t > p + 3
    risk_mean = (
        equal
        if lam == 1
        else oracle + (1 - lam) ** 2 * oracle * (p - 1) / (t - p - 1) + lam**2 * (equal - oracle)
        if available and mean_exists
        else None
    )
    forecast_mean = (
        equal
        if lam == 1
        else ((1 - lam**2) * oracle * (t - p) / (t - 1) + lam**2 * equal if available else None)
    )
    risk_quantiles = forecast_quantiles = None
    if lam == 1:
        risk_quantiles = {str(q): equal / oracle for q in QUANTILES}
    elif available and f is not None:
        # Under isotropy equal weight is the oracle, so fixed weight shrink scales
        # the squared estimation error exactly. No such F claim for other targets.
        if lam == 0 or np.array_equal(population["equal_weights"], population["oracle_weights"]):
            risk_quantiles = {
                str(q): float(
                    1 + (1 - lam) ** 2 * (p - 1) / (t - p + 1) * f.ppf(q, p - 1, t - p + 1)
                )
                for q in QUANTILES
            }
    if available and lam == 0 and chi2 is not None:
        forecast_quantiles = {str(q): float(chi2.ppf(q, t - p) / (t - 1)) for q in QUANTILES}
    return {
        "estimator_available": available,
        "population_variance_mean_status": "finite"
        if available and mean_exists
        else "infinite"
        if available
        else "unavailable",
        "population_variance_has_finite_variance": bool(available and variance_exists),
        "expected_population_variance": risk_mean,
        "expected_forecast_variance": forecast_mean,
        "population_risk_ratio_quantiles": risk_quantiles,
        "population_quantile_law": "point_mass"
        if lam == 1
        else "scaled_F"
        if risk_quantiles
        else None,
        "sample_gmv_forecast_ratio_quantiles": forecast_quantiles,
        "scipy_quantiles_available": scipy is not None,
        "risk_F_degrees_of_freedom": [p - 1, t - p + 1] if available and lam == 0 else None,
        "forecast_chi_square_degrees_of_freedom": t - p if available and lam == 0 else None,
    }


def fit_sample_gmv(observations):
    """Centered sample covariance; singular designs are unavailable, never regularized."""
    x = np.asarray(observations)
    if x.ndim != 2 or min(x.shape) < 2 or x.dtype.kind not in "iuf" or not np.isfinite(x).all():
        raise ValueError("observations must be finite real matrix with both dimensions >=2")
    x = np.asarray(x, dtype=float)
    t, p = x.shape
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            centered = x - x.mean(axis=0)
            sample = centered.T @ centered / (t - 1)
        except FloatingPointError as exc:
            raise ValueError("sample covariance not numerically representable") from exc
    diagnostics = {
        "centered_degrees_of_freedom": t - 1,
        "sample_covariance": sample,
        "weights": None,
        "failure": None,
        "condition_number": None,
    }
    if t <= p:
        diagnostics["failure"] = "centered degrees of freedom below asset count"
        return diagnostics
    try:
        np.linalg.cholesky(sample)
        eigenvalues = np.linalg.eigvalsh(sample)
        direction = np.linalg.solve(sample, np.ones(p))
        denominator = float(direction.sum())
        if denominator <= 0 or not math.isfinite(denominator):
            raise ValueError("invalid GMV normalizer")
        weights = direction / denominator
        if not np.isfinite(weights).all() or abs(float(weights.sum()) - 1) > 1e-8:
            raise ValueError("invalid budget residual")
        diagnostics["weights"] = weights
        condition = float(eigenvalues[-1] / eigenvalues[0])
        diagnostics["condition_number"] = (
            condition if math.isfinite(condition) and condition > 0 else None
        )
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        diagnostics["failure"] = str(exc)
    return diagnostics


def evaluate(observations, covariance, shrinkages=SHRINKAGES):
    """Fit once; population covariance enters risk scoring only, never the fit."""
    fit = fit_sample_gmv(observations)
    p = fit["sample_covariance"].shape[0]
    if np.shape(covariance) != (p, p):
        raise ValueError("population covariance has incompatible dimension")
    truth = population_reference(covariance)
    equal, oracle = truth["equal_weights"], truth["oracle_variance"]
    records = {}
    for lam in shrinkages:
        lam = _shrinkage(lam)
        if lam < 1 and fit["weights"] is None:
            records[str(lam)] = {"available": False, "failure": fit["failure"]}
            continue
        weights = equal if lam == 1 else (1 - lam) * fit["weights"] + lam * equal
        risk = float(weights @ covariance @ weights)
        forecast = float(weights @ fit["sample_covariance"] @ weights)
        row = {
            "available": True,
            "failure": None,
            "weights": weights.tolist(),
            "population_variance": risk,
            "forecast_variance": forecast,
            "population_risk_ratio": risk / oracle,
            "forecast_risk_ratio": forecast / oracle,
            "squared_weight_norm": float(weights @ weights),
            "gross_leverage": float(np.abs(weights).sum()),
            "max_abs_weight": float(np.max(np.abs(weights))),
            "negative_weight_count": int(np.sum(weights < 0)),
            "budget_residual": float(weights.sum() - 1),
        }
        if not all(
            math.isfinite(row[k]) for k in row if k not in ("available", "failure", "weights")
        ):
            row = {"available": False, "failure": "nonfinite metric"}
        records[str(lam)] = row
    return {
        "condition_number": fit["condition_number"],
        "sample_fit_failure": fit["failure"],
        "methods": records,
    }


def _summary(values, *, mean_exists=True, variance_exists=True):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {
            "available": 0,
            "empirical_mean": None,
            "monte_carlo_standard_error": None,
            "quantiles": None,
        }
    constant = bool(np.min(a) == np.max(a))
    mean = float(a[0]) if constant else float(a.mean())
    deviation = 0.0 if constant else float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return {
        "available": len(a),
        "empirical_mean": mean if mean_exists else None,
        "monte_carlo_standard_error": deviation / math.sqrt(len(a))
        if variance_exists and len(a) > 1
        else None,
        "quantiles": {str(q): float(np.quantile(a, q)) for q in QUANTILES},
        "maximum_observed": float(a.max()),
        "mean_interpretation": "finite population mean"
        if mean_exists
        else "mean does not exist; no sample-mean inference",
        "mcse_interpretation": "finite variance model"
        if variance_exists
        else "infinite variance model; MCSE suppressed",
    }


def _exceedance(values, threshold, expected):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"available": 0, "rate": None, "expected_rate": expected}
    count = len(a)
    successes = int(np.sum(a > threshold))
    rate = successes / count
    z = 1.959963984540054
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return {
        "available": count,
        "exceedances": successes,
        "rate": rate,
        "expected_rate": expected,
        "threshold": threshold,
        "monte_carlo_standard_error": math.sqrt(rate * (1 - rate) / (count - 1))
        if count > 1
        else None,
        "wilson_95": [
            0.0 if successes == 0 else center - half,
            1.0 if successes == count else center + half,
        ],
    }


def run_lab(
    *,
    repetitions=2000,
    n_assets=8,
    train_size=20,
    seed=137,
    families=FAMILIES,
    shrinkages=SHRINKAGES,
):
    repetitions = _integer(repetitions, "repetitions", 2)
    p = _integer(n_assets, "n_assets", 2)
    t = _integer(train_size, "train_size", 2)
    seed = _integer(seed, "seed", 0)
    families, shrinkages = tuple(families), tuple(_shrinkage(x) for x in shrinkages)
    if (
        not families
        or len(set(families)) != len(families)
        or any(x not in FAMILIES for x in families)
    ):
        raise ValueError("families must be distinct known families")
    if not shrinkages or len(set(shrinkages)) != len(shrinkages):
        raise ValueError("shrinkages must be distinct and nonempty")
    scenarios = []
    for family in families:
        covariance = true_covariance(family, p)
        scale = np.linalg.cholesky(covariance)
        population = population_reference(covariance)
        reference = {str(lam): exact_reference(t, p, covariance, lam) for lam in shrinkages}
        records = []
        digest = hashlib.sha256()
        for rep in range(repetitions):
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, FAMILIES.index(family), p, t, rep, 7329])
            )
            observations = rng.normal(size=(t, p)) @ scale.T
            digest.update(np.asarray(observations.shape, dtype="<u8").tobytes())
            digest.update(np.asarray(observations, dtype="<f8").tobytes())
            record = evaluate(observations, covariance, shrinkages)
            record["replication"] = rep
            records.append(record)
        summaries = {}
        for lam in shrinkages:
            name = str(lam)
            rows = [r["methods"][name] for r in records if r["methods"][name]["available"]]
            mean_exists = lam == 1 or t > p + 1
            risk_variance_exists = lam == 1 or t > p + 3
            metrics = {}
            for metric in (
                "population_variance",
                "forecast_variance",
                "population_risk_ratio",
                "forecast_risk_ratio",
                "squared_weight_norm",
                "gross_leverage",
                "max_abs_weight",
                "negative_weight_count",
            ):
                quadratic = metric in (
                    "population_variance",
                    "population_risk_ratio",
                    "squared_weight_norm",
                )
                linear = metric in ("gross_leverage", "max_abs_weight")
                metrics[metric] = _summary(
                    [r[metric] for r in rows],
                    mean_exists=mean_exists if quadratic else True,
                    variance_exists=risk_variance_exists
                    if quadratic
                    else lam == 1 or t > p + 1
                    if linear
                    else True,
                )
            summaries[name] = {
                "available": len(rows),
                "unavailable": repetitions - len(rows),
                "metrics": metrics,
                "comparison_to_equal_weight": _summary(
                    [r["population_variance"] - population["equal_variance"] for r in rows],
                    mean_exists=mean_exists,
                    variance_exists=risk_variance_exists,
                ),
                "selection_caveat": "means describe successful fits only if any numerical failures occur",
            }
            for metric, key, point_mass in (
                ("population_risk_ratio", "population_risk_ratio_quantiles", lam == 1),
                ("forecast_risk_ratio", "sample_gmv_forecast_ratio_quantiles", False),
            ):
                thresholds = reference[name][key]
                summaries[name][metric + "_reference_exceedances"] = (
                    {
                        q: _exceedance(
                            [r[metric] for r in rows],
                            threshold,
                            0.0 if point_mass else 1 - float(q),
                        )
                        for q, threshold in thresholds.items()
                    }
                    if thresholds
                    else None
                )
        scenarios.append(
            {
                "family": family,
                "true_covariance": covariance.tolist(),
                "oracle_weights": population["oracle_weights"].tolist(),
                "oracle_variance": population["oracle_variance"],
                "equal_weight_variance": population["equal_variance"],
                "reference": reference,
                "input_sha256": digest.hexdigest(),
                "summary": summaries,
                "records": records,
            }
        )
    return {
        "schema_version": 1,
        "lab": "covariance_estimation",
        "synthetic_only": True,
        "config": {
            "repetitions": repetitions,
            "n_assets": p,
            "train_size": t,
            "seed": seed,
            "families": list(families),
            "shrinkages": list(shrinkages),
        },
        "protocol": {
            "sample_covariance": "centered, denominator T-1",
            "constraint": "sum(weights)=1; shorting and leverage allowed",
            "weight_blend": "(1-lambda)*sample_GMV + lambda*equal_weight; not covariance shrinkage",
            "population_truth": "risk scoring and reference only; not passed to sample estimator",
        },
        "caveats": [
            "Known Gaussian population covariance, not market evidence or native allocator validation.",
            "No execution, transaction costs, financing, leverage limit or return forecast is modeled.",
            "In-sample forecast can decrease while population risk increases.",
            "At T=p+1 population risk has infinite mean for lambda<1; quantiles are descriptive.",
            "At T=p+2 or p+3 risk mean exists but variance is infinite; conventional MCSE is suppressed.",
            "Equal weight is optimal under isotropy, not under every covariance matrix.",
            "Failed fits are retained; successful-only means do not establish unconditional exact-law agreement.",
        ],
        "scenarios": scenarios,
    }


def _provenance():
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
        "numpy": np.__version__,
        "scipy": scipy.__version__ if scipy is not None else None,
        "python": platform.python_version(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        ("repetitions", 2000),
        ("n-assets", 8),
        ("train-size", 20),
        ("seed", 137),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--shrinkages", nargs="+", type=float, default=list(SHRINKAGES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    try:
        report = run_lab(**{k: v for k, v in vars(args).items() if k != "out"})
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    report["provenance"] = _provenance()
    report["performance"] = {"wall_seconds": time.perf_counter() - start}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
