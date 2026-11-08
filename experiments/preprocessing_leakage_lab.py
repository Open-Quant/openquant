"""Synthetic PCA information-boundary audit; no production APIs or investment claims.

Compare a training-only frozen PCA transform with a deliberately transductive PCA
fit that sees future feature rows (never future labels). See PREPROCESSING_LEAKAGE_LAB.md.
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

FAMILIES = (
    "stationary_signal",
    "stationary_null",
    "signal_variance_rises",
    "nuisance_variance_rises",
)


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _array(value: Any, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or not array.size or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty real {ndim}-dimensional array")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def fit_pca_ridge(
    train_x: Any,
    train_y: Any,
    *,
    n_components: int = 1,
    ridge: float = 1.0,
    preprocessing_x: Any = None,
) -> dict[str, Any]:
    """Fit PCA and ridge with an unpenalized intercept; return original-space model.

    Default PCA sees train_x only. preprocessing_x deliberately allows a different
    PCA fitting population for the contaminated experimental comparator. Its rows
    are features, never labels. No feature standardization or hyperparameter tuning.
    Ridge minimizes sum of squared residuals + ridge * squared coefficient norm.
    """
    train_x = _array(train_x, "train_x", 2)
    train_y = _array(train_y, "train_y", 1)
    if len(train_x) != len(train_y) or len(train_x) < 3:
        raise ValueError("training data need at least three aligned rows")
    fit_x = train_x if preprocessing_x is None else _array(preprocessing_x, "preprocessing_x", 2)
    if fit_x.shape[1] != train_x.shape[1] or len(fit_x) < 3:
        raise ValueError("PCA data need matching feature width and at least three rows")
    n_components = _integer(n_components, "n_components", 1)
    if n_components > min(fit_x.shape[1], len(fit_x) - 1):
        raise ValueError("n_components exceeds centered PCA dimensions")
    if isinstance(ridge, bool) or not np.isscalar(ridge):
        raise TypeError("ridge must be a positive finite scalar")
    ridge = float(ridge)
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be positive and finite")
    # Finite inputs can still overflow arithmetic; fail closed instead of emitting NaN.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            pca_mean = fit_x.mean(axis=0)
            _, singular_values, vt = np.linalg.svd(fit_x - pca_mean, full_matrices=False)
            components = vt[:n_components]
            z = (train_x - pca_mean) @ components.T
            z_mean = z.mean(axis=0)
            y_mean = float(train_y.mean())
            centered_z = z - z_mean
            weights = np.linalg.solve(
                centered_z.T @ centered_z + ridge * np.eye(n_components),
                centered_z.T @ (train_y - y_mean),
            )
            coefficient = components.T @ weights
            intercept = y_mean - float((pca_mean @ components.T + z_mean) @ weights)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise ValueError("PCA/ridge arithmetic is not numerically representable") from exc
    if not np.isfinite(coefficient).all() or not math.isfinite(intercept):
        raise ValueError("PCA/ridge produced nonfinite parameters")
    return {
        "coefficient": coefficient,
        "intercept": intercept,
        "pca_mean": pca_mean,
        "components": components,
        "singular_values": singular_values,
        "pca_fit_rows": len(fit_x),
        "training_rows": len(train_x),
    }


def predict(model: dict[str, Any], features: Any) -> np.ndarray:
    features = _array(features, "features", 2)
    if features.shape[1] != len(model["coefficient"]):
        raise ValueError("prediction features have wrong width")
    with np.errstate(over="raise", invalid="raise"):
        try:
            out = features @ model["coefficient"] + model["intercept"]
        except FloatingPointError as exc:
            raise ValueError("prediction arithmetic is not representable") from exc
    if not np.isfinite(out).all():
        raise ValueError("prediction produced nonfinite values")
    return out


def generate(family: str, train_size: int, holdout_size: int, n_features: int, seed: Any):
    """Independent point-observation rows; Gaussian covariance shifts, not market data."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    train_size = _integer(train_size, "train_size", 4)
    holdout_size = _integer(holdout_size, "holdout_size", 4)
    n_features = _integer(n_features, "n_features", 2)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(train_size + holdout_size, n_features)) * 0.5
    train_sd = [2.0, 1.0]
    test_sd = [2.0, 1.0]
    if family == "signal_variance_rises":
        train_sd, test_sd = [1.0, 2.0], [4.0, 2.0]
    elif family == "nuisance_variance_rises":
        test_sd = [2.0, 4.0]
    x[:train_size, :2] *= np.asarray(train_sd) / 0.5
    x[train_size:, :2] *= np.asarray(test_sd) / 0.5
    y = rng.normal(scale=0.5, size=len(x))
    if family != "stationary_null":
        y += x[:, 0]
    return (
        x,
        y,
        {
            "train_first_two_std": train_sd,
            "holdout_first_two_std": test_sd,
            "other_feature_std": 0.5,
            "label_noise_std": 0.5,
            "label_signal": "none" if family == "stationary_null" else "feature_0",
            "rows_independent": True,
            "label_span": "one point per row; no train/holdout label overlap",
        },
    )


def _summary(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "monte_carlo_standard_error": float(a.std(ddof=1) / math.sqrt(len(a))),
    }


def run_lab(
    *,
    repetitions: int = 200,
    train_size: int = 128,
    holdout_size: int = 128,
    n_features: int = 6,
    ridge: float = 1.0,
    seed: int = 19,
    families: Any = FAMILIES,
) -> dict[str, Any]:
    """Paired one-component comparison plus all-component rotation-invariance control."""
    repetitions = _integer(repetitions, "repetitions", 2)
    train_size = _integer(train_size, "train_size", 4)
    holdout_size = _integer(holdout_size, "holdout_size", 4)
    n_features = _integer(n_features, "n_features", 2)
    if n_features >= train_size:
        raise ValueError("n_features must be smaller than train_size for full-component control")
    seed = _integer(seed, "seed", 0)
    families = tuple(families)
    if (
        not families
        or len(set(families)) != len(families)
        or any(f not in FAMILIES for f in families)
    ):
        raise ValueError("families must be distinct known families")
    scenarios = []
    for family in families:
        digest = hashlib.sha256()
        records = []
        for replication in range(repetitions):
            x, y, parameters = generate(
                family,
                train_size,
                holdout_size,
                n_features,
                np.random.SeedSequence([seed, FAMILIES.index(family), replication]),
            )
            for array in (x, y):
                digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
                digest.update(np.asarray(array, dtype="<f8").tobytes(order="C"))
            tx, ty, hx, hy = x[:train_size], y[:train_size], x[train_size:], y[train_size:]
            record = {"replication": replication}
            for count, prefix in ((1, "one_component"), (n_features, "all_components")):
                honest = fit_pca_ridge(tx, ty, n_components=count, ridge=ridge)
                global_fit = fit_pca_ridge(
                    tx, ty, n_components=count, ridge=ridge, preprocessing_x=x
                )
                honest_pred, global_pred = predict(honest, hx), predict(global_fit, hx)
                honest_mse = float(np.mean((hy - honest_pred) ** 2))
                global_mse = float(np.mean((hy - global_pred) ** 2))
                record[prefix] = {
                    "training_only_mse": honest_mse,
                    "future_features_mse": global_mse,
                    "future_minus_training_mse": global_mse - honest_mse,
                    "max_prediction_difference": float(np.max(np.abs(global_pred - honest_pred))),
                    "training_only_coefficient": honest["coefficient"].tolist(),
                    "future_features_coefficient": global_fit["coefficient"].tolist(),
                    "training_only_intercept": honest["intercept"],
                    "future_features_intercept": global_fit["intercept"],
                }
            record["training_mean_baseline_mse"] = float(np.mean((hy - ty.mean()) ** 2))
            records.append(record)
        summary = {
            "training_mean_baseline_mse": _summary(
                [r["training_mean_baseline_mse"] for r in records]
            )
        }
        for prefix in ("one_component", "all_components"):
            summary[prefix] = {
                metric: _summary([r[prefix][metric] for r in records])
                for metric in (
                    "training_only_mse",
                    "future_features_mse",
                    "future_minus_training_mse",
                )
            }
            summary[prefix]["future_features_better_fraction"] = (
                sum(r[prefix]["future_minus_training_mse"] < 0 for r in records) / repetitions
            )
            summary[prefix]["max_prediction_difference"] = max(
                r[prefix]["max_prediction_difference"] for r in records
            )
        scenarios.append(
            {
                "family": family,
                "generator_parameters": parameters,
                "dataset_sha256": digest.hexdigest(),
                "summary": summary,
                "replications": records,
            }
        )
    return {
        "schema_version": 1,
        "lab": "preprocessing_leakage",
        "synthetic_only": True,
        "config": {
            "repetitions": repetitions,
            "train_size": train_size,
            "holdout_size": holdout_size,
            "n_features": n_features,
            "ridge": float(ridge),
            "seed": seed,
            "families": list(families),
        },
        "protocol": {
            "train_range": [0, train_size],
            "holdout_range": [train_size, train_size + holdout_size],
            "training_only_pca_range": [0, train_size],
            "future_features_pca_range": [0, train_size + holdout_size],
            "ridge_label_range_both_methods": [0, train_size],
            "hyperparameters": "fixed before simulation; no holdout tuning",
            "loss": "mean squared prediction error; lower is better",
            "paired_difference": "future_features_mse minus training_only_mse; negative favors future access",
        },
        "caveats": [
            "Global PCA uses future covariates, not future labels; this violates a frozen deployment boundary.",
            "A declared transductive problem has a different information contract.",
            "Synthetic point labels do not overlap; this does not execute OpenQuant purging or PCA APIs.",
            "Future access may improve or worsen prediction; no universal inflation claim.",
            "MCSE describes independent simulated replications, not real strategy profitability.",
            "Full-component isotropic ridge is rotation invariant; tiny differences are floating-point noise.",
        ],
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        ("repetitions", 200),
        ("train-size", 128),
        ("holdout-size", 128),
        ("n-features", 6),
        ("seed", 19),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    try:
        report = run_lab(**{k: v for k, v in vars(args).items() if k != "out"})
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    source = Path(__file__).resolve()
    try:
        git_sha = subprocess.check_output(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unavailable"
    report["provenance"] = {
        "git_sha": git_sha,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    report["performance"] = {"wall_seconds": time.perf_counter() - start}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Synthetic preprocessing audit: {args.out}")


if __name__ == "__main__":
    main()
