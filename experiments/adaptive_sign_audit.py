"""Exact sign diagnostics after adaptive selection; synthetic evidence only.

Validation p-values after search are deliberately contaminated. A separately
simulated audit tests sign balance, not expected return or investment skill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import adaptive_holdout_lab as adaptive  # noqa: E402
from experiments import validation_lab as base  # noqa: E402

CALIBRATED_NULLS = ("iid_null", "correlated_null", "duplicated_null", "heavy_tailed_null")


def _alpha(alpha: Any) -> float:
    if isinstance(alpha, (bool, np.bool_)) or np.iscomplexobj(alpha) or not np.isscalar(alpha):
        raise TypeError("alpha must be a real probability")
    alpha = float(alpha)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    return alpha


def sign_test(returns: Any, *, alpha: float = 0.05) -> dict[str, Any]:
    """One-sided sign test, H0: Pr(return > 0 | return != 0) <= 1/2.

    Zero signs are dropped. Calibration requires independent fair nonzero signs
    under the boundary null. Integer arithmetic computes the tail and compares
    it with the exact binary value of alpha; p_value is a float presentation.
    """
    alpha = _alpha(alpha)
    values = np.asarray(returns)
    if values.ndim != 1 or values.dtype.kind not in "iuf" or not len(values):
        raise ValueError("returns must be a nonempty one-dimensional real array")
    if not np.isfinite(values).all():
        raise ValueError("returns must be finite")
    positive, negative = int(np.sum(values > 0)), int(np.sum(values < 0))
    n = positive + negative
    out = {
        "positive": positive,
        "negative": negative,
        "ties": len(values) - n,
        "n_nonzero": n,
        "p_value": None,
        "rejected": None,
        "tail_integer": None,
        "denominator_power_of_two": n,
    }
    if n:
        tail = sum(math.comb(n, k) for k in range(positive, n + 1))
        alpha_num, alpha_den = alpha.as_integer_ratio()
        out.update(
            p_value=tail / (1 << n),
            rejected=tail * alpha_den <= alpha_num * (1 << n),
            tail_integer=str(tail),
        )
    return out


def attainable_boundary(n: int, alpha: float) -> dict[str, Any]:
    """Exact iid no-ties null rejection size at the configured sample length."""
    n = base._integer(n, "n", 1)
    alpha = _alpha(alpha)
    numerator, denominator = alpha.as_integer_ratio()
    tail = 0
    accepted_tail = 0
    critical = None
    for positives in range(n, -1, -1):
        tail += math.comb(n, positives)
        if tail * denominator <= numerator * (1 << n):
            critical, accepted_tail = positives, tail
        else:
            break
    return {
        "n_nonzero": n,
        "critical_positive_count": critical,
        "minimum_p_value": 1 / (1 << n),
        "exact_null_rejection_rate": accepted_tail / (1 << n),
        "null_rejection_tail_integer": str(accepted_tail),
        "denominator_power_of_two": n,
        "assumption": "iid fair signs, no ties; actual per-record nonzero n is retained",
    }


def _mixture(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """One common positive scale preserves signs and avoids simple overflow."""
    active = weights != 0
    used = values[:, active]
    scale = np.max(np.abs(used)) if used.size else 0
    return (used / scale) @ weights[active] if scale else np.zeros(len(values))


def evaluate(training: Any, validation: Any, audit: Any, *, rounds: int, alpha: float):
    """Freeze choices using training/validation before inspecting the audit matrix."""
    alpha = _alpha(alpha)
    choice = adaptive.search_validation(training, validation, rounds=rounds)
    valid, heldout = base._matrix(validation), base._matrix(audit)
    if heldout.shape[1] != len(choice["training_order"]):
        raise ValueError("audit must have the same candidate columns")
    final = choice["snapshots"][-1]
    weights = np.asarray(final["adaptive_weights"])
    frozen, single = choice["training_order"][0], final["single_candidate"]
    return {
        "training_order": choice["training_order"],
        "selected_weights": final["adaptive_weights"],
        "frozen_candidate": frozen,
        "single_candidate": single,
        "undefined_proposals_cumulative": final["undefined_proposals_cumulative"],
        "tests": {
            "adaptive_validation": sign_test(_mixture(valid, weights), alpha=alpha),
            "adaptive_audit": sign_test(_mixture(heldout, weights), alpha=alpha),
            "frozen_validation": sign_test(valid[:, frozen], alpha=alpha),
            "frozen_audit": sign_test(heldout[:, frozen], alpha=alpha),
            "single_validation": sign_test(valid[:, single], alpha=alpha),
            "single_audit": sign_test(heldout[:, single], alpha=alpha),
        },
    }


def _wilson(successes: int, count: int) -> list[float] | None:
    if count == 0:
        return None
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [
        0.0 if successes == 0 else max(0.0, center - half),
        1.0 if successes == count else min(1.0, center + half),
    ]


def _rate(records: list[dict], metric: str) -> dict[str, Any]:
    values = [record["tests"][metric]["rejected"] for record in records]
    available = [value for value in values if value is not None]
    successes = sum(available)
    return {
        "available": len(available),
        "unavailable": len(values) - len(available),
        "rejections": successes,
        "rejection_rate": successes / len(available) if available else None,
        "wilson_95_interval": _wilson(successes, len(available)),
    }


def _paired(records: list[dict], left: str, right: str) -> dict[str, Any]:
    values = [
        int(r["tests"][left]["rejected"]) - int(r["tests"][right]["rejected"])
        for r in records
        if r["tests"][left]["rejected"] is not None and r["tests"][right]["rejected"] is not None
    ]
    return {
        "available_pairs": len(values),
        "unavailable_pairs": len(records) - len(values),
        "mean": float(np.mean(values)) if values else None,
        "monte_carlo_standard_error": float(np.std(values, ddof=1) / math.sqrt(len(values)))
        if len(values) > 1
        else None,
    }


def run_lab(
    *,
    repetitions: int = 1000,
    sample_size: int = 252,
    n_candidates: int = 32,
    rounds: int = 16,
    seed: int = 71,
    alpha: float = 0.05,
    families: Any = base.FAMILIES,
) -> dict[str, Any]:
    repetitions = base._integer(repetitions, "repetitions", 2)
    sample_size = base._integer(sample_size, "sample_size", 4)
    n_candidates = base._integer(n_candidates, "n_candidates", 1)
    rounds = base._integer(rounds, "rounds", 0)
    seed = base._integer(seed, "seed", 0)
    alpha = _alpha(alpha)
    if rounds >= n_candidates:
        raise ValueError("rounds must be smaller than n_candidates")
    families = tuple(families)
    if (
        not families
        or len(set(families)) != len(families)
        or any(f not in base.FAMILIES for f in families)
    ):
        raise ValueError("families must be a nonempty distinct selection of known families")
    scenarios = []
    for family in families:
        records, digest = [], hashlib.sha256()
        for repetition in range(repetitions):
            segments = []
            for segment in range(3):
                values = base.simulate_family(
                    family,
                    sample_size,
                    n_candidates,
                    seed=np.random.SeedSequence(
                        [seed, base.FAMILIES.index(family), repetition, segment, 8401]
                    ),
                )
                digest.update(np.asarray(values.shape, dtype="<u8").tobytes())
                digest.update(np.asarray(values, dtype="<f8").tobytes())
                segments.append(values)
            record = evaluate(*segments, rounds=rounds, alpha=alpha)
            record["replication"] = repetition
            records.append(record)
        scenarios.append(
            {
                "family": family,
                "input_sha256": digest.hexdigest(),
                "interpretation": "iid_symmetric_null"
                if family in CALIBRATED_NULLS
                else "serial_dependence_violates_sign_test"
                if family == "serial_null"
                else "positive_drift_power",
                "rates": {name: _rate(records, name) for name in records[0]["tests"]},
                "paired_rejection_differences": {
                    "adaptive_validation_minus_audit": _paired(
                        records, "adaptive_validation", "adaptive_audit"
                    ),
                    "adaptive_minus_frozen_audit": _paired(
                        records, "adaptive_audit", "frozen_audit"
                    ),
                    "adaptive_minus_single_audit": _paired(
                        records, "adaptive_audit", "single_audit"
                    ),
                },
                "records": records,
            }
        )
    return {
        "schema_version": 1,
        "lab": "adaptive_sign_audit",
        "synthetic_only": True,
        "config": {
            "repetitions": repetitions,
            "sample_size": sample_size,
            "n_candidates": n_candidates,
            "rounds": rounds,
            "seed": seed,
            "alpha": alpha,
            "families": list(families),
        },
        "protocol": {
            "test": "one-sided sign test for Pr(return > 0 | return != 0) > 1/2",
            "attainable_no_ties_boundary": attainable_boundary(sample_size, alpha),
            "rejection": "exact binomial tail <= exact binary alpha",
            "ties": "zero signs dropped; all ties unavailable",
            "selection": "final round of fixed adaptive Sharpe search; sign tests do not select weights",
            "segments": "independent simulated training, validation and audit using adaptive lab streams",
        },
        "caveats": [
            "Adaptive and single-validation p-values are post-selection and not calibrated.",
            "Independent audit calibration requires iid fair nonzero signs; zero mean alone is insufficient.",
            "The serial control deliberately violates within-audit independence.",
            "This tests sign balance, not expected return, Sharpe, economic value or profitability.",
            "Exact binomial tails are discrete and can be conservative; float p_value is a presentation only.",
            "Wilson intervals describe replication-level rejection rates, not simultaneous guarantees.",
            "Unavailable rates condition on available outcomes; all original records are retained.",
        ],
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        ("repetitions", 1000),
        ("sample-size", 252),
        ("n-candidates", 32),
        ("rounds", 16),
        ("seed", 71),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--families", nargs="+", choices=base.FAMILIES, default=list(base.FAMILIES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    try:
        report = run_lab(**{k: v for k, v in vars(args).items() if k != "out"})
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "unavailable"
    report["provenance"] = {
        "git_sha": sha,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_sha256": {
            Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (adaptive, base)
        },
    }
    report["performance"] = {"wall_seconds": time.perf_counter() - start}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
