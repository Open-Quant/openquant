"""Independent-unit sign-test lab; grouping dependent rows is not independence.

Gaussian controls isolate when block sums have iid signs and when they do not.
No real market data, model selection, or production API changes.
"""

from __future__ import annotations

import argparse
import functools
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
from experiments import adaptive_sign_audit as audit  # noqa: E402

FAMILIES = ("independent_blocks", "continuous_path", "shared_factor_blocks")
SHARED_CORRELATION = 0.5


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _real(value: Any, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or np.iscomplexobj(value)
        or not isinstance(value, (int, float, np.integer, np.floating))
    ):
        raise TypeError(f"{name} must be real")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def block_sum_variance(block_size: int, phi: float) -> float:
    """Var(sum X_t) for stationary unit-variance AR(1), covariance phi**abs(t-s)."""
    m = _integer(block_size, "block_size", 1)
    phi = _real(phi, "phi")
    if not 0 <= phi < 1:
        raise ValueError("this experiment requires 0 <= phi < 1")
    return m + 2 * math.fsum((m - lag) * phi**lag for lag in range(1, m))


def independent_block_reference(
    n_blocks: int, block_size: int, phi: float, drift: float, alpha: float
):
    """Exact model formula, evaluated in floating point; not an empirical calibration.

    Each block sum ~ N(m*drift, variance), giving q=P(sum>0). Independent
    blocks make their positive count Binomial(n_blocks,q). Gaussian continuity
    means zero ties have probability zero in this reference model.
    """
    n = _integer(n_blocks, "n_blocks", 1)
    variance = block_sum_variance(block_size, phi)
    drift = _real(drift, "drift")
    boundary = audit.attainable_boundary(n, alpha)
    q = 0.5 * math.erfc(-block_size * drift / math.sqrt(2 * variance))
    critical = boundary["critical_positive_count"]
    if critical is None:
        power = 0.0
    elif q == 1:
        power = 1.0
    elif q == 0:
        power = 0.0
    elif q == 0.5:
        power = boundary["exact_null_rejection_rate"]
    else:
        # log combinatorics avoids overflowing an intermediate coefficient.
        power = math.fsum(
            math.exp(
                math.lgamma(n + 1)
                - math.lgamma(k + 1)
                - math.lgamma(n - k + 1)
                + k * math.log(q)
                + (n - k) * math.log1p(-q)
            )
            for k in range(critical, n + 1)
        )
        power = min(1.0, max(0.0, power))
    return {
        "block_sum_variance": variance,
        "block_positive_probability": q,
        "critical_positive_count": critical,
        "rejection_probability": power,
        "arithmetic": "closed-form Gaussian/binomial model, floating-point evaluation",
    }


def _ar_rows(rng: np.random.Generator, n_rows: int, size: int, phi: float) -> np.ndarray:
    values = rng.normal(size=(n_rows, size))
    innovation_sd = math.sqrt(1 - phi * phi)
    for t in range(1, size):
        values[:, t] = phi * values[:, t - 1] + innovation_sd * values[:, t]
    return values


def generate(family: str, *, n_blocks: int, block_size: int, phi: float, drift: float, seed: Any):
    if family not in FAMILIES:
        raise ValueError("unknown family")
    n_blocks = _integer(n_blocks, "n_blocks", 1)
    block_size = _integer(block_size, "block_size", 1)
    block_sum_variance(block_size, phi)
    phi, drift = float(phi), _real(drift, "drift")
    rng = np.random.default_rng(seed)
    if family == "continuous_path":
        values = _ar_rows(rng, 1, n_blocks * block_size, phi).reshape(n_blocks, block_size)
    else:
        values = _ar_rows(rng, n_blocks, block_size, phi)
        if family == "shared_factor_blocks":
            common = _ar_rows(rng, 1, block_size, phi)
            values = (
                math.sqrt(1 - SHARED_CORRELATION) * values + math.sqrt(SHARED_CORRELATION) * common
            )
    return values + drift


@functools.lru_cache(maxsize=16384)
def _cached_sign_test(positive: int, negative: int, ties: int, alpha: float):
    # Sign-test sufficient statistics avoid repeating expensive identical exact tails.
    values = np.concatenate((np.ones(positive), -np.ones(negative), np.zeros(ties)))
    return audit.sign_test(values, alpha=alpha)


def evaluate(values: Any, *, alpha: float):
    alpha = audit._alpha(alpha)
    array = np.asarray(values)
    if (
        array.ndim != 2
        or not array.size
        or array.dtype.kind not in "iuf"
        or not np.isfinite(array).all()
    ):
        raise ValueError("values must be a finite nonempty real matrix (blocks, observations)")
    array = np.asarray(array, dtype=float)
    with np.errstate(over="raise", invalid="raise"):
        try:
            sums = array.sum(axis=1)
        except FloatingPointError as exc:
            raise ValueError("block sums are not representable") from exc
    out = {}
    for name, sample in (("raw_observations", array.ravel()), ("block_sums", sums)):
        positive, negative = int(np.sum(sample > 0)), int(np.sum(sample < 0))
        out[name] = dict(
            _cached_sign_test(positive, negative, sample.size - positive - negative, float(alpha))
        )
    return {"tests": out, "block_sums": sums.tolist()}


def _stream(
    seed: int,
    family: str,
    n_blocks: int,
    block_size: int,
    phi: float,
    drift: float,
    repetition: int,
):
    # Hash exact parameter representation; invocation/grid ordering cannot alter paths.
    key = json.dumps([family, n_blocks, block_size, phi.hex(), drift.hex()], separators=(",", ":"))
    words = (
        np.frombuffer(hashlib.sha256(key.encode()).digest(), dtype="<u4").astype(np.uint32).tolist()
    )
    return np.random.SeedSequence([seed, repetition, 3911, *words])


def run_lab(
    *,
    repetitions: int = 1000,
    n_blocks: int = 16,
    block_size: int = 16,
    phi: float = 0.6,
    drift: float = 0.0,
    alpha: float = 0.05,
    seed: int = 113,
    families: Any = FAMILIES,
):
    repetitions = _integer(repetitions, "repetitions", 2)
    n_blocks = _integer(n_blocks, "n_blocks", 1)
    block_size = _integer(block_size, "block_size", 1)
    seed = _integer(seed, "seed", 0)
    phi, drift = _real(phi, "phi"), _real(drift, "drift")
    # This lab studies null and positive fixed shifts; no posthoc shift choice.
    if drift < 0:
        raise ValueError("drift must be nonnegative")
    variance = block_sum_variance(block_size, phi)
    alpha = audit._alpha(alpha)
    families = tuple(families)
    if (
        not families
        or len(set(families)) != len(families)
        or any(f not in FAMILIES for f in families)
    ):
        raise ValueError("families must be distinct known families")
    reference = independent_block_reference(n_blocks, block_size, phi, drift, alpha)
    scenarios = []
    for family in families:
        records, digest = [], hashlib.sha256()
        for replication in range(repetitions):
            values = generate(
                family,
                n_blocks=n_blocks,
                block_size=block_size,
                phi=phi,
                drift=drift,
                seed=_stream(seed, family, n_blocks, block_size, phi, drift, replication),
            )
            digest.update(np.asarray(values.shape, dtype="<u8").tobytes())
            digest.update(np.asarray(values, dtype="<f8").tobytes())
            record = evaluate(values, alpha=alpha)
            record["replication"] = replication
            records.append(record)
        rates = {name: audit._rate(records, name) for name in records[0]["tests"]}
        for rate in rates.values():
            n, p = rate["available"], rate["rejection_rate"]
            rate["monte_carlo_standard_error"] = math.sqrt(p * (1 - p) / (n - 1)) if n > 1 else None
        exact_valid = (
            n_blocks == 1
            or family == "independent_blocks"
            or (family == "continuous_path" and phi == 0)
        )
        scenarios.append(
            {
                "family": family,
                "input_sha256": digest.hexdigest(),
                "rates": rates,
                "independent_block_reference_valid": exact_valid,
                "exact_block_rejection_probability": reference["rejection_probability"]
                if exact_valid
                else None,
                "raw_iid_sign_assumption_valid": (
                    (family == "independent_blocks" and (phi == 0 or block_size == 1))
                    or (family == "continuous_path" and (phi == 0 or n_blocks * block_size == 1))
                    or (
                        family == "shared_factor_blocks"
                        and n_blocks == 1
                        and (phi == 0 or block_size == 1)
                    )
                ),
                "paired_raw_minus_block": audit._paired(records, "raw_observations", "block_sums"),
                "records": records,
            }
        )
    return {
        "schema_version": 1,
        "lab": "cluster_sign",
        "synthetic_only": True,
        "config": {
            "repetitions": repetitions,
            "n_blocks": n_blocks,
            "block_size": block_size,
            "phi": phi,
            "drift": drift,
            "alpha": alpha,
            "seed": seed,
            "families": list(families),
        },
        "protocol": {
            "marginal_variance": 1.0,
            "within_block_sum_variance": variance,
            "shared_factor_cross_block_sum_correlation": SHARED_CORRELATION,
            "test_target": "Pr(unit > 0 | unit != 0) > 1/2; unit is raw observation or block sum",
            "independent_block_reference": reference,
            "block_boundary": audit.attainable_boundary(n_blocks, alpha),
            "raw_boundary_if_iid": audit.attainable_boundary(n_blocks * block_size, alpha),
        },
        "caveats": [
            "Grouping rows does not establish independence; shared factors survive resets of idiosyncratic states.",
            "Continuous-path block sums remain dependent for phi>0; shared-factor block sums have correlation 0.5.",
            "Exact sign calibration requires independent fair nonzero signs, not merely zero mean.",
            "Gaussian symmetry and finite moments link positive mean to positive sign probability only in this model.",
            "Raw and block-sum tests have different experimental units and generally different estimands.",
            "Wilson intervals/MCSE describe independent simulation replications, not simultaneous inference.",
            "All-zero units are unavailable; a too-small nonzero count can make rejection unattainable.",
            "No adaptive selection, investment claim, market data or production API validation is involved.",
        ],
        "scenarios": scenarios,
    }


def _provenance():
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "unavailable"
    return {
        "git_sha": sha,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_sha256": {
            Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (audit, audit.adaptive, audit.base)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        ("repetitions", 1000),
        ("n-blocks", 16),
        ("block-size", 16),
        ("seed", 113),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("phi", 0.6), ("drift", 0.0), ("alpha", 0.05)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    try:
        report = run_lab(**{k: v for k, v in vars(args).items() if k != "out"})
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    report["provenance"] = _provenance()
    report["performance"] = {"wall_seconds": time.perf_counter() - start}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
