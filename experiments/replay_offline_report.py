"""Replay trusted local synthetic lab reports; this is not an authenticity check.

Only fixed, locally installed lab modules are executed. A report cannot name code,
commands or import paths. Exact comparisons deliberately fail across floating-point
environments when results differ; failure is not itself evidence of fraud or a bug.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np

LABS = {
    "selection_validation": "validation_lab.py",
    "adaptive_holdout": "adaptive_holdout_lab.py",
    "preprocessing_leakage": "preprocessing_leakage_lab.py",
    "adaptive_sign_audit": "adaptive_sign_audit.py",
    "bounded_finite_family": "bounded_finite_family_lab.py",
    "cluster_sign": "cluster_sign_lab.py",
    "selection_interval": "selection_interval_lab.py",
    "selection_budget": "selection_budget_lab.py",
    "sequential_sign": "sequential_sign_lab.py",
    "multiple_sequential": "multiple_sequential_lab.py",
    "payoff_target": "payoff_target_lab.py",
    "mean_robustness": "mean_robustness_lab.py",
    "delayed_label": "delayed_label_lab.py",
    "constraint_solver": "constraint_solver_lab.py",
    "zero_fit_probability": "zero_fit_probability_lab.py",
    "partial_identification": "partial_identification_lab.py",
    "point_in_time": "point_in_time_lab.py",
    "informative_missingness": "informative_missingness_lab.py",
    "covariance_robustness": "covariance_robustness_lab.py",
    "rare_event_uncertainty": "rare_event_uncertainty_lab.py",
    "covariance_estimation": "covariance_estimation_lab.py",
}
DEPENDENCIES = {
    "selection_validation": (),
    "adaptive_holdout": ("validation_lab.py",),
    "preprocessing_leakage": (),
    "adaptive_sign_audit": ("adaptive_holdout_lab.py", "validation_lab.py"),
    "bounded_finite_family": (),
    "cluster_sign": ("adaptive_sign_audit.py", "adaptive_holdout_lab.py", "validation_lab.py"),
    "selection_interval": (),
    "selection_budget": ("selection_interval_lab.py",),
    "sequential_sign": (),
    "multiple_sequential": ("sequential_sign_lab.py",),
    "payoff_target": ("sequential_sign_lab.py",),
    "mean_robustness": (),
    "delayed_label": (),
    "constraint_solver": ("covariance_robustness_lab.py",),
    "zero_fit_probability": (),
    "partial_identification": ("selection_interval_lab.py",),
    "point_in_time": (),
    "informative_missingness": (),
    "covariance_robustness": (),
    "rare_event_uncertainty": ("selection_interval_lab.py",),
    "covariance_estimation": (),
}
OPTIONAL_RUNTIME_PACKAGES = {"covariance_estimation": ("scipy",)}
EXTRA_RUNTIME_PACKAGES = {
    "selection_interval": ("scipy",),
    "selection_budget": ("scipy",),
    "covariance_robustness": ("scipy",),
    "constraint_solver": ("scipy",),
    "zero_fit_probability": ("scipy",),
    "partial_identification": ("scipy",),
    "rare_event_uncertainty": ("scipy",),
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def read_report(path: Path) -> dict[str, Any]:
    """Read strict JSON, transparently decompressing gzip identified by its header.

    The full decompressed document is read into memory, as for plain JSON.
    """
    with path.open("rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
        stream.seek(0)
        if compressed:
            try:
                with gzip.GzipFile(fileobj=stream, mode="rb") as decoded:
                    raw = decoded.read()
            except (OSError, EOFError, zlib.error) as exc:
                raise ValueError("invalid or truncated gzip report") from exc
        else:
            raw = stream.read()
    return parse_report_bytes(raw)


def parse_report_bytes(raw: bytes) -> dict[str, Any]:
    """Parse one strict UTF-8 JSON report, without checking scientific claims."""
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("report must be a JSON object")

    def finite_tree(node: Any) -> None:
        if isinstance(node, float) and not math.isfinite(node):
            raise ValueError("nonfinite JSON number")
        if isinstance(node, dict):
            for child in node.values():
                finite_tree(child)
        elif isinstance(node, list):
            for child in node:
                finite_tree(child)

    finite_tree(value)
    return value


def first_difference(expected: Any, observed: Any, path: str = "$") -> str | None:
    """Locate the first exact structural/numeric mismatch, without dumping input data."""
    if isinstance(expected, dict) and isinstance(observed, dict):
        if expected.keys() != observed.keys():
            return f"{path}: object keys differ"
        for key in sorted(expected):
            difference = first_difference(expected[key], observed[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(expected, list) and isinstance(observed, list):
        if len(expected) != len(observed):
            return f"{path}: list lengths differ"
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            difference = first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if type(expected) is not type(observed):
        return f"{path}: value types differ"
    if isinstance(expected, float) and not (math.isfinite(expected) and math.isfinite(observed)):
        return f"{path}: nonfinite numeric value"
    if expected != observed:
        return f"{path}: values differ"
    return None


def _load_lab(name: str) -> Any:
    if name not in LABS:
        raise ValueError(f"unknown lab: {name}")
    path = Path(__file__).resolve().parent / LABS[name]
    spec = importlib.util.spec_from_file_location(f"_openquant_replay_{name}", path)
    if spec is None or spec.loader is None:
        raise ValueError("lab module unavailable")
    module = importlib.util.module_from_spec(spec)
    # Dataclass definitions and ordinary sibling imports require standard import state.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _report_identity(report: dict[str, Any], *, lab: str):
    """Check the report identity without importing or executing a lab."""
    if lab not in LABS:
        raise ValueError(f"unknown lab: {lab}")
    if report.get("lab", lab) != lab:
        raise ValueError("report lab identifier differs from requested lab")
    if (
        type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
        or report.get("synthetic_only") is not True
    ):
        raise ValueError("expected a schema-1 synthetic report")
    config = report.get("config")
    provenance = report.get("provenance")
    if not isinstance(config, dict) or not isinstance(provenance, dict):
        raise ValueError("report needs config and provenance objects")
    if provenance.get("numpy") != np.__version__:
        raise ValueError("NumPy version mismatch; exact replay requires the recorded version")
    versions = {"numpy": np.__version__}
    optional = OPTIONAL_RUNTIME_PACKAGES.get(lab, ())
    for package in (*EXTRA_RUNTIME_PACKAGES.get(lab, ()), *optional):
        try:
            version = importlib.import_module(package).__version__
        except ImportError as exc:
            if package not in optional:
                raise ValueError(f"{package} is required to replay {lab}") from exc
            version = None
        if package not in provenance or provenance[package] != version:
            raise ValueError(f"{package} version mismatch; exact replay requires recorded versions")
        versions[package] = version
    source = Path(__file__).resolve().parent / LABS[lab]
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if provenance.get("source_sha256") != actual_hash:
        raise ValueError("lab source hash mismatch or missing source hash")
    dependencies = {
        name: hashlib.sha256((source.parent / name).read_bytes()).hexdigest()
        for name in DEPENDENCIES[lab]
    }
    if provenance.get("dependency_sha256", {}) != dependencies:
        raise ValueError("lab dependency source hash mismatch or missing dependency hashes")
    return config, versions, actual_hash


def check_report_identity(report: dict[str, Any], *, lab: str) -> dict[str, Any]:
    """Check source/runtime identity only; scientific contents remain unchecked."""
    config, versions, actual_hash = _report_identity(report, lab=lab)
    return {
        "schema_version": 1,
        "lab": lab,
        "identity_matched": True,
        "scientific_replay_performed": False,
        "source_sha256": actual_hash,
        "runtime_versions": versions,
        "config": config,
        "limitations": [
            "Scientific contents and configuration semantics have not been verified.",
            "Matching hashes are not evidence of authenticity or real-market profitability.",
        ],
    }


def replay_report(report: dict[str, Any], *, lab: str) -> dict[str, Any]:
    """Regenerate scientific results from a trusted local report's recorded config.

    Input sizes determine work: callers must review config before replaying untrusted
    or very large reports. No source supplied by the report is ever executed.
    """
    config, versions, actual_hash = _report_identity(report, lab=lab)
    module = _load_lab(lab)
    regenerated = module.run_lab(**config)
    # Roundtrip to the exact JSON representation written by the lab's CLI (tuples etc.).
    regenerated = json.loads(json.dumps(regenerated, allow_nan=False))
    ignored = {"provenance", "performance"}
    saved = {key: value for key, value in report.items() if key not in ignored}
    current = {key: value for key, value in regenerated.items() if key not in ignored}
    difference = first_difference(saved, current)

    def digest(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()

    return {
        "schema_version": 1,
        "lab": lab,
        "matched": difference is None,
        "first_difference": difference,
        "saved_scientific_sha256": digest(saved),
        "replayed_scientific_sha256": digest(current),
        "source_sha256": actual_hash,
        "numpy": np.__version__,
        "runtime_versions": versions,
        "comparison": "exact scientific output, excluding provenance and performance",
        "limitations": [
            "Replay checks reproducibility, not authenticity; reports and hashes can be rewritten.",
            "A matching synthetic result is not evidence of real-market profitability.",
            "Machine and floating-point differences can cause exact replay failure.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--lab", required=True, choices=LABS)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--check-identity",
        action="store_true",
        help="check code/runtime identity without rerunning or validating scientific results",
    )
    parser.add_argument(
        "--runtime-context",
        action="store_true",
        help="append current-process runtime diagnostics without changing identity or equality",
    )
    args = parser.parse_args()
    if args.out and (
        args.out.resolve() == args.report.resolve()
        or (args.out.exists() and args.report.exists() and os.path.samefile(args.out, args.report))
    ):
        parser.error("output must not overwrite the input report")
    try:
        operation = check_report_identity if args.check_identity else replay_report
        result = operation(read_report(args.report), lab=args.lab)
        if args.runtime_context:
            if __package__:
                from .runtime_context import collect_runtime_context
            else:
                from runtime_context import collect_runtime_context

            result["execution_runtime"] = collect_runtime_context()
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=args.out.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, args.out)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(encoded, end="")
    if not args.check_identity and not result["matched"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
