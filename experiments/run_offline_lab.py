"""Run an allowlisted synthetic lab from explicit, trusted local JSON configuration.

This is not a sandbox or a compute budget. Review configuration before running;
small inputs can request expensive studies. Use a fresh process and unchanged checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import replay_offline_report as replay  # noqa: E402

MAX_CONFIG_BYTES = 1024 * 1024
MAX_CONFIG_DEPTH = 64


def list_labs() -> dict[str, Any]:
    """Return registry metadata without importing individual scientific modules."""
    return {
        "schema_version": 1,
        "labs": [
            {
                "name": name,
                "source_file": source,
                "dependency_source_files": list(replay.DEPENDENCIES[name]),
                "required_runtime_packages": [
                    "numpy",
                    *replay.EXTRA_RUNTIME_PACKAGES.get(name, ()),
                ],
                "optional_runtime_packages": list(replay.OPTIONAL_RUNTIME_PACKAGES.get(name, ())),
            }
            for name, source in replay.LABS.items()
        ],
    }


def _require_lab(lab: str) -> None:
    if lab not in replay.LABS:
        raise ValueError(f"unknown lab: {lab}")


def _load_lab(lab: str) -> Any:
    _require_lab(lab)
    try:
        return replay._load_lab(lab)
    except ImportError as exc:
        raise ValueError(
            f"cannot load {lab}: missing dependency {exc.name or str(exc)}; "
            "see experiments/requirements-offline-research.txt"
        ) from exc


def lab_defaults(lab: str) -> dict[str, Any]:
    """Return JSON-ready keyword defaults without calling run_lab."""
    signature = inspect.signature(_load_lab(lab).run_lab)
    defaults = {}
    for name, parameter in signature.parameters.items():
        if (
            parameter.kind
            not in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            or parameter.default is inspect.Parameter.empty
        ):
            raise ValueError(f"{lab} does not expose a complete keyword-default configuration")
        defaults[name] = parameter.default
    return json.loads(json.dumps(defaults, allow_nan=False))


def _read_config_bytes(path: Path) -> bytes:
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_CONFIG_BYTES + 1)
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError("config exceeds the 1 MiB size limit")
    return raw


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in config: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant in config: {value}")


def _parse_config(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except RecursionError as exc:
        raise ValueError(f"config nesting exceeds {MAX_CONFIG_DEPTH} levels") from exc
    if not isinstance(value, dict):
        raise ValueError("config must be a JSON object of keyword arguments")
    pending = [(value, 1)]
    while pending:
        node, depth = pending.pop()
        if isinstance(node, float) and not math.isfinite(node):
            raise ValueError("nonfinite JSON number in config")
        if isinstance(node, (dict, list)):
            if depth > MAX_CONFIG_DEPTH:
                raise ValueError(f"config nesting exceeds {MAX_CONFIG_DEPTH} levels")
            children = node.values() if isinstance(node, dict) else node
            pending.extend((child, depth + 1) for child in children)
    return value


def read_config(path: Path) -> dict[str, Any]:
    """Read a bounded strict UTF-8 JSON keyword object, without loading a lab."""
    return _parse_config(_read_config_bytes(path))


def _source_hashes(lab: str) -> dict[str, str]:
    directory = Path(replay.__file__).resolve().parent
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in (replay.LABS[lab], *replay.DEPENDENCIES[lab])
    }


def _add_legacy_cli_metadata(lab: str, module: Any, report: dict, elapsed: float) -> None:
    # These five run_lab functions omit metadata supplied by their original
    # CLIs. Never repair metadata that is present, or add it to any other lab.
    if lab not in {
        "selection_validation",
        "preprocessing_leakage",
        "covariance_estimation",
        "adaptive_sign_audit",
        "cluster_sign",
    }:
        return
    if "provenance" not in report:
        if lab not in {"preprocessing_leakage", "adaptive_sign_audit"}:
            report["provenance"] = module._provenance()
        else:
            source = Path(module.__file__).resolve()
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
                "numpy": replay.np.__version__,
                "python": platform.python_version(),
            }
            if lab == "preprocessing_leakage":
                report["provenance"]["platform"] = platform.platform()
            else:
                report["provenance"]["dependency_sha256"] = {
                    Path(dependency.__file__).name: hashlib.sha256(
                        Path(dependency.__file__).read_bytes()
                    ).hexdigest()
                    for dependency in (module.adaptive, module.base)
                }
    if "performance" not in report:
        report["performance"] = {"wall_seconds": elapsed}
        if lab == "selection_validation":
            config = report["config"]
            count = (
                len(config["families"])
                * config["repetitions"]
                * config["n_candidates"]
                * (config["train_size"] + config["gap"] + config["holdout_size"])
            )
            report["performance"].update(
                candidate_observations=count, candidate_observations_per_second=count / elapsed
            )


def _publish_report(raw: bytes, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory link creation is atomic and exclusive. Never fall back
        # to rename/replace: a competing creator or symlink must survive intact.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_configured_lab(lab: str, config: Path, output: Path) -> dict[str, Any]:
    """Run once, verify source/runtime identity, and exclusively publish a report.

    Config is a file path. Receipt hashes cover exact input and output bytes, not
    canonicalized JSON. Identity checks are not scientific replay or authenticity.
    """
    _require_lab(lab)
    config, output = Path(config), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("output already exists; refusing overwrite")
    raw_config = _read_config_bytes(config)
    kwargs = _parse_config(raw_config)
    before = _source_hashes(lab)
    module = _load_lab(lab)
    inspect.signature(module.run_lab).bind(**kwargs)
    started = time.perf_counter()
    report = module.run_lab(**kwargs)
    _add_legacy_cli_metadata(lab, module, report, time.perf_counter() - started)
    raw_report = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if _source_hashes(lab) != before:
        raise ValueError("lab source or dependency source changed during run")
    identity = replay.check_report_identity(replay.parse_report_bytes(raw_report), lab=lab)
    if identity["source_sha256"] != before[replay.LABS[lab]]:
        raise ValueError("lab source changed during identity verification")
    _publish_report(raw_report, output)
    return {
        "schema_version": 1,
        "operation": "configured_offline_lab_run",
        "lab": lab,
        "config_path": str(config.resolve()),
        "output": str(output.resolve()),
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "report_sha256": hashlib.sha256(raw_report).hexdigest(),
        "config_bytes": len(raw_config),
        "report_bytes": len(raw_report),
        "source_sha256": identity["source_sha256"],
        "dependency_sha256": {name: before[name] for name in replay.DEPENDENCIES[lab]},
        "runtime_versions": identity["runtime_versions"],
        "identity_matched": True,
        "scientific_replay_performed": False,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list registered labs without loading them")
    defaults = commands.add_parser("defaults", help="print JSON defaults without running a study")
    defaults.add_argument("--lab", choices=replay.LABS, required=True)
    run = commands.add_parser("run", help="run one reviewed configuration into a new report")
    run.add_argument("--lab", choices=replay.LABS, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            result = list_labs()
        elif args.command == "defaults":
            result = lab_defaults(args.lab)
        else:
            result = run_configured_lab(args.lab, args.config, args.out)
    except (ValueError, TypeError, OSError, ImportError, ArithmeticError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
