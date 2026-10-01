#!/usr/bin/env python3
"""Smoke-test an installed openquant wheel: the docs-site Quickstart, step 3.

Run it with the interpreter of a clean virtual environment the wheel was installed into,
from outside the repository so the source tree cannot shadow the installed package:

    python scripts/release/smoke_wheel.py --dist-name NAME --version X.Y.Z [--abi3]

It checks that `openquant` was imported from site-packages, that the installed
distribution has the expected name and version, that the compiled extension is the
stable-ABI build when --abi3 is given, and then runs the Quickstart's flywheel iteration
(docs-site/src/content/docs/quickstart.md) and checks the shape of its result.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import math
import sys
from pathlib import Path

# Every public submodule, as tested by python/tests/test_binding_contract.py.
SUBMODULES = (
    "adapters",
    "backtesting_engine",
    "bars",
    "cross_validation",
    "data",
    "evaluation",
    "feature_diagnostics",
    "feature_importance",
    "hyperparameter_tuning",
    "pipeline",
    "research",
    "viz",
)


def fail(message: str) -> None:
    print(f"smoke test FAILED: {message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--dist-name", required=True, help="distribution name (pyproject)")
    parser.add_argument("--version", required=True, help="expected distribution version")
    parser.add_argument("--abi3", action="store_true", help="expect a stable-ABI extension")
    args = parser.parse_args()

    import openquant
    import openquant._core as core

    pkg_dir = Path(openquant.__file__).resolve().parent
    print(f"python {sys.version.split()[0]} on {sys.platform}")
    print(f"openquant imported from {pkg_dir}")
    if "site-packages" not in pkg_dir.parts:
        fail(f"openquant was imported from {pkg_dir}, not from an installed wheel")

    installed = importlib.metadata.version(args.dist_name)
    if installed != args.version:
        fail(f"{args.dist_name} {installed} is installed, expected {args.version}")
    print(f"distribution {args.dist_name} {installed}")

    ext = Path(core.__file__ or "").name
    print(f"extension {ext}")
    # abi3 builds are `_core.abi3.so` on Unix and a plain `_core.pyd` on Windows; a
    # per-version build carries the interpreter tag (`_core.cpython-311-...so`, `.cp311-...pyd`).
    if args.abi3 and ext not in ("_core.abi3.so", "_core.pyd"):
        fail(f"expected a stable-ABI (abi3) extension, got {ext}")

    for name in SUBMODULES:
        importlib.import_module(f"openquant.{name}")

    # docs-site/src/content/docs/quickstart.md, step 3.
    from openquant.research import make_synthetic_futures_dataset, run_flywheel_iteration

    dataset = make_synthetic_futures_dataset(n_bars=192, seed=7)
    result = run_flywheel_iteration(dataset)

    leakage = result["leakage_checks"]
    if not (leakage["timestamps_increasing"] and leakage["event_indices_sorted"]):
        fail(f"leakage checks did not pass: {leakage}")

    summary = result["summary"].transpose(
        include_header=True, header_name="metric", column_names=["value"]
    )
    metrics = dict(summary.iter_rows())
    for key in ("portfolio_sharpe", "realized_sharpe", "net_total_return", "turnover"):
        value = metrics.get(key)
        if value is None or not math.isfinite(value):
            fail(f"summary metric {key} is {value!r}")
        print(f"  {key:<20} {value:>12.6f}")

    # The Quickstart's synthetic candidate is rejected; a promoted one means a regression.
    if result["promotion"]["promote_candidate"] is not False:
        fail(f"the Quickstart candidate was promoted: {result['promotion']}")
    print("smoke test passed")


if __name__ == "__main__":
    main()
