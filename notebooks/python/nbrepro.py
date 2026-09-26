"""Reproducibility footer for the notebooks (the last cell of every ``NN_*.ipynb``).

The notebook contract (``docs-site/src/content/docs/workflows/research-notebook-contract.md``)
ends every notebook with a ``## Reproducibility`` section whose code cell calls
:func:`footer`. It records the four things needed to reproduce the run: the git
commit, a content hash of the data, the package version and the seed.

What goes where, and why:

* **stdout**: what the notebook computed from, which must not change unless the
  notebook changes: the data hash, the seed, config digests, trial counts. The
  committed-output check (``just notebooks-verify``) compares it.
* **stderr**: what differs between machines or commits without the results
  changing: the git commit, the package and dependency versions, and any
  platform-dependent hash passed in ``stderr=``. The committed-output check
  ignores stderr, so a dependency bump (a new polars, say) cannot make a
  notebook "stale".

When the notebook runner executes the repository's notebooks
(``just notebooks-run``, which sets ``OPENQUANT_NOTEBOOK_PINNED=1``), the
commit and the versions are not printed at all: the committed notebook is
pinned by the commit that contains it, whose ``uv.lock`` and ``Cargo.lock``
fix every version, and CI re-executes it at that commit and fails if the
outputs differ. Printing the parent commit and the local versions there would
only rewrite every notebook on every run. Run a notebook any other way (in
Jupyter, or with ``execute_notebook_cells.py`` on your own data) and the real
values are printed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

PINNED_ENV = "OPENQUANT_NOTEBOOK_PINNED"
PACKAGE = "pyopenquant"
_WIDTH = 14


def _line(label: str, value: Any, *, err: bool = False) -> None:
    print(f"{label + ':':<{_WIDTH}}{value}", file=sys.stderr if err else sys.stdout)


def _version(dist: str) -> str:
    try:
        return version(dist)
    except PackageNotFoundError:
        return "not installed"


def git_commit() -> str:
    """``HEAD``'s sha, with ``(dirty working tree)`` when there are uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable (not a git checkout)"
    return f"{sha} (dirty working tree)" if dirty else sha


def config_digest(config: Any) -> str:
    """12 hex digits of SHA-256 over ``config`` as sorted-key JSON (``research_run_manifest``'s)."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def frame_hash(frame: Any, *, decimals: int | None = 10) -> str:
    """``openquant.data.dataset_hash`` of a polars frame, floats rounded to ``decimals``.

    Rounding keeps the hash identical across platforms whose math libraries
    differ in the last bit (data generated in memory with ``exp``/``sin``);
    pass ``decimals=None`` to hash the exact values.
    """
    import polars as pl
    from openquant.data import dataset_hash

    if decimals is not None:
        frame = frame.with_columns(pl.col(pl.Float32, pl.Float64).round(decimals))
    return dataset_hash(frame)


def footer(
    *,
    seed: int | list[int] | None,
    data_hash: str,
    config: Any = None,
    stderr: dict[str, Any] | None = None,
    **fields: Any,
) -> None:
    """Print the reproducibility footer.

    ``seed`` is the seed (or seeds) every random draw derives from; ``None``
    only when nothing is random, and it is then printed as such. ``data_hash``
    is the content hash of the data (``fetch(..., return_meta=True)``'s
    ``dataset_hash``, or :func:`frame_hash` of data built in memory).
    ``config``, when given, is printed as its :func:`config_digest`. Extra
    keyword ``fields`` go to stdout; ``stderr`` holds values that are correct
    but platform-dependent (a hash over floats drawn from numpy, say).
    """
    _line("data hash", data_hash)
    _line("seed", "none (nothing is random)" if seed is None else seed)
    if config is not None:
        _line("config", config_digest(config))
    for key, value in fields.items():
        _line(key.replace("_", " "), value)
    for key, value in (stderr or {}).items():
        _line(key, value, err=True)
    if os.environ.get(PINNED_ENV) == "1":
        _line("git sha", "the commit containing this notebook (CI re-runs it there)", err=True)
        _line(
            "package", f"{PACKAGE} built from that commit; dependencies from its uv.lock", err=True
        )
        return
    _line("git sha", git_commit(), err=True)
    _line("package", f"{PACKAGE} {_version(PACKAGE)}", err=True)
    _line(
        "python",
        f"{platform.python_version()} | numpy {_version('numpy')} | polars {_version('polars')}",
        err=True,
    )
