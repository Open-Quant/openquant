"""Execute one notebook in a real Jupyter kernel and write it back with its outputs.

This replaced a plain ``exec()`` runner that kept stdout only. Running the
notebook through nbclient + ipykernel keeps every output a reader would see in
Jupyter: rich reprs (polars tables as HTML), ``display()`` calls and inline
matplotlib figures (``image/png``).

The written notebook is normalised so a re-run that computes the same thing
produces the same file: no execution timestamps (``record_timing=False``), no
Python patch version in the metadata, and deterministic cell ids. Any cell error
stops the run and raises :class:`NotebookExecutionError` after writing the
partially executed notebook, so the traceback can be read in the output file.

``run_notebooks.py`` (``just notebooks-run``) calls :func:`execute_notebook` for
every runbook; use this script directly for a single notebook.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

DEFAULT_TIMEOUT_S = 600
KERNEL_NAME = "python3"


class NotebookExecutionError(RuntimeError):
    """A cell raised; the message carries the notebook name and the cell's traceback."""


def normalize_notebook(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Strip what changes between runs of the same code: timings, versions, random ids."""
    nb.nbformat = 4
    nb.nbformat_minor = 5
    nb.metadata = nbformat.from_dict(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": KERNEL_NAME},
            "language_info": {"name": "python"},
        }
    )
    for idx, cell in enumerate(nb.cells):
        # Positional ids: stable across runs, unlike the random ones nbformat mints.
        cell["id"] = f"cell-{idx:02d}"
        cell.metadata.pop("execution", None)
        cell.metadata.pop("collapsed", None)
        cell.metadata.pop("scrolled", None)
        if cell.cell_type == "code":
            for output in cell.get("outputs", []):
                output.get("metadata", {}).pop("execution", None)
    return nb


def execute_notebook(
    path: Path,
    out_path: Path | None = None,
    *,
    timeout: int = DEFAULT_TIMEOUT_S,
    env: dict[str, str] | None = None,
) -> nbformat.NotebookNode:
    """Execute ``path`` with its own directory as the working directory; write to ``out_path``.

    ``out_path`` defaults to ``path`` (in place). ``env`` adds variables to the
    kernel's environment. Raises :class:`NotebookExecutionError` on the first
    failing cell.
    """
    path = Path(path)
    destination = Path(out_path) if out_path is not None else path
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=KERNEL_NAME,
        allow_errors=False,
        record_timing=False,
        resources={"metadata": {"path": str(path.resolve().parent)}},
    )
    saved_env = dict(os.environ)
    os.environ.update(env or {})
    error: CellExecutionError | None = None
    try:
        client.execute()
    except CellExecutionError as exc:
        error = exc
    finally:
        os.environ.clear()
        os.environ.update(saved_env)
    normalize_notebook(nb)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, destination)
    if error is not None:
        raise NotebookExecutionError(
            f"{path.name} failed; partial output in {destination}\n{error}"
        )
    return nb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute a notebook in a Jupyter kernel (nbclient)"
    )
    parser.add_argument("notebook", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output notebook path (default: execute in-place)",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S, help="per-cell seconds")
    args = parser.parse_args()
    execute_notebook(args.notebook, args.out, timeout=args.timeout)
    print(f"executed: {args.notebook} -> {args.out or args.notebook}")


if __name__ == "__main__":
    main()
