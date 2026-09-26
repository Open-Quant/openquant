"""Execute every research runbook headlessly (``just notebooks-run``) and check committed outputs.

    python notebooks/python/scripts/run_notebooks.py                   # execute in place + figures
    python notebooks/python/scripts/run_notebooks.py --only 05         # one notebook (number or stem)
    python notebooks/python/scripts/run_notebooks.py --check           # execute to a temp dir, compare
    python notebooks/python/scripts/run_notebooks.py --jobs 0          # one notebook per CPU at a time
    python notebooks/python/scripts/run_notebooks.py --against-git HEAD  # compare tree to a commit

Every ``notebooks/python/NN_*.ipynb`` runs unless ``EXCLUDED`` lists it with a
reason. Each notebook runs in a fresh ipykernel (``execute_notebook_cells.py``),
``--jobs N`` of them at a time (the notebooks are independent; see ``execute_all``);
a cell error fails that notebook, the others still run, and the exit status is 1
if any failed. Market data comes from ``openquant.data.fetch`` with its default
source, the committed SYNTHETIC sample, so no run touches the network.

The default mode rewrites the notebooks with their outputs and the figures in
``docs-site/public/figures/notebooks/``. A full run first deletes that
directory's SVGs, so a figure a notebook no longer draws does not linger.

``--check`` and ``--against-git`` report committed outputs that are stale. CI
runs the notebooks in place and then ``--against-git HEAD``, which compares the
fresh files with the committed ones without executing anything twice. The
comparison ignores what may legitimately differ between machines: PNG bytes
(only the number and type of outputs is compared; the same for raster tiles
inside SVGs), stderr, SVG path ids, and
floating-point noise (numbers are compared to ``SIG_DIGITS`` significant
digits). A changed source cell, a new output line or a missing or extra figure
is stale.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from execute_notebook_cells import NotebookExecutionError, execute_notebook  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_DIR = REPO_ROOT / "notebooks" / "python"
FIGURE_DIR = REPO_ROOT / "docs-site" / "public" / "figures" / "notebooks"

# Notebooks deliberately not run, with the reason. An entry also exempts the
# notebook from the staleness check. Empty: every runbook runs offline.
EXCLUDED: dict[str, str] = {}

SIG_DIGITS = 4
_FLOAT = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?")
_SVG_ID = re.compile(r"\b([mpc]|image)[0-9a-f]{10}\b")  # ids hashed from the content
# Raster tiles matplotlib embeds in an SVG (imshow, colorbars): libpng/zlib
# builds differ between platforms, so the bytes do, like the notebook PNGs.
_SVG_PNG = re.compile(r"data:image/png;base64,\s*[A-Za-z0-9+/=\s]+")


def discover() -> list[Path]:
    return sorted(NOTEBOOK_DIR.glob("[0-9][0-9]_*.ipynb"))


def _select(paths: list[Path], only: list[str]) -> list[Path]:
    if not only:
        return paths
    picked = [p for p in paths if any(p.name.startswith(o) or p.stem == o for o in only)]
    if not picked:
        raise SystemExit(f"no notebook matches {only}")
    return picked


def _round(match: re.Match[str]) -> str:
    text = f"{float(match.group(0)):.{SIG_DIGITS}g}"
    return "0" if text == "-0" else text


def normalize_text(text: str) -> str:
    return _FLOAT.sub(_round, text)


def notebook_fingerprint(text: str) -> list[str]:
    """The lines of a notebook that the staleness check compares."""
    nb = json.loads(text)
    lines: list[str] = []
    for idx, cell in enumerate(nb["cells"]):
        lines.append(f"## cell {idx} [{cell['cell_type']}]")
        lines.extend("".join(cell["source"]).splitlines())
        # Where the kernel splits stdout into stream outputs depends on flush timing, so
        # consecutive stdout outputs are joined and compared as one.
        stdout: list[str] = []

        def flush_stdout() -> None:
            if stdout:
                lines.append("-> stream:stdout")
                lines.extend(normalize_text("".join(stdout)).splitlines())
                stdout.clear()

        for out in cell.get("outputs", []):
            kind = out["output_type"]
            if kind == "stream" and out["name"] == "stderr":
                continue  # warnings and one-off notices (font cache builds) vary by machine
            if kind == "stream":
                stdout.append("".join(out["text"]))
                continue
            flush_stdout()
            if kind == "error":
                lines.append(f"-> error {out['ename']}: {out['evalue']}")
            else:
                for mime, data in sorted(out.get("data", {}).items()):
                    lines.append(f"-> {kind} {mime}")
                    if mime.startswith("image/") and mime != "image/svg+xml":
                        continue
                    if isinstance(data, list):
                        data = "".join(data)
                    if not isinstance(data, str):
                        data = json.dumps(data, sort_keys=True, indent=1)
                    lines.extend(normalize_text(data).splitlines())
        flush_stdout()
    return lines


def figure_fingerprint(text: str) -> list[str]:
    text = _SVG_PNG.sub("data:image/png;base64,PNG", _SVG_ID.sub(r"\1ID", text))
    return normalize_text(text).splitlines()


@dataclass
class Result:
    name: str
    status: str  # "ok", "failed", "excluded", "stale"
    seconds: float = 0.0
    detail: str = ""


def _diff(a: list[str], b: list[str], label: str, limit: int = 40) -> str:
    diff = list(
        difflib.unified_diff(a, b, f"committed/{label}", f"fresh/{label}", lineterm="", n=1)
    )
    more = f"\n... {len(diff) - limit} more diff lines" if len(diff) > limit else ""
    return "\n".join(diff[:limit]) + more


def compare(
    names: list[str],
    committed: Callable[[str], str | None],
    fresh: Callable[[str], str | None],
    committed_figures: dict[str, Callable[[], str]] | None,
    fresh_figures: dict[str, Callable[[], str]] | None,
) -> list[Result]:
    """Staleness results for notebooks ``names`` and, when given, the figure sets."""
    results: list[Result] = []
    for name in names:
        old, new = committed(name), fresh(name)
        if old is None or new is None:
            results.append(Result(name, "stale", detail="notebook missing on one side"))
            continue
        a, b = notebook_fingerprint(old), notebook_fingerprint(new)
        results.append(
            Result(name, "ok") if a == b else Result(name, "stale", detail=_diff(a, b, name))
        )
    if committed_figures is not None and fresh_figures is not None:
        problems = [
            f"not committed: {n}" for n in sorted(fresh_figures.keys() - committed_figures.keys())
        ]
        problems += [
            f"no longer drawn: {n}" for n in sorted(committed_figures.keys() - fresh_figures.keys())
        ]
        for n in sorted(committed_figures.keys() & fresh_figures.keys()):
            a, b = (
                figure_fingerprint(committed_figures[n]()),
                figure_fingerprint(fresh_figures[n]()),
            )
            if a != b:
                problems.append(f"changed: {n}\n{_diff(a, b, n, limit=12)}")
        results.append(
            Result(
                "figures/notebooks/*.svg", "stale" if problems else "ok", detail="\n".join(problems)
            )
        )
    return results


def _warm_matplotlib() -> None:
    """Build matplotlib's font cache here, not inside the first notebook's output."""
    try:
        import matplotlib.font_manager  # noqa: F401
    except ImportError:
        pass


def _execute_one(path: Path, out: Path, timeout: int, env: dict[str, str]) -> Result:
    started = time.monotonic()
    try:
        execute_notebook(path, out, timeout=timeout, env=env)
    except NotebookExecutionError as exc:
        return Result(path.name, "failed", time.monotonic() - started, str(exc))
    return Result(path.name, "ok", time.monotonic() - started)


def execute_all(
    paths: list[Path], out_dir: Path | None, fig_dir: Path, timeout: int, jobs: int = 1
) -> list[Result]:
    """Execute ``paths``; with ``jobs > 1``, that many notebooks at a time.

    Each notebook already runs in its own kernel with its own working state, writes
    only its own output file and its own ``nbNN-*`` figures, and reads market data
    through the atomically written cache, so running several at once changes
    nothing but the wall-clock time. Workers are processes, not threads:
    ``execute_notebook`` swaps ``os.environ`` around each run. Results come back
    in ``paths`` order whatever the completion order.
    """
    _warm_matplotlib()
    env = {
        "OPENQUANT_FIGURE_DIR": str(fig_dir),
        "PYTHONHASHSEED": "0",
        "MPLBACKEND": "Agg",
    }
    results: dict[str, Result] = {}
    todo: list[tuple[Path, Path]] = []
    for path in paths:
        if path.name in EXCLUDED:
            results[path.name] = Result(path.name, "excluded", detail=EXCLUDED[path.name])
        else:
            todo.append((path, out_dir / path.name if out_dir is not None else path))
    if jobs <= 1 or len(todo) <= 1:
        for path, out in todo:
            results[path.name] = _execute_one(path, out, timeout, env)
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, len(todo))) as pool:
            futures = {
                pool.submit(_execute_one, path, out, timeout, env): path for path, out in todo
            }
            for future in as_completed(futures):
                result = future.result()
                print(f"{result.name}: {result.status} ({result.seconds:.1f}s)", flush=True)
                results[result.name] = result
    return [results[p.name] for p in paths]


def _read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def _git_show(ref: str, path: Path) -> str | None:
    rel = path.relative_to(REPO_ROOT).as_posix()
    proc = subprocess.run(
        ["git", "show", f"{ref}:{rel}"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return proc.stdout if proc.returncode == 0 else None


def _git_figures(ref: str) -> dict[str, Callable[[], str]]:
    rel = FIGURE_DIR.relative_to(REPO_ROOT).as_posix()
    proc = subprocess.run(
        ["git", "ls-tree", "--name-only", f"{ref}:{rel}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    names = [n for n in proc.stdout.split() if n.endswith(".svg")] if proc.returncode == 0 else []
    return {n: (lambda n=n: _git_show(ref, FIGURE_DIR / n) or "") for n in names}


def _dir_figures(directory: Path) -> dict[str, Callable[[], str]]:
    return {p.name: (lambda p=p: p.read_text(encoding="utf-8")) for p in directory.glob("*.svg")}


def _report(results: list[Result]) -> int:
    width = max(len(r.name) for r in results)
    for r in results:
        seconds = f"{r.seconds:6.1f}s" if r.seconds else ""
        print(f"{r.name:<{width}}  {r.status:<8}  {seconds}")
    bad = [r for r in results if r.status in {"failed", "stale"}]
    for r in bad + [r for r in results if r.status == "excluded"]:
        print(f"\n--- {r.name}: {r.status}\n{r.detail}")
    if any(r.status == "stale" for r in bad):
        print(
            "\nCommitted notebook outputs are stale: run `just notebooks-run` and commit the result."
        )
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="execute to a temp dir and compare")
    mode.add_argument(
        "--against-git", metavar="REF", help="compare the working tree with REF; no run"
    )
    mode.add_argument("--list", action="store_true", help="list notebooks and exclusions")
    parser.add_argument("--only", action="append", default=[], help="notebook number or stem")
    parser.add_argument("--timeout", type=int, default=600, help="per-cell timeout in seconds")
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="notebooks to execute at once (default 1; 0 = one per CPU)",
    )
    args = parser.parse_args()
    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)

    paths = _select(discover(), args.only)
    full = not args.only
    names = [p.name for p in paths if p.name not in EXCLUDED]
    if args.list:
        for p in paths:
            print(f"{p.name}: {'excluded: ' + EXCLUDED[p.name] if p.name in EXCLUDED else 'runs'}")
        return 0

    if args.against_git:
        ref = args.against_git
        return _report(
            compare(
                names,
                lambda n: _git_show(ref, NOTEBOOK_DIR / n),
                lambda n: _read(NOTEBOOK_DIR / n),
                _git_figures(ref) if full else None,
                _dir_figures(FIGURE_DIR) if full else None,
            )
        )

    if args.check:
        with tempfile.TemporaryDirectory(prefix="openquant-notebooks-") as tmp:
            out_dir, fig_dir = Path(tmp), Path(tmp) / "figures"
            results = execute_all(paths, out_dir, fig_dir, args.timeout, jobs)
            ran = [r.name for r in results if r.status == "ok"]
            results = [r for r in results if r.status != "ok"] + compare(
                ran,
                lambda n: _read(NOTEBOOK_DIR / n),
                lambda n: _read(out_dir / n),
                _dir_figures(FIGURE_DIR) if full else None,
                _dir_figures(fig_dir) if full else None,
            )
            return _report(results)

    if full and FIGURE_DIR.exists():
        for old in FIGURE_DIR.glob("*.svg"):
            old.unlink()
    return _report(execute_all(paths, None, FIGURE_DIR, args.timeout, jobs))


if __name__ == "__main__":
    sys.exit(main())
