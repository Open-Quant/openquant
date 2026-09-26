"""Check every notebook against the notebook contract (``just notebooks-lint``).

    python3 notebooks/python/scripts/lint_notebooks.py            # every NN_*.ipynb
    python3 notebooks/python/scripts/lint_notebooks.py PATH ...   # just these

The contract is written down in
``docs-site/src/content/docs/workflows/research-notebook-contract.md``; this
script enforces the parts a machine can check. Standard library only, so it
runs without the extension or a kernel. For each notebook:

* **Kind.** The first cell is Markdown whose first line is ``# Runbook: <title>``
  (a research notebook) or ``# API tour: <title>`` (a tour), and it is the only
  level-1 heading.
* **Sections.** The level-2 (``## ``) headings of a runbook are exactly
  ``RESEARCH_SECTIONS``, in that order (``###`` subsections are free). A tour
  starts with ``## Setup``, ends with ``## Reproducibility`` and has none of
  ``RESEARCH_ONLY``: a tour tests nothing, so it may not claim a hypothesis or
  a promotion.
* **Footer.** The last cell is a code cell under ``## Reproducibility`` that
  calls ``nbrepro.footer(...)`` with ``seed=`` and ``data_hash=``, and its
  committed outputs carry the footer lines: ``data hash:`` and ``seed:`` on
  stdout, ``git sha:`` and ``package:`` on stderr.
* **Committed outputs.** Every code cell has been executed (no cell without an
  execution count), no output is an error, no output contains a machine path
  (a temp directory or a home directory), and no stdout names a package with
  its version: versions belong on stderr (see ``nbrepro.py``), or a dependency
  bump makes the committed outputs stale.

Exit status 1 if any notebook breaks a rule; every problem is printed as
``path: rule: message``.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_DIR = REPO_ROOT / "notebooks" / "python"

KINDS = {"Runbook:": "research", "API tour:": "tour"}
RESEARCH_SECTIONS = [
    "Setup",
    "Hypothesis",
    "Data",
    "Method",
    "Results",
    "Analysis",
    "Promotion decision",
    "Self-review checklist",
    "Reproducibility",
]
RESEARCH_ONLY = {"Hypothesis", "Promotion decision", "Self-review checklist"}
FOOTER_CALL = ("nbrepro", "footer")
FOOTER_KEYWORDS = ("seed", "data_hash")
FOOTER_STDOUT = ("data hash:", "seed:")
FOOTER_STDERR = ("git sha:", "package:")

# Absolute paths that differ by machine: macOS and Linux temp dirs, home dirs, CI checkouts.
MACHINE_PATH = re.compile(r"(/var/folders/|/private/var/|/tmp/|/Users/[^/\s]+/|/home/[^/\s]+/)")
# "polars 1.44.2", "numpy: 2.4.6", "pyopenquant==0.1.0", "Python 3.13.1" ...
PACKAGE_VERSION = re.compile(
    r"\b(python|numpy|polars|pandas|scipy|matplotlib|pyarrow|openquant|pyopenquant)\b"
    r"[\s:=|(v]{1,4}\d+\.\d+(\.\d+)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Problem:
    path: Path
    rule: str
    message: str

    def __str__(self) -> str:
        try:
            shown = self.path.relative_to(REPO_ROOT)
        except ValueError:
            shown = self.path
        return f"{shown}: {self.rule}: {self.message}"


def discover() -> list[Path]:
    return sorted(NOTEBOOK_DIR.glob("[0-9][0-9]_*.ipynb"))


def _source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def _text(value: object) -> str:
    return "".join(value) if isinstance(value, list) else str(value)


def headings(markdown: str) -> list[tuple[int, str]]:
    """``(level, title)`` of each ATX heading outside fenced code blocks."""
    found: list[tuple[int, str]] = []
    fence: str | None = None
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            fence = None if fence == marker else (fence or marker)
            continue
        if fence is not None:
            continue
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if m:
            found.append((len(m.group(1)), m.group(2)))
    return found


def notebook_kind(first_cell: dict | None) -> str | None:
    if not first_cell or first_cell.get("cell_type") != "markdown":
        return None
    lines = _source(first_cell).lstrip().splitlines()
    if not lines or not lines[0].startswith("# "):
        return None
    title = lines[0][2:].strip()
    for prefix, kind in KINDS.items():
        if title.startswith(prefix) and title[len(prefix) :].strip():
            return kind
    return None


def _calls_footer(source: str) -> tuple[bool, set[str]]:
    """Whether ``source`` calls ``nbrepro.footer``, and the keyword names it passes."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False, set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and (node.func.value.id, node.func.attr) == FOOTER_CALL
        ):
            return True, {k.arg for k in node.keywords if k.arg is not None}
    return False, set()


def _output_texts(cell: dict) -> list[tuple[str, str]]:
    """``(channel, text)`` for each text output: ``stdout``, ``stderr`` or ``result``."""
    texts: list[tuple[str, str]] = []
    for out in cell.get("outputs", []):
        kind = out.get("output_type")
        if kind == "stream":
            texts.append((out.get("name", "stdout"), _text(out.get("text", ""))))
        elif kind in {"execute_result", "display_data"}:
            data = out.get("data", {})
            for mime in ("text/plain", "text/html", "text/markdown"):
                if mime in data:
                    texts.append(("result", _text(data[mime])))
    return texts


def lint_notebook(path: Path) -> list[Problem]:
    problems: list[Problem] = []

    def bad(rule: str, message: str) -> None:
        problems.append(Problem(path, rule, message))

    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [Problem(path, "read", str(exc))]
    cells: list[dict] = nb.get("cells", [])

    # ---- kind and title ------------------------------------------------------------------------
    kind = notebook_kind(cells[0] if cells else None)
    if kind is None:
        bad(
            "kind",
            "the first cell must be Markdown starting '# Runbook: <title>' or "
            "'# API tour: <title>'",
        )

    # ---- section headings ----------------------------------------------------------------------
    sections: list[tuple[str, int]] = []  # (title, cell index)
    h1_count = 0
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        for level, title in headings(_source(cell)):
            if level == 1:
                h1_count += 1
                if idx != 0:
                    bad("title", f"cell {idx}: level-1 heading {title!r} outside the first cell")
            elif level == 2:
                sections.append((title, idx))
    if h1_count > 1:
        bad("title", f"{h1_count} level-1 headings; only the title may be one")

    titles = [t for t, _ in sections]
    if kind == "research" and titles != RESEARCH_SECTIONS:
        bad(
            "sections",
            f"a runbook's '## ' sections must be exactly {RESEARCH_SECTIONS} in order; "
            f"found {titles}",
        )
    if kind == "tour":
        if not titles or titles[0] != "Setup":
            bad("sections", f"an API tour's first '## ' section must be Setup; found {titles[:1]}")
        if not titles or titles[-1] != "Reproducibility":
            bad(
                "sections",
                f"an API tour's last '## ' section must be Reproducibility; found {titles[-1:]}",
            )
        claimed = sorted(RESEARCH_ONLY.intersection(titles))
        if claimed:
            bad(
                "sections",
                f"an API tour tests nothing, so it may not have {claimed}; "
                "make it a runbook or drop them",
            )

    # ---- reproducibility footer ----------------------------------------------------------------
    last = cells[-1] if cells else None
    repro_idx = next((i for t, i in reversed(sections) if t == "Reproducibility"), None)
    if last is None or last.get("cell_type") != "code":
        bad("footer", "the last cell must be the code cell that calls nbrepro.footer(...)")
    else:
        last_idx = len(cells) - 1
        if repro_idx is None or repro_idx >= last_idx:
            bad("footer", "the footer cell must come after the '## Reproducibility' heading")
        elif any(i > repro_idx for _, i in sections):
            bad("footer", "the footer cell must be in the Reproducibility section")
        called, keywords = _calls_footer(_source(last))
        if not called:
            bad("footer", "the last cell does not call nbrepro.footer(...)")
        else:
            missing = [k for k in FOOTER_KEYWORDS if k not in keywords]
            if missing:
                bad("footer", f"nbrepro.footer(...) is missing {missing}")
        streams: dict[str, str] = {}
        for channel, text in _output_texts(last):
            streams[channel] = streams.get(channel, "") + text
        out_lines = [ln.lstrip() for ln in streams.get("stdout", "").splitlines()]
        err_lines = [ln.lstrip() for ln in streams.get("stderr", "").splitlines()]
        expected = [(label, "stdout", out_lines) for label in FOOTER_STDOUT] + [
            (label, "stderr", err_lines) for label in FOOTER_STDERR
        ]
        for label, channel, lines in expected if called else []:
            if not any(ln.startswith(label) for ln in lines):
                bad(
                    "footer",
                    f"the committed footer output has no {label!r} line on {channel}; "
                    "run `just notebooks-run`",
                )

    # ---- committed outputs ---------------------------------------------------------------------
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code" or not _source(cell).strip():
            continue
        if cell.get("execution_count") is None:
            bad("outputs", f"cell {idx} has not been executed; run `just notebooks-run`")
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                bad("outputs", f"cell {idx} has an error output ({out.get('ename')})")
        for channel, text in _output_texts(cell):
            m = MACHINE_PATH.search(text)
            if m:
                bad(
                    "volatile",
                    f"cell {idx} {channel} contains a machine path ({m.group(0)}...); "
                    "print a description instead",
                )
            if channel != "stderr":
                m = PACKAGE_VERSION.search(text)
                if m:
                    bad(
                        "volatile",
                        f"cell {idx} {channel} prints a package version ({m.group(0)!r}); "
                        "versions go to stderr (nbrepro.footer does this)",
                    )
    return problems


def main(argv: list[str]) -> int:
    paths = [Path(a).resolve() for a in argv] or discover()
    if not paths:
        print("no notebooks found", file=sys.stderr)
        return 1
    problems = [p for path in paths for p in lint_notebook(path)]
    for problem in problems:
        print(problem)
    counts: dict[str, int] = {}
    for path in paths:
        try:
            first = json.loads(path.read_text(encoding="utf-8")).get("cells", [None])[0]
        except (OSError, json.JSONDecodeError, IndexError):
            first = None
        k = notebook_kind(first) or "unknown"
        counts[k] = counts.get(k, 0) + 1
    summary = ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
    status = f"{len(problems)} problem(s)" if problems else "all comply"
    print(f"notebook contract: {len(paths)} notebooks ({summary}); {status}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
