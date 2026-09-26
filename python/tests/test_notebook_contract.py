"""The notebook contract (`just notebooks-lint`) and the `nbrepro.footer` it requires."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = REPO_ROOT / "notebooks" / "python"

RESEARCH = [
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
FOOTER = "import nbrepro\n\nnbrepro.footer(seed=1, data_hash='sha256:x')"
FOOTER_OUT = [
    {"output_type": "stream", "name": "stdout", "text": "data hash:    sha256:x\nseed:  1\n"},
    {"output_type": "stream", "name": "stderr", "text": "git sha:  abc\npackage:  pyopenquant\n"},
]


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


lint = _load(NOTEBOOKS / "scripts" / "lint_notebooks.py", "openquant_nb_lint")
nbrepro = _load(NOTEBOOKS / "nbrepro.py", "openquant_nb_repro")


def _md(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def _code(src: str, outputs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": 1,
        "outputs": outputs or [],
        "source": [src],
    }


def _write(tmp_path: Path, title: str, sections: list[str], **kw: Any) -> Path:
    cells = [_md(f"# {title}\n\nIntro.")]
    for s in sections:
        cells += [_md(f"## {s}"), _code("x = 1")]
    cells[-1] = _code(kw.get("footer", FOOTER), kw.get("footer_out", FOOTER_OUT))
    cells[1:1] = kw.get("extra", [])
    path = tmp_path / "99_example.ipynb"
    path.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4}))
    return path


def _rules(path: Path) -> list[str]:
    return [p.rule for p in lint.lint_notebook(path)]


def test_committed_notebooks_comply() -> None:
    problems = [str(p) for path in lint.discover() for p in lint.lint_notebook(path)]
    assert problems == []


def test_compliant_runbook_and_tour(tmp_path: Path) -> None:
    assert _rules(_write(tmp_path, "Runbook: example", RESEARCH)) == []
    tour = _write(tmp_path, "API tour: example", ["Setup", "Anything", "Reproducibility"])
    assert _rules(tour) == []


def test_kind_is_required(tmp_path: Path) -> None:
    assert "kind" in _rules(_write(tmp_path, "Example", ["Setup", "Reproducibility"]))


def test_runbook_sections_exact_and_ordered(tmp_path: Path) -> None:
    swapped = RESEARCH.copy()
    swapped[1], swapped[2] = swapped[2], swapped[1]
    assert "sections" in _rules(_write(tmp_path, "Runbook: x", swapped))
    missing = [s for s in RESEARCH if s != "Promotion decision"]
    assert "sections" in _rules(_write(tmp_path, "Runbook: x", missing))


def test_tour_may_not_claim_research_sections(tmp_path: Path) -> None:
    path = _write(tmp_path, "API tour: x", ["Setup", "Hypothesis", "Reproducibility"])
    assert "sections" in _rules(path)


def test_footer_call_keywords_and_outputs(tmp_path: Path) -> None:
    tour = ["Setup", "Reproducibility"]
    no_call = _write(tmp_path, "API tour: x", tour, footer="print('seed: 1')")
    assert "footer" in _rules(no_call)
    no_seed = _write(tmp_path, "API tour: x", tour, footer="nbrepro.footer(data_hash='h')")
    assert "footer" in _rules(no_seed)
    unexecuted = _write(tmp_path, "API tour: x", tour, footer_out=FOOTER_OUT[:1])
    assert "footer" in _rules(unexecuted)


def test_volatile_outputs_are_rejected(tmp_path: Path) -> None:
    version_out = [
        {"output_type": "stream", "name": "stdout", "text": "numpy 2.4.6 | polars 1.1\n"}
    ]
    temp_path = [{"output_type": "stream", "name": "stderr", "text": "reg: /tmp/oq-1/a.json\n"}]
    extra = [_code("print(1)", version_out), _code("print(2)", temp_path)]
    path = _write(tmp_path, "API tour: x", ["Setup", "Reproducibility"], extra=extra)
    assert _rules(path).count("volatile") == 2


def test_headings_in_code_fences_are_ignored() -> None:
    text = "## Setup\n```bash\n# not a heading\n```\n### Sub"
    assert lint.headings(text) == [(2, "Setup"), (3, "Sub")]


def test_footer_streams(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(nbrepro.PINNED_ENV, raising=False)
    nbrepro.footer(seed=7, data_hash="sha256:abc", config={"a": 1}, trials=3, stderr={"mc": "h"})
    out, err = capsys.readouterr()
    assert out.splitlines() == [
        "data hash:    sha256:abc",
        "seed:         7",
        f"config:       {nbrepro.config_digest({'a': 1})}",
        "trials:       3",
    ]
    assert "git sha:" in err and "package:      pyopenquant" in err and "numpy" in err
    assert "mc:" in err

    monkeypatch.setenv(nbrepro.PINNED_ENV, "1")
    nbrepro.footer(seed=None, data_hash="sha256:abc")
    out, err = capsys.readouterr()
    assert "seed:         none (nothing is random)" in out
    assert "the commit containing this notebook" in err
    assert "numpy" not in err  # no versions in committed outputs


def test_config_digest_matches_research_manifest() -> None:
    from openquant.research import research_run_manifest

    config = {"b": [1, 2], "a": 0.5}
    assert nbrepro.config_digest(config) == research_run_manifest(config)["config_digest"]
