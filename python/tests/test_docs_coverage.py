"""Coverage claims must be explicit and apply only to their named surface."""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def coverage(tmp_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "docs_coverage", ROOT / "scripts/docs/check_coverage.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__dict__.update(
        ROOT=tmp_path,
        MODULE_PAGES=tmp_path / "modules",
        LIB_RS=tmp_path / "lib.rs",
        PY_SRC=tmp_path / "python",
        ALLOWLIST=tmp_path / "coverage_allowlist.toml",
    )
    module.MODULE_PAGES.mkdir()
    module.LIB_RS.write_text("pub mod data_structures;\n", encoding="utf-8")
    module.PY_SRC.mkdir()
    (module.PY_SRC / "bars.py").write_text("", encoding="utf-8")
    return module


def write_page(coverage: ModuleType, alias: str = "", body: str = "") -> None:
    (coverage.MODULE_PAGES / "data-structures.md").write_text(
        f"---\nmodule: data_structures\n{alias}---\n{body}", encoding="utf-8"
    )


def test_python_module_claim_covers_existing_bars_page(coverage: ModuleType) -> None:
    write_page(coverage, "python_module: bars\n")
    assert coverage.gaps(date(2026, 10, 10)) == ([], [])


def test_unknown_python_module_claim_fails(coverage: ModuleType) -> None:
    write_page(coverage, "python_module: nonexistent\n")
    _, failures = coverage.gaps(date(2026, 10, 10))
    assert any("python_module" in failure and "nonexistent" in failure for failure in failures)


def test_python_claim_cannot_cover_same_named_rust_module(coverage: ModuleType) -> None:
    coverage.LIB_RS.write_text("pub mod data_structures;\npub mod bars;\n", encoding="utf-8")
    write_page(coverage, "python_module: bars\n")
    _, failures = coverage.gaps(date(2026, 10, 10))
    assert len(failures) == 1
    assert "public rust module `bars` has no module page" in failures[0]


def test_prose_and_api_mentions_do_not_claim_module_coverage(coverage: ModuleType) -> None:
    write_page(
        coverage,
        'python_api:\n  - "bars.build_time_bars"\n',
        "Mentions openquant.bars.\npython_module: bars\n",
    )
    _, failures = coverage.gaps(date(2026, 10, 10))
    assert len(failures) == 1
    assert "public python module `bars` has no module page" in failures[0]


def test_documented_alias_rejects_stale_allowlist_entry(coverage: ModuleType) -> None:
    write_page(coverage, "python_module: bars\n")
    coverage.ALLOWLIST.write_text(
        '[python.bars]\nreason = "Needs a page"\nexpires = 2026-12-31\n', encoding="utf-8"
    )
    rows, failures = coverage.gaps(date(2026, 10, 10))
    assert rows == []
    assert len(failures) == 1
    assert "python.bars is exempted" in failures[0]
    assert "drop the entry" in failures[0]


def test_repository_bars_claim_matches_existing_documentation() -> None:
    page = ROOT / "docs-site/src/content/docs/modules/data-structures.md"
    text = page.read_text(encoding="utf-8")
    assert 'python_module: "bars"' in text.split("---", 2)[1]
    assert "from openquant import bars" in text
