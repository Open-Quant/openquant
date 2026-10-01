"""scripts/release/release_info.py: version checks and release notes for release.yml."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "release" / "release_info.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("openquant_release_info", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


ri = _load()

CHANGELOG = """# Changelog

Intro text.

## Unreleased

### Added

- Something new

## [0.2.0] - 2026-11-01

### Fixed

- A fix

```text
## 9.9.9 inside a code fence is not a heading
```

## 0.1.0 - 2026-10-01

### Added

- The first release
"""


def _write_tree(root: Path, py: str, core: str, bindings: str, toolchain: str = "1.98.1") -> None:
    (root / "crates" / "openquant").mkdir(parents=True)
    (root / "crates" / "pyopenquant").mkdir(parents=True)
    (root / "pyproject.toml").write_text(f'[project]\nname = "some-dist"\nversion = "{py}"\n')
    (root / "crates" / "openquant" / "Cargo.toml").write_text(
        f'[package]\nname = "openquant"\nversion = "{core}"\n'
    )
    (root / "crates" / "pyopenquant" / "Cargo.toml").write_text(
        f'[package]\nname = "pyopenquant"\nversion = "{bindings}"\n'
    )
    (root / "rust-toolchain.toml").write_text(f'[toolchain]\nchannel = "{toolchain}"\n')


def test_metadata_reads_name_version_and_toolchain(tmp_path: Path) -> None:
    _write_tree(tmp_path, "1.2.3", "1.2.3", "1.2.3")
    md = ri.read_metadata(tmp_path)
    assert (md.name, md.version, md.rust_toolchain) == ("some-dist", "1.2.3", "1.98.1")


def test_metadata_rejects_a_version_mismatch(tmp_path: Path) -> None:
    _write_tree(tmp_path, "1.2.3", "1.2.3", "1.2.4")
    with pytest.raises(ri.ReleaseError, match="pyopenquant/Cargo.toml has 1.2.4"):
        ri.read_metadata(tmp_path)


def test_repository_versions_agree() -> None:
    md = ri.read_metadata(REPO_ROOT)
    assert md.version
    assert md.name


@pytest.mark.parametrize("tag", ["v1.2.3", "refs/tags/v1.2.3"])
def test_check_tag_accepts_the_matching_tag(tag: str) -> None:
    ri.check_tag(tag, "1.2.3")


@pytest.mark.parametrize("tag", ["1.2.3", "v1.2.4", "v1.2.3-rc.1"])
def test_check_tag_rejects_other_tags(tag: str) -> None:
    with pytest.raises(ri.ReleaseError, match="does not match"):
        ri.check_tag(tag, "1.2.3")


def test_notes_for_a_version_section() -> None:
    assert ri.extract_notes(CHANGELOG, "0.1.0") == "### Added\n\n- The first release\n"


def test_notes_for_a_bracketed_heading_keep_code_fences() -> None:
    notes = ri.extract_notes(CHANGELOG, "0.2.0")
    assert notes.startswith("### Fixed\n\n- A fix\n")
    assert "## 9.9.9 inside a code fence" in notes
    assert "first release" not in notes


def test_notes_require_a_version_section_unless_unreleased_allowed() -> None:
    with pytest.raises(ri.ReleaseError, match="no `## 0.3.0` section"):
        ri.extract_notes(CHANGELOG, "0.3.0")
    notes = ri.extract_notes(CHANGELOG, "0.3.0", allow_unreleased=True)
    assert notes == "### Added\n\n- Something new\n"


def test_notes_reject_an_empty_section() -> None:
    with pytest.raises(ri.ReleaseError, match="empty"):
        ri.extract_notes("# C\n\n## 1.0.0\n\n## 0.9.0\n\n- x\n", "1.0.0")


def test_repository_changelog_yields_notes_for_the_current_version() -> None:
    md = ri.read_metadata(REPO_ROOT)
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert ri.extract_notes(changelog, md.version, allow_unreleased=True).strip()


def test_cli_metadata_checks_the_tag(capsys: pytest.CaptureFixture[str]) -> None:
    md = ri.read_metadata(REPO_ROOT)
    assert ri.main(["metadata", "--tag", f"v{md.version}"]) == 0
    assert f"version={md.version}" in capsys.readouterr().out
    assert ri.main(["metadata", "--tag", "v0.0.0-not-this"]) == 1
    assert "does not match" in capsys.readouterr().err
