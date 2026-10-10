"""Installed-wheel identity checks use the distribution's actual RECORD paths."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/release/smoke_wheel.py"
spec = importlib.util.spec_from_file_location("openquant_smoke_wheel", SCRIPT)
assert spec is not None and spec.loader is not None
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


@pytest.fixture
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    root = tmp_path / "site-packages"
    package = root / "openquant"
    package.mkdir(parents=True)
    init = package / "__init__.py"
    init.write_text("# installed package\n")
    core = package / "_core.abi3.so"
    core.write_bytes(b"extension fixture")
    metadata = root / "smoke_fixture_dist-1.2.3.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: smoke-fixture-dist\nVersion: 1.2.3\n")
    record = metadata / "RECORD"
    with record.open("w", newline="") as stream:
        csv.writer(stream).writerows(
            [("openquant/__init__.py", "", ""), ("openquant/_core.abi3.so", "", "")]
        )
    monkeypatch.syspath_prepend(str(root))
    pkg_module = ModuleType("openquant")
    pkg_module.__file__ = str(init)
    core_module = ModuleType("openquant._core")
    core_module.__file__ = str(core)
    monkeypatch.setitem(sys.modules, "openquant", pkg_module)
    monkeypatch.setitem(sys.modules, "openquant._core", core_module)
    pkg_module.__dict__["_core"] = core_module
    # The numerical Quickstart is covered by wheel CI; isolate its input here.
    research = ModuleType("openquant.research")
    research.__dict__["make_synthetic_futures_dataset"] = lambda **kwargs: None
    research.__dict__["run_flywheel_iteration"] = lambda dataset: {
        "leakage_checks": {"timestamps_increasing": True, "event_indices_sorted": True},
        "summary": SimpleNamespace(
            transpose=lambda **kwargs: SimpleNamespace(
                iter_rows=lambda: [
                    (name, 0.0)
                    for name in (
                        "portfolio_sharpe",
                        "realized_sharpe",
                        "net_total_return",
                        "turnover",
                    )
                ]
            )
        ),
        "promotion": {"promote_candidate": False},
    }
    monkeypatch.setitem(sys.modules, "openquant.research", research)
    monkeypatch.setattr(smoke, "SUBMODULES", ())
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--dist-name", "smoke-fixture-dist", "--version", "1.2.3", "--abi3"],
    )
    return SimpleNamespace(
        root=root, package=pkg_module, core=core_module, record=record, metadata=metadata
    )


def test_matching_installed_wheel(
    installed: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    smoke.main()
    assert "smoke test passed" in capsys.readouterr().out


@pytest.mark.parametrize("module", ["package", "core"])
def test_rejects_import_from_other_install(
    installed: SimpleNamespace, tmp_path: Path, module: str, capsys: pytest.CaptureFixture[str]
) -> None:
    imported = getattr(installed, module)
    stale = tmp_path / "other" / "site-packages" / "openquant" / Path(imported.__file__).name
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale install")
    imported.__file__ = str(stale)
    with pytest.raises(SystemExit, match="1"):
        smoke.main()
    assert "not recorded by" in capsys.readouterr().err


@pytest.mark.parametrize("missing", ["__init__.py", "_core.abi3.so", "both"])
def test_rejects_files_not_owned_by_distribution(
    installed: SimpleNamespace, missing: str, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = ["openquant/__init__.py,,\n", "openquant/_core.abi3.so,,\n"]
    (installed.root / "unrelated.py").write_text("# another distribution's module\n")
    installed.record.write_text(
        "unrelated.py,,\n"
        + "".join(row for row in rows if missing != "both" and missing not in row)
    )
    with pytest.raises(SystemExit, match="1"):
        smoke.main()
    assert "not recorded by" in capsys.readouterr().err


def test_rejects_missing_record(
    installed: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    installed.record.unlink()
    with pytest.raises(SystemExit, match="1"):
        smoke.main()
    assert "no installed file record" in capsys.readouterr().err


def test_accepts_windows_abi3_filename(installed: SimpleNamespace) -> None:
    extension = Path(installed.core.__file__)
    renamed = extension.with_name("_core.pyd")
    extension.rename(renamed)
    installed.core.__file__ = str(renamed)
    installed.record.write_text("openquant/__init__.py,,\nopenquant/_core.pyd,,\n")
    smoke.main()
