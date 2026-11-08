"""The static Python inventory must not promote nested definitions to module APIs."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def inventory() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "api_inventory", ROOT / "scripts/generate_api_inventory.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scan_respects_scope_without_executing_module(
    inventory: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(inventory, "PY_SRC", tmp_path)
    (tmp_path / "example.py").write_text(
        '''"""Example source:
def docstring_example():
    pass
"""
raise RuntimeError("The scanner must not import this module")

@decorator_not_imported
async def async_api():
    async def nested_async():
        pass

@decorator_not_imported
def public_api():
    def nested_helper():
        pass
    return nested_helper

class PublicClass:
    def method(self):
        pass

    @property
    def property_name(self):
        pass

def _private_helper():
    pass
''',
        encoding="utf-8",
    )
    (tmp_path / "classes_only.py").write_text(
        "class PublicClass:\n    def method(self): pass\n", encoding="utf-8"
    )
    (tmp_path / "__init__.py").write_text("def package_export(): pass\n", encoding="utf-8")
    assert inventory.scan_python() == {"example": ["async_api", "public_api"]}


def test_scan_sorts_and_deduplicates_overload_definitions(
    inventory: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(inventory, "PY_SRC", tmp_path)
    (tmp_path / "example.py").write_text(
        "@overload\ndef zebra(x: int): ...\n"
        "@overload\ndef zebra(x: str): ...\n"
        "def zebra(x): return x\n"
        "def alpha(): pass\n",
        encoding="utf-8",
    )
    assert inventory.scan_python() == {"example": ["alpha", "zebra"]}


def test_repository_inventory_does_not_export_methods_or_nested_helpers(
    inventory: ModuleType,
) -> None:
    result = inventory.scan_python()
    assert set(result["adapters"]) == {
        "to_pandas",
        "to_polars_backtest_frame",
        "to_polars_event_frame",
        "to_polars_frontier_frame",
        "to_polars_indicator_matrix",
        "to_polars_signal_frame",
        "to_polars_weights_frame",
    }
    assert "draw" not in result["cross_validation"]
    assert "fetch_symbol" not in result["data"]
    assert "record" not in result["evaluation"]
