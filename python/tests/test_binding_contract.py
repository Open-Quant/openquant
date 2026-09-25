"""Contract between the compiled extension and the Python package.

Every public submodule registered on ``openquant._core`` must be

(a) importable as ``openquant.<name>`` (``import openquant.<name>`` and
    ``from openquant import <name>`` both work, and ``<name>`` is listed in
    ``openquant.__all__``), with every public callable of the compiled
    submodule reachable through it, and
(b) covered by a value test.

Mechanism for (b) is a file naming convention: a submodule ``<name>`` is
covered when ``python/tests/test_core_<name>.py`` exists and defines at least
one ``test_*`` function that mentions ``<name>``. The eight submodules that
predate this convention are covered by older files, listed explicitly in
``LEGACY_TEST_FILES``; do not add new entries there.

So when a new submodule is registered in ``crates/pyopenquant/src/lib.rs``
this file fails until the submodule is re-exported from
``python/openquant/__init__.py`` and ``test_core_<name>.py`` is added.
"""

import importlib
import re
import types
from pathlib import Path

import openquant
import pytest
from openquant import _core

TESTS_DIR = Path(__file__).resolve().parent

# Submodules whose tests predate the `test_core_<name>.py` convention.
LEGACY_TEST_FILES = {
    "bars": "test_bars_module.py",
    "data": "test_data_module.py",
    "filters": "test_bindings_contract.py",
    "labeling": "test_bindings_contract.py",
    "pipeline": "test_pipeline_api.py",
    "portfolio": "test_bindings_contract.py",
    "risk": "test_bindings_contract.py",
    "sampling": "test_bindings_contract.py",
}

# Names where `openquant.<name>` is a pure-Python module wrapping the compiled
# one, so the compiled callables are not re-exported one-to-one.
PYTHON_WRAPPERS = {
    "backtesting_engine",
    "bars",
    "cross_validation",
    "data",
    "feature_importance",
    "hyperparameter_tuning",
    "pipeline",
}

_TEST_DEF_RE = re.compile(r"^def (test_\w+)\(", re.MULTILINE)


def _core_submodules():
    return sorted(
        name
        for name, value in vars(_core).items()
        if isinstance(value, types.ModuleType) and not name.startswith("_")
    )


def _public_callables(module):
    return sorted(
        name for name, value in vars(module).items() if callable(value) and not name.startswith("_")
    )


CORE_SUBMODULES = _core_submodules()


def test_core_submodules_are_discovered():
    # Guards the introspection itself: an empty walk would make every
    # parametrized check below vacuously green.
    lib_rs = TESTS_DIR.parents[1] / "crates" / "pyopenquant" / "src" / "lib.rs"
    registered = re.findall(r"^\s*(\w+)::register\(py, m\)\?;", lib_rs.read_text(), re.MULTILINE)
    # 32 today, fast_ewma included; README.md and the setup docs quote this count.
    assert len(registered) == 32
    assert CORE_SUBMODULES == sorted(registered)
    for name in CORE_SUBMODULES:
        assert _public_callables(getattr(_core, name)), f"_core.{name} exposes no callables"


@pytest.mark.parametrize("name", CORE_SUBMODULES)
def test_core_submodule_is_importable_from_package(name):
    try:
        module = importlib.import_module(f"openquant.{name}")
    except ImportError as exc:
        pytest.fail(
            f"openquant._core.{name} is not importable as openquant.{name}: {exc}. "
            "Re-export it in python/openquant/__init__.py."
        )
    assert getattr(openquant, name, None) is module, (
        f"openquant.{name} attribute and `import openquant.{name}` disagree"
    )
    assert name in openquant.__all__, f"'{name}' is missing from openquant.__all__"

    if name in PYTHON_WRAPPERS:
        assert module is not getattr(_core, name)
        return
    core_module = getattr(_core, name)
    for fn in _public_callables(core_module):
        assert getattr(module, fn, None) is getattr(core_module, fn), (
            f"openquant.{name}.{fn} is not the compiled openquant._core.{name}.{fn}"
        )


@pytest.mark.parametrize("name", CORE_SUBMODULES)
def test_core_submodule_has_value_test(name):
    filename = LEGACY_TEST_FILES.get(name, f"test_core_{name}.py")
    path = TESTS_DIR / filename
    assert path.is_file(), (
        f"openquant._core.{name} has no value test: expected python/tests/{filename} "
        "(see this file's docstring)."
    )
    source = path.read_text(encoding="utf-8")
    assert _TEST_DEF_RE.search(source), f"python/tests/{filename} defines no test_* function"
    assert re.search(rf"\b{re.escape(name)}\b", source), (
        f"python/tests/{filename} never references '{name}'"
    )


def test_legacy_test_file_map_is_not_stale():
    unknown = sorted(set(LEGACY_TEST_FILES) - set(CORE_SUBMODULES))
    assert not unknown, f"LEGACY_TEST_FILES names submodules that no longer exist: {unknown}"
    shadowed = sorted(
        name for name in LEGACY_TEST_FILES if (TESTS_DIR / f"test_core_{name}.py").exists()
    )
    assert not shadowed, (
        f"{shadowed} now have test_core_<name>.py files; drop them from LEGACY_TEST_FILES"
    )
