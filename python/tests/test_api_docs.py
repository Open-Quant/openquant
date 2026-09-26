"""Every compiled function carries a signature and a docstring (issue #54).

The static gate (scripts/generate_api_inventory.py --check) reads the Rust source; this
test checks what a Python user actually gets from the built extension.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import pydoc
import sys
import types
from pathlib import Path

import openquant
from openquant import _core


def _load_allowlist() -> dict[str, str]:
    path = Path(__file__).resolve().parents[2] / "scripts" / "pyopenquant_bindings.py"
    spec = importlib.util.spec_from_file_location("pyopenquant_bindings", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their module through sys.modules.
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    allowlist: dict[str, str] = module.UNDOCUMENTED_ALLOWLIST
    return allowlist


UNDOCUMENTED_ALLOWLIST = _load_allowlist()

STUBS = Path(openquant.__file__).resolve().parent / "_core"


def _compiled_functions() -> list[tuple[str, str, object]]:
    out = []
    for mod_name, mod in sorted(vars(_core).items()):
        if not isinstance(mod, types.ModuleType):
            continue
        for fn_name, fn in sorted(vars(mod).items()):
            if not fn_name.startswith("_") and callable(fn):
                out.append((mod_name, fn_name, fn))
    return out


def test_every_compiled_function_has_a_signature() -> None:
    missing = [
        f"{m}.{n}"
        for m, n, fn in _compiled_functions()
        if getattr(fn, "__text_signature__", None) is None
    ]
    assert not missing
    # And inspect can read it.
    for _, _, fn in _compiled_functions():
        inspect.signature(fn)  # type: ignore[arg-type]


def test_every_compiled_function_outside_the_allowlist_has_a_docstring() -> None:
    missing = [
        f"{m}.{n}"
        for m, n, fn in _compiled_functions()
        if m not in UNDOCUMENTED_ALLOWLIST and not (fn.__doc__ or "").strip()
    ]
    assert not missing, f"bound functions without a docstring: {missing}"


def test_every_compiled_module_has_a_stub() -> None:
    modules = {m for m, _, _ in _compiled_functions()}
    assert modules == {p.stem for p in STUBS.glob("*.pyi")} - {"__init__"}
    assert (STUBS.parent / "py.typed").exists()


def test_help_shows_signature_and_description() -> None:
    text = pydoc.plain(pydoc.render_doc(openquant.labeling.triple_barrier_labels))
    assert "triple_barrier_labels(" in text
    assert "vertical_barrier_times=None" in text
    assert "Parameters" in text and "Returns" in text
    assert "AFML" in text


def test_core_submodules_are_importable_by_name() -> None:
    assert importlib.import_module("openquant._core.labeling") is _core.labeling
