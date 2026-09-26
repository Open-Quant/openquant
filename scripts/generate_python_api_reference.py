#!/usr/bin/env python3
"""Generate the docs site's Python API reference from the built ``openquant`` package.

Needs the extension built into the current interpreter (``maturin develop``). For every
public module it records each public function's signature and docstring:

- compiled modules (``openquant._core.*``): names, ``__text_signature__`` and ``__doc__``
  come from the extension itself; the typed signature shown on the page comes from the
  generated stub (``python/openquant/_core/<module>.pyi``), and the two are cross-checked
  so a stale stub fails here rather than rendering a wrong signature. A compiled class
  (a ``#[pyclass]``) is listed with its constructor's parameters as its signature, its
  public methods, and its properties (shown as ``: <type>``);
- pure-Python modules: members come from the imported module, their signatures from the
  source (so the text is exactly what was written, independent of the installed polars or
  numpy versions) and their docstrings from ``__doc__`` (never inherited, so the output
  does not depend on the Python version).

Output is ``docs-site/src/data/pythonApiReference.json``, rendered by
``docs-site/src/pages/api/python/``. It is deterministic: sorted, with no timestamps.

    python scripts/generate_python_api_reference.py          # write
    python scripts/generate_python_api_reference.py --check  # fail if stale
"""

from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
import sys
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs-site" / "src" / "data" / "pythonApiReference.json"
STUBS = ROOT / "python" / "openquant" / "_core"


def _stub_signatures(module: str) -> dict[str, str]:
    path = STUBS / f"{module}.pyi"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = _format_def(node)
    return out


def _stub_classes(module: str) -> dict[str, ast.ClassDef]:
    path = STUBS / f"{module}.pyi"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def _is_property(node: ast.FunctionDef) -> bool:
    return any(isinstance(d, ast.Name) and d.id == "property" for d in node.decorator_list)


def compiled_class(name: str, cls: type, node: ast.ClassDef, import_path: str) -> dict:
    """A compiled class's entry, cross-checked against the runtime type."""
    defs = {sub.name: sub for sub in node.body if isinstance(sub, ast.FunctionDef)}
    stub_public = {n for n in defs if not n.startswith("_")}
    runtime_public = {n for n in vars(cls) if not n.startswith("_")}
    if stub_public != runtime_public:
        raise SystemExit(
            f"{import_path}.{name}: stub and extension disagree on members "
            f"(no stub: {sorted(runtime_public - stub_public)}, "
            f"not in extension: {sorted(stub_public - runtime_public)}); "
            "rebuild the extension and run scripts/generate_python_stubs.py"
        )
    signature = ""
    ctor = defs.get("__new__")
    if ctor is not None:
        # The class line shows the constructor's parameters, without `-> Self`.
        bare = copy.copy(ctor)
        bare.returns = None
        signature = _format_def(bare, drop_self=True)
        runtime_sig = [p.name for p in inspect.signature(cls).parameters.values()]
        stub_sig = [a.arg for a in ctor.args.posonlyargs + ctor.args.args][1:]
        stub_sig += [a.arg for a in ctor.args.kwonlyargs]
        if stub_sig != runtime_sig:
            raise SystemExit(
                f"{import_path}.{name}: stub constructor parameters {stub_sig} != runtime "
                f"{runtime_sig}"
            )
    methods = []
    for mname in sorted(stub_public):
        sub = defs[mname]
        doc = inspect.cleandoc(getattr(cls, mname).__doc__ or "")
        if _is_property(sub):
            ret = ast.unparse(sub.returns) if sub.returns is not None else "Any"
            methods.append({"name": mname, "signature": f": {ret}", "doc": doc})
        else:
            methods.append(
                {"name": mname, "signature": _format_def(sub, drop_self=True), "doc": doc}
            )
    return {
        "name": name,
        "signature": signature,
        "doc": inspect.cleandoc(cls.__doc__ or ""),
        "methods": methods,
    }


def _format_def(node: ast.FunctionDef | ast.AsyncFunctionDef, drop_self: bool = False) -> str:
    args = node.args
    if drop_self and (args.posonlyargs or args.args):
        args = ast.arguments(
            posonlyargs=args.posonlyargs[1:] if args.posonlyargs else [],
            args=args.args if args.posonlyargs else args.args[1:],
            vararg=args.vararg,
            kwonlyargs=args.kwonlyargs,
            kw_defaults=args.kw_defaults,
            kwarg=args.kwarg,
            defaults=args.defaults,
        )
    params = _format_params(args)
    ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    one_line = f"({', '.join(params)}){ret}"
    if len(node.name) + len(one_line) <= 88:
        return one_line
    # Black-style: one parameter per line.
    return "(\n" + "".join(f"    {p},\n" for p in params) + f"){ret}"


def _format_params(args: ast.arguments) -> list[str]:
    def one(arg: ast.arg, default: ast.expr | None) -> str:
        text = arg.arg
        if arg.annotation is not None:
            text += f": {ast.unparse(arg.annotation)}"
            if default is not None:
                text += f" = {ast.unparse(default)}"
        elif default is not None:
            text += f"={ast.unparse(default)}"
        return text

    positional = args.posonlyargs + args.args
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults))
    defaults += list(args.defaults)
    out = [one(a, d) for a, d in zip(positional, defaults, strict=True)]
    if args.posonlyargs:
        out.insert(len(args.posonlyargs), "/")
    if args.vararg is not None:
        out.append("*" + one(args.vararg, None))
    elif args.kwonlyargs:
        out.append("*")
    out += [one(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True)]
    if args.kwarg is not None:
        out.append("**" + one(args.kwarg, None))
    return out


def _runtime_param_names(func: Any) -> list[str]:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return []
    return [p.name for p in sig.parameters.values()]


def compiled_module(name: str, module: types.ModuleType, import_path: str, slug: str) -> dict:
    stub = _stub_signatures(name)
    stub_classes = _stub_classes(name)
    public = {n: f for n, f in vars(module).items() if not n.startswith("_") and callable(f)}
    runtime = {n: f for n, f in public.items() if not isinstance(f, type)}
    runtime_classes = {n: c for n, c in public.items() if isinstance(c, type)}
    if set(runtime_classes) != set(stub_classes):
        raise SystemExit(
            f"openquant._core.{name}: stub and extension disagree on classes "
            f"(no stub: {sorted(set(runtime_classes) - set(stub_classes))}, "
            f"not in extension: {sorted(set(stub_classes) - set(runtime_classes))}); "
            "rebuild the extension and run scripts/generate_python_stubs.py"
        )
    classes = [
        compiled_class(cname, runtime_classes[cname], stub_classes[cname], import_path)
        for cname in sorted(runtime_classes)
    ]
    missing_stub = sorted(set(runtime) - set(stub))
    extra_stub = sorted(set(stub) - set(runtime))
    if missing_stub or extra_stub:
        raise SystemExit(
            f"openquant._core.{name}: stub and extension disagree "
            f"(no stub: {missing_stub}, not in extension: {extra_stub}); "
            "rebuild the extension and run scripts/generate_python_stubs.py"
        )
    functions = []
    for fname in sorted(runtime):
        func = runtime[fname]
        if getattr(func, "__text_signature__", None) is None:
            raise SystemExit(f"{import_path}.{fname} has no __text_signature__")
        stub_names = [
            a.split(":")[0].split("=")[0].strip().lstrip("*") for a in _split_params(stub[fname])
        ]
        stub_names = [n for n in stub_names if n]
        if stub_names != _runtime_param_names(func):
            raise SystemExit(
                f"{import_path}.{fname}: stub parameters {stub_names} != runtime "
                f"{_runtime_param_names(func)}"
            )
        functions.append(
            {
                "name": fname,
                "signature": stub[fname],
                "doc": inspect.cleandoc(func.__doc__ or ""),
            }
        )
    return {
        "slug": slug,
        "import": import_path,
        "kind": "compiled",
        "doc": inspect.cleandoc(module.__doc__ or ""),
        "functions": functions,
        "classes": classes,
    }


def _split_params(signature: str) -> list[str]:
    inner = signature[1 : signature.rindex(")")] if ")" in signature else ""
    out, depth, cur = [], 0, []
    for ch in inner:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return [p.strip() for p in out if p.strip()]


def python_module(name: str, module: types.ModuleType) -> dict:
    source_file = inspect.getsourcefile(module)
    assert source_file is not None
    tree = ast.parse(Path(source_file).read_text(encoding="utf-8"))
    defs = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    functions, classes = [], []
    for member_name in sorted(vars(module)):
        if member_name.startswith("_"):
            continue
        obj = getattr(module, member_name)
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        node = defs.get(member_name)
        if node is None:
            continue
        if isinstance(node, ast.ClassDef):
            methods = []
            for sub in node.body:
                if not isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if sub.name.startswith("_") and sub.name != "__init__":
                    continue
                is_static = any(
                    isinstance(d, ast.Name) and d.id == "staticmethod" for d in sub.decorator_list
                )
                methods.append(
                    {
                        "name": sub.name,
                        "signature": _format_def(sub, drop_self=not is_static),
                        "doc": inspect.cleandoc(ast.get_docstring(sub) or ""),
                    }
                )
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            classes.append(
                {
                    "name": member_name,
                    "signature": f"({bases})" if bases else "",
                    "doc": inspect.cleandoc(obj.__doc__ or ""),
                    "methods": methods,
                }
            )
        elif callable(obj):
            functions.append(
                {
                    "name": member_name,
                    "signature": _format_def(node),
                    "doc": inspect.cleandoc(obj.__doc__ or ""),
                }
            )
    return {
        "slug": name,
        "import": f"openquant.{name}",
        "kind": "python",
        "doc": inspect.cleandoc(module.__doc__ or ""),
        "functions": functions,
        "classes": classes,
    }


def build() -> dict:
    import openquant
    from openquant import _core

    reexported = set(openquant._CORE_REEXPORTS)
    modules: list[dict] = []
    for name in sorted(vars(_core)):
        mod = getattr(_core, name)
        if not isinstance(mod, types.ModuleType):
            continue
        if name in reexported:
            modules.append(compiled_module(name, mod, f"openquant.{name}", name))
        else:
            # Wrapped by a pure-Python module of the same name; reachable as _core.<name>.
            modules.append(compiled_module(name, mod, f"openquant._core.{name}", f"core-{name}"))
    for name in sorted(openquant.__all__):
        if name in reexported:
            continue
        modules.append(python_module(name, getattr(openquant, name)))
    modules.sort(key=lambda m: m["import"])
    return {
        "generatedBy": "scripts/generate_python_api_reference.py",
        "modules": modules,
    }


def render() -> str:
    return json.dumps(build(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if the output is stale")
    args = parser.parse_args()
    text = render()
    if args.check:
        existing = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if existing != text:
            print(
                "python API reference is stale; rebuild the extension and run "
                "scripts/generate_python_api_reference.py"
            )
            return 1
        print("python API reference up to date")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
