#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyopenquant_bindings import UNDOCUMENTED_ALLOWLIST, documented_items  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CRATES = ROOT / "crates"
# Every crate whose `src/` tree is part of the public API surface. Scanned
# recursively, so nested modules (e.g. openquant/src/util/*.rs) are included.
RUST_CRATES = ("openquant", "pyopenquant")
PY_SRC = ROOT / "python" / "openquant"
OUT = ROOT / "docs-site" / "src" / "data" / "apiInventory.ts"

RUST_FN_RE = re.compile(r"^pub\s+fn\s+([a-zA-Z0-9_]+)\s*\(")
# Top-level public items (column 0) and the rustdoc page kind each one gets.
RUST_ITEM_RE = re.compile(
    r"^pub\s+(?:async\s+)?(fn|struct|enum|trait|type|const|static)\s+([A-Za-z0-9_]+)"
)
RUSTDOC_KIND = {"const": "constant"}
# `impl Foo`, `impl<T> Foo<T>`; not `impl Trait for Foo` (those methods are not `pub fn`).
RUST_IMPL_RE = re.compile(r"^impl(?:<[^>]*>)?\s+([A-Za-z0-9_]+)(?:<[^>]*>)?\s*\{?\s*$")
RUST_METHOD_RE = re.compile(r"^\s+pub\s+(?:const\s+)?(?:async\s+)?fn\s+([a-zA-Z0-9_]+)")
PY_FN_RE = re.compile(r"^def\s+([a-zA-Z0-9_]+)\s*\(")


def rust_module_key(crate: str, src_root: Path, path: Path) -> str:
    """Rust-style module path, e.g. ``openquant::util::volatility``.

    Keying by crate + path (not by ``path.stem``) is required now that the scan
    recurses and covers more than one crate: several module names, such as
    ``filters`` and ``volatility``, exist in both crates and would otherwise
    silently overwrite each other.
    """
    parts = path.relative_to(src_root).with_suffix("").parts
    return "::".join((crate,) + parts)


def scan_rust() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for crate in RUST_CRATES:
        src_root = CRATES / crate / "src"
        for path in sorted(src_root.rglob("*.rs")):
            # Crate/module roots only re-export; they declare no API of their own.
            if path.name in {"lib.rs", "mod.rs"}:
                continue
            module = rust_module_key(crate, src_root, path)
            fns: list[str] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                m = RUST_FN_RE.match(line.strip())
                if not m:
                    continue
                fns.append(m.group(1))
            if fns:
                if module in out:
                    raise SystemExit(f"duplicate rust module key: {module}")
                out[module] = sorted(set(fns))
    return out


def public_rust_modules() -> set[str]:
    """Module paths rustdoc documents: every `pub mod` reachable from the crate root."""
    src = CRATES / "openquant" / "src"
    out: set[str] = set()

    def walk(prefix: str, file: Path, directory: Path) -> None:
        for line in file.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^pub\s+mod\s+([A-Za-z0-9_]+)\s*;", line)
            if not m:
                continue
            name = m.group(1)
            path = f"{prefix}::{name}"
            out.add(path)
            sub_dir = directory / name
            if (sub_dir / "mod.rs").exists():
                walk(path, sub_dir / "mod.rs", sub_dir)
            elif (directory / f"{name}.rs").exists() and sub_dir.is_dir():
                walk(path, directory / f"{name}.rs", sub_dir)

    walk("openquant", src / "lib.rs", src)
    return out


def scan_rust_items() -> dict[str, dict[str, str]]:
    """Public items of the openquant crate, keyed by module, with their rustdoc kind.

    The docs site turns each entry into a link to its rustdoc page
    (`<module path>/<kind>.<name>.html`, or `#method.<name>` on the type's page).
    """
    src_root = CRATES / "openquant" / "src"
    public = public_rust_modules()
    out: dict[str, dict[str, str]] = {}
    for path in sorted(src_root.rglob("*.rs")):
        if path.name in {"lib.rs", "mod.rs"}:
            continue
        module = rust_module_key("openquant", src_root, path)
        if module not in public:
            continue
        items: dict[str, str] = {}
        impl_of: str | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#[cfg(test)]"):
                break
            m = RUST_ITEM_RE.match(line)
            if m:
                items[m.group(2)] = RUSTDOC_KIND.get(m.group(1), m.group(1))
                continue
            m = RUST_IMPL_RE.match(line)
            if m:
                impl_of = m.group(1)
                continue
            if line.startswith("}"):
                impl_of = None
                continue
            m = RUST_METHOD_RE.match(line)
            if m and impl_of:
                items[f"{impl_of}::{m.group(1)}"] = "method"
        if items:
            out[module] = dict(sorted(items.items()))
    return out


def check_binding_docstrings() -> list[str]:
    """Every bound Python function, class, method and property needs a `///` doc comment.

    The comment becomes `__doc__`; a class's also documents its constructor. Modules in
    UNDOCUMENTED_ALLOWLIST are exempt, and the allowlist may only shrink: an entry whose
    module is now fully documented, or that names no binding module, is an error too.
    """
    items = documented_items()
    modules = {path.stem for path, _, _, _ in items}
    errors: list[str] = []
    missing_by_module: dict[str, list[str]] = {}
    for path, line, name, doc in items:
        if not doc.strip():
            missing_by_module.setdefault(path.stem, []).append(
                f"{path.relative_to(ROOT)}:{line}: {name}"
            )
    for stem, entries in sorted(missing_by_module.items()):
        if stem in UNDOCUMENTED_ALLOWLIST:
            continue
        errors.extend(f"undocumented binding: {e}" for e in entries)
    for stem in sorted(UNDOCUMENTED_ALLOWLIST):
        if stem not in modules:
            errors.append(f"UNDOCUMENTED_ALLOWLIST names {stem!r}, which has no bindings")
        elif stem not in missing_by_module:
            errors.append(
                f"{stem}.rs is fully documented: remove it from UNDOCUMENTED_ALLOWLIST "
                "in scripts/pyopenquant_bindings.py"
            )
    return errors


def scan_python() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for path in sorted(PY_SRC.glob("*.py")):
        if path.name == "__init__.py":
            continue
        module = path.stem
        fns: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            m = PY_FN_RE.match(line.strip())
            if not m:
                continue
            name = m.group(1)
            if name.startswith("_"):
                continue
            fns.append(name)
        if fns:
            out[module] = sorted(set(fns))
    return out


def as_ts(
    rust_api: dict[str, list[str]],
    python_api: dict[str, list[str]],
    rust_items: dict[str, dict[str, str]],
) -> str:
    payload = {
        "generatedAt": "generated-by-scripts/generate_api_inventory.py",
        "rust": rust_api,
        "rustItems": rust_items,
        "python": python_api,
    }
    body = json.dumps(payload, indent=2, sort_keys=True)
    return (
        "// Generated file. Do not edit manually.\nexport const apiInventory = "
        + body
        + " as const;\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate docs-site API inventory snapshot")
    parser.add_argument("--check", action="store_true", help="Fail if generated output differs")
    args = parser.parse_args()

    rust_api = scan_rust()
    python_api = scan_python()
    ts = as_ts(rust_api, python_api, scan_rust_items())

    if args.check:
        failed = False
        existing = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if existing != ts:
            print("api inventory is stale; run scripts/generate_api_inventory.py")
            failed = True
        doc_errors = check_binding_docstrings()
        for err in doc_errors:
            print(err)
        if doc_errors:
            print(
                "every #[pyfunction] needs a /// doc comment (it is the Python __doc__); "
                "see scripts/pyopenquant_bindings.py"
            )
            failed = True
        if failed:
            return 1
        print("api inventory up to date; every binding outside the allowlist has a docstring")
        return 0

    OUT.write_text(ts, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
