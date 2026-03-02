#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
RUST_SRC = ROOT / "crates" / "openquant" / "src"
PY_SRC = ROOT / "python" / "openquant"
OUT = ROOT / "docs-site" / "src" / "data" / "apiInventory.ts"

RUST_FN_RE = re.compile(r"^pub\s+fn\s+([a-zA-Z0-9_]+)\s*\(")
PY_FN_RE = re.compile(r"^def\s+([a-zA-Z0-9_]+)\s*\(")


def scan_rust() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for path in sorted(RUST_SRC.glob("*.rs")):
        if path.name in {"lib.rs"}:
            continue
        module = path.stem
        fns: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            m = RUST_FN_RE.match(line.strip())
            if not m:
                continue
            fns.append(m.group(1))
        if fns:
            out[module] = sorted(set(fns))
    return out


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


def as_ts(rust_api: dict[str, list[str]], python_api: dict[str, list[str]]) -> str:
    payload = {
        "generatedAt": "generated-by-scripts/generate_api_inventory.py",
        "rust": rust_api,
        "python": python_api,
    }
    body = json.dumps(payload, indent=2, sort_keys=True)
    return "// Generated file. Do not edit manually.\nexport const apiInventory = " + body + " as const;\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate docs-site API inventory snapshot")
    parser.add_argument("--check", action="store_true", help="Fail if generated output differs")
    args = parser.parse_args()

    rust_api = scan_rust()
    python_api = scan_python()
    ts = as_ts(rust_api, python_api)

    if args.check:
        existing = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if existing != ts:
            print("api inventory is stale; run scripts/generate_api_inventory.py")
            return 1
        print("api inventory up to date")
        return 0

    OUT.write_text(ts, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
