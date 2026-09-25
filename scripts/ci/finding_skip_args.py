#!/usr/bin/env python3
"""Print `--skip <test>` arguments for tests ignored with a `FINDING:` reason.

A `#[ignore = "FINDING: ..."]` test records a known divergence from the AFML
reference that has not been fixed yet: it is expected to fail. The nightly
sweep runs every other ignored test (`-- --include-ignored`) and passes these
arguments so the known findings do not mask a new failure. When a finding is
fixed, its `#[ignore]` is removed and the test joins the normal suite.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

IGNORE_RE = re.compile(r'#\[ignore\s*=\s*"FINDING:')
FN_RE = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)")


def finding_tests(root: Path) -> list[str]:
    names: list[str] = []
    for path in sorted(root.rglob("*.rs")):
        if "target" in path.parts:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not IGNORE_RE.search(line):
                continue
            for follow in lines[i + 1 : i + 10]:
                match = FN_RE.match(follow)
                if match:
                    names.append(match.group(1))
                    break
            else:
                sys.exit(f"{path}:{i + 1}: FINDING ignore without a following fn")
    return names


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("crates")
    names = finding_tests(root)
    print(" ".join(f"--skip {name}" for name in names))
    for name in names:
        print(f"skipping known finding: {name}", file=sys.stderr)


if __name__ == "__main__":
    main()
