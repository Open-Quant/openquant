#!/usr/bin/env python3
"""Recompute documentation presence/status counts and fail when their tables drift.

The dashboard counts discovered modules, declared page statuses and selected
headings. It does not execute examples, assess citations or verify review claims.

  --check (CI)  compares the marked table regions with the current tree; date-only
                drift in the informational measurement stamp is ignored.
  --write       rewrites owned regions and the measured-on date when coverage and
                marker checks pass; it does not update authorship/review stamps.

Regions are delimited by ``<!-- coverage:begin:NAME -->`` and matching end
markers. Surrounding prose is maintained separately and is not rewritten.
Undocumented public modules fail unless ``coverage_allowlist.toml`` records a
reason and an unexpired date; stale or invalid allowlist entries also fail.
"""

from __future__ import annotations

import argparse
import re
import tomllib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = ROOT / "docs-site" / "src" / "content" / "docs"
MODULE_PAGES = DOCS_ROOT / "modules"
COVERAGE_PAGE = DOCS_ROOT / "coverage.md"
LIB_RS = ROOT / "crates" / "openquant" / "src" / "lib.rs"
PY_SRC = ROOT / "python" / "openquant"
ALLOWLIST = ROOT / "docs-site" / "coverage_allowlist.toml"

# Same taxonomy, same order, as docs-site/scripts/check-content-schema.mjs.
STATUS_ORDER = ("generated", "draft", "authored", "reviewed", "validated")
STATUS_MEANS = {
    "generated": "Emitted by a script from its declared source; no human review claimed.",
    "draft": "Hand-written, known incomplete. Claims nothing.",
    "authored": "Written against the code; does not establish executed examples or human review.",
    "reviewed": "A human read the page end to end.",
    "validated": "Reviewed *and* checked against the code.",
}

# The sections the module page generator emits for every module, versus the two
# it emits only for entries that fill in `keyParameters` / `commonPitfalls`.
BASE_SECTIONS = ("Concept Overview", "When to Use", "Related Modules")
FULL_SECTIONS = ("Key Parameters", "Common Pitfalls")

PUB_MOD_RE = re.compile(r"^\s*pub\s+mod\s+([a-zA-Z0-9_]+)\s*;\s*$", re.MULTILINE)
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
SCALAR_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")


def unquote(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
        return v[1:-1]
    return v


def frontmatter(path: Path) -> dict[str, str]:
    """Top-level scalar keys only — status and explicit module claims."""
    match = FRONTMATTER_RE.match(path.read_text(encoding="utf-8"))
    if not match:
        return {}
    out: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.startswith((" ", "\t", "#")):
            continue
        kv = SCALAR_RE.match(line)
        if kv and kv.group(2).strip():
            out[kv.group(1)] = unquote(kv.group(2))
    return out


def doc_pages() -> list[Path]:
    return sorted(p for p in DOCS_ROOT.rglob("*") if p.suffix in {".md", ".mdx"})


def status_tally() -> dict[str, int]:
    tally = dict.fromkeys(STATUS_ORDER, 0)
    for page in doc_pages():
        status = frontmatter(page).get("status")
        if status in tally:
            tally[status] += 1
    return tally


def documented_modules(
    surfaces: dict[str, list[str]],
) -> tuple[dict[str, set[str]], list[str]]:
    """Primary module claims plus validated, Python-only wrapper module claims.

    A page can document a Rust module and a differently named Python wrapper
    (for example, data_structures and bars). Only explicit `python_module`
    frontmatter claims that wrapper; API references and prose are not claims.
    """
    names: dict[str, set[str]] = {surface: set() for surface in surfaces}
    failures: list[str] = []
    for page in sorted(MODULE_PAGES.glob("*.md")):
        meta = frontmatter(page)
        module = meta.get("module")
        if module:
            for claimed in names.values():
                claimed.add(module)
            # A Rust child page documents its parent namespace, not a Python module.
            if "::" in module:
                names["rust"].add(module.split("::", 1)[0])
        python_module = meta.get("python_module")
        if python_module:
            if python_module not in surfaces["python"]:
                failures.append(
                    f"{page.relative_to(ROOT)}: python_module `{python_module}` "
                    "is not a public Python module"
                )
            else:
                names["python"].add(python_module)
    return names, failures


def module_depth() -> tuple[int, int, int]:
    """Counts: generated with both selected headings, other generated, non-generated.

    Non-generated status is a classification, not evidence of authorship quality,
    citations or executed examples. Generated pages are grouped by FULL_SECTIONS only.
    """
    full = base = handwritten = 0
    for page in sorted(MODULE_PAGES.glob("*.md")):
        meta = frontmatter(page)
        if not meta.get("module"):
            continue  # the index page
        if meta.get("status") != "generated":
            handwritten += 1
            continue
        text = page.read_text(encoding="utf-8")
        headings = {h for h in BASE_SECTIONS + FULL_SECTIONS if f"\n## {h}\n" in text}
        if all(h in headings for h in FULL_SECTIONS):
            full += 1
        else:
            base += 1
    return full, base, handwritten


def public_surfaces() -> dict[str, list[str]]:
    rust = sorted(set(PUB_MOD_RE.findall(LIB_RS.read_text(encoding="utf-8"))))
    python = sorted(p.stem for p in PY_SRC.glob("*.py") if p.name != "__init__.py")
    return {"rust": rust, "python": python}


def load_allowlist() -> dict[str, dict[str, dict[str, str]]]:
    if not ALLOWLIST.exists():
        return {}
    with ALLOWLIST.open("rb") as fh:
        return tomllib.load(fh)


def gaps(today: date) -> tuple[list[tuple[str, str, str, str]], list[str]]:
    """Undocumented public modules, plus any problems with their exemptions.

    Returns the rows for the gap table — (surface, module, expires, reason) —
    and a list of failures. A gap without a live exemption is a failure; so is
    an exemption for a module that is in fact documented, because a stale
    exemption is how an allowlist stops meaning anything.
    """
    surfaces = public_surfaces()
    documented, failures = documented_modules(surfaces)
    allow = load_allowlist()
    rows: list[tuple[str, str, str, str]] = []

    labels = {
        "rust": "Rust (`crates/openquant/src/lib.rs`)",
        "python": "Python (`python/openquant/`)",
    }
    for surface, modules in surfaces.items():
        entries = allow.get(surface, {})
        for module in modules:
            if module in documented[surface]:
                if module in entries:
                    failures.append(
                        f"{surface}.{module} is exempted in "
                        f"{ALLOWLIST.relative_to(ROOT)} but is documented — drop the entry"
                    )
                continue
            entry = entries.get(module)
            if entry is None:
                failures.append(
                    f"public {surface} module `{module}` has no module page and no "
                    f"entry in {ALLOWLIST.relative_to(ROOT)}"
                )
                continue
            reason = str(entry.get("reason", "")).strip()
            expires = entry.get("expires")
            if not reason:
                failures.append(f"{surface}.{module}: allowlist entry needs a `reason`")
            if not isinstance(expires, date):
                failures.append(
                    f"{surface}.{module}: allowlist entry needs `expires` as a bare "
                    f"YYYY-MM-DD date (got {expires!r})"
                )
            elif expires < today:
                failures.append(
                    f"{surface}.{module}: exemption expired on {expires} — document "
                    f"the module or justify a new date"
                )
            rows.append(
                (
                    labels[surface],
                    module,
                    expires.isoformat() if isinstance(expires, date) else str(expires),
                    reason or "—",
                )
            )
    return rows, failures


def render_status_tally(tally: dict[str, int]) -> str:
    lines = ["| `status` | Pages | Means |", "|---|---|---|"]
    for status in STATUS_ORDER:
        lines.append(f"| `{status}` | {tally[status]} | {STATUS_MEANS[status]} |")
    lines.append(f"| **Total** | **{sum(tally.values())}** | |")
    return "\n".join(lines)


def render_module_depth(full: int, base: int, handwritten: int) -> str:
    return "\n".join(
        [
            "| | Count |",
            "|---|---|",
            f"| Modules with a documentation page | {full + base + handwritten} |",
            f"| …not marked generated (status classification only) | {handwritten} |",
            f"| …generated, containing both **Key Parameters** and **Common Pitfalls**"
            f" headings | {full} |",
            f"| …generated, missing at least one of those two headings | {base} |",
        ]
    )


def render_gaps(rows: list[tuple[str, str, str, str]]) -> str:
    if not rows:
        return "Every public module on both surfaces has a documentation page."
    lines = ["| Surface | Undocumented | Exempt until | Why |", "|---|---|---|---|"]
    for surface, module, expires, reason in rows:
        lines.append(f"| {surface} | `{module}` | {expires} | {reason} |")
    return "\n".join(lines)


def region_re(name: str) -> re.Pattern[str]:
    return re.compile(
        rf"(<!-- coverage:begin:{re.escape(name)} -->\n)(.*?)(\n<!-- coverage:end:{re.escape(name)} -->)",
        re.DOTALL,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail if the page has drifted (default)")
    mode.add_argument("--write", action="store_true", help="rewrite the page's generated regions")
    args = parser.parse_args()

    today = date.today()
    tally = status_tally()
    full, base, handwritten = module_depth()
    rows, failures = gaps(today)

    regions = {
        "status-tally": render_status_tally(tally),
        "module-depth": render_module_depth(full, base, handwritten),
        "gaps": render_gaps(rows),
        "measured": f"Numbers below were regenerated on **{today.isoformat()}**.",
    }

    page = COVERAGE_PAGE.read_text(encoding="utf-8")
    drifted: list[str] = []
    for name, want in regions.items():
        pattern = region_re(name)
        match = pattern.search(page)
        if not match:
            failures.append(
                f"{COVERAGE_PAGE.relative_to(ROOT)} is missing the "
                f"`<!-- coverage:begin:{name} -->` region"
            )
            continue
        if match.group(2) != want:
            drifted.append(name)

        # `--check` must not rewrite the file; building the new text is cheap
        # and lets one code path serve both modes.
        def replace_region(match: re.Match[str], replacement: str = want) -> str:
            return match.group(1) + replacement + match.group(3)

        page = pattern.sub(replace_region, page, count=1)

    # The measured-on date moves every day, so on --check it is informational:
    # a gate that fails because a week went by teaches people to disable it.
    drifted = [name for name in drifted if name != "measured"]

    if args.write:
        if failures:
            for failure in failures:
                print(f"- {failure}")
            print("\nRefusing to write: fix the coverage gaps above first.")
            return 1
        COVERAGE_PAGE.write_text(page, encoding="utf-8")
        print(f"wrote {COVERAGE_PAGE.relative_to(ROOT)}")
        return 0

    print(
        f"docs pages: {sum(tally.values())} "
        f"({', '.join(f'{tally[s]} {s}' for s in STATUS_ORDER if tally[s])})"
    )
    print(
        f"module pages: {full + base + handwritten} "
        f"({handwritten} non-generated, {full} generated with both selected headings, "
        f"{base} other generated)"
    )
    print(f"undocumented public modules: {len(rows) or 'none'}")

    if drifted:
        failures.append(
            "coverage.md is stale in: "
            + ", ".join(drifted)
            + ". Run `python3 scripts/docs/check_coverage.py --write`."
        )
    if failures:
        print("\nCoverage check FAILED:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nCoverage check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
