#!/usr/bin/env python3
"""Release metadata for .github/workflows/release.yml. Standard library only (Python 3.11+).

The PyPI distribution name lives only in pyproject.toml (`[project] name`); the workflow
reads it from here and never spells it out, so renaming the package is a one-line change.

Subcommands:

  metadata [--tag TAG] [--github-output]
      Print the distribution name, the version and the pinned Rust toolchain. Fails if
      pyproject.toml, crates/openquant/Cargo.toml and crates/pyopenquant/Cargo.toml disagree
      on the version, or, with --tag, if the tag is not `v<version>`.

  notes VERSION [--allow-unreleased] [--out FILE]
      Extract the release notes for VERSION from CHANGELOG.md: the body of the `## <version>`
      section (`## 0.1.0 - 2026-10-01`, `## [0.1.0] - 2026-10-01`, `## v0.1.0` all match).
      With --allow-unreleased, fall back to the `## Unreleased` section when there is no
      version section yet (dry runs). A release tag must have its own section.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

VERSIONED_MANIFESTS = (
    Path("crates/openquant/Cargo.toml"),
    Path("crates/pyopenquant/Cargo.toml"),
)


class ReleaseError(Exception):
    """A release precondition failed; the message says which."""


@dataclass(frozen=True)
class Metadata:
    name: str
    version: str
    rust_toolchain: str


def _load_toml(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ReleaseError(f"{path} not found") from exc


def _table(doc: dict[str, object], key: str, path: Path) -> dict[str, object]:
    value = doc.get(key)
    if not isinstance(value, dict):
        raise ReleaseError(f"{path}: no [{key}] table")
    return value


def _string(table: dict[str, object], key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ReleaseError(f"{where}: `{key}` must be a non-empty string")
    return value


def read_metadata(root: Path = ROOT) -> Metadata:
    """Name and version from pyproject.toml, checked against the crate manifests."""
    pyproject = root / "pyproject.toml"
    project = _table(_load_toml(pyproject), "project", pyproject)
    dynamic = project.get("dynamic", [])
    if isinstance(dynamic, list) and "version" in dynamic:
        raise ReleaseError(f"{pyproject}: the version must be static, not dynamic")
    name = _string(project, "name", f"{pyproject} [project]")
    version = _string(project, "version", f"{pyproject} [project]")

    mismatches = []
    for rel in VERSIONED_MANIFESTS:
        manifest = root / rel
        package = _table(_load_toml(manifest), "package", manifest)
        crate_version = _string(package, "version", f"{manifest} [package]")
        if crate_version != version:
            mismatches.append(f"{rel} has {crate_version}")
    if mismatches:
        raise ReleaseError(
            f"pyproject.toml has version {version} but "
            + ", ".join(mismatches)
            + "; bump all three together"
        )

    toolchain_file = root / "rust-toolchain.toml"
    toolchain = _table(_load_toml(toolchain_file), "toolchain", toolchain_file)
    channel = _string(toolchain, "channel", f"{toolchain_file} [toolchain]")
    return Metadata(name=name, version=version, rust_toolchain=channel)


def check_tag(tag: str, version: str) -> None:
    """A release tag must be exactly `v<version>`."""
    tag = tag.removeprefix("refs/tags/")
    if tag != f"v{version}":
        raise ReleaseError(
            f"tag {tag!r} does not match the package version {version} "
            f"(expected 'v{version}'); bump the versions or fix the tag"
        )


_HEADING = re.compile(r"^##(?!#)\s*(.*?)\s*$")


def _heading_version(title: str) -> str | None:
    """`[0.1.0] - 2026-10-01` -> `0.1.0`; None for a heading without a version."""
    m = re.match(r"\[?v?(\d+\.\d+\.\d+[0-9A-Za-z.+-]*?)\]?(?:\s|$)", title)
    return m.group(1) if m else None


def _sections(changelog: str) -> list[tuple[str, str]]:
    """(heading title, body) for every `## ` section, in order."""
    sections: list[tuple[str, list[str]]] = []
    in_fence = False
    for line in changelog.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            sections.append((m.group(1), []))
        elif sections:
            sections[-1][1].append(line)
    return [(title, "\n".join(body).strip("\n")) for title, body in sections]


def extract_notes(changelog: str, version: str, *, allow_unreleased: bool = False) -> str:
    """The CHANGELOG.md section for `version` (or Unreleased, if allowed), without its heading."""
    sections = _sections(changelog)
    for title, body in sections:
        if _heading_version(title) == version:
            if not body.strip():
                raise ReleaseError(f"CHANGELOG.md: the section for {version} is empty")
            return body + "\n"
    if allow_unreleased:
        for title, body in sections:
            if title.strip("[]").lower() == "unreleased":
                if not body.strip():
                    raise ReleaseError("CHANGELOG.md: the Unreleased section is empty")
                return body + "\n"
    raise ReleaseError(
        f"CHANGELOG.md has no `## {version}` section; move the Unreleased entries under "
        f"`## {version} - YYYY-MM-DD` before tagging"
    )


def _write_outputs(values: dict[str, str]) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if not out:
        raise ReleaseError("--github-output given but GITHUB_OUTPUT is not set")
    with open(out, "a", encoding="utf-8") as fh:
        for key, value in values.items():
            fh.write(f"{key}={value}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)

    meta = sub.add_parser("metadata", help="name, version and toolchain; version checks")
    meta.add_argument("--tag", help="release tag (vX.Y.Z or refs/tags/vX.Y.Z) to check")
    meta.add_argument("--github-output", action="store_true", help="append to $GITHUB_OUTPUT")

    notes = sub.add_parser("notes", help="release notes from CHANGELOG.md")
    notes.add_argument("version")
    notes.add_argument("--allow-unreleased", action="store_true")
    notes.add_argument("--changelog", type=Path, default=ROOT / "CHANGELOG.md")
    notes.add_argument("--out", type=Path, help="write here instead of stdout")

    args = parser.parse_args(argv)
    try:
        if args.command == "metadata":
            md = read_metadata()
            if args.tag:
                check_tag(args.tag, md.version)
            values = {"name": md.name, "version": md.version, "rust_toolchain": md.rust_toolchain}
            for key, value in values.items():
                print(f"{key}={value}")
            if args.github_output:
                _write_outputs(values)
        else:
            text = extract_notes(
                args.changelog.read_text(encoding="utf-8"),
                args.version,
                allow_unreleased=args.allow_unreleased,
            )
            if args.out:
                args.out.write_text(text, encoding="utf-8")
            else:
                sys.stdout.write(text)
    except ReleaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
