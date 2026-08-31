---
title: Local Build Setup
description: Build the Rust core, run its tests, and run the documentation quality gates.
status: reviewed
last_validated: '2026-08-30'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 2
---

This is the maintainer path: compile the Rust core and run the gates CI
runs. If you are here to *use* OpenQuant rather than to change it, the
[Quickstart](/quickstart/) is shorter and ends with a result.

## Build and test the Rust core

```bash
cargo build --all-features
cargo test --workspace --lib --tests --all-features -- --skip test_sadf_test
```

The `--skip` is deliberate. `test_sadf_test` is annotated `#[ignore]` in
`crates/openquant/tests/structural_breaks.rs` as a "long-running
hotspot", so it does not run by default; the explicit skip keeps the
command honest if the annotation is ever removed. Run it on its own when
you have touched `structural_breaks`:

```bash
cargo test -p openquant --test structural_breaks test_sadf_test -- --ignored
```

Lint and format, mirroring the `lint` recipe in the `justfile`:

```bash
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D clippy::correctness -D clippy::suspicious
```

## Documentation quality gates

```bash
cd docs-site
npm install
npx astro build
node scripts/check-links.mjs
python3 ../scripts/generate_api_inventory.py --check
node scripts/check-content-schema.mjs
```

`package.json` wraps these as `bun run check:docs`, and the deploy
workflow uses Bun — but nothing here needs Bun as a runtime, and the npm
form above is what these commands were last verified with. If you have
Bun, `bun run check:docs` runs all four in order.

Expected checkpoints:

- `astro build` completes and reports the page count.
- `check:links` returns zero broken internal links.
- `check:api-drift` is clean.
- `check:content-schema` prints a per-status tally of all docs files.

## When a gate fails

**`check:links` reports broken links.** Run `astro build` first. The link
checker reads the built output in `dist/`, not the Markdown sources, so a
stale or absent `dist/` produces phantom failures. Internal links must
resolve under the `/openquant` base path once built — a link that works
in `astro dev` can still 404 in production if it omits the base.

**`check:api-drift` fails.** The inventory under `scripts/` no longer
matches the code. If you changed a public API on purpose, regenerate the
inventory from the repository root and commit the result; if you did not,
you have changed a public API by accident and the gate is doing its job.

**`check:content-schema` fails.** The message names the file and the rule.
The taxonomy is defined at the top of
`docs-site/scripts/check-content-schema.mjs`:

| `status` | Means | Date field it must carry |
|---|---|---|
| `generated` | emitted from `src/data/moduleDocs.ts`; nobody has read it | `last_generated` |
| `draft` | hand-written, known incomplete; claims nothing | **none** |
| `reviewed` | a human read the page end to end | `last_validated` |
| `validated` | reviewed *and* checked against the code it describes | `last_validated` |

Two rules catch most failures. A `draft` may not carry `last_validated` —
drop the field or raise the status. And a stamp may not predate the
page's last content change: if you edited a page, either re-read it and
bump the date, or lower the status to `draft`. The gate reads the change
date from `git log` plus the working-tree mtime, so an uncommitted edit
counts.

The coloured pill a reader sees at the top of every page is rendered from
`status` by the `Banner` override in `src/components/DocStatusBanner.astro`,
so there is nothing to keep in step by hand.

## Next

- [Python Bindings Setup](/setup/python-bindings/) — the PyO3 extension
- [Troubleshooting](/setup/troubleshooting/) — compile and link failures
