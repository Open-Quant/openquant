---
title: Local Build Setup
description: Build the Rust core, run its tests, and run the documentation quality gates.
status: authored
last_authored: '2026-11-05'
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
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

## Documentation quality gates

For a local preview only, install the locked site dependencies and start Astro:

```bash
cd docs-site
bun install --frozen-lockfile
bun run dev
```

Without Bun, use `npm ci` and `npm run dev`. The site itself runs on Node;
Bun is the package manager and script runner used by CI.

For the **full documentation gate**, first build the Python extension using
[the contributor setup](https://github.com/Open-Quant/openquant/blob/main/CONTRIBUTING.md#setup),
then run:

```bash
cd docs-site
bun run check:docs
```

This needs the pinned Rust toolchain, Python dependencies and the built extension.
It runs the checks declared in
[`docs-site/package.json`](https://github.com/Open-Quant/openquant/blob/main/docs-site/package.json):

1. Build the site and stage Rust API documentation with `stage:rustdoc`.
2. Check the generated agent index and test the navigation checker, then check
   built paths, anchors and redirects.
3. Check API drift, page status/schema, theme contrast, coverage and generated pages.
4. Compile Rust snippets on module pages, execute runnable Python snippets and
   check the generated Python reference against the built extension.

With npm, invoke those scripts individually with `npm run SCRIPT`; the
aggregate `check:docs` command currently uses `bun run` internally.
For example, a build and link check must include rustdoc staging:

```bash
npm run build
npm run stage:rustdoc
npm run test:links
npm run check:links
```

A preview or a successful Astro build is **not** a full docs pass. If Rust or
the extension is unavailable, record those checks as not run and still run the
independent schema, contrast, API inventory, coverage and generated-page checks.

## When a gate fails

**`check:links` reports broken links.** Run the site build and `stage:rustdoc` first. The link
checker reads the built output in `dist/`, not the Markdown sources, so a
stale or absent `dist/`, or missing staged rustdoc, produces failures.
The checker also verifies fragment anchors and relative links, not just files. Internal links must
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
| `authored` | complete and checked, not yet read by a human | `last_authored` |
| `reviewed` | a human read the page end to end | `last_validated` |
| `validated` | reviewed *and* checked against the code it describes | `last_validated` |

Two rules catch most failures. A `draft` may not carry `last_validated` —
drop the field; do not invent a validation. A stamp may not predate the
page's last content change. AI-authored changes stop at `authored` with a current
`last_authored`; `reviewed` and `validated` require a human. Use `draft` when
the page is incomplete. The gate reads the change
date from `git log` plus the working-tree mtime, so an uncommitted edit
counts.

The coloured pill a reader sees at the top of every page is rendered from
`status` by the `Banner` override in `src/components/DocStatusBanner.astro`,
so there is nothing to keep in step by hand.

## Next

- [Python Bindings Setup](/setup/python-bindings/) — the PyO3 extension
- [Troubleshooting](/setup/troubleshooting/) — compile and link failures
