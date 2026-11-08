---
title: Local Build Setup
description: Build the Rust core, run its tests, and run the documentation quality gates.
status: authored
last_authored: '2026-11-07'
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
2. Run the design/accessibility, agent-index and navigation-checker tests, then
   check built paths, anchors and redirects.
3. Check API drift, page status/schema, theme contrast, coverage and generated pages.
4. Compile Rust snippets on module pages and the complete Rust core workflow,
   execute runnable Python snippets and check the generated Python reference
   against the built extension.

The design/accessibility checks inspect rendered markup and source/style rules. They
do not exercise keyboard, screen-reader, zoom or mobile interactions in a browser.
The aggregate stops at its first failing command; later checks have not run.
Record the failing command and run unaffected checks separately when useful.

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

**`check:api-drift` fails.** Read the error first: the source-derived inventory
may be stale, or the binding documentation checks may have found a missing contract.
The generated file is `docs-site/src/data/apiInventory.ts`; the generator is under
`scripts/`. This is a symbol/source scan, not a Rust/Python behavioral parity test.
Inspect the public API diff before regenerating from the repository root:

```bash
python3 scripts/generate_api_inventory.py --check
# Only after reviewing the intended source changes:
python3 scripts/generate_api_inventory.py
```

**`check:python-reference` fails.** This is a different artifact:
`docs-site/src/data/pythonApiReference.json`. The generator checks compiled function
names and parameter names against the stubs, then combines compiled documentation
with public Python source. Confirm the selected package and extension paths before
rebuilding or regenerating; an old extension or another installed checkout can
produce a misleading result. See [bindings setup](/setup/python-bindings/).

**A Python example fails, times out or prints different output.** The checker writes
one standalone script per fence under `target/doc-python-examples/`, then runs it
from the repository root. Its final failure report names that file; inspect it and
reproduce the same inputs before changing the page. Each runnable fence has its own process,
so a variable or import from an earlier fence is not available.

You can prefer the prepared environment from the repository root:

```bash
DOC_PYTHON="$PWD/.venv/bin/python" node docs-site/scripts/check-doc-python-examples.mjs
```

`DOC_PYTHON` is the first candidate, **not a strict interpreter selection**. If its
`import openquant` probe fails, the gate tries the checkout's `.venv` interpreters,
then `python3`. Read the `via ...` path in the gate's opening summary. When a local
native extension is present under `python/openquant/`, the checker prepends the
checkout's Python source to `PYTHONPATH`; otherwise an installed package can satisfy
the import. Confirm the actual package and extension paths before treating a pass
as evidence for this checkout.

| Report | Meaning and next check |
| --- | --- |
| `FAIL` | The process exited unsuccessfully. Read the traceback and the saved script; verify package paths, fixture paths and the API contract. |
| `TIMEOUT` | A runnable block exceeded the default 180,000 ms limit, or terminated with SIGTERM. Check build mode and workload before setting `DOC_PYTHON_TIMEOUT_MS` to a larger positive millisecond value. |
| `STALE` | The process succeeded, but its stdout differs from the following eligible `text` fence after normalizing line endings, trailing spaces and outer whitespace. Check the expected numbers and their conventions; the comparison does not use a numerical tolerance. |
| `skip` | The fence received a syntax check only. Its imports, calls and displayed results were not executed. |

A Python fence's output can be matched to a later `text` fence before the next Python
fence, even with prose between them. Use `doc-output=skip` only for text that is not
program output; use the documented per-fence execution skip for examples that need
unavailable caller data. Skipping a block is a coverage exception to report, not a
fix for a runnable example that fails.

**Coverage or generated pages are stale.** Run the owning generator from the
repository root, then review its diff:

```bash
python3 scripts/docs/check_coverage.py --check
python3 scripts/docs/generate_site_pages.py --check
# After the underlying documentation/notebook/source change is ready:
python3 scripts/docs/check_coverage.py --write
python3 scripts/docs/generate_site_pages.py --write
```

Coverage counts module claims and page classifications; it does not establish
citation quality, numerical correctness or example execution. Generated-gallery
updates do not rewrite authored runbook pages. Update those separately when the
canonical notebook's interpretation changes.

**`check:content-schema` fails.** The message names the file and the rule.
The taxonomy is defined at the top of
`docs-site/scripts/check-content-schema.mjs`:

| `status` | Means | Date field it must carry |
|---|---|---|
| `generated` | emitted from the page's declared source; claims no human review | `last_generated` |
| `draft` | hand-written, known incomplete; claims nothing | **none** |
| `authored` | written against the code, not yet read by a human; not a test receipt | `last_authored` |
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
