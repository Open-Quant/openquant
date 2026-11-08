---
title: Coverage Dashboard
description: What is documented, what is not, and the commands that produce those numbers.
status: authored
last_authored: '2026-11-07'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 90
---

The tables below are generated from source-file discovery, frontmatter and selected
headings. They measure documentation presence and declared status, not correctness
or executed-example coverage.

Run both commands from the repository root:

```bash
python3 scripts/docs/check_coverage.py --check
# After an intentional documentation change:
python3 scripts/docs/check_coverage.py --write
```

From `docs-site/`, `bun run check:coverage` runs the same check. The generator owns
only the marked regions; edit the surrounding prose directly. A stale-table error
means the recorded tables differ from the current tree, not that the new content
has passed review. `--write` refreshes the informational measurement date inside
the page; `--check` ignores date-only drift. It does not update `last_authored` or
provide a new review stamp.

<!-- coverage:begin:measured -->
Numbers below were regenerated on **2026-11-07**.
<!-- coverage:end:measured -->

## Documentation status

`check:content-schema` validates allowed status values and date freshness. It
cannot establish that the claimed review happened; the tally counts declarations,
not independently verified review evidence.

<!-- coverage:begin:status-tally -->
| `status` | Pages | Means |
|---|---|---|
| `generated` | 4 | Emitted by a script from its declared source; no human review claimed. |
| `draft` | 0 | Hand-written, known incomplete. Claims nothing. |
| `authored` | 71 | Written against the code; does not establish executed examples or human review. |
| `reviewed` | 1 | A human read the page end to end. |
| `validated` | 0 | Reviewed *and* checked against the code. |
| **Total** | **76** | |
<!-- coverage:end:status-tally -->

The headline number is the last row of that table: **no page on this site
is `validated`.** Until recently every page claimed `status: validated`
with an identical `last_validated` date, because the value was a string
literal in the page generator rather than a record of anyone's review.

## Module documentation depth

Module pages began as output of `src/data/moduleDocs.ts`; that file now supplies
navigation metadata. The table classifies pages carrying `module:` frontmatter:
a status other than `generated` counts in the second row. For generated pages,
the checker looks for the `Key Parameters` and `Common Pitfalls` headings to
separate the two template categories. It does not inspect citations, run examples,
or verify that every base-template section is present.

<!-- coverage:begin:module-depth -->
| | Count |
|---|---|
| Modules with a documentation page | 45 |
| …not marked generated (status classification only) | 45 |
| …generated, containing both **Key Parameters** and **Common Pitfalls** headings | 0 |
| …generated, missing at least one of those two headings | 0 |
<!-- coverage:end:module-depth -->

Future module changes belong in the page itself. Update the navigation metadata only
when the module's name, summary, subject or API surface changes.

## Code with no documentation entry at all

The gate derives this from `pub mod` in `crates/openquant/src/lib.rs` and the
modules in `python/openquant/`, against the `module:` frontmatter of the pages
under `src/content/docs/modules/`, including explicit `python_module:` wrapper
claims. This is module-level coverage, not a per-function or recursively discovered
Rust-item audit. A gap fails the gate unless
`docs-site/coverage_allowlist.toml` carries a reason and an unexpired date for
it.

<!-- coverage:begin:gaps -->
Every public module on both surfaces has a documentation page.
<!-- coverage:end:gaps -->

`util` also appears in the raw module list, but it is a parent namespace: two
of its children (`util::fast_ewma`, `util::volatility`) have pages, and the
third, `util::stats`, holds the descriptive-statistics helpers the other modules
share and is documented in rustdoc only. It is not a gap, and the gate does not
count it as one.

`openquant.bars` is documented on the [`data_structures`](/modules/data-structures/)
page. Its explicit `python_module: bars` claim connects that page to the differently
named Python wrapper; it does not claim coverage for an unrelated Rust module.
The Rust preprocessing interfaces have their own
[`data_processing`](/modules/data-processing/) page, separate from Python ingestion
and caching on [`data`](/modules/data/).

## What this page does not measure

Honestly: most of what you would want.

- **Prose quality.** `status` records whether a human read a page, not
  whether the page is good.
- **Whether the prose is right.** The gates count pages, sections and
  modules. Nothing checks that a sentence describing an algorithm is true.
- **API parity between Rust and Python.** `check:api-drift`
  (`scripts/generate_api_inventory.py --check`) tracks the *inventory* of
  public symbols, not whether both surfaces are documented equivalently.

Example checks are separate from these coverage counts:
`check:examples` compiles the module-guide Rust snippets and the complete Rust core
workflow against the real crate; it does not execute them or check their printed output.
`check:python-examples` executes runnable Markdown/MDX Python fences and syntax-checks
explicitly skipped fragments. Confirm the interpreter and import paths: when a local
extension is absent, the checker can fall back to an installed package. Neither
example gate executes notebook cells or code embedded in API docstrings. Consult the gate output for the current
counts and exclusions. Both checks run in the `docs-checks` CI job alongside this
page's gate.
