---
title: Coverage Dashboard
description: What is documented, what is not, and the commands that produce those numbers.
status: authored
last_authored: '2026-10-10'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 90
---

Every number on this page comes with the command that produced it. Run the
command; if the answer differs, the page is out of date and the command is
right. A coverage page maintained any other way tells you what someone
hoped was true on the day they wrote it.

That used to be an aspiration. The three tables below are now generated:

```bash
# verify — this is the step CI runs
cd docs-site && bun run check:coverage
# regenerate after a real change (path relative to the repo root)
python3 scripts/docs/check_coverage.py --write
```

The gate recomputes each table from the tree and fails if this page disagrees
with it, so the numbers cannot quietly rot again. Everything outside the
generated regions is prose a human owns.

<!-- coverage:begin:measured -->
Numbers below were regenerated on **2026-10-10**.
<!-- coverage:end:measured -->

## Documentation status

`status` is enforced by a second gate, `check:content-schema`, which also
refuses a review stamp that predates the page's last content change — so the
tally below cannot be bulk-applied.

<!-- coverage:begin:status-tally -->
| `status` | Pages | Means |
|---|---|---|
| `generated` | 4 | Emitted by a script from its source (`moduleDocs.ts`, the notebooks, `benchmarks/`, `CHANGELOG.md`). Nobody has read it. |
| `draft` | 0 | Hand-written, known incomplete. Claims nothing. |
| `authored` | 59 | Hand-written and complete, examples executed by the docs gates. No human has read it. |
| `reviewed` | 3 | A human read the page end to end. |
| `validated` | 0 | Reviewed *and* checked against the code. |
| **Total** | **66** | |
<!-- coverage:end:status-tally -->

The headline number is the last row of that table: **no page on this site
is `validated`.** Until recently every page claimed `status: validated`
with an identical `last_validated` date, because the value was a string
literal in the page generator rather than a record of anyone's review.

## Module documentation depth

Module pages began as output of `src/data/moduleDocs.ts` and are being replaced, one AFML
chapter at a time, by pages written by hand: each cites its source, has only the sections it
needs, and runs its examples. For a module with Python bindings that is a Python block whose
printed output the docs checks compare with the page; for a Rust-only module it is a Rust
snippet that also runs as a test under `crates/openquant/tests/docs_*`. The first
row below counts those. All 42 module pages are now authored Markdown; `moduleDocs.ts`
contains only navigation metadata. Authored status records executed examples and sourced
content, not human review or validation.

<!-- coverage:begin:module-depth -->
| | Count |
|---|---|
| Modules with a documentation page | 42 |
| …written by hand, with a citation and executed examples | 42 |
| …generated, carrying the full template (**Key Parameters** and **Common Pitfalls** on top of the base sections) | 0 |
| …generated, carrying the base template only | 0 |
<!-- coverage:end:module-depth -->

Future module changes belong in the page itself. Update the navigation metadata only
when the module's name, summary, subject or API surface changes.

## Code with no documentation entry at all

The gate derives this from `pub mod` in `crates/openquant/src/lib.rs` and the
modules in `python/openquant/`, against the `module:` frontmatter of the pages
under `src/content/docs/modules/`. A gap fails the gate unless
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

The examples are covered, though, and by execution rather than by counting:
`check:examples` compiles every documented Rust snippet against the real
crate and `check:python-examples` runs every documented Python block against
the built extension. Both are in the `docs-checks` CI job alongside this page's
gate.
