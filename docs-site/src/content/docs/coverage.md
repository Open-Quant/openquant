---
title: Coverage Dashboard
description: What is documented, what is not, and the commands that produce those numbers.
status: reviewed
last_validated: '2026-08-30'
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
Numbers below were regenerated on **2026-08-30**.
<!-- coverage:end:measured -->

## Documentation status

`status` is enforced by a second gate, `check:content-schema`, which also
refuses a review stamp that predates the page's last content change — so the
tally below cannot be bulk-applied.

<!-- coverage:begin:status-tally -->
| `status` | Pages | Means |
|---|---|---|
| `generated` | 40 | Emitted from `src/data/moduleDocs.ts`. Nobody has read it. |
| `draft` | 5 | Hand-written, known incomplete. Claims nothing. |
| `reviewed` | 12 | A human read the page end to end. |
| `validated` | 0 | Reviewed *and* checked against the code. |
| **Total** | **57** | |
<!-- coverage:end:status-tally -->

The headline number is the last row of that table: **no page on this site
is `validated`.** Until recently every page claimed `status: validated`
with an identical `last_validated` date, because the value was a string
literal in the page generator rather than a record of anyone's review.

## Module documentation depth

Module pages are emitted from `src/data/moduleDocs.ts`. Every entry now fills
in `conceptOverview`, `whenToUse` and `relatedModules`, so every page carries
those sections; the split below is over the two optional ones,
`keyParameters` and `commonPitfalls`.

<!-- coverage:begin:module-depth -->
| | Count |
|---|---|
| Modules with a documentation page | 39 |
| …carrying the full template (**Key Parameters** and **Common Pitfalls** on top of the base sections) | 10 |
| …carrying the base template only | 29 |
<!-- coverage:end:module-depth -->

A base-template page is not a stub — it has a concept overview, mathematical
foundations, usage examples, an API reference and risk notes. What it lacks is
the parameter table telling you which knobs matter and the pitfalls section
telling you how the module is usually misused. Both are the parts a reader
reaches for second, and both are a `moduleDocs.ts` edit rather than a per-page
rewrite.

## Code with no documentation entry at all

The gate derives this from `pub mod` in `crates/openquant/src/lib.rs` and the
modules in `python/openquant/`, against the `module:` frontmatter of the pages
under `src/content/docs/modules/`. A gap fails the gate unless
`docs-site/coverage_allowlist.toml` carries a reason and an unexpired date for
it.

<!-- coverage:begin:gaps -->
| Surface | Undocumented | Exempt until | Why |
|---|---|---|---|
| Rust (`crates/openquant/src/lib.rs`) | `data_processing` | 2026-12-31 | Internal preprocessing helpers; no stable public surface to document yet. |
| Python (`python/openquant/`) | `bars` | 2026-12-31 | Needs a page — it is a whole stage of the Python Core Workflow. Tracked, not accepted. |
<!-- coverage:end:gaps -->

`util` also appears in the raw module list, but it is a parent namespace whose
two children (`util::fast_ewma`, `util::volatility`) both have pages; it is not
a gap, and the gate does not count it as one.

`openquant.bars` is the sharper miss of the two. It has no module page
even though [Python Core Workflow](/workflows/python-core-workflow/)
makes it a whole stage, and its memory behaviour under large inputs is
non-obvious enough to have earned an entry in
[Troubleshooting](/setup/troubleshooting/#python-runs-out-of-memory-building-bars-from-a-large-dataset).

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
