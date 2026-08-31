---
title: Coverage Dashboard
description: What is documented, what is not, and the commands that produce those numbers.
status: draft
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

Numbers below were measured on **2026-08-30**.

## Documentation status

```bash
cd docs-site && node scripts/check-content-schema.mjs
```

That gate runs in CI and prints a tally of every page by `status`. As of
the date above:

| `status` | Pages | Means |
|---|---|---|
| `generated` | 40 | Emitted from `src/data/moduleDocs.ts`. Nobody has read it. |
| `draft` | 8 | Hand-written, known incomplete. Claims nothing. |
| `reviewed` | 11 | A human read the page end to end. |
| `validated` | 0 | Reviewed *and* checked against the code. Nothing has earned this yet. |
| **Total** | **59** | |

The headline number is the last row of that table: **no page on this site
is `validated`.** Until recently every page claimed `status: validated`
with an identical `last_validated` date, because the value was a string
literal in the page generator rather than a record of anyone's review.
The taxonomy in `docs-site/scripts/check-content-schema.mjs` now refuses
a stamp that predates the page's last content change, so the tally above
cannot be bulk-applied.

## Module documentation depth

```bash
cd docs-site
grep -c '^    module:' src/data/moduleDocs.ts          # documented modules
grep -c '^    conceptOverview:' src/data/moduleDocs.ts # of those, the enriched tier
```

| | Count |
|---|---|
| Modules with a documentation entry | 39 |
| …of which get the **enriched** template (`conceptOverview`, `whenToUse`, `keyParameters`, `commonPitfalls`, `relatedModules`) | 12 |
| …of which fall through to the **stub** template (`Subject` + one sentence per heading) | 27 |

The stub tier is 69% of the module pages. Those pages have a heading
skeleton, one sentence under each heading, and a closing "Implementation
Notes" section that is a verbatim reprint of the page's own `risk_notes`
frontmatter. Depth on those pages is a generator problem, not a per-page
problem: adding `conceptOverview` to an entry in `moduleDocs.ts` moves it
to the enriched tier.

## Code with no documentation entry at all

```bash
docd=$(grep -o 'module: "[^"]*"' docs-site/src/data/moduleDocs.ts | sed 's/module: "//;s/"//' | sort)
comm -13 <(echo "$docd") <(grep "^pub mod" crates/openquant/src/lib.rs | sed 's/pub mod //;s/;//' | sort)
comm -13 <(echo "$docd") <(ls python/openquant/*.py | xargs -n1 basename | sed 's/\.py//' | grep -v __init__ | sort)
```

| Surface | Undocumented |
|---|---|
| Rust (`crates/openquant/src/lib.rs`) | `data_processing` |
| Python (`python/openquant/`) | `bars` |

`util` also appears in the raw diff, but it is a parent module whose two
children (`util::fast_ewma`, `util::volatility`) both have pages; it is
not a gap.

`openquant.bars` is the sharper miss of the two. It has no module page
even though [Python Core Workflow](/workflows/python-core-workflow/)
makes it a whole stage, and its memory behaviour under large inputs is
non-obvious enough to have earned an entry in
[Troubleshooting](/setup/troubleshooting/#python-runs-out-of-memory-building-bars-from-a-large-dataset).

## What this page does not measure

Honestly: most of what you would want.

- **Prose quality.** `status` records whether a human read a page, not
  whether the page is good.
- **Whether the examples run.** Nothing in CI executes the code blocks on
  these pages. The commands on the setup pages were executed by hand; the
  module pages' snippets were not, and several of them do not compile.
- **API parity between Rust and Python.** `check:api-drift`
  (`scripts/generate_api_inventory.py --check`) tracks the *inventory* of
  public symbols, not whether both surfaces are documented equivalently.
  That gate is **currently failing** on this branch.

:::note[Why this page is still `draft`]
Three of the four tables above are hand-transcribed from commands run
once, on one day. That is better than the prose bullets this page used to
carry, and every figure can be re-derived in seconds — but it is still a
snapshot a human has to refresh, which is exactly the failure mode the
page warns about in its first paragraph.

The fix is to emit this page from the gate that already computes the
tally. `check-content-schema.mjs` builds the status counts on every run
and throws them away after printing. Until it writes them out instead,
this page stays `draft`.
:::
