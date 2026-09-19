# Design selection: design-direction (#56)

- Canonical brief / RQ / delivery slice: `docs/design/production-readiness-brief.md` · RQ-015 · `design-direction`
- Decision owner and source: Sean Koval — chose **A** on 2026-09-19 ("go with A"), as proposed: A taking B's table treatment, with one further revision to clear findings 1 and 3
- Decision status: reviewed
- Effective rubric ID/version: `oq-identity` v1 (frozen in commit `3179fb6`, before any candidate existed)
- Declared budget, actual use and remaining authorization: 3 candidates + 1 revision each. Used: 3 candidates, 1 revision each (a shared fix to the reading measure and the 390px overflow), then **one owner-authorised revision of A** (r2). No paid resources.
- Evaluator: self-review by the generating session. Independence was unavailable, so the ratings below are a proposal, not a verdict.

## How to look at them

```sh
python3 -m http.server 4477 --directory docs/design/identity/candidates
# http://localhost:4477/a-monograph.html   (append #dark or #light, or use the button)
# http://localhost:4477/b-ledger.html
# http://localhost:4477/c-blueprint.html
```

Rebuild with `python3 docs/design/identity/build_candidates.py`, which also rewrites `contrast.md`.
Screenshots used for the ratings are in `evidence/`.

## Retained candidates

| Candidate / blob | Direction | Accent from logo |
| --- | --- | --- |
| `a-monograph.html` `2b28ddb697` | The book this library implements, set properly: Newsreader serif reading column on warm paper, ruled tables, IBM Plex Sans for navigation, IBM Plex Mono for code. Light first. | magenta → ink `#97196a` / `#f29ad6` |
| `b-ledger.html` `6f976339e8` | A trading terminal's ledger: dark first, IBM Plex Mono headings and numerals, IBM Plex Sans body, full-grid hairline tables, `//` section marks. | yellow → amber `#735a00` / `#e9c84a` |
| `c-blueprint.html` `eae10744a6` | An engineering drawing: IBM Plex Sans Condensed headings, Source Sans 3 body, a faint measured grid behind the title block only. Light first. | cyan → blue `#065b88` / `#62c6f1` |

## Required behavior checks

| Check | A | B | C | Evidence |
| --- | --- | --- | --- | --- |
| B1 every text pair ≥ 4.5:1, both themes | pass (min 6.35) | pass (min 5.62) | pass (min 5.65) | `contrast.md` |
| B2 one accent, from a logo hue | pass | pass | pass | tokens in `build_candidates.py` |
| B3 no page scroll at 390px; code scrolls in its box | pass after revision (failed before) | pass after revision | pass after revision | measured in browser, both themes |
| B4 fonts on Google Fonts, used weights loaded, none banned | pass | pass | pass | `document.fonts` dump |
| B5 no glow / blur / drop-shadow / pill / radius > 4px | pass | pass | pass (grid at 7% opacity, title block only) | grep of built HTML |
| B6 links identifiable without color | pass | pass | pass | screenshots |

## Subjective ratings (0–4, required threshold 3)

| Criterion | A · Monograph | B · Ledger | C · Blueprint |
| --- | --- | --- | --- |
| S1 Reading comfort | **4** — 70 chars/line; serif column, italic lede and quiet rules read like a well-set chapter, including at 390px | **2** — the monospaced formula overflows and scrolls even at 1440px; the mono H1 breaks onto two lines; the status line wraps | **3** — 70 chars/line, clear hierarchy, nothing memorable |
| S2 Domain character | **3** — monograph type, ruled tables, small-cap labels trace to the book; recognisable from a crop | **4** — unmistakably a quant terminal from any fragment | **2** — the grid says "engineering", not finance; it could sit on any developer tool unchanged |
| S3 Restraint | **4** — first thing seen is the title and the sentence under it | **3** — `//` marks and full grid lines are ornament, but light | **3** — the boxed title block is the first thing seen |
| S4 Data legibility | **2** — tabular numerals and strong rules, but the six-column table is **clipped at 1440px** (PSR column cut off) and numeric headers render oversized | **4** — the table fits, aligns, and reads like a statement | **2** — same clipping as A |
| S5 Brand coherence | **3** — flat mark works at 24px; magenta ink visibly relates to it | **3** — amber relates to the mark's yellow | **3** — blue relates to the mark's cyan |
| Meets threshold on all? | **no** (S4) | **no** (S1) | **no** (S2, S4) |

## Decision

- Selected candidate/revision: **none yet.** Proposed: **A · Monograph**, taking B's table treatment.
- Contract outcome: **needs-work.** No candidate meets every required threshold within the declared budget.
- Stop reason: budget-exhausted.
- Rationale: A is the only direction that is excellent at the thing this site is mostly for — reading long pages of prose, formulas and code — and its one failure is mechanical, not a matter of taste. B has the strongest character and by far the best tables, but monospace headings and formulas fight the content on exactly the pages that matter; B's failure is inherent to its premise. C fails on character, which no revision fixes. The two strengths are compatible: a serif reading column with ledger-grade tables is what a printed quantitative monograph actually looks like.
- Residual findings, all observed, none fixed (the revision budget was spent):
  1. **A, C:** six-column table clipped at 1440px. Cause is shared base CSS: numeric *header* cells inherit the larger mono size meant for data cells, and tables are held to the prose measure. Fix: header cells keep the small label size; tables may extend beyond the measure up to the content width.
  2. **B:** monospaced formula overflows at every width; mono H1 wraps.
  3. **All, dark theme:** native scrollbars render white. Needs `color-scheme: dark`.
  4. Candidates are static mocks. Starlight's own chrome (search, TOC, pagination, asides, KaTeX) is unverified until #57.
- Required follow-up / owner / artifacts for next session:
  1. **Owner chooses** a direction — A as proposed, B, C, or a hybrid.
  2. With that choice, the owner authorises one further revision to clear findings 1 and 3, after which the tokens are frozen in `tokens.md` and the banner is drawn in the chosen palette. Both are deliberately not produced yet: they depend on the choice.
  3. #57 then implements the tokens in `docs-site/src/styles/starlight.css`.
- Already direction-independent and delivered: `openquant-mark.svg`, the flat pinwheel (no tile, gradient or glow), which every candidate uses at 24px.

## Revision r2 of A (owner-authorised, 2026-09-19)

Changes: numeric header cells keep the label size (finding 1); `color-scheme` follows the theme (finding 3); B's column hairlines added inside A's heavy top and bottom rules.

Measured in the browser, both themes: at 1440px the table is no longer clipped and the formula does not overflow; at 390px there is no page scroll and the table and formula scroll inside their own boxes (what B3 requires). Evidence: `evidence/a-monograph-r2-light-table.jpeg`, `evidence/a-monograph-r2-dark-table.jpeg`.

Re-rating under rubric v1: **S4 Data legibility 2 → 4** (fits, aligns, reads like a statement; dark scrollbars now dark). S1 4, S2 3, S3 4, S5 3 unchanged. All required checks pass and every rated criterion is ≥ 3.

- Selected candidate/revision: **A · Monograph, r2** (`candidates/a-monograph.html` as committed with this section).
- Contract outcome: **accepted** (self-rated; chosen by the owner).
- Stop reason: contract-satisfied.
- Frozen outputs: `tokens.md`, `openquant-mark.svg`, `openquant-banner.svg`.
- Residual, carried to #57: Starlight chrome unverified against these tokens; banner text is live SVG text. B and C are retained for the record; findings 1 and 3 were fixed in shared CSS, so only B's formula overflow (finding 2) remains in the retained files.
