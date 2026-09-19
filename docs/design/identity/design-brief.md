# Design brief: one visual identity for OpenQuant

## Identity and decisions

- Design/work ID and owner: `design-direction` (GitHub #56); owner Sean Koval
- Status: draft
- Canonical product brief and OUT/RQ IDs: `docs/design/production-readiness-brief.md` — OUT-006, RQ-015
- Delivery slice and specification references: `docs/design/production-readiness-slices.md`; no formal spec (visual decision)
- Decision source: proposed choice. The owner's stated problem (2026-09-18): the GitHub Pages docs "look AI sloppy — design, colors, quality".
- Reviewed task decisions, owner and evidence: owner approved the work record for #56 (2026-09-18) and asked for work to start (2026-09-19). No direction has been chosen.
- Unresolved material decisions and next action: which candidate direction to adopt — owner decides from `design-selection.md`.

## Intent and scope

- Audience and primary journey: a quant researcher or Rust/Python engineer who lands on the docs, decides within a minute whether the library is serious, then reads long module pages containing prose, formulas, code and numeric tables.
- Current observed problem and evidence (audit 2026-09-18, screenshots at 1440px/390px, light/dark):
  - Three unrelated palettes: logo neon magenta `#e44de0` / cyan `#52c8f5` / yellow `#f0e050` with a Gaussian-blur glow on a dark rounded tile; README and docs banners a purple-to-blue gradient `#1b1648 → #36257b → #0d64c8` with radial glows; site accent navy `#0f5f9a`.
  - App-landing styling on a reference site: 16px-radius gradient hero card, radial washes on the main pane, `backdrop-filter` blur header, pill CTA, drop-shadow glow behind the logo, 10–12px radii on tables/code/blockquotes.
  - Type: Manrope body with `letter-spacing: 0.01em`; headings request weight 750, which is not loaded. The docs banner names Space Grotesk, never loaded, and reads "Ship Smarter / Astro-native docs".
  - Draft status pill white on `#a8760d` = 3.99:1 (fails AA). Light hairline `#c9d8e8` on `#f3f6fb` = 1.34:1.
  - Body text contrast is already good (14.9:1 light, 15.6:1 dark) and must stay so.
- Intended observable outcome: one palette and type system shared by logo, banner and site, drawn from the domain (a book of financial mathematics, implemented as a systems library), that a reader could not mistake for a generic generated template.
- In scope: direction, tokens (color, type scale, spacing, radii) for light and dark, a flat logo mark, one banner. Non-goals: implementing it in Starlight (#57), page content (#53), information architecture.
- Existing brand/components/navigation to preserve: the three-segment pinwheel mark (shape and its three hues are the org's identity); Starlight's layout, navigation and search; IBM Plex Mono for code (already loaded, already good).
- References and what each establishes: polars / uv / ruff docs — restraint and density suit a systems library; scikit-learn — long reference pages with math need a calm reading column; the AFML book itself — the content is a monograph of formulas and tables; trading terminals — tabular numerals and hairline ledgers are the native visual language of the domain.
- Compatibility, permissions, recovery and accessibility target: Google Fonts only (the site already loads from it); WCAG 2.1 AA contrast for all text; links distinguishable without color; no motion required to read; works without JavaScript.
- States, viewports, fixtures and access limitations: light and dark; 1440px and 390px; one fixed content fixture (header, sidebar, status pill, H1, lede, prose with link and inline code, formula, code block, output block, numeric table, aside) rendered identically by every candidate as a standalone HTML file. Candidates are static mocks, not Starlight builds.

## Evaluation contract before generation

- Rubric path/version: `docs/design/identity/design-rubric.md` v1, criteria B1–B6 and S1–S5, all tracing to RQ-015.
- Required behavior checks, evidence methods and thresholds: see rubric; contrast computed by script from each candidate's declared tokens; layout checked by screenshot at both viewports and themes.
- Candidate identity: file path plus git blob hash at evaluation time.
- Generator/tool choice: hand-written HTML/CSS by Claude; Playwright for screenshots. No paid tool.
- Evaluator: self-review by the generating session — **independence is unavailable**, so ratings are a proposal for the owner, not a verdict.
- Iteration ceiling: three initial candidates plus at most one revision each. Decision owner: Sean Koval.
- Budget status: proposed (no paid resources; bounded by the iteration ceiling).
- Stop/plateau rule: stop when at least one candidate passes every required check and meets threshold 3 on every required rated criterion, or when the ceiling is reached.

## Next handoff

See `design-selection.md` for the evaluated candidates, the proposed selection and the exact files to open.
