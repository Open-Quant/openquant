# OpenQuant identity tokens — A · Monograph

Status: **Monograph selected 2026-09-19; refined 2026-11-05.** Current source of truth
for the site. The CSS tokens and `check:contrast` measure the shipped palette;
`build_candidates.py` and `contrast.md` retain the original candidate evidence.

## Color

One accent per theme, derived from the logo's magenta `#e44de0` and shifted in lightness only as AA contrast requires.
The logo's cyan and yellow appear **only in the mark**.

| Token | Light | Dark | Use |
| --- | --- | --- | --- |
| `ground` | `#fbf9f4` | `#17140f` | page |
| `surface` | `#f2eee4` | `#221e17` | code blocks, inline code |
| `text` | `#1d1a16` | `#ece6d8` | body, headings, heavy table rules |
| `muted` | `#5c554b` | `#aaa190` | captions, code comments, status line |
| `accent` | `#97196a` | `#f29ad6` | links, current nav item, eyebrow labels, code keywords, status pill ground |
| `rule` | `#cdc4b1` | `#4c4536` | hairlines (not text: 1.65:1 / 1.93:1) |

Lowest required text contrast: 6.35:1 (muted on surface, light). `color-scheme` follows the theme.

## Type

| Role | Face | Size / leading | Notes |
| --- | --- | --- | --- |
| Body | Newsreader 400, optical size auto | 1.19rem / 1.62 | measure `52ch` ≈ 70 characters per line |
| H1 | Newsreader 600 | 3.1rem / 1.08, tracking −0.015em | 2.1rem below 760px |
| H2 | Newsreader 600 | 1.7rem / 1.2 | hairline above, 1.4rem padding |
| Lede | Newsreader 400 upright | 1.45rem / 1.5 | 1.25rem below 480px |
| Emphasis | Newsreader 600 | inherits body size | loaded weight, not synthesized 700 |
| Navigation and labels | IBM Plex Sans 400/600 | .92rem · sidebar groups .78rem uppercase | |
| Table body and headings | IBM Plex Sans 400/600 | body .92rem / 1.55; headings .82rem / 1.4 | sentence-case headings, `text` on `surface` |
| Code, numerals in tables | IBM Plex Mono 400/600 | .86rem / 1.6 · table data .9em, `tabular-nums` | |

Loaded from Google Fonts: `Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400`, `IBM+Plex+Sans:wght@400;600`,
`IBM+Plex+Mono:wght@400;600`. No other weights may be requested.

## Shape, rules and space

- Radius: 2px (pill, inline code), 3px (code blocks, inputs). Nothing above 4px.
- Tables: 2px `text` rule above and below; 1px `rule` hairlines between rows **and columns**; numeric columns right-aligned in mono; header cells use the table heading size above.
- Output blocks and asides: 3px `accent` rule on the left. Links: underlined, 1px, offset .18em.
- No gradients, glows, blurs, drop shadows or pill-shaped buttons.
- Spacing scale (rem): .25 · .5 · 1 · 1.5 · 2 · 3 · 5. Section gap 3rem; page padding 3rem / 2rem (2rem / 1rem below 760px).

## Assets

- `openquant-mark.svg` — flat open-center pinwheel, no tile. Intended for 16px and larger; validate at favicon size when revising geometry.
- `openquant-banner.svg` — 1280×320, paper ground, heavy-rule / hairline pair. Text is live SVG text and falls back to the viewer's serif on GitHub; outline it in #57 if exact type matters there.

## Original implementation evidence (#57, 2026-09-20)

Implemented in `docs-site/src/styles/starlight.css` (tokens as `--oq-*`, mapped onto Starlight's `--sl-*`
palette) and `docs-site/src/styles/code-themes.mjs` (Expressive Code themes). `bun run check:contrast`
reads the tokens from that stylesheet and fails CI if a text pair drops below AA or the code themes
use a colour that is not a token.

Starlight chrome checked in the browser, both themes (`evidence/site/chrome-*.jpeg`): search modal,
table of contents, pagination, asides, KaTeX display formulas, tables, code frames, the status line.
Departures from the static mock, all forced by Starlight's markup:

- The three status words (generated / draft / reviewed) share the one accent pill; the word carries
  the status. The old per-status colours included the pair that failed AA.
- Starlight's five aside hues collapse onto the accent; asides differ by title and icon.
- Numeric table columns are detected from markdown alignment (`---:`), not a class.

Banner text is still live SVG text and falls back to the viewer's serif on GitHub.

## Reading and diagram refinement (2026-11-05)

This is an evolution of the selected Monograph direction, not a new palette.
The requested logo update, stronger text hierarchy and diagrams are implemented
in the site and shared assets. Original candidate files remain a historical
record, not the current source for these refinements.

- The pinwheel has a larger open center and separated blades, retaining the
  three organization hues. The site, favicon and identity asset are identical;
  both banners use the same paths. The SVG includes a title and description.
- Header wordmark: Newsreader 600 at 1.6rem, with a 2rem mark. The landing-page
  eyebrow identifies the two language surfaces; the lede is upright for easier
  scanning. The decorative hero mark reduces to 88px below 480px.
- Bold prose uses the loaded 600 weight instead of browser-synthesized 700.
  Body text remains Newsreader 400. Emphasis is reserved for useful distinctions.
- Tables use IBM Plex Sans at .92rem/1.55, with .82rem sentence-case semibold
  headings on `surface`. Header text uses `text`, not the small muted label style.
  Numeric cells retain mono, tabular alignment. Decorative rules retain `rule`.
- The research map is a semantic four-step ordered list. Stage text remains
  available without JavaScript; decorative SVGs are hidden from assistive
  technology. Text and chart strokes use `text`, `muted` and `accent`, all above
  4.5:1 on `ground` in either theme. No data is encoded by color alone.
- At 960px the map becomes two columns, then one column at 480px. Its caption
  explicitly distinguishes a conceptual workflow from measured results.

References reviewed for this refinement:

- [Observable Plot](https://observablehq.github.io/plot/): layered marks and small
  multiples suggest compact, task-specific diagrams rather than decorative art.
- [marimo documentation](https://docs.marimo.io/): separate getting-started,
  examples and reference paths make a technical system easier to enter.
- [Butterick, Bold or italic](https://practicaltypography.com/bold-or-italic.html):
  selective emphasis preserves contrast between important and ordinary prose.

These are design principles, not copied visual assets. No dependencies or remote
scripts were added. `test:design` protects semantic stage labels, local module
links, accessible glyph treatment, logo consistency and loaded type weights.
