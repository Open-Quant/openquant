# OpenQuant identity tokens — A · Monograph

Status: **frozen 2026-09-19.** Chosen by the owner; source of truth for #57. Generated values live in
`build_candidates.py` (`CANDIDATES["a-monograph"]`); contrast for every pair is in `contrast.md`.

## Color

One accent per theme, derived from the logo's magenta `#e44de0` and shifted in lightness only as AA contrast requires.
The logo's cyan and yellow appear **only in the mark**.

| Token | Light | Dark | Use |
| --- | --- | --- | --- |
| `ground` | `#fbf9f4` | `#17140f` | page |
| `surface` | `#f2eee4` | `#221e17` | code blocks, inline code |
| `text` | `#1d1a16` | `#ece6d8` | body, headings, heavy table rules |
| `muted` | `#5c554b` | `#aaa190` | captions, table heads, code comments, status line |
| `accent` | `#97196a` | `#f29ad6` | links, current nav item, eyebrow labels, code keywords, status pill ground |
| `rule` | `#cdc4b1` | `#4c4536` | hairlines (not text: 1.65:1 / 1.93:1) |

Lowest required text contrast: 6.35:1 (muted on surface, light). `color-scheme` follows the theme.

## Type

| Role | Face | Size / leading | Notes |
| --- | --- | --- | --- |
| Body | Newsreader 400, optical size auto | 1.19rem / 1.62 | measure `52ch` ≈ 70 characters per line |
| H1 | Newsreader 600 | 3.1rem / 1.08, tracking −0.015em | 2.1rem below 760px |
| H2 | Newsreader 600 | 1.7rem / 1.2 | hairline above, 1.4rem padding |
| Lede, formulas | Newsreader 400 italic | 1.45rem / 1.45 · 1.3rem | |
| Navigation, labels, table heads | IBM Plex Sans 400/600 | .92rem · labels .74rem uppercase, tracking .10–.14em | |
| Code, numerals in tables | IBM Plex Mono 400/600 | .86rem / 1.6 · table data .9em, `tabular-nums` | |

Loaded from Google Fonts: `Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400`, `IBM+Plex+Sans:wght@400;600`,
`IBM+Plex+Mono:wght@400;600`. No other weights may be requested.

## Shape, rules and space

- Radius: 2px (pill, inline code), 3px (code blocks, inputs). Nothing above 4px.
- Tables: 2px `text` rule above and below; 1px `rule` hairlines between rows **and columns**; numeric columns right-aligned in mono; header cells stay at label size.
- Output blocks and asides: 3px `accent` rule on the left. Links: underlined, 1px, offset .18em.
- No gradients, glows, blurs, drop shadows or pill-shaped buttons.
- Spacing scale (rem): .25 · .5 · 1 · 1.5 · 2 · 3 · 5. Section gap 3rem; page padding 3rem / 2rem (2rem / 1rem below 760px).

## Assets

- `openquant-mark.svg` — flat pinwheel, no tile. Use from 16px up.
- `openquant-banner.svg` — 1280×320, paper ground, heavy-rule / hairline pair. Text is live SVG text and falls back to the viewer's serif on GitHub; outline it in #57 if exact type matters there.

## Implementation (#57, 2026-09-20)

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
