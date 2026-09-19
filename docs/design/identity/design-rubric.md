# Design rubric: OpenQuant visual identity

- Rubric ID/version: `oq-identity` v1 — frozen before any candidate was generated
- Brief/work references and canonical RQ IDs: `design-brief.md`; work `design-direction` (#56); RQ-015
- Status: frozen
- Decision owner and task approval source: Sean Koval; work record approved 2026-09-18
- Candidate access, states, viewports and fixtures: `candidates/<id>.html`, light and dark, 1440px and 390px, shared content fixture
- Iteration ceiling: 3 candidates + 1 revision each. Budget: proposed, no paid resources.
- Stop/plateau condition: a candidate passes all required checks and rates ≥ 3 on all required criteria, or the ceiling is reached.

## Required and optional behavior checks

| ID / RQ | Required? | Setup | Action | Expected observable result | Evidence | Severity |
| --- | --- | --- | --- | --- | --- | --- |
| B1 / RQ-015 | yes | tokens, both themes | compute WCAG contrast for every text-on-ground pair (body, muted, link, code, pill, table head) | every pair ≥ 4.5:1 | `contrast.md` from script | blocker |
| B2 / RQ-015 | yes | tokens | inspect | exactly one accent hue per theme, and it is one of the three logo hues (shifted in lightness only as contrast requires) | tokens + note | blocker |
| B3 / RQ-015 | yes | 390px, both themes | load page | no horizontal page scroll; code and tables scroll inside their own box | screenshot | blocker |
| B4 / RQ-015 | yes | source | inspect | every font is on Google Fonts; every weight used is loaded; none of Inter, Roboto, Open Sans, Lato, Arial, system stack, Space Grotesk, Manrope | source | blocker |
| B5 / RQ-015 | yes | source | grep | no glow, `blur(`, `drop-shadow`, pill-shaped buttons, or radius above 4px; gradients only as a functional texture at ≤ 8% opacity | source | major |
| B6 / RQ-015 | yes | both themes | view links in prose | links identifiable without color (underline) | screenshot | major |

## Subjective criteria and anchors

Scale 0–4; required threshold 3 on each.

| ID | What serves this brief | 1: major shortcomings | 3: meets | 4: exceeds usefully | Evidence |
| --- | --- | --- | --- | --- | --- |
| S1 Reading comfort | A long page of prose, formulas and code is comfortable to read | measure over ~85ch or under ~50ch, cramped leading, headings that shout over the text | 60–75ch measure, clear hierarchy from size/weight alone, code and prose sit together | a reader would choose to read a chapter here rather than in the PDF | desktop screenshot |
| S2 Domain character | It looks like it belongs to quantitative finance and to this library | could be swapped onto any SaaS product unchanged | at least two deliberate choices traceable to the domain (e.g. ledger rules, tabular numerals, monograph type) | the identity is recognisable from a cropped fragment without the logo | both screenshots |
| S3 Restraint | Decoration never competes with content | ornament is the first thing seen | the first thing seen is the title and the first sentence | removing any remaining element would lose information | desktop screenshot |
| S4 Data legibility | Numeric tables and code are first-class | proportional digits, misaligned decimals, low-contrast rules | tabular right-aligned numerals, visible hairlines, readable code at 390px | tables read like a well-set financial statement | both screenshots |
| S5 Brand coherence | Logo, banner and site are one system | logo looks pasted in from another product | logo hues and site accent visibly relate; mark works flat at 24px | the mark and the site would be recognisable as one family in monochrome | header crop |

## Selection and stopping

Required checks must pass and required criteria must meet threshold before acceptance. Scores compare eligible candidates only and never offset a failed check. Self-review is recorded as such; the owner's choice overrides these ratings.
