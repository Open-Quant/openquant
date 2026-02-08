# Docs Writing Prompt: CHAPTER 2 -> etf_trick

## Scope
- Chapter: `CHAPTER 2`
- Module: `etf_trick` (`/module/etf-trick`)
- Source files:
  - `docs-site/src/data/moduleDocs.ts` (module content)
  - `docs-site/src/pages/modules.astro` (subject/chapter entry point)
  - `docs-site/src/pages/module/[slug].astro` (module page rendering)

## Required AFML grounding
Use AFML MCP retrieval before writing:
1. `afml_search(query="2.3.1 Standard Bars", chapter="CHAPTER 2", top_k=3)`
2. `afml_search(query="ETF Trick single future roll", chapter="CHAPTER 2", top_k=3)`
3. `afml_get_context(query="ETF Trick single future roll", chapter="CHAPTER 2", top_k=2, window=1)`

Use chunk ids and page ranges in your internal notes.

## Content contract (must be complete)
1. High-level purpose and failure modes.
2. AFML math framing tied to API behavior.
3. Minimum 2 executable Rust examples (realistic, not toy-only).
4. Implementation caveats and production constraints.
5. Cross-links to adjacent modules in same chapter/subject.

## UI/UX contract (not AI slop)
1. Keep typography intentional and consistent.
2. Improve scanability with clear section rhythm and spacing.
3. Avoid generic gradient-heavy cards everywhere.
4. Keep code blocks readable on mobile and desktop.
5. Verify rendered page in Astro before marking complete.

## Current AFML anchors
- Chapter anchor: `afml-S00416-c000058` pages 55-55 (2.3.1 Standard Bars)
- Module anchor: `` pages -1--1 ()
