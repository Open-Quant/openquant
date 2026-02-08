# Docs Writing Prompt: CHAPTER 3 -> labeling

## Scope
- Chapter: `CHAPTER 3`
- Module: `labeling` (`/module/labeling`)
- Source files:
  - `docs-site/src/data/moduleDocs.ts` (module content)
  - `docs-site/src/pages/modules.astro` (subject/chapter entry point)
  - `docs-site/src/pages/module/[slug].astro` (module page rendering)

## Required AFML grounding
Use AFML MCP retrieval before writing:
1. `afml_search(query="3.8 THE QUANTAMENTAL WAY", chapter="CHAPTER 3", top_k=3)`
2. `afml_search(query="Triple-Barrier Method meta-labeling", chapter="CHAPTER 3", top_k=3)`
3. `afml_get_context(query="Triple-Barrier Method meta-labeling", chapter="CHAPTER 3", top_k=2, window=1)`

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
- Chapter anchor: `afml-S00471-c000103` pages 88-89 (3.8 THE QUANTAMENTAL WAY)
- Module anchor: `afml-S00471-c000103` pages 88-89 (3.8 THE QUANTAMENTAL WAY)
