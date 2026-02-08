---
name: afml-docs-loop
description: Build and run an AFML-grounded, chapter-by-chapter documentation loop for OpenQuant docs. Use this when you need consistent module docs, hierarchical chapter coverage, and iterative progress tracking with MCP-backed prompts.
---

# AFML Docs Loop

Use this skill when the goal is to produce complete, consistent OpenQuant docs aligned to AFML structure.

## What this skill does

1. Builds a hierarchical docs backlog (`chapter -> module section`) from AFML + current module docs.
2. Generates per-section writing prompts grounded in AFML MCP retrieval anchors.
3. Runs a loop: claim next section, write docs, review UI, mark done.

## Required script

- Main orchestrator: `scripts/afml_docs_loop.py`
- Wrapper: `scripts/run_afml_docs_loop.sh`

## Quick start

```bash
# 1) initialize plan + prompt files
skills/afml-docs-loop/scripts/run_afml_docs_loop.sh init

# 2) inspect progress
skills/afml-docs-loop/scripts/run_afml_docs_loop.sh status

# 3) run MCP evidence loop across sections (writes evidence.json/evidence.md)
skills/afml-docs-loop/scripts/run_afml_docs_loop.sh evidence

# 4) claim next section and print its prompt
skills/afml-docs-loop/scripts/run_afml_docs_loop.sh next --print-prompt

# 5) mark section complete
skills/afml-docs-loop/scripts/run_afml_docs_loop.sh done <section_id> --note "what was updated"

# 6) export loop status into Astro data
skills/afml-docs-loop/scripts/run_afml_docs_loop.sh export
```

## Iterative workflow (mandatory)

For each section:

1. Run `evidence` to refresh AFML retrieval artifacts for all targeted sections.
2. Claim next section with `next --print-prompt`.
3. Use AFML MCP tools (`afml_search`, `afml_get_context`) to gather chapter/module grounding.
4. Update module docs in `docs-site/src/data/moduleDocs.ts`.
5. Validate rendering in Astro and apply the UI rubric in `references/ui-ux-rubric.md`.
6. Mark section complete with `done`.
7. Repeat until `status` shows no pending sections.

## File targets

- Module content source: `docs-site/src/data/moduleDocs.ts`
- Module renderer: `docs-site/src/pages/module/[slug].astro`
- Subject index: `docs-site/src/pages/modules.astro`
- State file: `docs/afml-docs-loop/state.json`
- MCP evidence: `docs/afml-docs-loop/evidence.json`, `docs/afml-docs-loop/evidence.md`
- Generated prompts: `docs/afml-docs-loop/prompts/*.md`

## Quality bar

- Use `references/doc-writing-template.md` for consistency.
- Use `references/ui-ux-rubric.md` to avoid generic/AI-slop design.
- Keep examples executable and API-accurate for the current crate.
