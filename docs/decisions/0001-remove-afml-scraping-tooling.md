# Decision: remove the AFML book-scraping tooling from the public repository

Status: accepted
Owner: Sean Koval
Date: 2026-09-19

## Context

The repository tracked a pipeline that crawled an online copy of *Advances in
Financial Machine Learning* (a copyrighted book), built a semantic index over it,
served it through a local MCP server, and drove a documentation-writing loop from
the results: `scripts/scrape_afml_crawl4ai.py`, `scripts/afml_semantic_index.py`,
`scripts/afml_mcp_server.py`, `scripts/afml_docs_loop.py`, `skills/afml-docs-loop/`,
the loop's state under `docs/afml-docs-loop/`, and five `docs-loop-*` recipes in
the `justfile`.

The tracked files contained no text from the book: the state and evidence files
hold section headings, page numbers and chunk ids only. The scraped content itself
(`afml/`) was never committed and is ignored. The concern is that a public,
open-source quant library documents and ships a way to scrape a copyrighted work.

## Options considered

- Keep as is, recording the decision.
- Rewrite as "bring your own copy": keep an indexer that takes a user-supplied
  path to a legally obtained copy and remove the scraper and any source URL.
- Remove the tooling from the repository.

## Decision

Remove it. The tooling is a private authoring aid, not part of the library, and
nothing in the build, tests, docs site or CI depends on it.

## Consequences

- The `docs-loop-*` recipes no longer exist. Module pages are reviewed by hand
  against the book (GitHub issue #53), citing chapter, section and snippet numbers.
- Earlier commits still contain the scripts; history is not rewritten.
- Local copies of the tooling and the untracked `afml/` directory may be kept for
  private use. `afml/`, `.hf-cache/` and `.crawl4ai-home/` stay in `.gitignore`.
- A bring-your-own-copy indexer can be reintroduced later as its own decision.

## Links

- GitHub issue #32
- `docs/design/production-readiness-brief.md` (unresolved decision 2)
