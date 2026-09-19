# Delivery slices: production readiness

Owner: Sean Koval
Status: reviewed
Review source: owner approved all 31 records on 2026-09-18 (issues-only publication)
Canonical brief: `docs/design/production-readiness-brief.md` (owns OUT-/RQ- IDs; not duplicated here)

Each row is one independently finishable work record in `.ai-dlc/work/<id>.toml`; the record holds scope, evidence and acceptance.
Local IDs are draft identifiers until published to the tracker.

## Hygiene

| Work ID | Pri | Outcome | Requirement | Spec | Depends on |
| --- | --- | --- | --- | --- | --- |
| `hyg-repo-cleanup` | P0 | Clean the repository root, ignore local tooling, and prune merged branches | RQ-001 | no | - |
| `hyg-afml-scrape-decision` | P0 | Decide whether the AFML book-scraping tooling stays in a public repository | RQ-001 | no | - |

## Release

| Work ID | Pri | Outcome | Requirement | Spec | Depends on |
| --- | --- | --- | --- | --- | --- |
| `rel-name-metadata` | P1 | Choose a distribution name and make package metadata and README accurate | RQ-016 | no | - |
| `rel-vendored-pyo3-polars` | P1 | Document or eliminate the vendored pyo3-polars patch | RQ-016 | no | - |
| `rel-publish-pipeline` | P2 | Add a tagged release pipeline: wheels, crates.io, changelog | RQ-016 | no | `rel-name-metadata`, `rel-vendored-pyo3-polars`, `code-clippy-clean`, `code-no-panics`, `test-python-binding-contract` |
| `rel-community-files` | P2 | Add contributor and community files | RQ-016 | no | - |

## Code quality

| Work ID | Pri | Outcome | Requirement | Spec | Depends on |
| --- | --- | --- | --- | --- | --- |
| `code-clippy-clean` | P0 | Make the workspace clippy-clean and use one lint flag set everywhere | RQ-003 | no | - |
| `code-typed-errors` | P1 | Replace String errors with typed errors across the Rust core | RQ-004 | yes | `code-clippy-clean` |
| `code-no-panics` | P0 | Remove panics from public API paths | RQ-005 | yes | `code-typed-errors` |
| `code-rustdoc` | P2 | Document the public Rust API and gate it | RQ-006 | no | `code-typed-errors` |
| `code-dedupe-utils` | P2 | Consolidate duplicated statistics helpers and test fixtures | RQ-003 | no | `code-clippy-clean` |

## Testing

| Work ID | Pri | Outcome | Requirement | Spec | Depends on |
| --- | --- | --- | --- | --- | --- |
| `test-python-binding-contract` | P0 | Test every Python submodule and expose them from the package | RQ-007 | no | - |
| `test-reference-values` | P1 | Add reference-checked tests for thin and inline-only modules | RQ-008 | no | - |
| `ci-close-gaps` | P1 | Make CI gate what it claims to gate | RQ-009 | no | `code-clippy-clean` |

## Research enablement

| Work ID | Pri | Outcome | Requirement | Spec | Depends on |
| --- | --- | --- | --- | --- | --- |
| `res-recover-stranded-work` | P0 | Recover or formally drop the four unmerged feature commits | RQ-002 | yes | - |
| `bind-validation-backtest` | P0 | Add Python bindings for purged CV, CPCV backtesting, feature importance and tuning | RQ-010 | yes | `res-recover-stranded-work` |
| `res-real-data-layer` | P0 | Build a real market data layer with cache, snapshot hash and recorded terms | RQ-011 | yes | - |
| `res-evaluation-module` | P1 | Add the research evaluation module (PSR, DSR, MinTRL, trial registry) | RQ-012 | yes | `test-python-binding-contract` |
| `res-notebook-runner` | P1 | Execute notebooks with rich outputs and run them in CI | RQ-012 | no | `res-real-data-layer` |
| `res-cell-contract` | P2 | Write down and lint the research notebook contract | RQ-012 | no | `res-notebook-runner` |

## Research runbooks

| Work ID | Pri | Outcome | Requirement | Spec | Depends on |
| --- | --- | --- | --- | --- | --- |
| `runbook-meta-labeling` | P1 | Runbook: triple-barrier labeling and meta-labeling on real data | RQ-012 | no | `bind-validation-backtest`, `res-real-data-layer`, `res-evaluation-module`, `res-notebook-runner` |
| `runbook-cpcv-dsr` | P1 | Runbook: CPCV backtest with PSR and deflated Sharpe | RQ-012 | no | `bind-validation-backtest`, `res-real-data-layer`, `res-evaluation-module`, `res-notebook-runner` |
| `runbook-fracdiff` | P2 | Runbook: fractional differentiation - stationarity versus memory | RQ-012 | no | `res-real-data-layer`, `res-notebook-runner`, `test-python-binding-contract` |
| `runbook-bet-sizing` | P2 | Runbook: bet sizing from predicted probabilities | RQ-012 | no | `runbook-meta-labeling` |
| `runbook-hrp-portfolio` | P2 | Runbook: HRP versus inverse-variance and CLA out of sample | RQ-012 | no | `res-real-data-layer`, `res-notebook-runner`, `test-python-binding-contract` |

## Docs content

| Work ID | Pri | Outcome | Requirement | Spec | Depends on |
| --- | --- | --- | --- | --- | --- |
| `docs-landing-rewrite` | P0 | Rewrite the landing page to describe the library, not the documentation | RQ-014 | no | - |
| `docs-module-review` | P1 | Human-review the 40 generated module pages and break the template | RQ-013 | no | `test-python-binding-contract`, `docs-landing-rewrite` |
| `docs-api-reference` | P1 | Publish a real API reference for Rust and Python | RQ-013 | no | `code-rustdoc` |
| `docs-research-gallery` | P2 | Add a research gallery and finish or remove stub pages | RQ-013 | no | `runbook-meta-labeling`, `runbook-cpcv-dsr` |

## Docs design

| Work ID | Pri | Outcome | Requirement | Spec | Depends on |
| --- | --- | --- | --- | --- | --- |
| `design-direction` | P0 | Set one visual identity: palette, type, logo | RQ-015 | no | - |
| `design-implement` | P1 | Implement the visual identity in the docs site and README | RQ-015 | no | `design-direction`, `docs-landing-rewrite` |

## Suggested order

1. No dependencies, start now: `hyg-repo-cleanup`, `hyg-afml-scrape-decision`, `res-recover-stranded-work`, `code-clippy-clean`, `test-python-binding-contract`, `res-real-data-layer`, `docs-landing-rewrite`, `design-direction`.
2. Then: `code-typed-errors` -> `code-no-panics`; `bind-validation-backtest`; `res-evaluation-module`; `res-notebook-runner`; `design-implement`.
3. Then the runbooks, which feed `docs-module-review` figures and `docs-research-gallery`.
4. Release last: `rel-publish-pipeline` after the quality and testing slices.

## Beads carry-over

OQ-nbr.7 -> `res-evaluation-module`; OQ-nbr.8, OQ-nbr.9 -> `res-notebook-runner`; OQ-nbr.10, OQ-g5v -> `res-cell-contract`; closed-but-unmerged OQ-nbr.4, OQ-nbr.5, OQ-det -> `res-recover-stranded-work`.

Decision: proceed.
