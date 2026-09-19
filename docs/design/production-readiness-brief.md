# Product brief: From AFML port to a credible, usable research library

Owner: Sean Koval
Status: draft
Canonical brief: `docs/design/production-readiness-brief.md` (owns the OUT-/RQ- IDs below)
Entry path: brownfield

## Audience and problem

Two audiences meet the project today and both bounce.

- **A quant researcher** who wants to use AFML methods from Python finds no
  installable package, a docs site whose landing page describes the
  documentation rather than the library, 40 of 57 pages carrying a "No human has
  reviewed this page" banner, and one real-data notebook whose "model" is a
  5-day momentum heuristic with no triple-barrier labels, no purged CV and no
  PSR/DSR. Nothing shows the tools producing a research result.
- **A contributor or reviewer** finds a green CI that hides 160 clippy warnings
  the release workflow would reject, ~92 public signatures returning
  `Result<_, String>`, public functions that panic on bad input, 17 Python
  submodules with no tests, and three closed tracker issues whose code never
  reached `main`.

The requested features (cleanup, tests, runbooks, prettier docs) are symptoms of
one underlying problem: **the library's claims are not yet backed by visible
evidence** — executed research, reference-checked tests, reviewed pages, or a
release.

Unknown: who the first real users are, and whether Python-first or Rust-first
adoption matters more. This brief assumes Python-first for research runbooks.

## Evidence and assumptions

Observed on 2026-09-18 at `main` = `d3e6dfa` by four read-only audits (commands
were run; nothing was modified).

- **Build health:** `cargo check`, `cargo fmt --check` and `cargo test
  --workspace` pass (1 ignored test, `test_sadf_test`, run nightly). CI, Docs
  Pages and the last 12 nightly runs are green.
- **Lint:** `cargo clippy --workspace --all-targets` emits 160 warning lines.
  `ci.yml` and `just clippy` deny only `correctness`/`suspicious`;
  `release.yml` uses `-D warnings` and therefore cannot pass.
- **Errors/panics:** 16 modules have error enums, 6 implement
  `std::error::Error`; ~92 signatures return `Result<_, String>`
  (`backtesting_engine` 21, `data_processing` 13). Public panics remain in
  `bet_sizing.rs`, `filters.rs`, `backtest_statistics.rs:107-164`,
  `data_structures.rs`, `util/volatility.rs`, `microstructural_features.rs:440`.
- **Bindings:** `crates/pyopenquant` registers 27 submodules; Python tests cover
  the original 9. No bindings exist for `cross_validation`,
  `backtesting_engine`, `feature_importance`, `hyperparameter_tuning`,
  `fingerprint`, `combinatorial_optimization`, `hpc_parallel`, `etf_trick`.
  `python/openquant/__init__.py` re-exports 6 of 27.
- **Tests:** 19 of 37 Rust test files load mlfinlab-derived fixtures; 18 use
  inline data only. `data_processing` (563 LOC) has 1 test. No doctests.
  67 `///` lines across ~9,900 source lines.
- **Research:** notebooks 01–05 are 3-cell toy calls on synthetic data; 06 is the
  only real-data notebook (inline Stooq scrape, heuristic signal, CUSUM
  threshold yielding 845 events on 900 bars); 07–08 are synthetic. The runner
  captures stdout only — no plots. `openquant.data` is used by no notebook.
- **Stranded work:** beads issues OQ-nbr.4, OQ-nbr.5 and OQ-det are closed but
  their single commits (`59ac6fa`, `25b7b24`, `27a2007`) are not on `main`.
  Open beads: OQ-nbr.7–.10, OQ-g5v.
- **Docs site:** 57 content pages — 40 `generated`, 5 `draft`, 12 `reviewed`,
  0 `validated`. All module pages share seven identical H2s. 28 of 40 module
  pages have no Python example; 3 pages show real output; zero images; no AFML
  section/snippet citations; no API signatures. Three unrelated palettes (neon
  logo, purple-gradient banners, navy site accent); white-on-`#a8760d` draft
  pill is 3.99:1 (fails AA); 491-line unused `Layout.astro`.
- **Release:** no tags, releases, CHANGELOG or CONTRIBUTING. PyPI name
  `openquant` is owned by another project; `pyopenquant` is free; crates.io
  `openquant` is free. Crate `homepage`/`documentation` URLs return 404.
  `[patch.crates-io] pyo3-polars` points at an undocumented vendored copy.
- **Hygiene:** untracked `afml/` is a scraped copy of the AFML book and is not
  ignored; `.worktrees/` holds 41G; 15 local branches are fully merged; tracked
  empty file `foo`.

- User decisions: the owner asked (2026-09-18) for the state to be assessed and
  turned into GitHub issues covering code improvement, cleanup, testing, real
  research runbooks, and documentation content and design. No individual ticket
  has been reviewed yet.
- Hypotheses: (H1) real-data runbooks are what makes the library credible —
  untested with users. (H2) Stooq data cannot be redistributed — terms not
  checked. (H3) the 8 unbound modules lack bindings because of closure/trait
  parameters — not verified. (H4) fixtures derived from mlfinlab tests have
  acceptable license provenance — not verified.

## Current behavior (brownfield)

- Inspected: Rust crate `openquant` (34 modules), `pyopenquant` bindings, Python
  package, notebooks, experiments, docs-site, CI workflows, GitHub repo state.
- Compatibility: the public Rust and Python function names and numeric results
  are the contract. Typed-error work changes error *types* (pre-1.0, unpublished,
  so acceptable) but must not change successful results; existing fixture tests
  must keep passing unchanged. Docs URLs already have 13 production redirects —
  any IA change must preserve them.
- Migration/recovery: nothing is published, so there are no external consumers
  to migrate. Every slice lands by PR and is revertible. Branch/worktree
  deletion is the only irreversible step and is limited to branches proven
  merged with `git branch --merged main` and `git cherry`.

## Options and trade-offs

| Option | User impact and confidence | Effort and dependencies |
| --- | --- | --- |
| A. Polish the docs design only | Removes the first-glance "slop" impression; does not fix that pages are unreviewed and examples are absent. Low confidence it changes credibility. | Small; independent. |
| B. Release first (PyPI/crates.io) | Makes the library installable, but ships panicking public APIs and untested bindings under a name that then cannot be changed cheaply. | Medium; blocked by naming and the vendored patch. |
| C. Evidence-first: foundations → bindings → real runbooks → docs built from runbook output → release | Each stage produces the evidence the next one displays. Runbooks exercise the bindings and expose bugs before release. Highest confidence, slowest to a visible result. | Large; ordered dependencies. |
| D. Do nothing | CI stays green; the project remains a private port. | None. |

## Selected outcome

OUT-001: The repository is clean, honest and safe to make visible — no stranded
work, no scraped book, accurate metadata.
OUT-002: The Rust core meets a stated quality bar — clippy-clean under one flag
set, typed errors, no panics on bad input, documented public API.
OUT-003: Tests back the claims — every Python submodule is tested, thin modules
gain reference-checked tests, CI gates what it says it gates.
OUT-004: Researchers can run credible AFML studies on real data from Python,
with executed outputs and plots committed.
OUT-005: The docs site reads as a reviewed reference — reviewed module pages
with citations, signatures, executed examples and plots.
OUT-006: The docs site has one restrained visual identity suited to a
quantitative library.
OUT-007: The library is installable under a name it owns, with a repeatable
release process.

Reason: option C. Design polish (A) on unreviewed content is lipstick; release
(B) before the quality bar fixes the name and API too early. Design work
(OUT-006) is independent and can run in parallel with everything else.

## Scope and exclusions

Included: the 31 delivery slices in
`docs/design/production-readiness-slices.md`.

Excluded: new AFML modules (e.g. the missing Ch. 18 entropy features), live
trading or execution, a hosted data service, GPU work, rewriting history to
drop old blobs, inspecting the legacy `origin/feature-generation` and
`origin/feature/tests` branches beyond a keep/delete decision.

Must survive: numeric results of existing fixture tests; the 13 docs redirects;
the docs CI gates added in PRs #17–#30.

## Success evidence

| Requirement | Outcome | Observable criterion | Evidence/source and status |
| --- | --- | --- | --- |
| RQ-001 | OUT-001 | `git status` on a fresh clone is clean; `afml/`, `.worktrees/`, `.hf-cache/`, `.crawl4ai-home/` are ignored; `foo` is gone; no local branch is both merged and present | Planned check |
| RQ-002 | OUT-001 | Each of `59ac6fa`, `25b7b24`, `27a2007`, `f31de34` is either re-ported to `main` or recorded as dropped with a reason | Planned; commits identified |
| RQ-003 | OUT-002 | `cargo clippy --workspace --all-targets -- -D warnings` passes and `ci.yml`, `justfile`, `release.yml` use the same flags | Observed: 160 warnings today |
| RQ-004 | OUT-002 | No public fn returns `Result<_, String>`; every error enum implements `std::error::Error` | Observed: ~92 and 10 of 16 today |
| RQ-005 | OUT-002 | Invalid input to any public fn returns `Err`, proven by tests for each site listed in the audit | Observed panic sites listed above |
| RQ-006 | OUT-002 | `RUSTDOCFLAGS="-D missing_docs" cargo doc` passes for the `openquant` crate and doctests run in CI | Observed: 67 doc lines today |
| RQ-007 | OUT-003 | A contract test enumerates every `_core` submodule and fails when one has no value test; all 27 pass | Observed: 9 of 27 tested |
| RQ-008 | OUT-003 | Each thin module named in the audit has tests checked against an external reference value | Observed: 18 files inline-only |
| RQ-009 | OUT-003 | CI runs Python tests on 3.11 and 3.13, compiles all benches, lints Python, and the nightly job runs the full suite | Observed gaps listed |
| RQ-010 | OUT-004 | `cross_validation`, `backtesting_engine`, `feature_importance`, `hyperparameter_tuning` are callable from Python with tests | Observed: no binding files |
| RQ-011 | OUT-004 | A documented data layer fetches, caches and hashes real daily OHLCV from a source whose terms are recorded | Hypothesis H2 open |
| RQ-012 | OUT-004 | Five runbooks execute end to end in CI from cached data and commit outputs and plots | Observed: 0 today |
| RQ-013 | OUT-005 | 0 pages with status `generated`; every module page has an AFML citation, signatures, one executed Python example with output | Observed: 40 generated |
| RQ-014 | OUT-005 | The landing page states what the library is, how to install it, and shows a runnable example with output | Observed: none of the three |
| RQ-015 | OUT-006 | One palette across logo, banners and site; all text/background pairs ≥ 4.5:1; no unused theme code | Observed: 3 palettes, pill 3.99:1 |
| RQ-016 | OUT-007 | `pip install <name>` and `cargo add openquant` work from the registries for a tagged release | Observed: nothing published |

## Next slice

`hyg-repo-cleanup` — it is reversible except for proven-merged branch deletion,
unblocks nothing else but removes the risk of committing `afml/`. In parallel,
`code-clippy-clean` and `design-direction` have no dependencies.

## Unresolved decisions

All owned by Sean Koval.

1. **PyPI distribution name** — `pyopenquant` is free; import name can stay
   `openquant`. Blocks `rel-name-metadata`.
2. **Keep the AFML scraping scripts public?** (`scripts/scrape_afml_crawl4ai.py`,
   `afml_mcp_server.py`, `afml_semantic_index.py`, `skills/afml-docs-loop/`.)
3. **Real-data source** — needs a source with recorded terms (H2).
4. **Modeling glue for runbooks** — re-port `notebook_modeling.rs` (`25b7b24`)
   or use scikit-learn from Python.
5. **Track `.beads/` or retire beads** in favour of GitHub Issues; decides the
   fate of `.gitattributes` and `AGENTS.md`.
6. **ai-dlc `verify.yml`** — the adopted workflow bootstraps ai-dlc in CI; a
   source-generated project has no release manifest, so it is expected to fail
   until ai-dlc publishes one. Keep, disable, or pin.

Decision: proceed — scope is bounded, nothing is published so compatibility
risk is low, and the six open decisions each block only their own slice. This
decision does not itself authorize implementation or publication.
