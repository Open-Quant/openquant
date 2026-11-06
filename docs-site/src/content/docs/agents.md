---
title: Agent guide
description: Entry points, canonical sources and verification steps for agents using or changing OpenQuant.
status: authored
last_authored: '2026-11-05'
audience:
  - coding-agent
  - platform-engineering
sidebar:
  order: 2
---

Use this path to find evidence and run a bounded task. For a guided explanation
of the first research result, follow the [human quickstart](/quickstart/).
Both paths use the same APIs and checks; this page is an index, not a second
API specification.

For programmatic discovery, read the [generated plain-text index](/llms.txt).
It lists these same guides and public Python references, with page status and
the source revision. It is rebuilt with the site, rather than maintained as a
second list of APIs.

## Start with the task

| Task | Read first | Check against |
| --- | --- | --- |
| Call an API from Python | [Python API](/api/python/), then its [module guide](/modules/) | Exact import, signature, return shape and documented limitations |
| Change the Rust core | [Rust workflow](/workflows/rust-core-workflow/) and [rustdoc](/api/rust/openquant/) | The implementation and its independent-reference tests |
| Run a research experiment | [Research workflow](/workflows/notebook-research-workflow/) and [runbooks](/runbooks/) | Dataset hash, seed, trial count, costs and out-of-sample evaluation |
| Edit documentation | [Local build](/setup/local-build/#documentation-quality-gates) | Source generators, example checks, links and page status |

Working in a checkout? Read
[AI-DLC.md](https://github.com/Open-Quant/openquant/blob/main/AI-DLC.md),
[CONTRIBUTING.md](https://github.com/Open-Quant/openquant/blob/main/CONTRIBUTING.md)
and any instructions applying to the files you change before editing. The
[repository skills](https://github.com/Open-Quant/openquant/tree/main/skills)
cover experiment runs, notebook execution, artifact auditing, feature discovery,
parity checks and promotion gates. Choose the skill matching the task rather
than running every workflow.

## Resolve the source of truth

The website describes the published documentation revision. When changing code,
use files from **your checkout's commit**, not a signature copied from a different
release. Start by recording `git rev-parse HEAD` and `git status --short`; preserve
unrelated changes.

| Question | Canonical repository source |
| --- | --- |
| Which Python names can I import? | `python/openquant/__init__.py`, public Python modules and generated `_core/*.pyi` stubs |
| What does a binding accept and return? | `crates/pyopenquant/src/` plus the built extension's signatures/docstrings |
| What is the Rust behavior? | `crates/openquant/src/` and `crates/openquant/tests/` |
| What versions and commands apply? | `rust-toolchain.toml`, `pyproject.toml`, lockfiles, `justfile` and `.github/workflows/ci.yml` |
| Where do research results come from? | Executed notebooks, dataset hashes and experiment manifests, not a prose summary |
| Which files generate the site? | The mapping below and `docs-site/package.json` |

The distribution name is `pyopenquant`; the Python import is `openquant`.
The [module index](/modules/#by-language-surface) distinguishes Rust, compiled
bindings and Python wrappers. Use the supported wrapper when one exists.
For example, Python purged CV is `openquant.cross_validation`, not a guessed
Rust-to-Python name.

### Generated documentation

Edit the source, then regenerate. Do not patch generated outputs alone.

| Output | Source / generator |
| --- | --- |
| Python type stubs | Rust bindings → `scripts/generate_python_stubs.py` |
| API inventory | Rust and Python source → `scripts/generate_api_inventory.py` |
| Python API reference | Built extension → `scripts/generate_python_api_reference.py` |
| Research gallery, performance and changelog pages | Notebooks, benchmarks, `CHANGELOG.md` → `scripts/docs/generate_site_pages.py --write` |
| Coverage dashboard | Current source and documentation tree → `scripts/docs/check_coverage.py --write` |

See [Contributing](https://github.com/Open-Quant/openquant/blob/main/CONTRIBUTING.md#generated-api-artefacts)
for the regeneration order. Rebuild the extension after Rust changes before
checking Python behavior: an older installed extension can make a test pass
against the wrong code.

## Run and verify

Follow the [local setup](/setup/local-build/) rather than installing a package
with a similar name. Start with the smallest relevant check, then run the
required aggregate gates for the change.

- **Rust:** format, clippy, the fast suite and doctests. Run the explicit slow
  SADF test when changing structural breaks. Benchmark hot-path changes.
- **Python:** rebuild the extension, run `pytest python/tests -q`, then the lint,
  type and stub checks from `CONTRIBUTING.md`.
- **Docs:** `bun run check:docs` from `docs-site/`. This includes rustdoc staging,
  so Rust is required even for a full documentation check. The
  [local build guide](/setup/local-build/#documentation-quality-gates) separates
  the lightweight preview from the full gate.
- **Notebooks:** execute the affected notebooks, verify committed outputs and
  run the notebook contract lint. Regenerate the gallery if results change.

Record the exact command, revision and outcome. Separate **passed**, **failed**
and **not run**; a missing compiler or extension is a blocker, not a successful
check. A site build alone does not prove the Python examples execute or numerical
results match an independent reference.

## Preserve research safeguards

- Use synthetic data for the documented smoke paths. A synthetic example is
  not evidence of a tradable strategy.
- Keep label end times, purging and embargo explicit. Do not disable a leakage
  guard merely to make an example pass.
- Record every tried configuration before selecting a winner. Report trial
  count, costs and out-of-sample results alongside Sharpe ratios.
- Keep reference fixtures independent of the implementation being tested.
  A value produced by the function under test cannot validate that function.
- Do not raise an AI-authored page to `reviewed` or `validated`. Those statuses
  require human review. Keep skipped examples visible and explain why they
  cannot run.

The full policy and its limits are in [Governance](/project/governance/).
Return to the [human quickstart](/quickstart/) when you need the rationale
behind an output, or the [module index](/modules/) for a different API.

## Ownership

Repository maintainers own this navigation guide. Its source links point to
existing behavior and contribution rules; it introduces no new release,
research-promotion or publication authority.
