---
title: Agent guide
description: Entry points, canonical sources and verification steps for agents using or changing OpenQuant.
status: authored
last_authored: '2026-11-07'
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
| Run or reproduce a research experiment | [Research workflow](/workflows/notebook-research-workflow/) and [runbooks](/runbooks/) | Notebook tier, dataset hash, seed, trial count, costs and out-of-sample evaluation |
| Select data available at a decision cutoff | [Point-in-time data workflow](/workflows/point-in-time-data/) | Observation versus availability time, revision selection and policy identity; synthetic checks do not verify a provider’s history |
| Load or adapt market data | [Data](/modules/data/) and [adapters](/modules/adapters/) | Timestamp units, row order, missing values and provenance; a converted frame is not a quality certificate |
| Connect market-data rows to named allocations | [Multi-asset allocation walkthrough](/workflows/multi-asset-allocation/) | Common observation times, matrix axes, weight names and fitting boundary; calculated weights are not a realized backtest |
| Select parameters and assess a final holdout | [Model-selection walkthrough](/workflows/model-selection-holdout/), then [cross-validation](/modules/cross-validation/) | Development-only search, fold-local preprocessing, final refit and an untouched assessment span |
| Interpret supplied evaluation results | [Evaluation](/modules/evaluation/) | Return conventions, label spans and the complete inventory of tried configurations |
| Resume or audit saved work | [Journey readback](/workflows/research-journey/#read-back-and-check-the-handoffs) and [grid artifacts](/examples/catalog/#run-a-configuration-grid-and-read-its-artifacts) | Saved inputs/configuration, exact output paths and artifact consistency; a leaderboard is not a trial registry |
| Diagnose a failed command | [Troubleshooting](/setup/troubleshooting/) | Exact command, interpreter/import paths and the first failing gate |
| Edit documentation | [Local build](/setup/local-build/#documentation-quality-gates) | Source generators, example checks, links and page status |

Working in a checkout? Read
[AI-DLC.md](https://github.com/Open-Quant/openquant/blob/main/AI-DLC.md),
[CONTRIBUTING.md](https://github.com/Open-Quant/openquant/blob/main/CONTRIBUTING.md)
and any instructions applying to the files you change before editing. The
[repository skills](https://github.com/Open-Quant/openquant/tree/main/skills)
cover experiment runs, notebook execution, artifact auditing, feature discovery,
parity checks and promotion gates. Choose the skill matching the task rather
than running every workflow. Read skill files in the selected checkout and check their
command descriptions against that checkout's scripts.

### Match a skill to its evidence

| Repository skill | What its entry point establishes | Read alongside it |
| --- | --- | --- |
| [flywheel-runner](https://github.com/Open-Quant/openquant/blob/main/skills/flywheel-runner/SKILL.md) | Synthetic configured runs and artifacts; no forecasting model is fitted by the CLI | [Artifact readback](/examples/catalog/#run-a-configuration-grid-and-read-its-artifacts) |
| [notebook-executor](https://github.com/Open-Quant/openquant/blob/main/skills/notebook-executor/SKILL.md) | The current executor uses a real Jupyter kernel and saves rich outputs; it is not a plain Python cell loop | [Execution and comparison scope](/workflows/notebook-research-workflow/) |
| [artifact-auditor](https://github.com/Open-Quant/openquant/blob/main/skills/artifact-auditor/SKILL.md) | Checks a selected run directory for required files, core columns and grid membership; does not recompute numerical results or prove provenance | [Catalog artifact checks](/examples/catalog/#run-a-configuration-grid-and-read-its-artifacts) |
| [parity-check](https://github.com/Open-Quant/openquant/blob/main/skills/parity-check/SKILL.md) | Scaffold tests compare selected outputs through shared implementation paths, not an independent numerical reference | [Validation layers](/setup/local-build/) |
| [feature-discovery](https://github.com/Open-Quant/openquant/blob/main/skills/feature-discovery/SKILL.md) | Feature screening and diagnostic calls; selection must stay inside training boundaries | [Feature diagnostics](/modules/feature-diagnostics/) |
| [promotion-gate](https://github.com/Open-Quant/openquant/blob/main/skills/promotion-gate/SKILL.md) | A summary of configured threshold flags, not proof of out-of-sample validity or publication authority | [Research assumptions](/workflows/python-core-workflow/) |

These are capability notes, not replacements for skill instructions. Select the exact
single-run or grid-run directory printed by the runner for the artifact audit; its
structural pass is only one part of reviewing data lineage and research validity.

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
| Python API reference | Built extension, generated stubs and public Python source → `scripts/generate_python_api_reference.py` |
| Research gallery, performance and changelog pages | Notebooks, benchmarks, `CHANGELOG.md` → `scripts/docs/generate_site_pages.py --write` |
| Coverage dashboard | Current source and documentation tree → `scripts/docs/check_coverage.py --write` |

Authored runbooks and generated gallery summaries can describe the same notebook. When a
result or interpretation changes, update its canonical notebook and check both destinations;
regenerating the gallery does not rewrite an authored runbook.

See [Contributing](https://github.com/Open-Quant/openquant/blob/main/CONTRIBUTING.md#generated-api-artefacts)
for the regeneration order. Rebuild the extension after Rust changes before
checking Python behavior: an older installed extension can make a test pass
against the wrong code.

The Python reference generator imports the package from its interpreter. Its
compiled signatures are checked against the stubs; its wrapper signatures come
from the imported Python source. Confirm both `openquant.__file__` and
`openquant._core.__file__` point to the intended checkout/build before regenerating
or validating examples. Matching the package version alone is insufficient:
different source revisions may share that version.

## Select the environment before running

Choose whether you are checking a source checkout or a packaged wheel. The
[bindings setup](/setup/python-bindings/) separates locked dependency installation from
the explicit native build; its [wheel check](/setup/python-bindings/#verify-the-packaged-installation-separately)
uses a separate environment and isolated imports. Do not use a checkout import as evidence
that a wheel contains the same files.

For an existing environment, record `sys.executable`, `openquant.__file__` and
`openquant._core.__file__` with the selected interpreter. Use the documented `--no-sync`
diagnostic commands to inspect it without triggering another project synchronization.
Restart a process after rebuilding its loaded extension. A notebook launches the named
`python3` kernel, so verify those paths inside the kernel as well.

Before rerunning research, identify the selected data source, registry and output paths.
Temporary notebook output is not a sandbox for cache or registry writes. The journey
script refuses to overwrite its report; the experiment runner also rejects an existing
configuration-derived output directory or a symlink at that target. Follow the linked artifact instructions and
use new output locations when preserving a baseline.

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

Keep example coverage precise. `check:examples` compiles Rust fences on module
pages and the complete Rust core workflow without running them; it does not scan every
Rust fence across the site.
`check:python-examples` scans the Markdown content tree and executes each runnable
Python fence in a separate interpreter. For runnable fences, a following `text` fence is
also compared with captured stdout unless marked `doc-output=skip`; rerunning code alone
does not verify its printed-output example. Fences marked `doc-check=skip` receive a
syntax check only. Neither gate executes code embedded in API docstrings or
notebooks. Check those through their own tests and notebook workflow, and report
any walkthrough that falls outside the gates as unverified until checked directly.

The Python example gate may fall back to an installed package when the checkout
has no local extension. Its success alone therefore does not establish that the
checkout's Python implementation ran. Rebuild locally and verify import locations
when using the result as evidence for a source change.

When handing work to another agent, preserve the base revision and uncommitted changes,
interpreter/import locations, configuration and dataset identity, registry/output paths,
and exact checks with their outcomes. A commit alone does not identify local edits or the
loaded binary. Include the saved artifacts and unresolved failures so the next agent can
read back evidence before rerunning or overwriting it. Distinguish an API tour from a research runbook using the
[notebook contract](/workflows/research-notebook-contract/); successful execution of a tour
is not experimental validation.

## Preserve research safeguards

- Use synthetic data for the documented smoke paths. A synthetic example is
  not evidence of a tradable strategy.
- Keep label end times, purging and embargo explicit. Do not disable a leakage
  guard merely to make an example pass.
- Define trial identity before selecting a winner. The [evaluation registry](/modules/evaluation/#what-to-watch-for)
  keys entries by configuration JSON: another data window or seed must be represented
  explicitly when it defines a different trial. Re-recording a key replaces its entry;
  output-directory counts and leaderboard rows are not automatically the registry's count.
  Report the trial inventory, costs and out-of-sample results together. Follow the
  [shared-registry guidance](/modules/evaluation/#refreshing-a-shared-registry) before
  reading another writer's results; reloading does not make concurrent writes safe.
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
