---
title: Governance
description: The rules OpenQuant holds its research, documentation, benchmarks and releases to, and where to report a problem.
status: authored
last_authored: '2026-09-26'
audience:
  - quant-dev
  - platform-engineering
risk_notes:
  - Naive random CV with overlapping labels is not acceptable.
  - A result reported without its trial count cannot be deflated, so it cannot be judged.
sidebar:
  order: 1
---

This page collects the project's rules in one place. Each rule names the code, check or file
that enforces it; where nothing enforces a rule yet, the page says so. It replaces five short
pages (methodology, reproducibility, benchmarks, versioning, support) that repeated each other
and made few concrete claims.

## Methodology and leakage controls

A result that leaks information from the test period into training cannot be trusted, however
good it looks. The library and the runbooks apply these controls:

- **Time-aware splits with purging and embargo** whenever label windows overlap:
  [`cross_validation`](/modules/cross-validation/) (`PurgedKFold`, and `cpcv_splits` for
  combinatorial purged CV). In Python, `feature_diagnostics.mda_importance` raises unless you pass
  the label end times (`event_end_indices`) or opt out explicitly with `allow_unpurged=True`, and
  its result reports whether the CV was purged.
- **Every configuration counts.** Record each backtested configuration in an
  `openquant.evaluation.TrialRegistry` before selecting one, and report the deflated Sharpe ratio
  computed from the registry's trial count next to any probabilistic Sharpe ratio. The registry
  is a JSON file, so the count survives restarts and is shared between sessions that use the same
  path. [Runbook 12](/runbooks/cpcv-deflated-sharpe/) shows what goes wrong without it.
- **Costs and assumptions are stated.** A runbook reports results net of an explicit cost per
  unit of turnover and names its universe and data source.
- **Controls with a known answer.** A runbook tests its method on data where the right answer is
  known: a no-signal control, where it must find nothing, and, where possible, a planted signal of
  known strength. The committed data is SYNTHETIC; nothing on this site describes a real market.
- **Pre-registration.** A runbook writes its hypotheses and pass/fail rules before the full run
  and ends with a promotion decision. The [research gallery](/runbooks/) lists every decision.

Enforcement is partial: `mda_importance` refuses unpurged CV and the runbooks follow the rules
above, but nothing stops a user's own code from calling a split without purging.

## Reproducibility

- **Notebooks are committed with their outputs.** `.github/workflows/notebooks.yml` executes every
  `notebooks/python/NN_*.ipynb` on pull requests that touch notebooks, Python or the crates, and
  nightly, and fails if the committed outputs or the exported figures differ from a fresh run by
  more than float noise (compared to 4 significant digits). See
  [Notebook Research Workflow](/workflows/notebook-research-workflow/).
- **Runbooks end with a reproducibility cell** that prints the package, numpy and polars versions,
  the hash of every dataset the run used, the seed and the git commit; most also print a hash of
  the run's configuration and its trial counts.
- **Datasets are identified by content.** `openquant.data.dataset_hash` hashes a frame's contents,
  and `openquant.data.record_dataset_hash` writes that hash and the data's provenance into a run
  manifest. `openquant.data.fetch` reads the committed SYNTHETIC sample by default, so every run
  in CI is offline and deterministic (see `DATA_SOURCES.md`).
- **Figures are deterministic.** `nbfigures.figure` writes SVGs without dates or random ids, so an
  unchanged figure is byte-identical after a re-run.

## Documentation status

Every page on this site declares a `status`, shown as the pill at the top of the page:
`generated` (emitted by a script from a source file, not read by anyone), `draft` (known to be
incomplete), `authored` (complete, examples executed, not yet read by a human), `reviewed` (read by
a human) and `validated` (also checked against the code). `check:content-schema` fails if a status
is missing or if the date behind it is older than the page's last change, and the Rust and Python
examples on every page are compiled or executed in CI. Pages written by an AI assistant stop at
`authored`. The [coverage dashboard](/coverage/) counts pages by status.

## Benchmarks

The committed results, what each benchmark measures and the regression gate are on the
[Performance](/project/performance/) page. In short: pull requests that touch `crates/openquant/`
run the tracked criterion benchmarks on their merge base and their head on the same runner, and
fail on a slowdown above 35% (or a per-benchmark override). A pull request that refreshes
`benchmarks/baseline_benchmarks.json` has to say why.

## Versioning and releases

OpenQuant is pre-release. The `openquant` crate, the `pyopenquant` extension and the Python
package are all at version 0.1.0, no version has been tagged, and nothing is published to
crates.io or PyPI. Until a release, only `main` is supported.

- **When a release is made,** `openquant` goes to crates.io; `pyopenquant` has `publish = false`
  and ships as a wheel built with maturin, because it depends on the vendored pyo3-polars patch.
  The steps are in `docs/publishing.md`.
- **Release gate.** Pushing a `v*` tag runs `.github/workflows/release.yml`: format, clippy with
  `-D warnings`, the fast test suite, a `cargo package` dry run, a compile check of every benchmark,
  and the long SADF test that is skipped on pull requests.
- **Changes are recorded** in `CHANGELOG.md`, rendered as the [Changelog](/project/changelog/)
  page. A change of behaviour ships with the update to the docs page of every module it touches.
  Pull request titles use conventional-commit prefixes (`feat`, `fix`, `docs`, ...), which is
  what the changelog's sections follow.

## Support and escalation

- **Bugs, wrong numbers and questions** go to [GitHub issues](https://github.com/Open-Quant/openquant/issues).
  The bug report form asks for the module and function, what happened, what you expected, the AFML
  section or other reference the behaviour should follow, a minimal reproducer and your
  environment. A numerical result that disagrees with its reference is a bug worth reporting even
  if you are not sure.
- **Security problems** are reported privately through a
  [GitHub security advisory](https://github.com/Open-Quant/openquant/security/advisories/new),
  never in a public issue. `SECURITY.md` defines what is in scope and aims for an acknowledgement
  within 7 days.
- **Contributions** follow `CONTRIBUTING.md`: open an issue first for anything larger than a small
  fix, link it from the pull request, and include a regression test that fails on the old code for
  every bug fix.
