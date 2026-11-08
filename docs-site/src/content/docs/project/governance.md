---
title: Governance
description: The rules OpenQuant holds its research, documentation, benchmarks and releases to, and where to report a problem.
status: authored
last_authored: '2026-11-07'
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
  path. It counts distinct explicit configuration keys, replacing a repeated key; it cannot
  infer omitted dataset or protocol choices. Concurrent writers can lose updates. Inspect the
  [trial identity contract](/workflows/research-notebook-contract/#trial-identity-is-part-of-the-method)
  before adapting a runbook. [Runbook 12](/runbooks/cpcv-deflated-sharpe/) illustrates selection bias.
- **Costs and assumptions are stated.** A runbook reports results net of an explicit cost per
  unit of turnover and names its universe and data source.
- **Controls with a known answer.** A runbook tests its method on data where the right answer is
  known: a no-signal control, with a predeclared false-positive criterion, and, where possible,
  a planted signal of known strength. A finite random control can show an apparent effect;
  inspect the uncertainty and declared criterion rather than requiring every estimate to be zero. The committed data is SYNTHETIC; nothing on this site describes a real market.
- **Pre-registration.** A runbook writes its hypotheses and pass/fail rules before the full run
  and ends with a promotion decision. The [research gallery](/runbooks/) lists every decision.

Enforcement is partial: `mda_importance` requires label spans unless the caller explicitly
opts out. Notebook lint checks structure, not whether hypotheses preceded analysis, every
trial was registered, or a method is free of leakage. The library cannot detect future
information already embedded in features or supplied probabilities. Review these research
claims separately from successful execution.

## Reproducibility

- **Notebooks are committed with their outputs.** The configured notebook workflow executes
  numbered notebooks on matching pull-request/main-push paths, nightly and manual dispatch,
  then compares sources, text outputs and exported SVGs. Text floats are rounded to four
  significant digits; stderr, raster-image bytes, execution counts, IDs and metadata are
  ignored. A partial `--only` check omits exported-figure comparison. See
  [Notebook Research Workflow](/workflows/notebook-research-workflow/).
- **Runbooks end with a reproducibility cell.** The footer records the supplied data hash,
  seed and optional configuration/trial fields. Its caller decides which inputs are covered.
  With `OPENQUANT_NOTEBOOK_PINNED=1`, actual commit and version strings are replaced by a
  stable reference to the containing commit and lockfiles. This marker does not verify the
  working tree, installed dependencies or loaded extension. Retain actual provenance when
  collecting an independent result.
- **Datasets have content identifiers.** `dataset_hash` canonicalizes a frame's contents;
  row and column ordering are deliberately normalized. A hash does not preserve the data or
  establish the order used in a calculation. `record_dataset_hash` updates the provided
  manifest dictionary in memory; the caller must serialize it. `fetch` defaults to the
  committed synthetic sample, avoiding a market-data network request under the documented
  defaults. Custom sources and inherited environment overrides need their own provenance.
- **Figure exports remove avoidable variability.** `nbfigures.figure` uses a fixed SVG hash
  salt and omits date/creator metadata. Byte stability still depends on the drawing code,
  data, Matplotlib and fonts. The comparator normalizes some SVG details; passing it does
  not prove visual equality. Inspect both themes when changing a figure.

## Documentation status

Every page on this site declares a `status`, shown as the pill at the top of the page:
`generated` (emitted by a script from a source file, not read by anyone), `draft` (known to be
incomplete), `authored` (written against the code, not yet read by a human), `reviewed` (read by
a human) and `validated` (also checked against the code). `check:content-schema` fails if a status
is missing or if the date behind it is older than the page's last change.

The example gates have narrower scopes than the status labels. `check:examples` compiles Rust
fences under `modules/` and the complete Rust core workflow without executing them;
other Rust fences are outside its scope.
`check:python-examples` scans Markdown and MDX pages throughout the content tree and runs each
eligible Python fence independently. Fences marked `doc-check=skip` receive syntax checks only,
and are listed in the gate output. Passing these checks does not verify every prose claim or
the statistical validity of an example. Pages written by an AI assistant stop at `authored`.
The [coverage dashboard](/coverage/) counts pages by status, not by test coverage.

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
  and ships as a wheel built with maturin (it is a Python extension module, not a Rust library).
  The steps are in `docs/publishing.md`.
- **Release gate.** Pushing a `v*` tag runs `.github/workflows/release.yml`: format, clippy with
  `-D warnings`, the fast test suite, a compile check of every benchmark, the long SADF test that
  is skipped on pull requests, `cargo publish --dry-run`, and abi3 wheels for Linux (x86_64,
  aarch64), macOS (x86_64, arm64) and Windows plus an sdist, each installed into a clean
  environment on Python 3.11 to 3.13 and run through the Quickstart. Only when all of that passes
  does it publish to PyPI and crates.io and create a GitHub Release with the version's changelog
  section as its notes. A manual Actions dispatch defaults to `dry_run=true`, which publishes
  nothing. Setting `dry_run=false` on a selected `v*` tag enables publication; the workflow
  rejects that publishing mode on a branch. Inspect the input and selected ref before dispatching.
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
