---
title: Notebook contract
description: The sections, reproducibility footer and committed-output rules every notebook under notebooks/python/ follows, what `just notebooks-lint` checks, and how a runbook differs from an API tour.
status: authored
last_authored: '2026-09-26'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 4
---

Every notebook under `notebooks/python/` is committed with its outputs and executed in CI, so a
reader takes its numbers as evidence. This contract makes that evidence checkable: a reader knows
where to find the claim, the data behind it and the decision, and anyone can rerun it and get the
same numbers. `just notebooks-lint` checks the structural rules on every pull request. How to run
the notebooks, and what else CI checks, is on [Notebook Research Workflow](/workflows/notebook-research-workflow/).

## Two kinds of notebook

A notebook is either a **runbook** (a research notebook) or an **API tour**. Its title says which.

| | Runbook | API tour |
|---|---|---|
| Title (first cell) | `# Runbook: <title>` | `# API tour: <title>` |
| Purpose | Tests a hypothesis stated before the first run and ends with a promotion decision | Shows how a few calls fit together, on small synthetic inputs |
| Sections | All nine below, in order | `## Setup` first, `## Reproducibility` last, free sections in between |
| May claim | Whatever the hypothesis, controls and trial count support | Nothing about performance. Its numbers illustrate the API; they are not evidence |
| Forbidden sections | None | `## Hypothesis`, `## Promotion decision`, `## Self-review checklist` |
| Reproducibility footer | Required | Required |
| Docs page | One under Runbooks | None required |

The notebooks today:

| Notebook | Kind |
|---|---|
| `01_event_labeling_and_pipeline` | API tour: `pipeline.run_mid_frequency_pipeline_frames` and CUSUM events |
| `02_purged_cv_and_seq_bootstrap` | API tour: indicator matrix, uniqueness and the sequential bootstrap |
| `03_feature_diagnostics` | API tour: `viz` payloads for feature importance and regimes |
| `04_portfolio_construction` | API tour: inverse-variance, minimum-volatility and maximum-Sharpe weights |
| `05_risk_overlays_and_reality_check` | API tour: `research.run_flywheel_iteration` and its drawdown payload |
| `07_feature_engineering_discovery_loop` | API tour: `feature_diagnostics.feature_screen_report` |
| `08_algo_wheel_experiments` | API tour: `research.run_flywheel_grid` |
| `09` to `13` | Runbooks, each with a page under Runbooks |

A tour that grows a question worth answering becomes a runbook: add the research sections, a
trial registry and controls, and change the title. Do not add a hypothesis to a tour and leave the
rest out.

## Runbook sections

The level-2 (`##`) headings of a runbook are exactly these, in this order. Use `###` subsections
for anything else. Runbooks 09 to 13 are the reference implementations.

1. **Setup.** Imports and one parameters cell. Every parameter that could change a result is set
   here, before the first full run, with a comment saying why it has that value. The seed is set
   here. The data source defaults to the committed SYNTHETIC sample (`openquant.data.fetch` with no
   source), and environment variables (`OPENQUANT_RUNBOOK_SOURCE` and similar) point it at the
   reader's own file without editing the notebook.
2. **Hypothesis.** Written before the first full run, and says so. Each hypothesis is falsifiable
   and states its pass criterion as a number: a test statistic and threshold, a deflated Sharpe
   ratio of at least 0.95, a count out of N paths. Say what would count against the candidate, and
   list secondary quantities that are reported but not tested.
3. **Data.** Where the data comes from, labelled SYNTHETIC when it is, and its content hash
   (`fetch(..., return_meta=True)` returns `dataset_hash`). Simulated data states its generator
   and seed. Controls go here too: a no-signal series, or series whose answer is known, so the
   method can be checked before it is believed.
4. **Method.** The pipeline, with the AFML snippet or paper each step comes from; the cost model;
   the cross-validation scheme with its purge and embargo; and the trial grid. Record every
   configuration in an `openquant.evaluation.TrialRegistry` before selecting among them, including
   configurations added after the first look (mark them post hoc).
5. **Results.** Tables and figures, with no interpretation beyond captions. Each result names the
   hypothesis it answers. Include the look-ahead and leakage checks as executed assertions, not
   prose.
6. **Analysis.** What the results say about each hypothesis, quoting the numbers in the outputs
   above: supported, not supported, or not testable here, and why. State the limits: what kind of
   data, how many paths, which choices were not varied.
7. **Promotion decision.** One sentence in bold that decides (promote, do not promote, or promote
   a narrower thing), then what later work should use or avoid, with caveats and follow-ups. A
   decision reached on synthetic data describes the *method*, and says so.
8. **Self-review checklist.** A Markdown checklist, ticked or not, covering at least: no look-ahead
   (tested), trial count recorded and deflated against, costs included, seeds fixed, data
   labelled, and what the notebook does not cover. An unticked item says why.
9. **Reproducibility.** The footer below, as the last cell.

Every annualised statistic states its annualisation factor, which is the bar frequency of the
data, not 252 unless the bars are daily. A Sharpe ratio annualised from one-minute synthetic bars
is large and means little; say so next to it.

## The reproducibility footer

The last cell of every notebook, runbook or tour, is a code cell under `## Reproducibility` that
calls `nbrepro.footer` (`notebooks/python/nbrepro.py`):

```python doc-check=skip
import nbrepro

nbrepro.footer(
    seed=SEED,
    data_hash=meta["dataset_hash"],  # or nbrepro.frame_hash(frame) for data built in memory
    config=CONFIG,  # printed as a 12-digit digest
    trials=registry.n_trials,  # any other keyword is printed on stdout
    stderr={"Monte Carlo hash": MC_HASH},  # correct but platform-dependent values
)
```

It records the four things a rerun needs: the **git commit**, the **data hash**, the **package
version** and the **seed**. They go to two streams:

- **stdout**: the data hash, the seed, the config digest and any other keyword. These describe
  what the notebook computed from, so they change only when the notebook does, and the
  committed-output check compares them.
- **stderr**: the git commit (with `(dirty working tree)` when there are uncommitted changes), the
  `pyopenquant` version and the Python, numpy and polars versions, plus any value passed in
  `stderr=`. These differ between machines and commits without the results changing, and the
  committed-output check ignores stderr. A dependency bump therefore cannot make a notebook stale
  just by printing a new version number.

When `just notebooks-run` executes the repository's notebooks, it sets
`OPENQUANT_NOTEBOOK_PINNED=1`, and the footer prints that the commit containing the notebook pins
the code and, through `uv.lock` and `Cargo.lock`, every version, instead of the local commit and
versions. The commit a committed notebook belongs to is the one that contains it, and CI
re-executes it at that commit and fails if the outputs differ, so the pinned text is accurate.
Printing the parent commit and the local versions would only rewrite every notebook on every run.
Run a notebook any other way (in Jupyter, or with `execute_notebook_cells.py` on your own data)
and the footer prints the real commit and versions.

`nbrepro.frame_hash(frame)` is `openquant.data.dataset_hash` with floats rounded to 10 decimals,
so data generated in memory with `exp` or `sin` hashes the same on every platform. Pass
`decimals=None` for the exact hash.

## Committed outputs

Notebooks are committed executed, and the outputs are part of the review:

- Every code cell has been executed, and no output is an error.
- No output contains a machine path: no temporary directory, home directory or CI checkout. Print
  a description ("temporary directory") instead.
- No stdout or displayed result names a package with its version. Versions go to stderr, which
  `nbrepro.footer` does.
- Nothing else volatile on stdout: no timings, no timestamps of the run. Print them to stderr if
  they are worth keeping at all.
- Figures are drawn with `nbfigures.figure(name, draw)`, which exports both site themes to
  `docs-site/public/figures/notebooks/`.

`just notebooks-run` executes the notebooks in a Jupyter kernel and rewrites them with their
outputs; `just notebooks-verify` (CI) fails if the committed outputs differ from a fresh run by more
than float noise. See `notebooks/python/README.md`.

## What `just notebooks-lint` checks

`notebooks/python/scripts/lint_notebooks.py` uses only the standard library and runs in the CI
`python-lint` job on every pull request. For each `notebooks/python/NN_*.ipynb` it checks:

- **kind**: the first cell is Markdown starting `# Runbook: ` or `# API tour: `, and no other cell
  has a level-1 heading;
- **sections**: a runbook's `##` headings are exactly the nine above, in order; a tour starts with
  Setup, ends with Reproducibility and has none of the research-only sections;
- **footer**: the last cell is a code cell in the Reproducibility section that calls
  `nbrepro.footer` with `seed=` and `data_hash=`, and its committed output has `data hash:` and
  `seed:` lines on stdout and `git sha:` and `package:` lines on stderr;
- **outputs**: every code cell is executed, no output is an error, no output contains a machine
  path, and no stdout or displayed result prints a package version.

It reports every problem as `path: rule: message` and exits 1 if there are any. Run it on one
notebook with `just notebooks-lint notebooks/python/11_meta_labeling_triple_barrier.ipynb`.

The lint checks structure, not substance. Whether a hypothesis was really written first, whether
the trial count is complete and whether the analysis follows from the numbers is for the reviewer,
with the self-review checklist as the starting point.

## Adding a notebook

1. Copy the closest existing notebook of the same kind; runbooks 09 to 13 show every section.
2. Number it `NN_short_name.ipynb` with the next free number.
3. For a runbook, write the hypothesis and fix the parameters cell before the first full run, and
   add a page under `docs-site/src/content/docs/runbooks/`.
4. End with the `nbrepro.footer` cell.
5. Run `just notebooks-run --only NN`, then `just notebooks-lint`, and commit the notebook and its
   figures.
