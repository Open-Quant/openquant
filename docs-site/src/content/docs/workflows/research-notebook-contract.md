---
title: Notebook contract
description: The sections, reproducibility footer and committed-output rules every notebook under notebooks/python/ follows, what `just notebooks-lint` checks, and how a runbook differs from an API tour.
status: authored
last_authored: '2026-11-07'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 4
---

The numbered `NN_*.ipynb` notebooks under `notebooks/python/` are committed with outputs
and selected by the CI runner. This contract makes their claims checkable: a reader knows
where to find the claim, its data and decision, and can rerun it under the stated environment
and comparison tolerances. API-tour outputs illustrate calls rather than establish research
results. `just notebooks-lint` checks the structural rules on every pull request. How to run
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
| Docs page | One under Runbooks, and an entry in the [research gallery](/runbooks/) | Listed under API tours in the gallery |

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
| `09` to `14` | Runbooks, each with a page under Runbooks |

A tour that grows a question worth answering becomes a runbook: add the research sections,
an appropriate trial record and controls, and change the title. Do not add a hypothesis to a tour and leave the
rest out.

## Runbook sections

The level-2 (`##`) headings of a runbook are exactly these, in this order. Use `###` subsections
for anything else. Runbooks 09 to 14 are the reference implementations.

1. **Setup.** Imports and one parameters cell. Every parameter that could change a result is set
   here, before the first full run, with a comment saying why it has that value. The seed is set
   here. The data source defaults to the committed SYNTHETIC sample (`openquant.data.fetch` with no
   source). Document how a reader supplies their own data. Several notebooks accept
   `OPENQUANT_RUNBOOK_SOURCE` and similar variables; notebook 10 instead exposes a `SOURCE`
   parameter. The runner does not translate one setting into the other.
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
   return-based strategy configuration in an `openquant.evaluation.TrialRegistry` before selecting
   among them, including configurations added after the first look (mark them post hoc).
   A feature sweep or variance-only comparison without Sharpe-based selection still needs an
   explicit inventory of configurations and post hoc choices; notebooks 09 and 10 record these
   without computing a DSR. If any fitted feature has
   memory (see below), say how the Method controls for it.
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
   (tested), trial count recorded and deflated against when evaluating selected Sharpe ratios,
   costs included where returns are modeled, seeds fixed, data
   labelled, the null control for features with memory (below) where one applies, and what the
   notebook does not cover. An unticked item says why.
9. **Reproducibility.** The footer below, as the last cell.

Every annualised statistic states the observation frequency of the **return stream being
scored**, not merely the frequency of the input prices. Overlapping event outcomes are not
one-minute account returns just because the labels came from minute bars. Construct an
account return path before assigning it a calendar annualisation factor, or report an
unannualised event statistic. Even a correctly annualised synthetic Sharpe is not evidence
of trading performance.

For revised or delayed data, record the availability basis and selection cutoff separately
from the observation period. Retain the exact selected values and versions, the selection
policy, raw snapshot and retrieval evidence alongside the notebook. A footer data hash
identifies content under its hashing rules; it does not prove that content was available
at a historical decision. Use the [point-in-time walkthrough](/workflows/point-in-time-data/)
and its [evidence checklist](/workflows/point-in-time-data/#interpret-the-evidence-before-using-a-result)
to separate synthetic checks, provider replay and live receipt evidence. The notebook
linter does not verify these availability claims.

## Trial identity is part of the method

`TrialRegistry.record` identifies a trial by canonical configuration JSON, not by its returns.
Recording the same configuration again replaces that entry. A persistent file therefore
preserves only the distinctions actually included in the configuration: include the data
revision/hash, evaluation period, relevant model and preprocessing settings, cost assumptions
and code/protocol revision when those distinguish experiments. Keep return frequency
consistent within the trial comparison. Do not assume that a footer hash also enters the
registry key.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from openquant.evaluation import TrialRegistry

with TemporaryDirectory() as directory:
    registry = TrialRegistry(Path(directory) / "trials.json")
    config = {"feature": "ret_1", "data_revision": "A", "cost_bps": 5.0}
    registry.record(config, [0.01, -0.02, 0.03])
    registry.record(config, [0.02, -0.01, 0.01])  # changed returns, same identity
    assert registry.n_trials == 1
    registry.record({**config, "data_revision": "B"}, [0.02, -0.01, 0.01])
    assert registry.n_trials == 2
```

This example demonstrates bookkeeping, not a rule to pool unrelated datasets into one DSR.
Define the comparison population and preserve all attempted configurations before selection.
The existing runbooks' keys cover their stated grids; when adapting a notebook, inspect its
actual `record(...)` call before relying on a persistent registry to track new choices.

## Features with memory: a null through the same splits

Purged k-fold and CPCV fit the model for a middle fold on the samples after it as well. Purging
and the embargo remove training labels that overlap the test fold; they do not remove what a
model learns from the later samples. For a feature with memory (a price level, a weakly
differenced price such as FFD at small $d$, a long moving average or volatility ratio) that is
enough to score above chance on random walks: runbook 14 found purged 5-fold CV giving the log
price an annualised net Sharpe ratio of 1.85 on the `SYN_*` sample
([#217](https://github.com/Open-Quant/openquant/issues/217); the
[`cross_validation` page](/modules/cross-validation/#features-with-memory-a-bias-purging-does-not-remove)
explains the mechanism). So a runbook that fits a model on any such feature must:

- **Run a no-signal null through the same splits,** with features *and* labels rebuilt from the
  null data, and judge the result against that null, not against chance. The null is a
  zero-signal version of the generator for simulated data, or
  `cross_validation.bootstrap_returns` (demeaned returns drawn with replacement) for real data,
  with `cross_validation.null_score_distribution` running the pipeline. Permuting the labels
  alone, or shuffling the returns without replacement (which pins every path to the real end
  point), is not this null.
- **Or evaluate walk-forward,** with `cross_validation.walk_forward_splits`, which tests the same
  folds with models fitted only on earlier samples. A runbook that compares features with
  different memory (levels against returns, say) should report both schemes, and the null
  through each.
- **Not rest a conclusion on pooled out-of-fold AUC** for such features. It compares events at
  different times and credits slowly varying scores even walk-forward; use accuracy, log loss or
  net returns against the null.

State in the Method which control the runbook uses, and tick it in the self-review checklist.
Runbooks 11 to 14 were audited against this rule in
[#217](https://github.com/Open-Quant/openquant/issues/217); their pages say what the audit found.

## The reproducibility footer

The last cell of every notebook, runbook or tour, is a code cell under `## Reproducibility` that
calls `nbrepro.footer` (`notebooks/python/nbrepro.py`):

```python doc-check=skip
import nbrepro

nbrepro.footer(
    seed=SEED,
    data_hash=meta["dataset_hash"],  # or nbrepro.frame_hash(frame) for data built in memory
    config=CONFIG,  # printed as a 12-character hexadecimal digest
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
`OPENQUANT_NOTEBOOK_PINNED=1`. The footer then prints a stable statement referring to the
containing commit and lockfiles instead of the actual local revision and versions. This avoids
rewriting environment text on every run, but the marker is not a provenance check: it does
not verify a clean checkout, locked dependencies or that the loaded extension matches source.
CI separately installs locked Python dependencies, builds the extension and executes the
notebooks before comparison. For local evidence, record the actual revision, dirty state,
interpreter and extension paths alongside the result.

In Jupyter or `execute_notebook_cells.py`, the footer prints actual commit/version values
when `OPENQUANT_NOTEBOOK_PINNED` is not `1`. An inherited value of `1` still suppresses them;
unset it when collecting provenance for an independent run.

`nbrepro.frame_hash(frame)` is `openquant.data.dataset_hash` with floats rounded to 10 decimal
places. This reduces sensitivity to small floating-point differences, but values on opposite
sides of a rounding boundary can still hash differently; it is not a guarantee of platform
invariance. Pass `decimals=None` to hash the unrounded frame.

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

`just notebooks-run` executes the notebooks in fresh Jupyter kernels and rewrites their
outputs. `just notebooks-verify` only compares current files with `HEAD`; it does not execute
anything. CI runs these two steps in that order. For a fresh comparison without rewriting
notebooks or site figures, use `just notebooks-run --check`.

The comparator checks cell sources and output structure, normalizes decimal output numbers
to four significant digits, ignores stderr and raster image bytes, and normalizes exported
SVG identifiers and embedded raster bytes. It does not prove visual equality. `--only`
limits notebooks and omits exported SVG comparison; a full run is needed for that figure
check. See `notebooks/python/README.md` and `scripts/run_notebooks.py` in that directory.

The comparator also ignores execution counts, cell IDs, and cell/notebook metadata.
Notebook lint separately requires execution counts on nonempty code cells; neither check
replaces recording the actual environment and source provenance for a research run.

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
- **outputs**: every nonempty code cell has an execution count, no output is an error, no output contains a machine
  path, and no stdout or displayed result prints a package version.

The lint cannot establish that a hypothesis preceded the first experiment, that every trial
was registered, or that the assertions detect leakage. Preserve a dated protocol and changes
to it, and review the method and trial keys separately from structural lint.

It reports every problem as `path: rule: message` and exits 1 if there are any. Run it on one
notebook with `just notebooks-lint notebooks/python/11_meta_labeling_triple_barrier.ipynb`.

The [research gallery](/runbooks/) is generated from the same headings
(`scripts/docs/generate_site_pages.py`): a notebook with `## Hypothesis` and `## Promotion
decision` sections is shown as a runbook, anything else as an API tour. After re-running a
notebook, regenerate it with `python3 scripts/docs/generate_site_pages.py --write`.

The lint checks structure, not substance. Whether a hypothesis was really written first, whether
the trial count is complete and whether the analysis follows from the numbers is for the reviewer,
with the self-review checklist as the starting point.

## Adding a notebook

1. Copy the closest existing notebook of the same kind; runbooks 09 to 14 show every section.
2. Number it `NN_short_name.ipynb` with the next free number.
3. For a runbook, write the hypothesis and fix the parameters cell before the first full run, and
   add a page under `docs-site/src/content/docs/runbooks/`.
4. End with the `nbrepro.footer` cell.
5. Run `just notebooks-run --only NN`, then `just notebooks-lint`, and commit the notebook and its
   figures.
