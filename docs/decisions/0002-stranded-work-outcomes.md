# Decision: outcomes for the stranded feature commits and legacy branches

Status: proposed (accepted when the pull request that adds it merges)
Owner: Sean Koval
Date: 2026-09-24

## Context

Beads issues OQ-nbr.4, OQ-nbr.5 and OQ-det were closed, but their commits never reached
`main`. Neither did a fourth commit, on `feat/afml-real-data-notebook`. Each of the four
branches is one commit ahead of an old `main` (brief RQ-002, GitHub issue #33). Two legacy
branches from 2020, `feature-generation` and `feature/tests`, share no history with `main`.
This record gives each of the six an outcome. It also answers the brief's unresolved
decision 4: re-port `notebook_modeling.rs` as modeling glue for runbooks, or use
scikit-learn from Python.

## Outcomes

| Commit / branch | Content | Outcome | Reason |
| --- | --- | --- | --- |
| `59ac6fa` `feat/oq-nbr-4-validation` | `split_with_diagnostics`, CPCV splits, `count_train_test_overlaps`, `naive_kfold_splits` | **Re-ported** by hand (issue #33 pull request) | Needed by the validation bindings (#42). Adapted to typed errors and the #65 purge window. The original `cpcv_paths` returned C(N, k) splits, so it is now `cpcv_splits`, and `cpcv_paths` returns AFML §12.4's φ[N, k] paths. The original one-sided embargo was not taken; #134 owns that decision. Spec: `openspec/changes/res-recover-stranded-work`. |
| `27a2007` `feat/oq-det-plot-artifacts` | SVG equity and drawdown charts from `experiments/run_pipeline.py` | **Re-ported** | Small, dependency-free and deterministic. Now also written for every grid sub-run. The `tomli` fallback was dropped because the project requires Python ≥ 3.11. |
| `25b7b24` `feat/oq-nbr-5-modeling-lab` | `notebook_modeling.rs`: threshold "primary" and "meta" models, a bagged-model wrapper; `sb_bagging` changes | **Dropped** | See decision 4 below. Its `sb_bagging` warm-up change was a workaround for the non-sequential bootstrap that #90 fixes properly (PR #125). |
| `f31de34` `feat/afml-real-data-notebook` | 12 generated per-chapter notebooks (07–18), a generator, a runner, Stooq helpers | **Dropped** after mining (ideas below) | The numbering collides with `main`'s 07 and 08. Every notebook re-downloads Stooq data with no cache or recorded terms (RQ-011). Several cells show placeholder numbers or re-implement library functions (hard-coded feature importances, a Python FFD, lagged correlations as a "frontier"). The CDaR cell uses the input #102 corrected. The interpretation cells are fixed text, not read from the outputs. |
| `feature-generation` (69 commits, 2020-08-11 → 2020-10-01) | Pre-Rust Python prototype: cookiecutter layout, standard bars, labeling, sampling, filters, a 773 KB `data/raw/raw_data.csv` | **Delete** | No common ancestor with `main`. Every module is superseded by the Rust crate (`data_structures`, `labeling`, `sampling`, `filters`, `risk_metrics`). The CSV has no recorded source or terms. |
| `feature/tests` (61 commits) | Tests for the prototype's standard bars | **Delete** | An ancestor of `feature-generation` (8 commits behind it, 0 ahead), so it holds nothing the other branch lacks. |

Commit counts are from `git rev-list --count origin/main..<branch>` on 2026-09-24. The
issue's 66 and 59 were counted differently. If the 2020 prototype's history should stay
reachable, one tag on `feature-generation`'s tip (`96948ac`) keeps both branches, since
`feature/tests` is its ancestor.

## Decision 4: modeling glue for runbooks

Use scikit-learn from Python. Do not re-port `notebook_modeling.rs`.

- `notebook_modeling.rs` is not a model library. Its primary model is a one-feature
  threshold passed through a sigmoid, and the meta model is the same stump on features
  augmented with the primary probability. A Rust model zoo is outside the library's scope,
  which is the methods of AFML.
- AFML's own snippets fit scikit-learn estimators, and the runbooks (`runbook-meta-labeling`,
  `runbook-cpcv-dsr`) are Python notebooks. What they need from OpenQuant is purged splits,
  sample weights, CPCV paths and statistics. That is `bind-validation-backtest` (#42), which
  this re-port unblocks.
- scikit-learn becomes a notebook/dev dependency when the first runbook needs it. It is not
  a runtime dependency of the `openquant` package.

## Ideas kept from `f31de34`

For `res-notebook-runner`, the runbooks and `docs-research-gallery`. No issues are filed from
this list.

1. One short notebook per AFML chapter, each mapped to the module that implements the
   chapter, as a gallery index. Number them in a separate range so they cannot collide.
2. Generate notebooks from a declarative spec (title, setup cell, body, commentary) and
   execute them all with one runner script. The runner fails on the first error.
3. Load a shared cached data panel once per notebook, from the real-data layer, instead of
   downloading inside each notebook.
4. Leakage demonstration: count train/test label overlaps under naive k-fold and under
   purged k-fold. `naive_kfold_splits` and `count_train_test_overlaps` now provide this.
5. Fractional differentiation: compare lag-1 autocorrelation of raw and FFD returns (for
   `runbook-fracdiff`, using `openquant.fracdiff`, not a re-implementation).
6. Plot an equity curve and drawdown for every pipeline run (done in `experiments/`).
7. Compare IVP, minimum-variance and maximum-Sharpe weights side by side (for
   `runbook-hrp-portfolio`, with HRP added).
8. Write interpretation cells from computed values, not fixed text.

## Consequences

- After the pull request merges, the maintainer deletes the six branches:
  `feat/oq-nbr-4-validation`, `feat/oq-det-plot-artifacts`, `feat/oq-nbr-5-modeling-lab`,
  `feat/afml-real-data-notebook`, `feature-generation` and `feature/tests`. The commits stay
  in this record by hash.
- `SequentiallyBootstrappedBaggingClassifier::predict_proba`, added in `25b7b24`, is not
  ported. If runbooks need bagged probabilities from Rust, add it under #90.

## Links

- GitHub issue #33; `.ai-dlc/work/res-recover-stranded-work.toml`
- `docs/design/production-readiness-brief.md` (RQ-002, unresolved decision 4)
- `openspec/changes/res-recover-stranded-work`
