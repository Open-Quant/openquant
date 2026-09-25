## Why

A backtest's Sharpe ratio is a finite-sample estimate that was selected for being large.
`backtest_stats` already implements the corrections (PSR, DSR, MinTRL), but they take
pre-computed moments, and nothing records how many configurations were tried — so the
deflated Sharpe ratio is computed against a trial count the researcher has to remember.
The research runbooks (`runbook-cpcv-dsr`, `runbook-meta-labeling`) need one place that
turns a returns series into these statistics and keeps an honest trial count across runs
(work record `res-evaluation-module`, issue #44, requirement RQ-012).

## What Changes

- New pure-Python module `openquant.evaluation`, a thin layer over the compiled
  `backtest_stats` and `strategy_risk` bindings:
  - `return_moments`, `probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`,
    `expected_max_sharpe`, `minimum_track_record_length` computed from a returns series;
  - `meta_label_metrics`: precision, recall and F1 of a meta-labeling overlay and of the
    primary model it filters;
  - `strategy_failure_probability` over `strategy_risk.estimate_strategy_failure_probability`;
  - `TrialRegistry`, `Trial`, `config_hash`: a JSON trial registry persisted atomically at a
    user-chosen path, whose count and Sharpe-ratio dispersion feed the DSR.
- `openquant.evaluation` is exported from the package.
- No change to the Rust statistics: they were checked against hand-computed values and
  match the published formulas.

## Capabilities

### New Capabilities
- `research-evaluation`: statistics that decide whether a backtest's Sharpe ratio is
  evidence of skill, computed from a returns series, and a persistent trial registry.

### Modified Capabilities

## Impact

- New file `python/openquant/evaluation.py`; `python/openquant/__init__.py` exports it.
- Tests `python/tests/test_evaluation.py` with a committed fixture
  `python/tests/fixtures/evaluation_returns.csv` and its generator.
- Docs: new authored page `docs-site/src/content/docs/modules/evaluation.md`, registered in
  `moduleDocs.ts`, the sidebar, the module index, the API inventory and the coverage page.
- No new dependencies: the module uses the standard library and the existing bindings.
