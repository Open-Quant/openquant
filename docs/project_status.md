# Project Status

## Current delivery state (2026-02-13)
- AFML gap modules tracked in epic `OQ-mef` are implemented in `crates/openquant/src/`:
  `ensemble_methods`, `hyperparameter_tuning`, `backtesting_engine`, `synthetic_backtesting`,
  `strategy_risk`, `hpc_parallel`, `combinatorial_optimization`, and `streaming_hpc`.
- Invalid arguments return `Err` rather than panicking. `OQ-mef.6` added `_checked` variants but
  left the panicking wrappers public; #36 (2026-09-20) removed the wrappers, so the plain name is
  the fallible one, and converted the remaining `assert!`/index panics in `bet_sizing`, `filters`,
  `backtest_statistics`, `data_structures`, `sampling`, `microstructural_features`, `ef3m`,
  `util::volatility` and `util::fast_ewma`. Covered by `crates/openquant/tests/invalid_input.rs`
  and `python/tests/test_invalid_input_raises.py`. The `expect`s that remain in non-test code
  guard internal invariants, not caller input. Not yet audited: slice indexing on mismatched
  lengths in modules outside that list.
- Notebook-first platform artifacts are present:
  Python bindings (`crates/pyopenquant`), Python API package (`python/openquant`),
  notebook starter packs (`notebooks/python`, `notebooks/rust`), experiment scaffold (`experiments/`),
  and CI smoke jobs (the `python` jobs in `.github/workflows/ci.yml`).

## Reconciliation status
- Reconciliation source of truth: `docs/reconciliation_closure_history.md`.
- Most previously closed deliverables are now present on `main`.
- Of the two follow-ups raised by that reconciliation:
  - `OQ-ojp` (docs-site notebook workflow page + navigation links) landed in PR #14.
  - `OQ-det` (experiment plot artifact outputs and tests) was closed in the tracker before its
    commit `27a2007` reached `main`; it was re-ported in PR #137 (issue #33).
    `experiments/run_pipeline.py` now writes `equity_curve.svg` and `drawdown.svg` for every
    run, including grid sub-runs. See `docs/decisions/0002-stranded-work-outcomes.md`.

## Quality and CI posture
- What each CI workflow enforces, and on which events, is tabulated in
  `docs/stabilization_productionization.md` (CI/Automation).
- Remaining work is tracked as GitHub issues #31-#61; the scope and evidence behind them is in
  `docs/design/production-readiness-brief.md`.
