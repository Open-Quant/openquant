# Project Status

## Current delivery state (2026-02-13)
- AFML gap modules tracked in epic `OQ-mef` are implemented in `crates/openquant/src/`:
  `ensemble_methods`, `hyperparameter_tuning`, `backtesting_engine`, `synthetic_backtesting`,
  `strategy_risk`, `hpc_parallel`, `combinatorial_optimization`, and `streaming_hpc`.
- Panic-based public API paths were migrated to typed errors under `OQ-mef.6`.
- Notebook-first platform artifacts are present:
  Python bindings (`crates/pyopenquant`), Python API package (`python/openquant`),
  notebook starter packs (`notebooks/python`, `notebooks/rust`), experiment scaffold (`experiments/`),
  and CI smoke workflows (`.github/workflows/python-bindings.yml`,
  `.github/workflows/notebooks-examples-smoke.yml`).

## Reconciliation status
- Reconciliation source of truth: `docs/reconciliation_closure_history.md`.
- Most previously closed deliverables are now present on `main`.
- Two acceptance-criteria mismatches remain tracked as open follow-ups:
  - `OQ-ojp`: docs-site notebook workflow page + navigation links.
  - `OQ-det`: experiment plot artifact outputs and tests.

## Quality and CI posture
- Core CI workflows are present for lint/test, benchmark regression, release checks, bindings smoke,
  and notebook/example smoke.
- Remaining work is focused on documentation-site parity and experiment artifact completeness
  (tracked by `OQ-ojp` and `OQ-det`).
