## 1. Bindings

- [x] 1.1 `_core.cross_validation` and `openquant.cross_validation` (numpy index arrays)
- [x] 1.2 `_core.backtesting_engine` (`run_cpcv` over precomputed returns) and `assemble_cpcv_paths`
- [x] 1.3 `_core.feature_importance` (MDI, replayed MDA / SFI) and the estimator-driven wrappers
- [x] 1.4 `hyperparameter_tuning::sample_param_sets` in Rust; `_core.hyperparameter_tuning` and `purged_search`

## 2. Tests

- [x] 2.1 Python parity tests on the Rust fixtures for every binding
- [x] 2.2 No train index overlaps a test span; embargo honoured
- [x] 2.3 Feature importance raises without spans
- [x] 2.4 scikit-learn integration (skipped when scikit-learn is absent)

## 3. Docs

- [x] 3.1 Python sections and `api_surface: both` on the four module pages; API inventory
