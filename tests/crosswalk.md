# Python → Rust test crosswalk

| Python tests (mlfinlab/tests) | Rust tests (openquant-rs) | Fixtures | Status |
| --- | --- | --- | --- |
| test_filters.py | crates/openquant/tests/filters.rs | tests/fixtures/filters/{dollar_bar_sample.csv, events.json} | ✅ ported |
| test_backtest_statistics.py | crates/openquant/tests/backtest_statistics.rs | tests/fixtures/backtest_statistics/dollar_bar_sample.csv | ✅ ported |
| test_bet_sizing.py | crates/openquant/tests/bet_sizing.rs | tests/fixtures/bet_sizing/{prob_dynamic_budget.json,reserve_fixture.json} | ✅ ported (reserve fit/return-params + broadcast semantics added) |
| test_cross_validation.py | crates/openquant/tests/cross_validation.rs | (inline fixtures) | ✅ ported |
| test_labels.py | crates/openquant/tests/labeling.rs | (inline fixtures) | ✅ ported |
| test_microstructural_features.py | crates/openquant/tests/microstructural_features.rs | tests/fixtures/microstructural_features/{tick_data.csv,tick_data_time_bars.csv,dollar_bar_sample.csv} | ✅ ported |
| test_portfolio_optimization.py | crates/openquant/tests/portfolio_optimization.rs | tests/fixtures/portfolio_optimization/{stock_prices.csv,mean_variance_fixture.json} | ✅ ported |
| test_sample_weights.py | crates/openquant/tests/sample_weights.rs | (inline fixtures) | ✅ ported |
| test_sampling.py | crates/openquant/tests/sampling.rs | (inline fixtures) | ✅ ported |
| test_run_data_structures.py | crates/openquant/tests/data_structures_run_imbalance.rs | (inline fixtures) | ✅ ported (run bars) |
| test_standard_data_structures.py | crates/openquant/tests/data_structures_standard.rs | (inline fixtures) | ✅ ported (standard bars) |
| test_time_data_structures.py | crates/openquant/tests/data_structures_standard.rs | (inline fixtures) | ✅ ported (time bars) |
| test_imbalance_data_structures.py | crates/openquant/tests/data_structures_run_imbalance.rs | (inline fixtures) | ✅ ported (imbalance bars) |
| test_structural_breaks.py | crates/openquant/tests/structural_breaks.rs | tests/fixtures/structural_breaks/dollar_bar_sample.csv | ✅ ported |
| test_ch10_snippets.py | crates/openquant/tests/ch10_snippets.rs | (inline fixtures) | ✅ ported |
| test_cla.py | crates/openquant/tests/cla.rs | tests/fixtures/portfolio_optimization/stock_prices.csv | ✅ ported |
| test_codependence.py | crates/openquant/tests/codependence.rs | tests/fixtures/codependence/random_state_42.csv | ✅ ported |
| test_ef3m.py | crates/openquant/tests/ef3m.rs | (inline fixtures) | ✅ ported |
| test_etf_trick.py | crates/openquant/tests/etf_trick.rs | tests/fixtures/etf_trick/{open_df.csv,close_df.csv,alloc_df.csv,costs_df.csv,rates_df.csv} | ✅ ported |
| test_fast_ewma.py | crates/openquant/tests/fast_ewma.rs | tests/fixtures/microstructural_features/tick_data.csv | ✅ ported |
| test_feature_importance.py | crates/openquant/tests/feature_importance.rs | (inline fixtures) | ✅ ported (core MDI/MDA/SFI + orthogonal/PCA + output file behavior) |
| test_fingerpint.py | crates/openquant/tests/fingerprint.rs | (inline fixtures) | ✅ ported (Rust-native deterministic models; linear/non-linear/pairwise + classification parity) |
| test_fracdiff.py | crates/openquant/tests/fracdiff.rs | tests/fixtures/backtest_statistics/dollar_bar_sample.csv | ✅ ported |
| test_futures_roll.py | crates/openquant/tests/futures_roll.rs | tests/fixtures/etf_trick/{open_df.csv,close_df.csv} | ✅ ported |
| test_hcaa.py | crates/openquant/tests/hcaa.rs | tests/fixtures/portfolio_optimization/stock_prices.csv | ✅ ported |
| test_hrp.py | crates/openquant/tests/hrp.rs | tests/fixtures/portfolio_optimization/stock_prices.csv | ✅ ported |
| test_onc.py | crates/openquant/tests/onc.rs | tests/fixtures/onc/breast_cancer.csv | ✅ ported |
| test_risk_metrics.py | crates/openquant/tests/risk_metrics.rs | tests/fixtures/portfolio_optimization/stock_prices.csv | ✅ ported |
| test_sb_bagging.py | crates/openquant/tests/sb_bagging.rs | (inline fixtures) | ✅ ported |
| test_volatility_features.py | crates/openquant/tests/volatility_features.rs | tests/fixtures/backtest_statistics/dollar_bar_sample.csv | ✅ ported |

Notes:
- Fixtures live under `openquant-rs/tests/fixtures/` and should be shared across Python/Rust tests.
- Update `Status` as modules are ported; add fixture file references and any tolerance notes.
