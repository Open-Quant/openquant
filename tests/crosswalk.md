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
| test_ef3m.py | (tbd) | (tbd) | |
| test_etf_trick.py | (tbd) | (tbd) | |
| test_fast_ewma.py | (tbd) | (tbd) | |
| test_feature_importance.py | (tbd) | (tbd) | |
| test_fingerpint.py | (tbd) | (tbd) | |
| test_fracdiff.py | (tbd) | (tbd) | |
| test_futures_roll.py | (tbd) | (tbd) | |
| test_hcaa.py | (tbd) | (tbd) | |
| test_hrp.py | (tbd) | (tbd) | |
| test_onc.py | (tbd) | (tbd) | |
| test_risk_metrics.py | (tbd) | (tbd) | |
| test_sb_bagging.py | (tbd) | (tbd) | |
| test_volatility_features.py | (tbd) | (tbd) | |

Notes:
- Fixtures live under `openquant-rs/tests/fixtures/` and should be shared across Python/Rust tests.
- Update `Status` as modules are ported; add fixture file references and any tolerance notes.
