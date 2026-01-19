# Python → Rust test crosswalk

| Python tests (mlfinlab/tests) | Rust tests (openquant-rs) | Fixtures | Status |
| --- | --- | --- | --- |
| test_filters.py | crates/openquant/tests/filters.rs | tests/fixtures/filters/{dollar_bar_sample.csv, events.json} | ✅ ported |
| test_backtest_statistics.py | (tbd) | (tbd) | |
| test_bet_sizing.py | (tbd) | (tbd) | |
| test_ch10_snippets.py | (tbd) | (tbd) | |
| test_cla.py | (tbd) | (tbd) | |
| test_codependence.py | (tbd) | (tbd) | |
| test_cross_validation.py | (tbd) | (tbd) | |
| test_ef3m.py | (tbd) | (tbd) | |
| test_etf_trick.py | (tbd) | (tbd) | |
| test_fast_ewma.py | (tbd) | (tbd) | |
| test_feature_importance.py | (tbd) | (tbd) | |
| test_fingerpint.py | (tbd) | (tbd) | |
| test_fracdiff.py | (tbd) | (tbd) | |
| test_futures_roll.py | (tbd) | (tbd) | |
| test_hcaa.py | (tbd) | (tbd) | |
| test_hrp.py | (tbd) | (tbd) | |
| test_imbalance_data_structures.py | (tbd) | (tbd) | |
| test_labels.py | (tbd) | (tbd) | |
| test_mean_variance.py | (tbd) | (tbd) | |
| test_microstructural_features.py | (tbd) | (tbd) | |
| test_onc.py | (tbd) | (tbd) | |
| test_risk_metrics.py | (tbd) | (tbd) | |
| test_run_data_structures.py | (tbd) | (tbd) | |
| test_sample_weights.py | (tbd) | (tbd) | |
| test_sampling.py | (tbd) | (tbd) | |
| test_sb_bagging.py | (tbd) | (tbd) | |
| test_standard_data_structures.py | (tbd) | (tbd) | |
| test_structural_breaks.py | (tbd) | (tbd) | |
| test_time_data_structures.py | (tbd) | (tbd) | |
| test_volatility_features.py | (tbd) | (tbd) | |

Notes:
- Fixtures live under `openquant-rs/tests/fixtures/` and should be shared across Python/Rust tests.
- Update `Status` as modules are ported; add fixture file references and any tolerance notes.
