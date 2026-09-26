//! Rust implementations of the methods in Marcos López de Prado, *Advances in Financial
//! Machine Learning* (Wiley, 2018; "AFML"), and of the portfolio and HPC tooling around them.
//!
//! Each module maps to one or more AFML chapters, and its documentation cites the section and
//! snippet it implements. Where a function departs from the book or from the `mlfinlab`
//! reference it mirrors, the item's documentation says so.
//!
//! # Modules by AFML chapter
//!
//! | Chapter | Modules |
//! | --- | --- |
//! | 2. Financial data structures | [`data_structures`], [`filters`], [`etf_trick`] |
//! | 3. Labeling | [`labeling`], [`util::volatility`] |
//! | 4. Sample weights | [`sampling`], [`sample_weights`], [`sb_bagging`] |
//! | 5. Fractionally differentiated features | [`fracdiff`] |
//! | 6. Ensemble methods | [`ensemble_methods`] |
//! | 7. Cross-validation in finance | [`cross_validation`] |
//! | 8. Feature importance | [`feature_importance`], [`fingerprint`] |
//! | 9. Hyper-parameter tuning | [`hyperparameter_tuning`] |
//! | 10. Bet sizing | [`bet_sizing`], [`ef3m`] |
//! | 11–12. Backtesting, walk-forward and CPCV | [`backtesting_engine`] |
//! | 13. Backtesting on synthetic data | [`synthetic_backtesting`] |
//! | 14. Backtest statistics | [`backtest_statistics`], [`risk_metrics`] |
//! | 15. Understanding strategy risk | [`strategy_risk`] |
//! | 16. Machine learning asset allocation | [`hrp`], [`hcaa`], [`onc`], [`cla`], [`portfolio_optimization`] |
//! | 17. Structural breaks | [`structural_breaks`] |
//! | 18–19. Entropy and microstructural features | [`microstructural_features`], [`codependence`] |
//! | 20. Multiprocessing and vectorization | [`hpc_parallel`], [`streaming_hpc`] |
//! | 21. Brute force and quantum computers | [`dynamic_allocation`], [`combinatorial_optimization`] |
//!
//! [`pipeline`] strings several of these together (events, signals, portfolio, risk,
//! backtest) for research workflows, [`data_processing`] loads, cleans and calendar-aligns
//! OHLCV bars, and [`util`] holds shared primitives such as [`util::fast_ewma`] and the common
//! [`util::InputError`].
//!
//! # Conventions
//!
//! - Series are ordered **oldest first**. Whether a function wants prices, log prices or
//!   returns, and whether returns are per period or annualised, is stated on each function.
//! - Fallible functions return a typed error per module (or [`util::InputError`]) rather
//!   than panicking; any remaining reachable panic is listed under `# Panics` on the item.
//! - Randomised functions take a seed or a generator, or have a variant that does, so that
//!   results can be reproduced.
//!
//! The same material, with worked examples and caveats, is on the documentation site's module
//! pages, which these docs follow.
//!
//! ```
//! use openquant::fracdiff::frac_diff_ffd;
//!
//! // Fractional differencing of order 1 is the ordinary first difference (AFML Snippet 5.3).
//! let diffed = frac_diff_ffd(&[1.0, 2.0, 4.0, 7.0], 1.0, 1e-5);
//! assert!(diffed[0].is_nan());
//! assert_eq!(&diffed[1..], &[1.0, 2.0, 3.0]);
//! ```
#![deny(missing_docs)]

pub mod backtest_statistics;
pub mod backtesting_engine;
pub mod bet_sizing;
pub mod cla;
pub mod codependence;
pub mod combinatorial_optimization;
pub mod cross_validation;
pub mod data_processing;
pub mod data_structures;
pub mod dynamic_allocation;
pub mod ef3m;
pub mod ensemble_methods;
pub mod etf_trick;
pub mod feature_importance;
pub mod filters;
pub mod fingerprint;
pub mod fracdiff;
pub mod hcaa;
pub mod hpc_parallel;
pub mod hrp;
pub mod hyperparameter_tuning;
pub mod labeling;
pub mod microstructural_features;
pub mod onc;
pub mod pipeline;
pub mod portfolio_optimization;
pub mod risk_metrics;
pub mod sample_weights;
pub mod sampling;
pub mod sb_bagging;
pub mod strategy_risk;
pub mod streaming_hpc;
pub mod structural_breaks;
pub mod synthetic_backtesting;
pub mod util;
