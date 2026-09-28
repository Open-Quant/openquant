mod helpers;

mod backtest_stats;
mod backtesting_engine;
mod bars;
mod bet_sizing;
mod cla;
mod codependence;
mod cross_validation;
mod data;
mod dynamic_allocation;
mod ef3m;
mod ensemble;
mod fast_ewma;
mod feature_importance;
mod filters;
mod fracdiff;
mod hcaa;
mod hrp;
mod hyperparameter_tuning;
mod labeling;
mod microstructural;
mod onc;
mod pipeline;
mod portfolio;
mod risk;
mod sample_weights;
mod sampling;
mod sb_bagging;
mod strategy_risk;
mod streaming_hpc;
mod structural_breaks;
mod synthetic_bt;
mod volatility;

use pyo3::prelude::*;

// PyO3 0.28 made modules declare free-threading support by default. The bindings are not
// tested on a free-threaded (3.13t/3.14t) interpreter, so keep the pre-0.28 behaviour: such an
// interpreter re-enables the GIL when it imports this module.
#[pymodule(gil_used = true)]
fn _core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    risk::register(py, m)?;
    filters::register(py, m)?;
    sampling::register(py, m)?;
    labeling::register(py, m)?;
    bars::register(py, m)?;
    data::register(py, m)?;
    bet_sizing::register(py, m)?;
    portfolio::register(py, m)?;
    pipeline::register(py, m)?;
    fracdiff::register(py, m)?;
    fast_ewma::register(py, m)?;
    volatility::register(py, m)?;
    codependence::register(py, m)?;
    backtest_stats::register(py, m)?;
    sample_weights::register(py, m)?;
    microstructural::register(py, m)?;
    strategy_risk::register(py, m)?;
    ensemble::register(py, m)?;
    structural_breaks::register(py, m)?;
    synthetic_bt::register(py, m)?;
    ef3m::register(py, m)?;
    streaming_hpc::register(py, m)?;
    hrp::register(py, m)?;
    hcaa::register(py, m)?;
    onc::register(py, m)?;
    cla::register(py, m)?;
    sb_bagging::register(py, m)?;
    cross_validation::register(py, m)?;
    backtesting_engine::register(py, m)?;
    feature_importance::register(py, m)?;
    hyperparameter_tuning::register(py, m)?;
    dynamic_allocation::register(py, m)?;
    Ok(())
}
