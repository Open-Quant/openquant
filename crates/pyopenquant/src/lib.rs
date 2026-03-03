mod helpers;

mod backtest_stats;
mod bars;
mod bet_sizing;
mod cla;
mod codependence;
mod data;
mod ef3m;
mod ensemble;
mod fast_ewma;
mod filters;
mod fracdiff;
mod hcaa;
mod hrp;
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

#[pymodule]
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
    Ok(())
}
