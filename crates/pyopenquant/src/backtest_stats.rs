use pyo3::prelude::*;

use crate::helpers::{format_naive_datetimes, pair_timestamps_values};

#[pyfunction(name = "sharpe_ratio")]
fn bs_sharpe_ratio(returns: Vec<f64>, entries_per_year: f64, risk_free_rate: f64) -> f64 {
    openquant::backtest_statistics::sharpe_ratio(&returns, entries_per_year, risk_free_rate)
}

#[pyfunction(name = "information_ratio")]
fn bs_information_ratio(returns: Vec<f64>, benchmark: f64, entries_per_year: f64) -> f64 {
    openquant::backtest_statistics::information_ratio(&returns, benchmark, entries_per_year)
}

#[pyfunction(name = "probabilistic_sharpe_ratio")]
fn bs_probabilistic_sharpe_ratio(
    observed_sr: f64,
    benchmark_sr: f64,
    number_of_returns: usize,
    skewness: f64,
    kurtosis: f64,
) -> f64 {
    openquant::backtest_statistics::probabilistic_sharpe_ratio(
        observed_sr,
        benchmark_sr,
        number_of_returns,
        skewness,
        kurtosis,
    )
}

#[pyfunction(name = "deflated_sharpe_ratio")]
#[pyo3(signature = (
    observed_sr,
    sr_estimates,
    number_of_returns,
    skewness,
    kurtosis,
    estimates_param=false,
    benchmark_out=false
))]
fn bs_deflated_sharpe_ratio(
    observed_sr: f64,
    sr_estimates: Vec<f64>,
    number_of_returns: usize,
    skewness: f64,
    kurtosis: f64,
    estimates_param: bool,
    benchmark_out: bool,
) -> f64 {
    openquant::backtest_statistics::deflated_sharpe_ratio(
        observed_sr,
        &sr_estimates,
        number_of_returns,
        skewness,
        kurtosis,
        estimates_param,
        benchmark_out,
    )
}

#[pyfunction(name = "minimum_track_record_length")]
fn bs_minimum_track_record_length(
    observed_sr: f64,
    benchmark_sr: f64,
    skewness: f64,
    kurtosis: f64,
    alpha: f64,
) -> f64 {
    openquant::backtest_statistics::minimum_track_record_length(
        observed_sr,
        benchmark_sr,
        skewness,
        kurtosis,
        alpha,
    )
}

#[pyfunction(name = "timing_of_flattening_and_flips")]
fn bs_timing_of_flattening_and_flips(
    timestamps: Vec<String>,
    positions: Vec<f64>,
) -> PyResult<Vec<String>> {
    let target = pair_timestamps_values(timestamps, positions, "timestamps", "positions")?;
    let result = openquant::backtest_statistics::timing_of_flattening_and_flips(&target);
    Ok(format_naive_datetimes(result))
}

#[pyfunction(name = "average_holding_period")]
fn bs_average_holding_period(
    timestamps: Vec<String>,
    positions: Vec<f64>,
) -> PyResult<Option<f64>> {
    let target = pair_timestamps_values(timestamps, positions, "timestamps", "positions")?;
    Ok(openquant::backtest_statistics::average_holding_period(&target))
}

#[pyfunction(name = "bets_concentration")]
fn bs_bets_concentration(returns: Vec<f64>) -> Option<f64> {
    openquant::backtest_statistics::bets_concentration(&returns)
}

#[pyfunction(name = "all_bets_concentration")]
fn bs_all_bets_concentration(
    timestamps: Vec<String>,
    returns: Vec<f64>,
) -> PyResult<(Option<f64>, Option<f64>, Option<f64>)> {
    let data = pair_timestamps_values(timestamps, returns, "timestamps", "returns")?;
    Ok(openquant::backtest_statistics::all_bets_concentration(&data))
}

#[pyfunction(name = "drawdown_and_time_under_water")]
#[pyo3(signature = (timestamps, returns, dollars=false))]
fn bs_drawdown_and_time_under_water(
    timestamps: Vec<String>,
    returns: Vec<f64>,
    dollars: bool,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let data = pair_timestamps_values(timestamps, returns, "timestamps", "returns")?;
    Ok(openquant::backtest_statistics::drawdown_and_time_under_water(&data, dollars))
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "backtest_stats")?;
    m.add_function(wrap_pyfunction!(bs_sharpe_ratio, &m)?)?;
    m.add_function(wrap_pyfunction!(bs_information_ratio, &m)?)?;
    m.add_function(wrap_pyfunction!(bs_probabilistic_sharpe_ratio, &m)?)?;
    m.add_function(wrap_pyfunction!(bs_deflated_sharpe_ratio, &m)?)?;
    m.add_function(wrap_pyfunction!(bs_minimum_track_record_length, &m)?)?;
    m.add_function(wrap_pyfunction!(bs_timing_of_flattening_and_flips, &m)?)?;
    m.add_function(wrap_pyfunction!(bs_average_holding_period, &m)?)?;
    m.add_function(wrap_pyfunction!(bs_bets_concentration, &m)?)?;
    m.add_function(wrap_pyfunction!(bs_all_bets_concentration, &m)?)?;
    m.add_function(wrap_pyfunction!(bs_drawdown_and_time_under_water, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("backtest_stats", m)?;
    Ok(())
}
