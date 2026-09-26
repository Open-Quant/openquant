use pyo3::prelude::*;

use crate::helpers::{format_naive_datetimes, pair_timestamps_values, to_py_err};

/// Annualised Sharpe ratio of per-period returns.
///
/// Computes `(mean - risk_free_rate) / std * sqrt(entries_per_year)` with the sample standard
/// deviation (ddof=1), AFML 14.7.1. Nothing is validated: an empty list or a single return
/// gives NaN, and constant returns give an infinite ratio.
///
/// Parameters
/// ----------
/// returns : list[float]
///     Per-period returns.
/// entries_per_year : float
///     Number of return periods per year (252 for daily).
/// risk_free_rate : float
///     Per-period risk-free rate, in the same units as `returns` (not annualised).
///
/// Returns
/// -------
/// float
///     The annualised Sharpe ratio.
#[pyfunction(name = "sharpe_ratio")]
fn bs_sharpe_ratio(returns: Vec<f64>, entries_per_year: f64, risk_free_rate: f64) -> f64 {
    openquant::backtest_statistics::sharpe_ratio(&returns, entries_per_year, risk_free_rate)
}

/// Annualised information ratio against a constant per-period benchmark return.
///
/// Equivalent to `sharpe_ratio(returns - benchmark, entries_per_year, 0.0)`: the mean excess
/// return over its sample standard deviation (ddof=1), times `sqrt(entries_per_year)`. Nothing
/// is validated; fewer than two returns give NaN.
///
/// Parameters
/// ----------
/// returns : list[float]
///     Per-period returns.
/// benchmark : float
///     Per-period benchmark return, in the same units as `returns`.
/// entries_per_year : float
///     Number of return periods per year (252 for daily).
///
/// Returns
/// -------
/// float
///     The annualised information ratio.
#[pyfunction(name = "information_ratio")]
fn bs_information_ratio(returns: Vec<f64>, benchmark: f64, entries_per_year: f64) -> f64 {
    openquant::backtest_statistics::information_ratio(&returns, benchmark, entries_per_year)
}

/// Probabilistic Sharpe ratio: the probability that the true Sharpe ratio exceeds a benchmark.
///
/// AFML 14.7.2 (Bailey and Lopez de Prado, 2012). Returns
/// `Phi((SR - SR*) sqrt(n - 1) / sqrt(1 - skew SR + (kurt - 1) / 4 SR^2))`. The Sharpe ratios
/// must be per-period (not annualised; passing an annualised ratio silently overstates the
/// confidence), and `kurtosis` is raw kurtosis (3 for normal returns), not excess kurtosis.
///
/// Parameters
/// ----------
/// observed_sr : float
///     Observed per-period Sharpe ratio.
/// benchmark_sr : float
///     Per-period Sharpe ratio to beat.
/// number_of_returns : int
///     Number of returns the Sharpe ratio was estimated from.
/// skewness : float
///     Skewness of the returns.
/// kurtosis : float
///     Raw kurtosis of the returns (3 for a normal distribution).
///
/// Returns
/// -------
/// float
///     A probability in `[0, 1]`.
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

/// Deflated Sharpe ratio: the PSR against the best Sharpe ratio expected from skill-less trials.
///
/// AFML 14.7.3 (Bailey and Lopez de Prado, 2014). The benchmark is
/// `SR_0 = sigma_SR * ((1 - g) Z^-1(1 - 1/N) + g Z^-1(1 - 1/(N e)))`, with `g` the
/// Euler-Mascheroni constant, and the result is
/// `probabilistic_sharpe_ratio(observed_sr, SR_0, number_of_returns, skewness, kurtosis)`.
/// Units follow `probabilistic_sharpe_ratio`: per-period Sharpe ratios, raw kurtosis. The
/// formula assumes independent trials; correlated variations of one idea overstate `N`.
///
/// Parameters
/// ----------
/// observed_sr : float
///     Observed per-period Sharpe ratio of the selected strategy.
/// sr_estimates : list[float]
///     With `estimates_param=False`, every trial's per-period Sharpe ratio (`sigma_SR` is
///     their population standard deviation, `N` their count). With `estimates_param=True`,
///     `[sigma_SR, N]`.
/// number_of_returns : int
///     Number of returns `observed_sr` was estimated from.
/// skewness : float
///     Skewness of the returns.
/// kurtosis : float
///     Raw kurtosis of the returns (3 for a normal distribution).
/// estimates_param : bool, default False
///     Whether `sr_estimates` is `[sigma_SR, N]` rather than the trials' Sharpe ratios.
/// benchmark_out : bool, default False
///     Return the benchmark `SR_0` instead of the deflated Sharpe ratio.
///
/// Returns
/// -------
/// float
///     The deflated Sharpe ratio (a probability in `[0, 1]`), or `SR_0` when `benchmark_out`
///     is True.
///
/// Raises
/// ------
/// ValueError
///     If `sr_estimates` has fewer than two values, or `estimates_param` is True and the
///     number of trials `sr_estimates[1]` is NaN or not above 1.
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
) -> PyResult<f64> {
    openquant::backtest_statistics::deflated_sharpe_ratio(
        observed_sr,
        &sr_estimates,
        number_of_returns,
        skewness,
        kurtosis,
        estimates_param,
        benchmark_out,
    )
    .map_err(to_py_err)
}

/// Minimum track record length: the returns needed for the PSR to reach `1 - alpha`.
///
/// AFML 14.7.2 (Bailey and Lopez de Prado, 2012). Returns
/// `1 + (1 - skew SR + (kurt - 1) / 4 SR^2) (Z^-1(1 - alpha) / (SR - SR*))^2`, a number of
/// return periods. Units follow `probabilistic_sharpe_ratio`: per-period Sharpe ratios, raw
/// kurtosis. Only meaningful when `observed_sr > benchmark_sr`: the difference is squared, so
/// an underperforming strategy gets a finite positive answer and an equal one gets infinity.
///
/// Parameters
/// ----------
/// observed_sr : float
///     Observed per-period Sharpe ratio.
/// benchmark_sr : float
///     Per-period Sharpe ratio to beat.
/// skewness : float
///     Skewness of the returns.
/// kurtosis : float
///     Raw kurtosis of the returns (3 for a normal distribution).
/// alpha : float
///     Significance level in `[0, 1]` (e.g. 0.05).
///
/// Returns
/// -------
/// float
///     The minimum number of return periods.
///
/// Raises
/// ------
/// ValueError
///     If `alpha` is outside `[0, 1]`.
#[pyfunction(name = "minimum_track_record_length")]
fn bs_minimum_track_record_length(
    observed_sr: f64,
    benchmark_sr: f64,
    skewness: f64,
    kurtosis: f64,
    alpha: f64,
) -> PyResult<f64> {
    openquant::backtest_statistics::minimum_track_record_length(
        observed_sr,
        benchmark_sr,
        skewness,
        kurtosis,
        alpha,
    )
    .map_err(to_py_err)
}

/// Timestamps at which a position was closed or reversed.
///
/// AFML Snippet 14.1. A flattening is a bar where the position goes from non-zero to zero; a
/// flip is a bar where the position changes sign. The result is sorted, deduplicated and
/// always ends with the last timestamp of the input, so an open position is treated as closed
/// there.
///
/// Parameters
/// ----------
/// timestamps : list[str]
///     Bar timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is accepted),
///     in increasing order.
/// positions : list[float]
///     Target position at each timestamp.
///
/// Returns
/// -------
/// list[str]
///     Timestamps of flattenings and flips plus the last timestamp (empty for empty input).
///
/// Raises
/// ------
/// ValueError
///     If `timestamps` and `positions` differ in length, or a timestamp does not parse.
#[pyfunction(name = "timing_of_flattening_and_flips")]
fn bs_timing_of_flattening_and_flips(
    timestamps: Vec<String>,
    positions: Vec<f64>,
) -> PyResult<Vec<String>> {
    let target = pair_timestamps_values(timestamps, positions, "timestamps", "positions")?;
    let result = openquant::backtest_statistics::timing_of_flattening_and_flips(&target);
    Ok(format_naive_datetimes(result))
}

/// Average holding period of a position series, in days.
///
/// AFML Snippet 14.2. Tracks a size-weighted average entry time as the position grows, and
/// records a holding time each time it shrinks or flips, weighted by the size exited. Returns
/// the weighted mean of those holding times in days (86,400 seconds; whole seconds only).
///
/// Parameters
/// ----------
/// timestamps : list[str]
///     Bar timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is accepted),
///     in increasing order.
/// positions : list[float]
///     Target position at each timestamp.
///
/// Returns
/// -------
/// float | None
///     The average holding period in days, or None if the input is empty or the position is
///     never reduced.
///
/// Raises
/// ------
/// ValueError
///     If `timestamps` and `positions` differ in length, or a timestamp does not parse.
#[pyfunction(name = "average_holding_period")]
fn bs_average_holding_period(
    timestamps: Vec<String>,
    positions: Vec<f64>,
) -> PyResult<Option<f64>> {
    let target = pair_timestamps_values(timestamps, positions, "timestamps", "positions")?;
    Ok(openquant::backtest_statistics::average_holding_period(&target))
}

/// Normalised Herfindahl-Hirschman concentration of a set of returns.
///
/// AFML Snippet 14.3. Each return's share is `r_i / sum(r)` and the index is
/// `(HHI - 1/n) / (1 - 1/n)`: 0 when every bet contributed equally, 1 when one bet made
/// everything. Mixing signs makes the shares meaningless; pass one sign at a time, as
/// `all_bets_concentration` does.
///
/// Parameters
/// ----------
/// returns : list[float]
///     Returns of the bets, all of one sign.
///
/// Returns
/// -------
/// float | None
///     The concentration, or None for two or fewer returns or when they sum to zero.
#[pyfunction(name = "bets_concentration")]
fn bs_bets_concentration(returns: Vec<f64>) -> Option<f64> {
    openquant::backtest_statistics::bets_concentration(&returns)
}

/// Concentration of positive returns, negative returns, and bets over time.
///
/// AFML Snippet 14.3. Each component is `bets_concentration` over: the non-negative returns,
/// the negative returns, and the number of bets per calendar day from the first to the last
/// date (days without bets count as zero). AFML and mlfinlab group the time component by
/// month, so the time index here is higher on data with gaps; compare it only with itself.
///
/// Parameters
/// ----------
/// timestamps : list[str]
///     Bet timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is accepted),
///     in increasing order; the first and last define the day range.
/// returns : list[float]
///     Return of each bet.
///
/// Returns
/// -------
/// tuple[float | None, float | None, float | None]
///     `(positive, negative, time)` concentrations; each is None when `bets_concentration`
///     has too few values or they sum to zero.
///
/// Raises
/// ------
/// ValueError
///     If `timestamps` and `returns` differ in length, or a timestamp does not parse.
#[pyfunction(name = "all_bets_concentration")]
fn bs_all_bets_concentration(
    timestamps: Vec<String>,
    returns: Vec<f64>,
) -> PyResult<(Option<f64>, Option<f64>, Option<f64>)> {
    let data = pair_timestamps_values(timestamps, returns, "timestamps", "returns")?;
    Ok(openquant::backtest_statistics::all_bets_concentration(&data))
}

/// Drawdowns and time under water of a cumulative series.
///
/// AFML Snippet 14.4, reproducing mlfinlab. Despite the parameter name, `returns` must be a
/// cumulative series (equity, NAV or cumulative PnL). For every high-water mark followed by a
/// dip, reports the drawdown (`1 - trough / peak` for a positive series, or `peak - trough`
/// with `dollars=True`) and the time under water in years (365.25 days) from that high-water
/// mark to the next one that itself had a drawdown, or to the end of the series. That is not
/// the time to recovery: new highs without a dip after them extend it.
///
/// Parameters
/// ----------
/// timestamps : list[str]
///     Timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is accepted), in
///     increasing order.
/// returns : list[float]
///     Cumulative series value at each timestamp.
/// dollars : bool, default False
///     Report drawdowns as `peak - trough` instead of `1 - trough / peak`.
///
/// Returns
/// -------
/// tuple[list[float], list[float]]
///     `(drawdowns, time_under_water_years)`, one entry per high-water mark that was followed
///     by a dip (both empty for empty input).
///
/// Raises
/// ------
/// ValueError
///     If `timestamps` and `returns` differ in length, or a timestamp does not parse.
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
