use pyo3::prelude::*;

use crate::helpers::to_py_err;

/// Chow-type Dickey-Fuller statistics for a single break at each candidate date.
///
/// AFML §17.3.1. For each candidate break index `tau` from `min_length` to
/// `len(log_prices) - min_length - 1`, regresses the log-price change `dy_t` on the lagged
/// level `y_{t-1}` with the regressor zeroed before `tau` (no intercept), and records the
/// t-statistic `beta / se(beta)`. Large values suggest a switch from a random walk to an
/// explosive process at `tau`. A singular regression gives `NaN`.
///
/// Parameters
/// ----------
/// log_prices : list[float]
///     Log prices, oldest first.
/// min_length : int
///     Minimum number of observations kept on each side of a candidate break.
///
/// Returns
/// -------
/// list[float]
///     One statistic per candidate break, `len(log_prices) - 2 * min_length` values; empty if
///     the series is shorter than `2 * min_length`.
///
/// Raises
/// ------
/// ValueError
///     If the regression has no rows (`min_length == 0` with a single price).
#[pyfunction(name = "get_chow_type_stat")]
fn sb_get_chow_type_stat(log_prices: Vec<f64>, min_length: usize) -> PyResult<Vec<f64>> {
    openquant::structural_breaks::get_chow_type_stat(&log_prices, min_length).map_err(to_py_err)
}

/// Chu-Stinchcombe-White CUSUM test on levels, with its critical values.
///
/// AFML §17.3.2. For each bar `t` from index 2 on, `sigma_t^2` is the mean of the squared
/// one-bar changes up to `t`, and the statistic is
/// `S_t = max_{n < t} (y_t - y_n) / (sigma_t * sqrt(t - n))` (absolute difference for
/// `"two_sided"`). The critical value is `sqrt(4.6 + ln(t - n))` at the maximising `n`.
///
/// Parameters
/// ----------
/// log_prices : list[float]
///     Log prices, oldest first; at least three.
/// test_type : str
///     `"one_sided"` (tests for an upward departure) or `"two_sided"`.
///
/// Returns
/// -------
/// tuple[list[float], list[float]]
///     `(critical_value, stat)`, each with `len(log_prices) - 2` values, for bars 2 onward.
///     Note the critical values come first.
///
/// Raises
/// ------
/// ValueError
///     If there are fewer than three prices or `test_type` is not `"one_sided"` or
///     `"two_sided"`.
#[pyfunction(name = "get_chu_stinchcombe_white_statistics")]
fn sb_get_chu_stinchcombe_white_statistics(
    log_prices: Vec<f64>,
    test_type: String,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let result =
        openquant::structural_breaks::get_chu_stinchcombe_white_statistics(&log_prices, &test_type)
            .map_err(to_py_err)?;
    Ok((result.critical_value, result.stat))
}

/// Supremum ADF and sub/super-martingale explosiveness statistics.
///
/// AFML §17.4.2-17.4.3, Snippets 17.1-17.4. Builds one regression row per bar from bar
/// `lags + 1` on, then returns, for each row from position `min_length` on, the supremum
/// over every backward-expanding window of at least `min_length` rows ending at that row of
/// the statistic below (`t` is the 0-based row position over the whole sample):
///
/// - `"linear"`: `dy` on `y_{t-1}`, `lags` lagged `dy`, a constant (if `add_const`) and `t`;
///   the t-statistic of `y_{t-1}`.
/// - `"quadratic"`: as `"linear"` plus `t^2` (Snippet 17.2 `ctt`).
/// - `"sm_poly_1"`: `y` on 1, `t`, `t^2`; `|beta(t^2)| / se`.
/// - `"sm_poly_2"`: `log y` on 1, `t`, `t^2`; `|beta(t^2)| / se`.
/// - `"sm_exp"`: `log y` on 1, `t`; `|beta(t)| / se`.
/// - `"sm_power"`: `log y` on 1, `log(t + 1)`; `|beta| / se`.
///
/// The `sm_*` models ignore `add_const` and `lags` beyond setting the first row, take the
/// absolute value because a trend of either sign is of interest, and need a positive series
/// when they take logs. Windows whose regression is singular are skipped; a row with no
/// usable window is `-inf`. Every window of every row is a separate regression, so the
/// cost grows quickly with the series length.
///
/// Parameters
/// ----------
/// series : list[float]
///     Log prices (or prices, for `sm_poly_1`), oldest first.
/// model : str
///     One of `"linear"`, `"quadratic"`, `"sm_poly_1"`, `"sm_poly_2"`, `"sm_exp"`,
///     `"sm_power"`.
/// add_const : bool
///     Whether the ADF models include a constant.
/// min_length : int
///     Minimum window length in regression rows; must be >= 1.
/// lags : int
///     Number of lagged differences in the ADF regression (lags `1..=lags`).
///
/// Returns
/// -------
/// list[float]
///     One statistic per regression row from position `min_length` on, i.e.
///     `len(series) - lags - 1 - min_length` values; empty if there are no more than
///     `min_length` rows.
///
/// Raises
/// ------
/// ValueError
///     If `model` is unknown, the series has fewer than two values or no more than
///     `lags + 1`, or `min_length` is 0 (an empty window).
#[pyfunction(name = "get_sadf")]
#[pyo3(signature = (series, model, add_const, min_length, lags))]
fn sb_get_sadf(
    series: Vec<f64>,
    model: String,
    add_const: bool,
    min_length: usize,
    lags: usize,
) -> PyResult<Vec<f64>> {
    openquant::structural_breaks::get_sadf(
        &series,
        &model,
        add_const,
        min_length,
        openquant::structural_breaks::SadfLags::Fixed(lags),
    )
    .map_err(to_py_err)
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "structural_breaks")?;
    m.add_function(wrap_pyfunction!(sb_get_chow_type_stat, &m)?)?;
    m.add_function(wrap_pyfunction!(sb_get_chu_stinchcombe_white_statistics, &m)?)?;
    m.add_function(wrap_pyfunction!(sb_get_sadf, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("structural_breaks", m)?;
    Ok(())
}
