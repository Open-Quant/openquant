use pyo3::prelude::*;

/// First `size` fractional-differencing weights of order `diff_amt`.
///
/// The weights of `(1 - B)^d` follow `w_0 = 1`, `w_k = -w_{k-1} (d - k + 1) / k` (AFML
/// Snippet 5.1). They are returned oldest lag first, so the last element is `w_0 = 1` and
/// the one before it is `-diff_amt`. No numeric-domain validation is performed after
/// argument conversion.
///
/// Parameters
/// ----------
/// diff_amt : float
///     Differencing order `d` (may be fractional).
/// size : int
///     Number of weights to return.
///
/// Returns
/// -------
/// list[float]
///     `size` weights, oldest lag first (empty when `size` is 0).
#[pyfunction(name = "get_weights")]
fn fracdiff_get_weights(diff_amt: f64, size: usize) -> Vec<f64> {
    openquant::fracdiff::get_weights(diff_amt, size)
}

/// Fixed-width-window fractional-differencing weights of order `diff_amt`.
///
/// Uses the recursion of `get_weights` and stops before the first weight whose absolute
/// value is below `thresh`, or once `lim` weights exist, whichever comes first (AFML
/// Snippet 5.3). A `thresh` of 0 or less (or NaN) never satisfies the strict stopping
/// comparison, even for zero weights: it means "cap only" and yields exactly `lim` weights.
/// `thresh` is a floor on a single weight's magnitude, not a share of cumulative weight as
/// in `frac_diff`.
///
/// Parameters
/// ----------
/// diff_amt : float
///     Differencing order `d` (may be fractional).
/// thresh : float
///     Minimum absolute weight to keep.
/// lim : int
///     Maximum number of weights.
///
/// Returns
/// -------
/// list[float]
///     At most `lim` weights, oldest lag first (last element `1.0`); empty when `lim` is 0.
#[pyfunction(name = "get_weights_ffd")]
fn fracdiff_get_weights_ffd(diff_amt: f64, thresh: f64, lim: usize) -> Vec<f64> {
    openquant::fracdiff::get_weights_ffd(diff_amt, thresh, lim)
}

/// Fractionally difference a series with an expanding window.
///
/// Output `t` applies the first `t + 1` weights of `(1 - B)^d` to `series[0..=t]` (AFML
/// Snippet 5.2). To compute the leading NaN count, generate `len(series)` oldest-first
/// weights, normalize their cumulative absolute sum, and count entries strictly greater
/// than `thresh`. This count depends on batch length. `thresh=1.0` skips nothing (the
/// mlfinlab default); AFML uses 0.01. Growing windows can introduce drift (AFML 5.5.1), so
/// prefer `frac_diff_ffd` for features. Pass levels (prices or log prices), not returns.
/// Numeric domains are not validated; a NaN input poisons every later output, even at
/// zero weight. Neither function guarantees stationarity. Cost is O(n^2).
///
/// Parameters
/// ----------
/// series : list[float]
///     Levels, oldest first.
/// diff_amt : float
///     Differencing order `d`.
/// thresh : float
///     Threshold on normalized cumulative absolute weights used to count leading NaNs.
///
/// Returns
/// -------
/// list[float]
///     Differenced series, same length as `series`, with leading NaNs.
#[pyfunction(name = "frac_diff")]
fn fracdiff_frac_diff(series: Vec<f64>, diff_amt: f64, thresh: f64) -> Vec<f64> {
    openquant::fracdiff::frac_diff(&series, diff_amt, thresh)
}

/// Fractionally difference a series with a fixed-width window (FFD).
///
/// The weights are `get_weights_ffd(diff_amt, thresh, len(series))`; each output is their
/// dot product with the matching trailing window, so the first `len(weights) - 1` outputs
/// are NaN (AFML Snippet 5.3). A threshold too small for the data gives a window as long as
/// the series and a single eligible output (which may still be nonfinite). Extending a
/// batch that hit its cap can change the window and earlier output availability. Each
/// dot product uses values at or before its own position. Pass levels (prices or log
/// prices), oldest first, not returns. Numeric domains are not validated; a NaN poisons
/// every window containing it, even at zero weight. Stationarity is not guaranteed.
///
/// Parameters
/// ----------
/// series : list[float]
///     Levels, oldest first.
/// diff_amt : float
///     Differencing order `d`.
/// thresh : float
///     Minimum absolute weight kept in the window (see `get_weights_ffd`).
///
/// Returns
/// -------
/// list[float]
///     Differenced series, same length as `series`, with leading NaNs.
#[pyfunction(name = "frac_diff_ffd")]
fn fracdiff_frac_diff_ffd(series: Vec<f64>, diff_amt: f64, thresh: f64) -> Vec<f64> {
    openquant::fracdiff::frac_diff_ffd(&series, diff_amt, thresh)
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "fracdiff")?;
    m.add_function(wrap_pyfunction!(fracdiff_get_weights, &m)?)?;
    m.add_function(wrap_pyfunction!(fracdiff_get_weights_ffd, &m)?)?;
    m.add_function(wrap_pyfunction!(fracdiff_frac_diff, &m)?)?;
    m.add_function(wrap_pyfunction!(fracdiff_frac_diff_ffd, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("fracdiff", m)?;
    Ok(())
}
