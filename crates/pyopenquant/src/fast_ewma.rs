use pyo3::prelude::*;

use crate::helpers::to_py_err;

/// Bias-corrected exponentially weighted moving average with span `window`.
///
/// Matches `mlfinlab.util.fast_ewma.ewma` and pandas' `ewm(span=window, adjust=True).mean()`:
/// with `alpha = 2 / (window + 1)`, output `t` is the weighted mean of every value so far,
/// with weight `(1 - alpha)^k` on the value `k` steps back. The first output equals the first
/// input, and `window = 1` returns the input unchanged. `window` is a span, not a hard
/// lookback. Values are not validated: a NaN makes that output and every later one NaN
/// (pandas would skip it).
///
/// Parameters
/// ----------
/// arr : list[float]
///     Input series, oldest first.
/// window : int
///     Span of the average; must be positive.
///
/// Returns
/// -------
/// list[float]
///     The moving average, same length as `arr` (empty for an empty input).
///
/// Raises
/// ------
/// ValueError
///     If `window` is 0.
#[pyfunction(name = "ewma")]
fn fast_ewma_ewma(arr: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    openquant::util::fast_ewma::ewma(&arr, window).map_err(to_py_err)
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "fast_ewma")?;
    m.add_function(wrap_pyfunction!(fast_ewma_ewma, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("fast_ewma", m)?;
    Ok(())
}
