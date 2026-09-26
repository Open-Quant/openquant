use pyo3::prelude::*;

use crate::helpers::to_py_err;
use pyo3::types::PyDict;

/// One M2N fit: `(mu_1, mu_2, sigma_1, sigma_2, p_1, error)`.
type M2nFitRow = (f64, f64, f64, f64, f64, f64);

/// The `order`-th centred moment `E[(x - m_1)^order]` from raw moments.
///
/// Uses the binomial expansion `sum_j C(order, j) (-m_1)^j m_{order - j}` with `m_0 = 1`, so
/// only the first `order` raw moments are read. Order 0 gives 1 and order 1 gives 0 (up to
/// rounding).
///
/// Parameters
/// ----------
/// moments : list[float]
///     Raw moments `[E[x], E[x^2], ...]`.
/// order : int
///     Order of the centred moment.
///
/// Returns
/// -------
/// float
///     The centred moment.
///
/// Raises
/// ------
/// ValueError
///     If `moments` has fewer than `max(order, 1)` entries.
#[pyfunction(name = "centered_moment")]
fn ef3m_centered_moment(moments: Vec<f64>, order: usize) -> PyResult<f64> {
    openquant::ef3m::centered_moment(&moments, order).map_err(to_py_err)
}

/// Raw moments `[E[x], E[x^2], ...]` from centred moments and the mean.
///
/// `central_moments` starts at the first centred moment, which is 0:
/// `[0, E[(x - mu)^2], E[(x - mu)^3], ...]`. Entry 0 of the output is `dist_mean` and entry
/// `n - 1` is `sum_k C(n, k) c_k mu^(n - k)` with `c_0 = 1`. The first centred moment is used
/// as given in that sum, so pass 0 there.
///
/// Parameters
/// ----------
/// central_moments : list[float]
///     Centred moments from order 1 upward, starting with 0.
/// dist_mean : float
///     Mean of the distribution.
///
/// Returns
/// -------
/// list[float]
///     Raw moments, as many as `central_moments` (at least one).
#[pyfunction(name = "raw_moment")]
fn ef3m_raw_moment(central_moments: Vec<f64>, dist_mean: f64) -> Vec<f64> {
    openquant::ef3m::raw_moment(&central_moments, dist_mean)
}

/// The most likely value of each mixture parameter over many EF3M fits.
///
/// López de Prado and Foreman (2014): for each of `mu_1`, `mu_2`, `sigma_1`, `sigma_2` and
/// `p_1` separately, the peak of a Gaussian kernel density over the fits, rounded to five
/// decimals (the `error` column is ignored). The bandwidth is `std * n^(-1/5)` (population
/// standard deviation, floored at `1e-6`) and the density is evaluated on `max(res, 10)`
/// evenly spaced points from the column's minimum to its maximum; a constant column returns
/// its value. The five modes may come from different runs and need not reproduce the
/// moments; check that the runs agree on which component is which.
///
/// Parameters
/// ----------
/// data : list[tuple[float, float, float, float, float, float]]
///     Fits as `(mu_1, mu_2, sigma_1, sigma_2, p_1, error)` rows, e.g. from `fit_m2n`.
/// res : int
///     Number of grid points for the density search (at least 10 are used).
///
/// Returns
/// -------
/// dict[str, float]
///     Keys `mu_1`, `mu_2`, `sigma_1`, `sigma_2` and `p_1`; empty if `data` is empty.
#[pyfunction(name = "most_likely_parameters")]
fn ef3m_most_likely_parameters(
    py: Python<'_>,
    data: Vec<(f64, f64, f64, f64, f64, f64)>,
    res: usize,
) -> PyResult<PyObject> {
    let rows: Vec<openquant::ef3m::FitResultRow> = data
        .into_iter()
        .map(|(mu_1, mu_2, sigma_1, sigma_2, p_1, error)| openquant::ef3m::FitResultRow {
            mu_1,
            mu_2,
            sigma_1,
            sigma_2,
            p_1,
            error,
        })
        .collect();
    let result = openquant::ef3m::most_likely_parameters(&rows, None, res);
    let d = PyDict::new(py);
    for (k, v) in result {
        d.set_item(k, v)?;
    }
    Ok(d.into_pyobject(py).unwrap().into_any().unbind())
}

/// Fit a mixture of two Gaussians to five raw moments with EF3M.
///
/// López de Prado and Foreman (2014); used in AFML §10.2 and §15.4.1. Each run searches a
/// grid of starting values `mu_2 = m_1 + i * epsilon * factor * sigma` (about
/// `1 / epsilon` starts), each from a random starting `p_1`, iterating the moment equations
/// until `p_1` moves by less than `epsilon`, and keeps the fit with the smallest squared
/// error on the five raw moments. `variant=1` fits four moments, `variant=2` all five (more
/// accurate; fitted about the mean so a negative `mu_2` can be recovered). Fits use an
/// unseeded RNG, so results vary between calls: run several and summarise them with
/// `most_likely_parameters`. The component labels can come back swapped between runs.
///
/// Parameters
/// ----------
/// moments : list[float]
///     Raw moments `[E[x], E[x^2], E[x^3], E[x^4], E[x^5]]`; at least five.
/// epsilon : float, default 1e-5
///     Convergence tolerance on `p_1` and spacing of the `mu_2` start grid; must be > 0
///     (0 never returns, a negative value yields no fits). Smaller values are slower.
/// factor : float, default 5.0
///     Width of the `mu_2` start grid in standard deviations.
/// n_runs : int, default 1
///     Number of independent searches, run serially.
/// variant : int, default 1
///     `1` to fit four moments or `2` to fit five.
/// max_iter : int, default 100000
///     Maximum iterations of one attempt from one start.
///
/// Returns
/// -------
/// list[tuple[float, float, float, float, float, float]]
///     Up to `n_runs` rows of `(mu_1, mu_2, sigma_1, sigma_2, p_1, error)`, where the sigmas
///     are standard deviations, `p_1` is the weight of component 1 and `error` is the sum of
///     squared differences between the target and implied raw moments. A run that finds no
///     admissible fit contributes no row.
///
/// Raises
/// ------
/// ValueError
///     If `moments` has fewer than five entries or `variant` is not 1 or 2.
#[pyfunction(name = "fit_m2n")]
#[pyo3(signature = (
    moments,
    epsilon=1e-5,
    factor=5.0,
    n_runs=1,
    variant=2,
    max_iter=100_000
))]
fn ef3m_fit_m2n(
    moments: Vec<f64>,
    epsilon: f64,
    factor: f64,
    n_runs: usize,
    variant: usize,
    max_iter: usize,
) -> PyResult<Vec<M2nFitRow>> {
    // One fit loop per run, each from its own random starting p_1: up to `n_runs` rows.
    let m2n = openquant::ef3m::M2N::new(moments, epsilon, factor, n_runs, variant, max_iter, 1);
    let results = m2n.mp_fit().map_err(to_py_err)?;
    Ok(results
        .into_iter()
        .map(|r| (r.mu_1, r.mu_2, r.sigma_1, r.sigma_2, r.p_1, r.error))
        .collect())
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "ef3m")?;
    m.add_function(wrap_pyfunction!(ef3m_centered_moment, &m)?)?;
    m.add_function(wrap_pyfunction!(ef3m_raw_moment, &m)?)?;
    m.add_function(wrap_pyfunction!(ef3m_most_likely_parameters, &m)?)?;
    m.add_function(wrap_pyfunction!(ef3m_fit_m2n, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("ef3m", m)?;
    Ok(())
}
