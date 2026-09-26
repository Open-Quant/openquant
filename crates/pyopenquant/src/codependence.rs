use pyo3::prelude::*;

use crate::helpers::to_py_err;

/// Angular distance `sqrt((1 - rho) / 2)` from the Pearson correlation `rho` of two series.
///
/// In `[0, 1]`: 0 for `rho = 1` and 1 for `rho = -1`, so a negatively correlated asset is
/// treated as a diversifier (suits long-only portfolios). López de Prado, Machine Learning
/// for Asset Managers (2020), chapter 3; the distance layer under AFML chapter 16. Pass
/// returns or other stationary series, not trending price levels.
///
/// Parameters
/// ----------
/// x : list[float]
///     First series.
/// y : list[float]
///     Second series, paired with `x`.
///
/// Returns
/// -------
/// float
///     The angular distance.
///
/// Raises
/// ------
/// ValueError
///     If `x` and `y` differ in length, have fewer than two observations, or either series
///     is constant.
#[pyfunction(name = "angular_distance")]
fn codependence_angular_distance(x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
    openquant::codependence::angular_distance(&x, &y).map_err(to_py_err)
}

/// Absolute angular distance `sqrt((1 - |rho|) / 2)` from the Pearson correlation of two series.
///
/// In `[0, sqrt(1/2)]` and 0 for `rho = +-1`: perfectly anti-correlated series are treated
/// as identical, as suits long-short portfolios. López de Prado, Machine Learning for Asset
/// Managers (2020), chapter 3. Pass returns or other stationary series, not price levels.
///
/// Parameters
/// ----------
/// x : list[float]
///     First series.
/// y : list[float]
///     Second series, paired with `x`.
///
/// Returns
/// -------
/// float
///     The absolute angular distance.
///
/// Raises
/// ------
/// ValueError
///     If `x` and `y` differ in length, have fewer than two observations, or either series
///     is constant.
#[pyfunction(name = "absolute_angular_distance")]
fn codependence_absolute_angular_distance(x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
    openquant::codependence::absolute_angular_distance(&x, &y).map_err(to_py_err)
}

/// Squared angular distance `sqrt((1 - rho^2) / 2)` from the Pearson correlation of two series.
///
/// Like `absolute_angular_distance`, 0 for `rho = +-1`; it spreads out high correlations
/// and compresses low ones. López de Prado, Machine Learning for Asset Managers (2020),
/// chapter 3. Pass returns or other stationary series, not price levels.
///
/// Parameters
/// ----------
/// x : list[float]
///     First series.
/// y : list[float]
///     Second series, paired with `x`.
///
/// Returns
/// -------
/// float
///     The squared angular distance.
///
/// Raises
/// ------
/// ValueError
///     If `x` and `y` differ in length, have fewer than two observations, or either series
///     is constant.
#[pyfunction(name = "squared_angular_distance")]
fn codependence_squared_angular_distance(x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
    openquant::codependence::squared_angular_distance(&x, &y).map_err(to_py_err)
}

/// Distance correlation of two series (Székely et al., 2007).
///
/// Double-centres the matrices of pairwise absolute differences within each series and
/// returns `dCov(x, y) / sqrt(dVar(x) * dVar(y))`, in `[0, 1]`, which is zero only when the
/// series are independent (unlike Pearson correlation). Memory is `O(n^2)` (about 16 MB at
/// 1,000 observations).
///
/// Parameters
/// ----------
/// x : list[float]
///     First series.
/// y : list[float]
///     Second series, paired with `x`.
///
/// Returns
/// -------
/// float
///     The distance correlation.
///
/// Raises
/// ------
/// ValueError
///     If `x` and `y` differ in length, have fewer than two observations, or either series
///     is constant.
#[pyfunction(name = "distance_correlation")]
fn codependence_distance_correlation(x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
    openquant::codependence::distance_correlation(&x, &y).map_err(to_py_err)
}

/// Histogram bin count that minimises the bias of entropy estimates.
///
/// Hacine-Gharbi et al. (2012). With `corr_coef=None` uses the univariate (marginal
/// entropy) rule; with the correlation `rho` of two series uses the bivariate (joint
/// entropy) rule `round(sqrt(1 + sqrt(1 + 24 N / (1 - rho^2))) / sqrt(2))`. A correlation
/// within `1e-4` of `+-1` falls back to the univariate rule.
///
/// Parameters
/// ----------
/// num_obs : int
///     Number of observations `N`; must be positive.
/// corr_coef : float | None, default None
///     Sample correlation of the two series for the bivariate rule; None for the
///     univariate rule.
///
/// Returns
/// -------
/// int
///     The bin count.
///
/// Raises
/// ------
/// ValueError
///     If `num_obs` is 0, or the rule does not yield a positive count (a NaN `corr_coef`).
#[pyfunction(name = "get_optimal_number_of_bins")]
#[pyo3(signature = (num_obs, corr_coef=None))]
fn codependence_get_optimal_number_of_bins(
    num_obs: usize,
    corr_coef: Option<f64>,
) -> PyResult<usize> {
    openquant::codependence::get_optimal_number_of_bins(num_obs, corr_coef).map_err(to_py_err)
}

/// Mutual information `I[X;Y] = H[X] + H[Y] - H[X,Y]` from an equal-width 2-D histogram.
///
/// Estimated from an `n_bins x n_bins` histogram with natural logarithms. With
/// `n_bins=None` the count comes from `get_optimal_number_of_bins` with the sample
/// correlation. With `normalize=True` the result is divided by `min(H[X], H[Y])` and lies
/// in `[0, 1]`. The estimate is biased upward on small samples; compare values only at
/// equal length and binning. Normalised mutual information is not a distance; use
/// `variation_of_information_score` for clustering. López de Prado, Machine Learning for
/// Asset Managers (2020), chapter 3.
///
/// Parameters
/// ----------
/// x : list[float]
///     First series.
/// y : list[float]
///     Second series, paired with `x`.
/// n_bins : int | None, default None
///     Bins per axis; None picks the optimal count.
/// normalize : bool, default False
///     Divide by the smaller marginal entropy.
///
/// Returns
/// -------
/// float
///     The (optionally normalised) mutual information, in nats.
///
/// Raises
/// ------
/// ValueError
///     If `x` and `y` differ in length or are empty, have fewer than two observations with
///     `n_bins=None`, `n_bins` is 0 or the bin rule fails, a series is constant with
///     `n_bins=None`, or a marginal entropy is zero with `normalize=True`.
#[pyfunction(name = "get_mutual_info")]
#[pyo3(signature = (x, y, n_bins=None, normalize=false))]
fn codependence_get_mutual_info(
    x: Vec<f64>,
    y: Vec<f64>,
    n_bins: Option<usize>,
    normalize: bool,
) -> PyResult<f64> {
    openquant::codependence::get_mutual_info(&x, &y, n_bins, normalize).map_err(to_py_err)
}

/// Variation of information `VI[X;Y] = H[X] + H[Y] - 2 I[X;Y]` of two series (Meilă, 2007).
///
/// A true metric, estimated from equal-width histograms with natural logarithms: the
/// uncertainty left in each variable once the other is known. With `normalize=True` it is
/// divided by the joint entropy `H[X,Y]` and lies in `[0, 1]`, with 0 meaning each variable
/// determines the other. With `n_bins=None` the count comes from
/// `get_optimal_number_of_bins` with the sample correlation. López de Prado, Machine
/// Learning for Asset Managers (2020), chapter 3.
///
/// Parameters
/// ----------
/// x : list[float]
///     First series.
/// y : list[float]
///     Second series, paired with `x`.
/// n_bins : int | None, default None
///     Bins per axis; None picks the optimal count.
/// normalize : bool, default False
///     Divide by the joint entropy.
///
/// Returns
/// -------
/// float
///     The (optionally normalised) variation of information.
///
/// Raises
/// ------
/// ValueError
///     If `x` and `y` differ in length or are empty, have fewer than two observations with
///     `n_bins=None`, `n_bins` is 0 or the bin rule fails, a series is constant with
///     `n_bins=None`, or the joint entropy is zero with `normalize=True`.
#[pyfunction(name = "variation_of_information_score")]
#[pyo3(signature = (x, y, n_bins=None, normalize=false))]
fn codependence_variation_of_information_score(
    x: Vec<f64>,
    y: Vec<f64>,
    n_bins: Option<usize>,
    normalize: bool,
) -> PyResult<f64> {
    openquant::codependence::variation_of_information_score(&x, &y, n_bins, normalize)
        .map_err(to_py_err)
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "codependence")?;
    m.add_function(wrap_pyfunction!(codependence_angular_distance, &m)?)?;
    m.add_function(wrap_pyfunction!(codependence_absolute_angular_distance, &m)?)?;
    m.add_function(wrap_pyfunction!(codependence_squared_angular_distance, &m)?)?;
    m.add_function(wrap_pyfunction!(codependence_distance_correlation, &m)?)?;
    m.add_function(wrap_pyfunction!(codependence_get_optimal_number_of_bins, &m)?)?;
    m.add_function(wrap_pyfunction!(codependence_get_mutual_info, &m)?)?;
    m.add_function(wrap_pyfunction!(codependence_variation_of_information_score, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("codependence", m)?;
    Ok(())
}
