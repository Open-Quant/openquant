use pyo3::prelude::*;

use crate::helpers::{matrix_from_rows, to_py_err};

/// Hierarchical Risk Parity weights (AFML chapter 16, Snippets 16.1-16.4).
///
/// Allocates without inverting the covariance matrix: (1) tree clustering on correlation
/// distances `d = sqrt((1 - rho) / 2)` by single linkage (section 16.4.1); (2)
/// quasi-diagonalisation so similar assets are adjacent (16.4.2); (3) recursive bisection,
/// splitting weight between halves in inverse proportion to their inverse-variance cluster
/// variances (16.4.3). The result is long-only and fully invested. The covariance is taken
/// from `covariance_matrix` if given, else estimated from `asset_returns`, else from
/// simple returns of `asset_prices`; HRP weights do not depend on its scale.
///
/// Parameters
/// ----------
/// asset_names : list[str]
///     One name per asset; its length fixes the number of assets `N`.
/// asset_prices : list[list[float]] | None, default None
///     Prices, one inner list per observation (oldest first) and `N` columns. Used only
///     when neither `asset_returns` nor `covariance_matrix` is given.
/// asset_returns : list[list[float]] | None, default None
///     Per-period returns, one inner list per observation and `N` columns. Used when no
///     `covariance_matrix` is given.
/// covariance_matrix : list[list[float]] | None, default None
///     `N x N` covariance, used as given when present.
/// resample_by : str | None, default None
///     For prices only: `"W"`/`"week"`/`"weekly"` keeps every 5th row,
///     `"M"`/`"month"`/`"monthly"` every 21st (case-insensitive); anything else, or None,
///     keeps every row.
/// use_shrinkage : bool, default False
///     Multiply the off-diagonal terms of an estimated covariance by 0.9 (a fixed shrink,
///     not Ledoit-Wolf); no effect on a supplied `covariance_matrix`.
/// distance : str | None, default None
///     Distance the tree is built on (case-insensitive): `"distance_of_distances"` (the
///     default; the Euclidean distance between columns of `d`, as AFML Snippet 16.4
///     computes) or `"correlation"` (cluster on `d` itself, as mlfinlab does).
///
/// Returns
/// -------
/// tuple[list[float], list[int]]
///     `(weights, ordered_indices)`: non-negative weights summing to 1, one per asset in
///     `asset_names` order, and the asset indices in quasi-diagonal (leaf) order.
///
/// Raises
/// ------
/// ValueError
///     If `distance` is unknown, a matrix is empty or ragged, or the core rejects the input
///     (e.g. no prices, returns or covariance given, empty `asset_names`, fewer than two
///     price or return rows, a zero price, a non-positive variance, or a matrix whose
///     shape disagrees with the number of asset names).
#[pyfunction(name = "allocate_hrp")]
#[pyo3(signature = (
    asset_names,
    asset_prices=None,
    asset_returns=None,
    covariance_matrix=None,
    resample_by=None,
    use_shrinkage=false,
    distance=None
))]
fn hrp_allocate(
    asset_names: Vec<String>,
    asset_prices: Option<Vec<Vec<f64>>>,
    asset_returns: Option<Vec<Vec<f64>>>,
    covariance_matrix: Option<Vec<Vec<f64>>>,
    resample_by: Option<String>,
    use_shrinkage: bool,
    distance: Option<String>,
) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let distance: openquant::hrp::HrpDistance = match distance {
        Some(name) => name.parse().map_err(to_py_err)?,
        None => openquant::hrp::HrpDistance::default(),
    };
    let prices_m = asset_prices.map(matrix_from_rows).transpose()?;
    let returns_m = asset_returns.map(matrix_from_rows).transpose()?;
    let cov_m = covariance_matrix.map(matrix_from_rows).transpose()?;

    let mut hrp = openquant::hrp::HierarchicalRiskParity::with_distance(distance);
    hrp.allocate(
        &asset_names,
        prices_m.as_ref(),
        returns_m.as_ref(),
        cov_m.as_ref(),
        resample_by.as_deref(),
        use_shrinkage,
    )
    .map_err(to_py_err)?;

    Ok((hrp.weights, hrp.ordered_indices))
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "hrp")?;
    m.add_function(wrap_pyfunction!(hrp_allocate, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("hrp", m)?;
    Ok(())
}
