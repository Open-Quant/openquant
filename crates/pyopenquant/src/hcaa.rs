use pyo3::prelude::*;

use crate::helpers::{matrix_from_rows, to_py_err};

/// Hierarchical Clustering-based Asset Allocation weights (Raffinot, 2017).
///
/// Builds a hierarchical tree (Ward linkage by default; see `linkage`) from the correlation
/// distance `d = sqrt(2 (1 - rho))` (AFML section 16.4, Snippets 16.1-16.2; see `distance`)
/// and splits weight down it from the root. At each of the top `optimal_num_clusters - 1` merges the left child receives a
/// share `alpha` set by `allocation_metric`, each side scored as its inverse-variance
/// portfolio: `"minimum_variance"`, `"minimum_standard_deviation"`, `"expected_shortfall"`
/// and `"conditional_drawdown_risk"` give `1 - risk_L / (risk_L + risk_R)`;
/// `"sharpe_ratio"` gives `sr_L / (sr_L + sr_R)` (minimum variance outside `[0, 1]`);
/// `"equal_weighting"` gives 0.5. Below the cut each cluster's weight is shared equally
/// (`"equal_weighting"`) or by inverse variance (every other metric). Raffinot's
/// gap-statistic choice of the cluster count is not implemented: None means no cut. Returns
/// from prices are simple returns; estimated expected returns are annualised by 252
/// periods, while covariance and tail measures are per-period.
///
/// Parameters
/// ----------
/// asset_names : list[str]
///     One name per asset; its length fixes the number of assets `N` and their order.
/// asset_prices : list[list[float]] | None, default None
///     Prices, one inner list per observation (oldest first) and `N` columns. Used only
///     when `asset_returns` is None.
/// asset_returns : list[list[float]] | None, default None
///     Per-period returns, one inner list per observation and `N` columns; takes
///     precedence over prices.
/// covariance_matrix : list[list[float]] | None, default None
///     `N x N` covariance; if None, the sample covariance of the returns.
/// expected_asset_returns : list[float] | None, default None
///     `N` expected returns for `"sharpe_ratio"`; if None they are estimated, but only
///     when `asset_prices` is given. Ignored by the other metrics.
/// allocation_metric : str, default "equal_weighting"
///     One of `"minimum_variance"`, `"minimum_standard_deviation"`, `"sharpe_ratio"`,
///     `"equal_weighting"`, `"expected_shortfall"`, `"conditional_drawdown_risk"`.
/// confidence_level : float, default 0.05
///     Tail probability for the two tail metrics (0.05 = worst 5%); not validated, clamped
///     into `[0, 1]` when the quantile is taken.
/// optimal_num_clusters : int | None, default None
///     Where to cut the tree, in `1..=N`; None means `N` (no cut).
/// resample_by : str | None, default None
///     For prices only: `"W"`/`"week"`/`"weekly"` keeps every 5th row,
///     `"M"`/`"month"`/`"monthly"` every 21st (case-insensitive); anything else, or None,
///     keeps every row.
/// calculate_expected_returns : str, default "mean"
///     How expected returns are estimated from prices for `"sharpe_ratio"`: `"mean"` or
///     `"exponential"` (exponentially weighted, span 500); case-insensitive.
/// distance : str | None, default None
///     Distance the tree is built on (case-insensitive): `"correlation"` (the default when
///     None; cluster on `d` itself, Mantegna's correlation distance, as Raffinot and
///     mlfinlab do) or `"distance_of_distances"` (the Euclidean distance between columns
///     of `d`, as AFML Snippet 16.4 and the HRP default compute).
/// linkage : str | None, default None
///     How the distance between clusters is measured when the tree is built
///     (case-insensitive), as scipy's `linkage(method=...)`: `"ward"` (the default when
///     None; Ward's minimum-variance criterion, R's `ward.D2`, the default of mlfinlab's and
///     R HierPortfolios' HCAA), `"average"` (mean pairwise distance), `"complete"` (largest
///     pairwise distance) or `"single"` (smallest pairwise distance; HRP's tree, and this
///     function's tree before the default changed to Ward).
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
///     If `distance` is not `"correlation"` or `"distance_of_distances"`, `linkage` is not
///     `"single"`, `"complete"`, `"average"` or `"ward"`, a matrix is empty
///     or ragged, or the core rejects the input (e.g. no prices, returns or covariance
///     given, empty `asset_names`, too few rows, a zero price, a non-positive covariance
///     diagonal, an unknown `allocation_metric` or `calculate_expected_returns`, mismatched
///     shapes, `"sharpe_ratio"` without expected returns or prices, a tail metric without a
///     return history, or `optimal_num_clusters` of 0 or above `N`).
#[pyfunction(name = "allocate_hcaa")]
#[pyo3(signature = (
    asset_names,
    asset_prices=None,
    asset_returns=None,
    covariance_matrix=None,
    expected_asset_returns=None,
    allocation_metric="equal_weighting",
    confidence_level=0.05,
    optimal_num_clusters=None,
    resample_by=None,
    calculate_expected_returns="mean",
    distance=None,
    linkage=None
))]
// Python keyword signature.
#[allow(clippy::too_many_arguments)]
fn hcaa_allocate(
    asset_names: Vec<String>,
    asset_prices: Option<Vec<Vec<f64>>>,
    asset_returns: Option<Vec<Vec<f64>>>,
    covariance_matrix: Option<Vec<Vec<f64>>>,
    expected_asset_returns: Option<Vec<f64>>,
    allocation_metric: &str,
    confidence_level: f64,
    optimal_num_clusters: Option<usize>,
    resample_by: Option<String>,
    calculate_expected_returns: &str,
    distance: Option<String>,
    linkage: Option<String>,
) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let distance: openquant::hcaa::HcaaDistance = match distance {
        Some(name) => name.parse().map_err(to_py_err)?,
        None => openquant::hcaa::HcaaDistance::default(),
    };
    let linkage: openquant::hcaa::HcaaLinkage = match linkage {
        Some(name) => name.parse().map_err(to_py_err)?,
        None => openquant::hcaa::HcaaLinkage::default(),
    };
    let prices_m = asset_prices.map(matrix_from_rows).transpose()?;
    let returns_m = asset_returns.map(matrix_from_rows).transpose()?;
    let cov_m = covariance_matrix.map(matrix_from_rows).transpose()?;

    let mut hcaa =
        openquant::hcaa::HierarchicalClusteringAssetAllocation::new(calculate_expected_returns)
            .with_distance(distance)
            .with_linkage(linkage);
    hcaa.allocate(
        &asset_names,
        prices_m.as_ref(),
        returns_m.as_ref(),
        cov_m.as_ref(),
        expected_asset_returns.as_deref(),
        allocation_metric,
        confidence_level,
        optimal_num_clusters,
        resample_by.as_deref(),
    )
    .map_err(to_py_err)?;

    Ok((hcaa.weights, hcaa.ordered_indices))
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "hcaa")?;
    m.add_function(wrap_pyfunction!(hcaa_allocate, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("hcaa", m)?;
    Ok(())
}
