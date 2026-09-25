//! Hierarchical Clustering-based Asset Allocation (HCAA): split weight down a single-linkage
//! cluster tree with a choice of risk measure.
//!
//! References: Raffinot (2017), *Hierarchical clustering-based asset allocation*, Journal of
//! Portfolio Management 44(2); AFML Chapter 16 (§16.4, the tree and quasi-diagonalisation
//! shared with HRP, Snippets 16.1–16.2).
//!
//! The tree is HRP's: single linkage on the correlation distance `sqrt(2 (1 - rho))`. Weight
//! starts at 1 at the root. At each of the top `k - 1` merges (`k` = `optimal_num_clusters`)
//! the node's weight is split between its children, the left one receiving a share `alpha`
//! set by `allocation_metric`; below that cut each subtree is one cluster whose weight is
//! shared equally (`"equal_weighting"`) or by inverse variance (every other metric). Each side
//! of a split is scored as its inverse-variance portfolio:
//!
//! | `allocation_metric` | `alpha` (left share) |
//! | --- | --- |
//! | `"minimum_variance"` | `1 - var_L / (var_L + var_R)` |
//! | `"minimum_standard_deviation"` | `1 - sd_L / (sd_L + sd_R)` |
//! | `"expected_shortfall"` | `1 - es_L / (es_L + es_R)` |
//! | `"conditional_drawdown_risk"` | `1 - cdd_L / (cdd_L + cdd_R)` |
//! | `"sharpe_ratio"` | `sr_L / (sr_L + sr_R)`, falling back to minimum variance outside `[0, 1]` |
//! | `"equal_weighting"` | `0.5` |
//!
//! What is **not** implemented: Raffinot's gap-statistic choice of the number of clusters.
//! With `optimal_num_clusters = None` there is no cut and every merge is split down to single
//! assets; pick the count yourself (for example with [`crate::onc`]).
//!
//! Conventions:
//! - Matrices are `T x N`: rows are dates in ascending order, columns are assets in the order
//!   of `asset_names`. Covariance is `N x N` in the same order.
//! - Returns derived from prices are simple returns `p_t / p_{t-1} - 1` (after optional
//!   resampling). Estimated expected returns are **annualised** by 252 periods; covariance and
//!   tail measures are per-period.
//! - `confidence_level` is the **tail probability** (e.g. 0.05) for both tail metrics.
//! - Expected shortfall and conditional drawdown are computed here (on each side's
//!   inverse-variance portfolio), not by [`crate::risk_metrics`]. Expected shortfall is returned
//!   as a positive loss; the drawdown is relative to the running peak of a wealth curve
//!   starting at 1.
//!
//! ```
//! use nalgebra::DMatrix;
//! use openquant::hcaa::{HcaaError, HierarchicalClusteringAssetAllocation};
//!
//! # fn main() -> Result<(), HcaaError> {
//! // a and b are correlated (0.8), c is independent and more volatile.
//! let covariance = DMatrix::from_row_slice(
//!     3,
//!     3,
//!     &[0.010, 0.016, 0.000, 0.016, 0.040, 0.000, 0.000, 0.000, 0.090],
//! );
//! let names: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();
//! let mut model = HierarchicalClusteringAssetAllocation::new("mean");
//! model.allocate(
//!     &names, None, None, Some(&covariance), None, "minimum_variance", 0.05, None, None,
//! )?;
//!
//! // The root splits c from {a, b}. {a, b} as an inverse-variance portfolio (0.8, 0.2) has
//! // variance 0.01312, so c gets 0.01312 / (0.01312 + 0.09).
//! assert_eq!(model.ordered_indices, vec![2, 0, 1]);
//! assert!((model.weights[2] - 0.01312 / 0.10312).abs() < 1e-12);
//! assert!((model.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
//! # Ok(())
//! # }
//! ```
#![deny(missing_docs)]

use crate::util::resample::{freq_step, resample_prices};
use nalgebra::DMatrix;

/// Errors returned by [`HierarchicalClusteringAssetAllocation::allocate`].
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum HcaaError {
    /// Missing or unusable data. Returned when no prices, returns or covariance are supplied;
    /// when `asset_names` is empty; when prices have fewer than two rows (after resampling) or a
    /// zero price is divided by; when a covariance must be estimated from fewer than two return
    /// rows; and when a covariance diagonal entry is not strictly positive.
    #[error("no data: supply asset prices or returns")]
    NoData,
    /// `allocation_metric` is not one of the six supported names.
    #[error("unknown allocation metric: {0}")]
    UnknownAllocationMetric(String),
    /// The expected-returns method given to
    /// [`HierarchicalClusteringAssetAllocation::new`] is neither `"mean"` nor `"exponential"`
    /// (checked only when it is needed).
    #[error("unknown returns method: {0}")]
    UnknownReturns(String),
    /// `"sharpe_ratio"` was requested with neither `expected_asset_returns` nor prices.
    #[error("the sharpe_ratio metric needs expected returns")]
    MissingExpectedReturnsForSharpe,
    /// A tail metric (`"expected_shortfall"` or `"conditional_drawdown_risk"`) was requested
    /// without a return history (neither returns nor prices).
    #[error("tail-risk metrics need asset returns")]
    MissingReturnsForTailRisk,
    /// Input shapes disagree; the message names which.
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(&'static str),
    /// `optimal_num_clusters` is zero or exceeds the number of assets.
    #[error(
        "optimal_num_clusters must be between 1 and {assets} (the asset count), got {requested}"
    )]
    InvalidNumClusters {
        /// The requested number of clusters.
        requested: usize,
        /// The number of assets.
        assets: usize,
    },
}

/// HCAA allocator; call [`allocate`](Self::allocate), then read the public fields.
///
/// The fields are empty until the first successful `allocate` and are overwritten by each
/// successful call; a call that returns an error leaves them as they were.
#[derive(Debug, Clone)]
pub struct HierarchicalClusteringAssetAllocation {
    /// Portfolio weights, one per asset in `asset_names` order; non-negative and summing to 1.
    pub weights: Vec<f64>,
    /// Asset indices in quasi-diagonal (dendrogram leaf) order.
    pub ordered_indices: Vec<usize>,
    /// Single-linkage merges in SciPy linkage convention: row `i` merges the two listed nodes
    /// (smaller id first) into node `N + i`, where ids below `N` are assets.
    pub clusters: Vec<[usize; 2]>,
    calculate_expected_returns: String,
}

impl Default for HierarchicalClusteringAssetAllocation {
    fn default() -> Self {
        Self::new("mean")
    }
}

impl HierarchicalClusteringAssetAllocation {
    /// Create an allocator that estimates expected returns (for `"sharpe_ratio"` from prices)
    /// with `calculate_expected_returns`: `"mean"` (annualised mean of per-period returns) or
    /// `"exponential"` (annualised exponentially weighted mean, span 500, newest weighted
    /// most). Matching is case-insensitive; an unknown name is only rejected when it is used,
    /// with [`HcaaError::UnknownReturns`]. [`Default`] uses `"mean"`.
    pub fn new(calculate_expected_returns: &str) -> Self {
        Self {
            weights: Vec::new(),
            ordered_indices: Vec::new(),
            clusters: Vec::new(),
            calculate_expected_returns: calculate_expected_returns.to_string(),
        }
    }

    /// Compute HCAA weights (Raffinot 2017) and store them with the tree in `self`.
    ///
    /// Arguments:
    /// - `asset_names` — one name per asset; its length `N` fixes the asset count and order.
    /// - `asset_prices` — optional `T x N` price matrix (rows ascending in time). Used only
    ///   when `asset_returns` is `None`; converted to simple returns after resampling.
    /// - `asset_returns` — optional `T x N` matrix of per-period returns; takes precedence
    ///   over prices.
    /// - `covariance_matrix` — optional `N x N` covariance; if `None` it is the sample
    ///   covariance (denominator `T - 1`) of the returns.
    /// - `expected_asset_returns` — `N` expected returns for `"sharpe_ratio"`; if `None` they
    ///   are estimated (annualised by 252) from the returns, but **only when `asset_prices`
    ///   is supplied**. Ignored by other metrics.
    /// - `allocation_metric` — one of `"minimum_variance"`, `"minimum_standard_deviation"`,
    ///   `"sharpe_ratio"`, `"equal_weighting"`, `"expected_shortfall"`,
    ///   `"conditional_drawdown_risk"` (see the [module table](self)).
    /// - `confidence_level` — tail probability for the two tail metrics (e.g. 0.05); not
    ///   validated, clamped into `[0, 1]` when the quantile is taken (nearest-rank, rounded).
    /// - `optimal_num_clusters` — where to cut the tree, in `1..=N`; `None` means `N` (no cut).
    /// - `resample_by` — for prices only: `"W"`/`"week"`/`"weekly"` keeps every 5th row,
    ///   `"M"`/`"month"`/`"monthly"` every 21st (the last row of each block; case-insensitive);
    ///   anything else keeps every row.
    ///
    /// A split whose `alpha` is not finite uses 0.5; `alpha` is clamped to `[0, 1]`.
    ///
    /// # Errors
    ///
    /// - [`HcaaError::NoData`] if prices, returns and covariance are all `None`, or
    ///   `asset_names` is empty, or prices have fewer than two rows after resampling or a zero
    ///   price in a denominator, or the covariance must be estimated from fewer than two
    ///   return rows, or a covariance diagonal entry is `<= 0`.
    /// - [`HcaaError::UnknownAllocationMetric`] for an unsupported `allocation_metric`.
    /// - [`HcaaError::DimensionMismatch`] if the (non-empty) returns do not have `N` columns,
    ///   the covariance is not `N x N`, or `expected_asset_returns` does not have length `N`
    ///   (checked only for `"sharpe_ratio"`).
    /// - [`HcaaError::MissingExpectedReturnsForSharpe`] for `"sharpe_ratio"` with neither
    ///   `expected_asset_returns` nor `asset_prices`.
    /// - [`HcaaError::UnknownReturns`] if expected returns must be estimated and the method
    ///   given to [`new`](Self::new) is unknown.
    /// - [`HcaaError::MissingReturnsForTailRisk`] for a tail metric without a return history.
    /// - [`HcaaError::InvalidNumClusters`] if `optimal_num_clusters` is `Some(0)` or exceeds
    ///   `N`.
    ///
    /// ```
    /// use nalgebra::DMatrix;
    /// use openquant::hcaa::{HcaaError, HierarchicalClusteringAssetAllocation};
    ///
    /// let covariance = DMatrix::from_row_slice(
    ///     3,
    ///     3,
    ///     &[0.010, 0.016, 0.000, 0.016, 0.040, 0.000, 0.000, 0.000, 0.090],
    /// );
    /// let names: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();
    /// let mut model = HierarchicalClusteringAssetAllocation::default();
    /// let metric = "minimum_standard_deviation";
    ///
    /// // No cut: the root splits c from {a, b} by standard deviation (0.3 vs sqrt(0.01312)),
    /// // then {a, b} is split by standard deviation too (0.1 vs 0.2: 2/3 to a).
    /// model.allocate(&names, None, None, Some(&covariance), None, metric, 0.05, None, None)?;
    /// let expected = [0.482459, 0.241230, 0.276311];
    /// for (w, e) in model.weights.iter().zip(expected) {
    ///     assert!((w - e).abs() < 1e-6);
    /// }
    ///
    /// // Cut into two clusters: {a, b} is one cluster, shared by inverse variance (0.8, 0.2).
    /// model.allocate(&names, None, None, Some(&covariance), None, metric, 0.05, Some(2), None)?;
    /// let expected = [0.578951, 0.144738, 0.276311];
    /// for (w, e) in model.weights.iter().zip(expected) {
    ///     assert!((w - e).abs() < 1e-6);
    /// }
    ///
    /// // Tail metrics need a return history, not just a covariance matrix.
    /// assert_eq!(
    ///     model.allocate(
    ///         &names, None, None, Some(&covariance), None, "expected_shortfall", 0.05, None, None,
    ///     ),
    ///     Err(HcaaError::MissingReturnsForTailRisk)
    /// );
    /// # Ok::<(), HcaaError>(())
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn allocate(
        &mut self,
        asset_names: &[String],
        asset_prices: Option<&DMatrix<f64>>,
        asset_returns: Option<&DMatrix<f64>>,
        covariance_matrix: Option<&DMatrix<f64>>,
        expected_asset_returns: Option<&[f64]>,
        allocation_metric: &str,
        confidence_level: f64,
        optimal_num_clusters: Option<usize>,
        resample_by: Option<&str>,
    ) -> Result<(), HcaaError> {
        if asset_prices.is_none() && asset_returns.is_none() && covariance_matrix.is_none() {
            return Err(HcaaError::NoData);
        }
        if !matches!(
            allocation_metric,
            "minimum_variance"
                | "minimum_standard_deviation"
                | "sharpe_ratio"
                | "equal_weighting"
                | "expected_shortfall"
                | "conditional_drawdown_risk"
        ) {
            return Err(HcaaError::UnknownAllocationMetric(allocation_metric.to_string()));
        }
        let n_assets = asset_names.len();
        if n_assets == 0 {
            return Err(HcaaError::NoData);
        }

        let returns_owned = if let Some(r) = asset_returns {
            r.clone_owned()
        } else if let Some(p) = asset_prices {
            let step = freq_step(resample_by);
            let sampled = resample_prices(p, step);
            returns_from_prices(&sampled)?
        } else {
            DMatrix::zeros(0, n_assets)
        };
        if returns_owned.ncols() != n_assets && returns_owned.nrows() > 0 {
            return Err(HcaaError::DimensionMismatch(
                "asset_returns columns != asset_names length",
            ));
        }

        let covariance_owned = if let Some(cov) = covariance_matrix {
            cov.clone_owned()
        } else {
            covariance(&returns_owned)?
        };
        if covariance_owned.nrows() != n_assets || covariance_owned.ncols() != n_assets {
            return Err(HcaaError::DimensionMismatch(
                "covariance matrix dimensions must equal number of assets",
            ));
        }

        let expected_owned = if allocation_metric == "sharpe_ratio" {
            if let Some(mu) = expected_asset_returns {
                if mu.len() != n_assets {
                    return Err(HcaaError::DimensionMismatch(
                        "expected_asset_returns length != asset_names length",
                    ));
                }
                mu.to_vec()
            } else if asset_prices.is_none() {
                return Err(HcaaError::MissingExpectedReturnsForSharpe);
            } else if self.calculate_expected_returns.eq_ignore_ascii_case("mean") {
                mean_expected_returns(&returns_owned)
            } else if self.calculate_expected_returns.eq_ignore_ascii_case("exponential") {
                exponential_expected_returns(&returns_owned, 500)
            } else {
                return Err(HcaaError::UnknownReturns(self.calculate_expected_returns.clone()));
            }
        } else {
            vec![0.0; n_assets]
        };

        if matches!(allocation_metric, "expected_shortfall" | "conditional_drawdown_risk")
            && returns_owned.nrows() == 0
        {
            return Err(HcaaError::MissingReturnsForTailRisk);
        }

        let num_clusters = optimal_num_clusters.unwrap_or(n_assets);
        if num_clusters == 0 || num_clusters > n_assets {
            return Err(HcaaError::InvalidNumClusters {
                requested: num_clusters,
                assets: n_assets,
            });
        }

        let corr = cov2corr(&covariance_owned)?;
        self.clusters = single_linkage_children(&corr);
        self.ordered_indices = quasi_diagonalization(n_assets, &self.clusters, 2 * n_assets - 2);
        let inputs = MetricInputs {
            expected: &expected_owned,
            returns: &returns_owned,
            cov: &covariance_owned,
            metric: allocation_metric,
            confidence_level,
        };
        let mut weights = vec![0.0; n_assets];
        allocate_down_tree(
            &self.clusters,
            n_assets,
            num_clusters,
            2 * n_assets - 2,
            1.0,
            &inputs,
            &mut weights,
        )?;
        self.weights = weights;

        Ok(())
    }
}

fn returns_from_prices(prices: &DMatrix<f64>) -> Result<DMatrix<f64>, HcaaError> {
    if prices.nrows() < 2 {
        return Err(HcaaError::NoData);
    }
    let rows = prices.nrows();
    let cols = prices.ncols();
    let mut out = DMatrix::zeros(rows - 1, cols);
    for r in 1..rows {
        for c in 0..cols {
            let prev = prices[(r - 1, c)];
            if prev == 0.0 {
                return Err(HcaaError::NoData);
            }
            out[(r - 1, c)] = prices[(r, c)] / prev - 1.0;
        }
    }
    Ok(out)
}

fn covariance(returns: &DMatrix<f64>) -> Result<DMatrix<f64>, HcaaError> {
    let rows = returns.nrows();
    let cols = returns.ncols();
    if rows < 2 {
        return Err(HcaaError::NoData);
    }
    let means: Vec<f64> = (0..cols).map(|c| returns.column(c).sum() / rows as f64).collect();
    let mut cov = DMatrix::zeros(cols, cols);
    for i in 0..cols {
        for j in i..cols {
            let mut s = 0.0;
            for r in 0..rows {
                s += (returns[(r, i)] - means[i]) * (returns[(r, j)] - means[j]);
            }
            s /= (rows - 1) as f64;
            cov[(i, j)] = s;
            cov[(j, i)] = s;
        }
    }
    Ok(cov)
}

fn mean_expected_returns(returns: &DMatrix<f64>) -> Vec<f64> {
    let rows = returns.nrows();
    let cols = returns.ncols();
    if rows == 0 {
        return vec![0.0; cols];
    }
    (0..cols).map(|c| returns.column(c).sum() / rows as f64 * 252.0).collect()
}

fn exponential_expected_returns(returns: &DMatrix<f64>, span: usize) -> Vec<f64> {
    let rows = returns.nrows();
    let cols = returns.ncols();
    if rows == 0 {
        return vec![0.0; cols];
    }
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut out = vec![0.0; cols];
    for c in 0..cols {
        let mut weight = 1.0;
        let mut num = 0.0;
        let mut denom = 0.0;
        for r in (0..rows).rev() {
            num += weight * returns[(r, c)];
            denom += weight;
            weight *= 1.0 - alpha;
        }
        out[c] = if denom > 0.0 { num / denom * 252.0 } else { 0.0 };
    }
    out
}

fn cov2corr(covariance: &DMatrix<f64>) -> Result<DMatrix<f64>, HcaaError> {
    let n = covariance.nrows();
    if n == 0 || covariance.ncols() != n {
        return Err(HcaaError::DimensionMismatch("covariance must be square and non-empty"));
    }
    let mut corr = DMatrix::zeros(n, n);
    let mut std = vec![0.0; n];
    for i in 0..n {
        let v = covariance[(i, i)];
        if v <= 0.0 {
            return Err(HcaaError::NoData);
        }
        std[i] = v.sqrt();
    }
    for i in 0..n {
        for j in 0..n {
            corr[(i, j)] = covariance[(i, j)] / (std[i] * std[j]);
        }
    }
    Ok(corr)
}

fn single_linkage_children(corr: &DMatrix<f64>) -> Vec<[usize; 2]> {
    #[derive(Clone)]
    struct Cluster {
        id: usize,
        members: Vec<usize>,
    }

    let n = corr.nrows();
    let mut distance = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let c = corr[(i, j)].clamp(-1.0, 1.0);
            distance[(i, j)] = (2.0 * (1.0 - c)).max(0.0).sqrt();
        }
    }

    let mut clusters: Vec<Cluster> = (0..n).map(|i| Cluster { id: i, members: vec![i] }).collect();
    let mut next_id = n;
    let mut children: Vec<[usize; 2]> = Vec::with_capacity(n.saturating_sub(1));
    let eps = 1e-12;

    while clusters.len() > 1 {
        let mut best_i = 0usize;
        let mut best_j = 1usize;
        let mut best_d = f64::INFINITY;
        let mut best_pair_ids = (
            clusters[best_i].id.min(clusters[best_j].id),
            clusters[best_i].id.max(clusters[best_j].id),
        );

        for i in 0..clusters.len() {
            for j in i + 1..clusters.len() {
                let mut d = f64::INFINITY;
                for &a in &clusters[i].members {
                    for &b in &clusters[j].members {
                        d = d.min(distance[(a, b)]);
                    }
                }
                let ids = (clusters[i].id.min(clusters[j].id), clusters[i].id.max(clusters[j].id));
                let better = d + eps < best_d || ((d - best_d).abs() <= eps && ids < best_pair_ids);
                if better {
                    best_i = i;
                    best_j = j;
                    best_d = d;
                    best_pair_ids = ids;
                }
            }
        }

        let (lo, hi) = if best_i < best_j { (best_i, best_j) } else { (best_j, best_i) };
        let right = clusters.remove(hi);
        let left = clusters.remove(lo);
        let mut members = left.members;
        members.extend(right.members);
        let left_id = left.id.min(right.id);
        let right_id = left.id.max(right.id);
        children.push([left_id, right_id]);
        clusters.push(Cluster { id: next_id, members });
        next_id += 1;
    }

    children
}

fn quasi_diagonalization(
    num_assets: usize,
    clusters: &[[usize; 2]],
    curr_index: usize,
) -> Vec<usize> {
    if curr_index < num_assets {
        return vec![curr_index];
    }
    let row = curr_index - num_assets;
    let left = clusters[row][0];
    let right = clusters[row][1];
    let mut out = quasi_diagonalization(num_assets, clusters, left);
    out.extend(quasi_diagonalization(num_assets, clusters, right));
    out
}

fn inverse_variance_weights(cov: &DMatrix<f64>, indices: &[usize]) -> Result<Vec<f64>, HcaaError> {
    let mut inv_diag: Vec<f64> = Vec::with_capacity(indices.len());
    for &i in indices {
        let v = cov[(i, i)];
        if v <= 0.0 {
            return Err(HcaaError::NoData);
        }
        inv_diag.push(1.0 / v);
    }
    let sum: f64 = inv_diag.iter().sum();
    if sum <= 0.0 {
        return Err(HcaaError::NoData);
    }
    Ok(inv_diag.into_iter().map(|x| x / sum).collect())
}

fn cluster_variance(cov: &DMatrix<f64>, indices: &[usize]) -> Result<f64, HcaaError> {
    let w = inverse_variance_weights(cov, indices)?;
    let mut v = 0.0;
    for (ii, &i) in indices.iter().enumerate() {
        for (jj, &j) in indices.iter().enumerate() {
            v += w[ii] * cov[(i, j)] * w[jj];
        }
    }
    Ok(v.max(0.0))
}

fn cluster_sharpe(
    expected: &[f64],
    cov: &DMatrix<f64>,
    indices: &[usize],
) -> Result<f64, HcaaError> {
    let w = inverse_variance_weights(cov, indices)?;
    let mut mu = 0.0;
    for (ii, &i) in indices.iter().enumerate() {
        mu += w[ii] * expected[i];
    }
    let var = cluster_variance(cov, indices)?;
    if var <= 0.0 {
        Ok(0.0)
    } else {
        Ok(mu / var.sqrt())
    }
}

fn quantile(mut values: Vec<f64>, q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let qn = q.clamp(0.0, 1.0);
    let idx = ((values.len() - 1) as f64 * qn).round() as usize;
    values[idx]
}

fn cluster_expected_shortfall(
    returns: &DMatrix<f64>,
    cov: &DMatrix<f64>,
    confidence_level: f64,
    indices: &[usize],
) -> Result<f64, HcaaError> {
    let w = inverse_variance_weights(cov, indices)?;
    let mut portfolio_returns = Vec::with_capacity(returns.nrows());
    for r in 0..returns.nrows() {
        let mut v = 0.0;
        for (ii, &idx) in indices.iter().enumerate() {
            v += returns[(r, idx)] * w[ii];
        }
        portfolio_returns.push(v);
    }
    let threshold = quantile(portfolio_returns.clone(), confidence_level);
    let tail: Vec<f64> = portfolio_returns.into_iter().filter(|x| *x <= threshold).collect();
    if tail.is_empty() {
        return Ok(0.0);
    }
    Ok(-tail.iter().sum::<f64>() / tail.len() as f64)
}

fn cluster_conditional_drawdown(
    returns: &DMatrix<f64>,
    cov: &DMatrix<f64>,
    confidence_level: f64,
    indices: &[usize],
) -> Result<f64, HcaaError> {
    let w = inverse_variance_weights(cov, indices)?;
    let mut wealth = Vec::with_capacity(returns.nrows() + 1);
    wealth.push(1.0);
    for r in 0..returns.nrows() {
        let mut ret = 0.0;
        for (ii, &idx) in indices.iter().enumerate() {
            ret += returns[(r, idx)] * w[ii];
        }
        let next = wealth.last().copied().unwrap_or(1.0) * (1.0 + ret);
        wealth.push(next);
    }
    let mut peak = wealth[0];
    let mut drawdowns = Vec::with_capacity(wealth.len());
    for v in wealth {
        if v > peak {
            peak = v;
        }
        let dd = if peak > 0.0 { (peak - v) / peak } else { 0.0 };
        drawdowns.push(dd);
    }
    let threshold = quantile(drawdowns.clone(), 1.0 - confidence_level);
    let tail: Vec<f64> = drawdowns.into_iter().filter(|x| *x >= threshold).collect();
    if tail.is_empty() {
        return Ok(0.0);
    }
    Ok(tail.iter().sum::<f64>() / tail.len() as f64)
}

/// What a split needs to score each side.
struct MetricInputs<'a> {
    expected: &'a [f64],
    returns: &'a DMatrix<f64>,
    cov: &'a DMatrix<f64>,
    metric: &'a str,
    confidence_level: f64,
}

/// The share of a node's weight that goes to its left child.
fn split_factor(left: &[usize], right: &[usize], m: &MetricInputs) -> Result<f64, HcaaError> {
    let left_var = cluster_variance(m.cov, left)?;
    let right_var = cluster_variance(m.cov, right)?;
    let alloc_factor = match m.metric {
        "minimum_variance" => 1.0 - left_var / (left_var + right_var + f64::EPSILON),
        "minimum_standard_deviation" => {
            let left_sd = left_var.sqrt();
            let right_sd = right_var.sqrt();
            1.0 - left_sd / (left_sd + right_sd + f64::EPSILON)
        }
        "sharpe_ratio" => {
            let left_sr = cluster_sharpe(m.expected, m.cov, left)?;
            let right_sr = cluster_sharpe(m.expected, m.cov, right)?;
            let raw = left_sr / (left_sr + right_sr + f64::EPSILON);
            if (0.0..=1.0).contains(&raw) {
                raw
            } else {
                1.0 - left_var / (left_var + right_var + f64::EPSILON)
            }
        }
        "expected_shortfall" => {
            let left_es = cluster_expected_shortfall(m.returns, m.cov, m.confidence_level, left)?;
            let right_es = cluster_expected_shortfall(m.returns, m.cov, m.confidence_level, right)?;
            1.0 - left_es / (left_es + right_es + f64::EPSILON)
        }
        "conditional_drawdown_risk" => {
            let left_cdd =
                cluster_conditional_drawdown(m.returns, m.cov, m.confidence_level, left)?;
            let right_cdd =
                cluster_conditional_drawdown(m.returns, m.cov, m.confidence_level, right)?;
            1.0 - left_cdd / (left_cdd + right_cdd + f64::EPSILON)
        }
        _ => 0.5,
    };
    Ok(if alloc_factor.is_finite() { alloc_factor.clamp(0.0, 1.0) } else { 0.5 })
}

/// Hand `weight` down the dendrogram from `node`. The top `num_clusters - 1` merges are split
/// between their two children by the metric; below that cut each node is one cluster, whose
/// weight is shared equally (`equal_weighting`) or by inverse variance (every other metric, the
/// portfolio the metrics assume inside a cluster).
fn allocate_down_tree(
    children: &[[usize; 2]],
    n_assets: usize,
    num_clusters: usize,
    node: usize,
    weight: f64,
    m: &MetricInputs,
    weights: &mut [f64],
) -> Result<(), HcaaError> {
    if node < n_assets {
        weights[node] = weight;
        return Ok(());
    }
    let row = node - n_assets;
    if row + num_clusters >= n_assets {
        let [left, right] = children[row];
        let alpha = split_factor(
            &quasi_diagonalization(n_assets, children, left),
            &quasi_diagonalization(n_assets, children, right),
            m,
        )?;
        allocate_down_tree(children, n_assets, num_clusters, left, weight * alpha, m, weights)?;
        return allocate_down_tree(
            children,
            n_assets,
            num_clusters,
            right,
            weight * (1.0 - alpha),
            m,
            weights,
        );
    }
    let members = quasi_diagonalization(n_assets, children, node);
    let within = if m.metric == "equal_weighting" {
        vec![1.0 / members.len() as f64; members.len()]
    } else {
        inverse_variance_weights(m.cov, &members)?
    };
    for (&asset, w) in members.iter().zip(within) {
        weights[asset] = weight * w;
    }
    Ok(())
}
