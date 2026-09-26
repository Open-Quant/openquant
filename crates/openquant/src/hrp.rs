//! Hierarchical Risk Parity (AFML chapter 16, Snippets 16.1–16.4).
//!
//! HRP allocates without inverting the covariance matrix, in three steps:
//!
//! 1. **Tree clustering** (§16.4.1): correlations become distances
//!    `d = sqrt((1 - rho) / 2)`, and by default assets are merged by single linkage on the
//!    distance between columns of that matrix, `d~_ij = sqrt(sum_n (d_ni - d_nj)^2)`, as
//!    AFML's Snippet 16.4 does. [`HrpDistance::Correlation`] clusters on `d` itself instead
//!    (mlfinlab's choice, and this library's before #167); the two often give different trees.
//! 2. **Quasi-diagonalisation** (§16.4.2): assets are reordered so similar ones are adjacent.
//! 3. **Recursive bisection** (§16.4.3): the ordered list is split in halves and weight is
//!    divided between the halves in inverse proportion to their inverse-variance cluster
//!    variances.
//!
//! The result is long-only and fully invested (weights sum to 1). From prices, returns are
//! simple returns and the covariance is the unannualised sample covariance (HRP weights do
//! not depend on the scale). Weights are in the column order of the inputs.
//!
//! ```
//! use nalgebra::DMatrix;
//! use openquant::hrp::{HierarchicalRiskParity, HrpDistance};
//!
//! # fn main() -> Result<(), openquant::hrp::HrpError> {
//! // Assets 0 and 1 are nearly the same bet; asset 2 is independent. All have variance 0.04.
//! let covariance = DMatrix::from_row_slice(3, 3, &[
//!     0.040, 0.036, 0.000,
//!     0.036, 0.040, 0.000,
//!     0.000, 0.000, 0.040,
//! ]);
//! let names: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();
//!
//! let mut model = HierarchicalRiskParity::new();
//! model.allocate(&names, None, None, Some(&covariance), None, false)?;
//!
//! assert_eq!(model.clusters[0], [0, 1]); // a and b merge first
//! assert!((model.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
//! // Bisection splits {a, b} from {c}: the pair's variance is 0.038, so c gets
//! // 0.038 / (0.038 + 0.040) and a and b share the rest equally.
//! assert!((model.weights[2] - 0.038 / 0.078).abs() < 1e-12);
//! assert!((model.weights[0] - model.weights[1]).abs() < 1e-12);
//!
//! // Clustering on the pairwise distances instead builds the same tree here (it need not).
//! let mut pairwise = HierarchicalRiskParity::with_distance(HrpDistance::Correlation);
//! pairwise.allocate(&names, None, None, Some(&covariance), None, false)?;
//! assert_eq!(pairwise.clusters, model.clusters);
//! # Ok(())
//! # }
//! ```

use crate::util::linkage::{distance_of_distances, quasi_diagonalization, single_linkage_children};
use crate::util::resample::{freq_step, resample_prices};
use nalgebra::DMatrix;

#[derive(Debug, PartialEq, thiserror::Error)]
/// Errors returned by [`HierarchicalRiskParity`].
pub enum HrpError {
    /// No input, no assets, too few observations, a zero price, or a non-positive variance.
    #[error("no data: supply asset prices, returns or a covariance matrix")]
    NoData,
    /// An input's shape disagrees with the number of asset names (the message says which).
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(&'static str),
    /// [`HierarchicalRiskParity::plot_clusters`] was called before `allocate`.
    #[error("no clusters yet: call allocate first")]
    MissingClusters,
    /// A distance name given to [`HrpDistance`]'s `FromStr` is not `"correlation"` or
    /// `"distance_of_distances"`.
    #[error("unknown distance: {0} (expected \"correlation\" or \"distance_of_distances\")")]
    UnknownDistance(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Which distance the single-linkage tree is built on (AFML §16.4.1).
///
/// Both start from the correlation distance `d_ij = sqrt((1 - rho_ij) / 2)`.
pub enum HrpDistance {
    /// Cluster on `d` itself, as a pairwise distance matrix. This is what mlfinlab does, and
    /// was this library's only behaviour before #167.
    Correlation,
    /// Cluster on the Euclidean distance between columns of `d`,
    /// `d~_ij = sqrt(sum_n (d_ni - d_nj)^2)`: two assets are close when they are at similar
    /// distances from every asset. This is the second step of AFML §16.4.1, and what Snippet
    /// 16.4's `sch.linkage(dist, 'single')` computes, because scipy reads a square matrix as
    /// one observation per row. The default: on the §16.5 Monte Carlo (10,000 runs) it gives
    /// lower out-of-sample variance than [`Correlation`](Self::Correlation) and about half the
    /// turnover.
    #[default]
    DistanceOfDistances,
}

impl std::str::FromStr for HrpDistance {
    type Err = HrpError;

    /// Parses `"correlation"` or `"distance_of_distances"` (case-insensitive).
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name.to_ascii_lowercase().as_str() {
            "correlation" => Ok(Self::Correlation),
            "distance_of_distances" => Ok(Self::DistanceOfDistances),
            _ => Err(HrpError::UnknownDistance(name.to_string())),
        }
    }
}

#[derive(Debug, Clone)]
/// A scipy-style dendrogram description from [`HierarchicalRiskParity::plot_clusters`].
///
/// Only `ivl` and `leaves` are meaningful; `icoord`, `dcoord` and `color_list` are
/// placeholders (every link at height 1). To draw the tree, use
/// [`HierarchicalRiskParity::clusters`] with the seriated distances.
pub struct HrpDendrogram {
    /// Placeholder x coordinates of each link.
    pub icoord: Vec<[f64; 4]>,
    /// Placeholder heights of each link (all 1).
    pub dcoord: Vec<[f64; 4]>,
    /// Asset names in leaf order.
    pub ivl: Vec<String>,
    /// Asset indices in leaf order.
    pub leaves: Vec<usize>,
    /// Placeholder link colours.
    pub color_list: Vec<String>,
}

#[derive(Debug, Clone, Default)]
/// Hierarchical Risk Parity allocator and its results.
///
/// Build with [`HierarchicalRiskParity::new`], call [`HierarchicalRiskParity::allocate`],
/// then read the public fields.
pub struct HierarchicalRiskParity {
    /// Portfolio weights, one per asset in input order, non-negative and summing to 1.
    pub weights: Vec<f64>,
    /// Correlation matrix with rows and columns in [`ordered_indices`](Self::ordered_indices)
    /// order.
    pub seriated_correlations: Option<DMatrix<f64>>,
    /// Distance matrix `sqrt((1 - rho) / 2)` in [`ordered_indices`](Self::ordered_indices)
    /// order.
    pub seriated_distances: Option<DMatrix<f64>>,
    /// Asset indices in quasi-diagonal (leaf) order.
    pub ordered_indices: Vec<usize>,
    /// The single-linkage merges, scipy-style: ids below `n` are assets and `n + k` is the
    /// cluster formed by merge `k`.
    pub clusters: Vec<[usize; 2]>,
    /// The distance the tree is built on; set it before calling [`allocate`](Self::allocate).
    pub distance: HrpDistance,
}

impl HierarchicalRiskParity {
    /// Creates an allocator with no results that clusters on [`HrpDistance::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an allocator with no results that clusters on `distance`.
    pub fn with_distance(distance: HrpDistance) -> Self {
        Self { distance, ..Self::default() }
    }

    /// Computes HRP weights and stores them with the tree in the public fields.
    ///
    /// `asset_names` fixes the number of assets `n`. Supply whichever of these you have
    /// (matrices have one row per observation, oldest first, and one column per asset):
    ///
    /// - `covariance_matrix` (`n x n`): always used as given when present;
    /// - `asset_returns`: used for the covariance when no covariance is given;
    /// - `asset_prices`: used (converted to simple returns, optionally resampled positionally
    ///   with `resample_by`: `"W"` keeps every 5th row, `"M"` every 21st) only when neither of
    ///   the others is given.
    ///
    /// `use_shrinkage` multiplies the off-diagonal terms of an estimated covariance by 0.9 (a
    /// fixed shrink, not Ledoit–Wolf); it has no effect on a supplied covariance.
    ///
    /// # Errors
    ///
    /// - [`HrpError::NoData`] if no input is given, `asset_names` is empty, fewer than two
    ///   price or return rows are available, a price used as a denominator is zero, or a
    ///   variance is not positive.
    /// - [`HrpError::DimensionMismatch`] if returns, prices or covariance disagree with the
    ///   number of asset names.
    #[allow(clippy::too_many_arguments)]
    pub fn allocate(
        &mut self,
        asset_names: &[String],
        asset_prices: Option<&DMatrix<f64>>,
        asset_returns: Option<&DMatrix<f64>>,
        covariance_matrix: Option<&DMatrix<f64>>,
        resample_by: Option<&str>,
        use_shrinkage: bool,
    ) -> Result<(), HrpError> {
        if asset_prices.is_none() && asset_returns.is_none() && covariance_matrix.is_none() {
            return Err(HrpError::NoData);
        }
        let n_assets = asset_names.len();
        if n_assets == 0 {
            return Err(HrpError::NoData);
        }

        let returns_owned = if let Some(r) = asset_returns {
            if r.ncols() != n_assets {
                return Err(HrpError::DimensionMismatch("asset_returns columns != asset_names"));
            }
            r.clone_owned()
        } else if covariance_matrix.is_none() {
            let prices = asset_prices.ok_or(HrpError::NoData)?;
            if prices.ncols() != n_assets {
                return Err(HrpError::DimensionMismatch("asset_prices columns != asset_names"));
            }
            let sampled = resample_prices(prices, freq_step(resample_by));
            returns_from_prices(&sampled)?
        } else {
            DMatrix::zeros(0, n_assets)
        };

        let covariance = if let Some(cov) = covariance_matrix {
            cov.clone_owned()
        } else {
            let raw_cov = covariance(&returns_owned)?;
            if use_shrinkage {
                shrink_covariance(&raw_cov, 0.1)
            } else {
                raw_cov
            }
        };

        if covariance.nrows() != n_assets || covariance.ncols() != n_assets {
            return Err(HrpError::DimensionMismatch("covariance dims != asset_names"));
        }

        let corr = cov2corr(&covariance)?;
        let distances = corr_to_distances(&corr);
        self.clusters = match self.distance {
            HrpDistance::Correlation => single_linkage_children(&distances),
            HrpDistance::DistanceOfDistances => {
                single_linkage_children(&distance_of_distances(&distances))
            }
        };
        self.ordered_indices = quasi_diagonalization(n_assets, &self.clusters, 2 * n_assets - 2);

        self.seriated_distances = Some(seriate_matrix(&distances, &self.ordered_indices));
        self.seriated_correlations = Some(seriate_matrix(&corr, &self.ordered_indices));
        self.weights = recursive_bisection_min_var(&covariance, &self.ordered_indices)?;
        Ok(())
    }

    /// Returns the leaf order of the last allocation as an [`HrpDendrogram`], labelled with
    /// `assets` (indexed by asset position).
    ///
    /// # Errors
    ///
    /// [`HrpError::MissingClusters`] before a successful [`allocate`](Self::allocate) with at
    /// least two assets.
    ///
    /// # Panics
    ///
    /// Panics if `assets` has fewer entries than the number of allocated assets.
    pub fn plot_clusters(&self, assets: &[String]) -> Result<HrpDendrogram, HrpError> {
        if self.clusters.is_empty() || self.ordered_indices.is_empty() {
            return Err(HrpError::MissingClusters);
        }
        let mut icoord = Vec::with_capacity(self.clusters.len());
        let mut dcoord = Vec::with_capacity(self.clusters.len());
        let mut color_list = Vec::with_capacity(self.clusters.len());
        for (i, _) in self.clusters.iter().enumerate() {
            let x0 = (i * 10) as f64;
            icoord.push([x0, x0 + 2.5, x0 + 7.5, x0 + 10.0]);
            dcoord.push([0.0, 1.0, 1.0, 0.0]);
            color_list.push("C0".to_string());
        }
        let leaves = self.ordered_indices.clone();
        let ivl = leaves.iter().map(|i| assets[*i].clone()).collect();
        Ok(HrpDendrogram { icoord, dcoord, ivl, leaves, color_list })
    }
}

fn returns_from_prices(prices: &DMatrix<f64>) -> Result<DMatrix<f64>, HrpError> {
    if prices.nrows() < 2 {
        return Err(HrpError::NoData);
    }
    let mut out = DMatrix::zeros(prices.nrows() - 1, prices.ncols());
    for r in 1..prices.nrows() {
        for c in 0..prices.ncols() {
            let prev = prices[(r - 1, c)];
            if prev == 0.0 {
                return Err(HrpError::NoData);
            }
            out[(r - 1, c)] = prices[(r, c)] / prev - 1.0;
        }
    }
    Ok(out)
}

fn covariance(returns: &DMatrix<f64>) -> Result<DMatrix<f64>, HrpError> {
    crate::util::stats::covariance(returns).ok_or(HrpError::NoData)
}

fn shrink_covariance(cov: &DMatrix<f64>, alpha: f64) -> DMatrix<f64> {
    let a = alpha.clamp(0.0, 1.0);
    let n = cov.nrows();
    let mut out = cov.clone_owned();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                out[(i, j)] *= 1.0 - a;
            }
        }
    }
    out
}

fn cov2corr(cov: &DMatrix<f64>) -> Result<DMatrix<f64>, HrpError> {
    let n = cov.nrows();
    if n == 0 || cov.ncols() != n {
        return Err(HrpError::DimensionMismatch("covariance must be square"));
    }
    let mut std = vec![0.0; n];
    for i in 0..n {
        let v = cov[(i, i)];
        if v <= 0.0 {
            return Err(HrpError::NoData);
        }
        std[i] = v.sqrt();
    }
    let mut corr = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            corr[(i, j)] = cov[(i, j)] / (std[i] * std[j]);
        }
    }
    Ok(corr)
}

fn corr_to_distances(corr: &DMatrix<f64>) -> DMatrix<f64> {
    let n = corr.nrows();
    let mut d = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let c = corr[(i, j)].clamp(-1.0, 1.0);
            d[(i, j)] = ((1.0 - c).max(0.0) / 2.0).sqrt();
        }
    }
    d
}

fn seriate_matrix(mat: &DMatrix<f64>, order: &[usize]) -> DMatrix<f64> {
    let n = order.len();
    let mut out = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            out[(i, j)] = mat[(order[i], order[j])];
        }
    }
    out
}

fn inverse_variance_weights(cov: &DMatrix<f64>, indices: &[usize]) -> Result<Vec<f64>, HrpError> {
    let mut inv_diag = Vec::with_capacity(indices.len());
    for &idx in indices {
        let v = cov[(idx, idx)];
        if v <= 0.0 {
            return Err(HrpError::NoData);
        }
        inv_diag.push(1.0 / v);
    }
    let sum: f64 = inv_diag.iter().sum();
    if sum <= 0.0 {
        return Err(HrpError::NoData);
    }
    Ok(inv_diag.into_iter().map(|x| x / sum).collect())
}

fn cluster_variance(cov: &DMatrix<f64>, indices: &[usize]) -> Result<f64, HrpError> {
    let w = inverse_variance_weights(cov, indices)?;
    let mut v = 0.0;
    for (ii, &i) in indices.iter().enumerate() {
        for (jj, &j) in indices.iter().enumerate() {
            v += w[ii] * cov[(i, j)] * w[jj];
        }
    }
    Ok(v.max(0.0))
}

fn recursive_bisection_min_var(
    covariance: &DMatrix<f64>,
    ordered_indices: &[usize],
) -> Result<Vec<f64>, HrpError> {
    let n = covariance.nrows();
    let mut weights = vec![1.0; n];
    let mut clustered = vec![ordered_indices.to_vec()];
    while !clustered.is_empty() {
        let mut split = Vec::new();
        for cluster in clustered {
            if cluster.len() > 1 {
                let mid = cluster.len() / 2;
                split.push(cluster[0..mid].to_vec());
                split.push(cluster[mid..].to_vec());
            }
        }
        if split.is_empty() {
            break;
        }
        for i in (0..split.len()).step_by(2) {
            let left = &split[i];
            let right = &split[i + 1];
            let lv = cluster_variance(covariance, left)?;
            let rv = cluster_variance(covariance, right)?;
            let a = 1.0 - lv / (lv + rv + f64::EPSILON);
            for &idx in left {
                weights[idx] *= a;
            }
            for &idx in right {
                weights[idx] *= 1.0 - a;
            }
        }
        clustered = split;
    }
    let sum: f64 = weights.iter().sum();
    if sum > 0.0 {
        for w in &mut weights {
            *w /= sum;
        }
    }
    Ok(weights)
}
