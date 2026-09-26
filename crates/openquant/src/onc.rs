//! Optimal Number of Clusters (ONC): partition a correlation matrix with k-means, choosing the
//! number of clusters by silhouette quality.
//!
//! Not from AFML. References: López de Prado, *Machine Learning for Asset Managers* (2020),
//! Chapter 4, §4.4 (Snippet 4.1, base clustering; Snippet 4.2, higher-level clustering);
//! López de Prado and Lewis (2019), *Detection of false investment strategies using
//! unsupervised learning methods*; Rousseeuw (1987) for the silhouette.
//!
//! The algorithm:
//! 1. Convert correlations to distances `d_ij = sqrt((1 - rho_ij) / 2)` (inputs clamped to
//!    `[-1, 1]`) and represent each item by its row of that distance matrix.
//! 2. Run k-means for every `k` from 2 to `max(N - 1, 2)`, `repeat` times each, and keep the
//!    partition with the highest t-statistic of the silhouettes, `mean(S) / std(S)`.
//! 3. Compute that t-statistic per cluster. If more than two clusters score below the average,
//!    pool their members, re-run the whole procedure on them, and keep the result only if its
//!    mean cluster t-statistic beats that of the clusters it replaced
//!    ([`check_improve_clusters`]).
//!
//! Conventions:
//! - The input is an `N x N` correlation matrix, `N >= 2`; rows and columns are items in the
//!   same order. Symmetry and a unit diagonal are not checked.
//! - Negative correlation is distance, not similarity: `rho = -1` is maximally far apart. Take
//!   absolute correlations first if a series and its mirror image should cluster together.
//! - Member indices in [`OncResult::clusters`] and the order of
//!   [`OncResult::silhouette_scores`] refer to the original row order.
//! - The result has at least two clusters unless every row of the matrix is the same (for
//!   example all ones), which gives one cluster of every item.
//! - k-means is seeded from a fixed value, the repetition number and `k`, so results are
//!   deterministic; there is no seed parameter. Cost grows at least as `N^3` (every `k`,
//!   `repeat` times, quadratic silhouettes), plus the recursion.
//!
//! ```
//! use nalgebra::DMatrix;
//! use openquant::onc::{get_onc_clusters, OncError};
//!
//! # fn main() -> Result<(), OncError> {
//! // Two blocks of three: 0.8 within a block, 0.1 across.
//! let block = |i: usize| i / 3;
//! let corr = DMatrix::from_fn(6, 6, |i, j| {
//!     if i == j {
//!         1.0
//!     } else if block(i) == block(j) {
//!         0.8
//!     } else {
//!         0.1
//!     }
//! });
//!
//! let result = get_onc_clusters(&corr, 3)?;
//! let mut found: Vec<Vec<usize>> = result.clusters.values().cloned().collect();
//! found.sort();
//! assert_eq!(found, vec![vec![0, 1, 2], vec![3, 4, 5]]);
//! assert_eq!(result.silhouette_scores.len(), 6);
//! assert!(result.silhouette_scores.iter().all(|s| *s > 0.5));
//! # Ok(())
//! # }
//! ```
#![deny(missing_docs)]

use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::BTreeMap;

/// Errors returned by [`get_onc_clusters`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum OncError {
    /// The correlation matrix is not square or has fewer than two rows.
    #[error("the correlation matrix must be square with at least two rows")]
    InvalidCorrelationMatrix,
    /// `repeat` is zero.
    #[error("repeat must be positive")]
    InvalidRepeat,
    /// No candidate partition could be selected. Not expected on a finite correlation matrix;
    /// it can occur when `NaN` entries make every candidate's quality score `NaN`.
    #[error("clustering failed to produce a partition")]
    ClusteringFailed,
}

/// Partition found by [`get_onc_clusters`].
#[derive(Debug, Clone)]
pub struct OncResult {
    /// The input correlation matrix with rows and columns permuted so that the members of each
    /// cluster are contiguous, clusters in label order.
    pub ordered_correlation: DMatrix<f64>,
    /// Cluster label (`0..number of clusters`) to the indices of its members, in the original
    /// row order of the input.
    pub clusters: BTreeMap<usize, Vec<usize>>,
    /// Silhouette score of every item, indexed by the original row order; a singleton
    /// cluster's member scores 0.
    pub silhouette_scores: Vec<f64>,
}

#[derive(Clone)]
struct ClusterState {
    ordered_correlation: DMatrix<f64>,
    clusters: BTreeMap<usize, Vec<usize>>,
    silhouette_scores: Vec<f64>,
}

/// Keep the re-clustered partition only if its mean cluster t-stat beats the mean t-stat of the
/// clusters that were re-clustered (MLAM Snippet 4.2); otherwise keep the old partition.
///
/// Returns `new_cluster` when `new_tstat_mean > mean_redo_tstat` and `old_cluster` otherwise
/// (ties and `NaN` keep the old one). Exposed for parity with mlfinlab; [`get_onc_clusters`]
/// calls it internally.
///
/// ```
/// use openquant::onc::check_improve_clusters;
///
/// assert_eq!(check_improve_clusters(2.0, 1.5, "old", "new"), "new");
/// assert_eq!(check_improve_clusters(1.5, 1.5, "old", "new"), "old");
/// ```
pub fn check_improve_clusters<T: Clone>(
    new_tstat_mean: f64,
    mean_redo_tstat: f64,
    old_cluster: T,
    new_cluster: T,
) -> T {
    if new_tstat_mean > mean_redo_tstat {
        new_cluster
    } else {
        old_cluster
    }
}

/// Partition the items of a correlation matrix with ONC (MLAM §4.4, Snippets 4.1–4.2).
///
/// `corr_mat` is an `N x N` correlation matrix (`N >= 2`; entries clamped to `[-1, 1]`,
/// symmetry and unit diagonal not checked). `repeat` is the number of k-means initialisations
/// per candidate `k`. The number of clusters is chosen by the silhouette t-statistic; see the
/// [module documentation](self) for the full procedure. The search starts at `k = 2`, so a
/// matrix with no structure still comes back partitioned: a low mean silhouette is the sign
/// that the clusters are not real. The exception is a matrix whose rows are all identical,
/// such as all ones: there is nothing to separate, every k-means centroid is the same point,
/// and the result is one cluster of every item, each scoring a silhouette of 0.
///
/// # Errors
///
/// - [`OncError::InvalidRepeat`] if `repeat == 0`.
/// - [`OncError::InvalidCorrelationMatrix`] if `corr_mat` is not square or has fewer than two
///   rows.
/// - [`OncError::ClusteringFailed`] if no candidate partition can be selected (only with
///   `NaN` entries).
///
/// ```
/// use nalgebra::DMatrix;
/// use openquant::onc::{get_onc_clusters, OncError};
///
/// // Items 0, 2 and 4 move together, as do 1, 3 and 5.
/// let corr = DMatrix::from_fn(6, 6, |i, j| {
///     if i == j {
///         1.0
///     } else if i % 2 == j % 2 {
///         0.9
///     } else {
///         0.0
///     }
/// });
/// let result = get_onc_clusters(&corr, 2).unwrap();
/// let mut found: Vec<Vec<usize>> = result.clusters.values().cloned().collect();
/// found.sort();
/// assert_eq!(found, vec![vec![0, 2, 4], vec![1, 3, 5]]);
/// // The ordered matrix puts each block on the diagonal.
/// let first = &result.clusters[&0];
/// assert_eq!(result.ordered_correlation[(0, 1)], corr[(first[0], first[1])]);
///
/// assert_eq!(get_onc_clusters(&corr, 0).unwrap_err(), OncError::InvalidRepeat);
/// assert_eq!(
///     get_onc_clusters(&DMatrix::from_element(1, 1, 1.0), 1).unwrap_err(),
///     OncError::InvalidCorrelationMatrix
/// );
/// ```
pub fn get_onc_clusters(corr_mat: &DMatrix<f64>, repeat: usize) -> Result<OncResult, OncError> {
    if repeat == 0 {
        return Err(OncError::InvalidRepeat);
    }
    if corr_mat.nrows() != corr_mat.ncols() || corr_mat.nrows() < 2 {
        return Err(OncError::InvalidCorrelationMatrix);
    }

    let state = cluster_kmeans_top(corr_mat, repeat)?;
    Ok(OncResult {
        ordered_correlation: state.ordered_correlation,
        clusters: state.clusters,
        silhouette_scores: state.silhouette_scores,
    })
}

fn cluster_kmeans_top(corr_mat: &DMatrix<f64>, repeat: usize) -> Result<ClusterState, OncError> {
    let max_num_clusters = corr_mat.ncols().saturating_sub(1).max(2);
    let base = cluster_kmeans_base(corr_mat, max_num_clusters, repeat)?;

    let mut cluster_quality: BTreeMap<usize, f64> = BTreeMap::new();
    for (k, members) in &base.clusters {
        let scores: Vec<f64> = members.iter().map(|&idx| base.silhouette_scores[idx]).collect();
        cluster_quality.insert(*k, tstat(&scores));
    }

    let avg_quality = {
        let vals: Vec<f64> = cluster_quality.values().copied().collect();
        if vals.is_empty() {
            0.0
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        }
    };

    let redo_clusters: Vec<usize> = cluster_quality
        .iter()
        .filter_map(|(k, q)| if *q < avg_quality { Some(*k) } else { None })
        .collect();

    if redo_clusters.len() <= 2 {
        return Ok(base);
    }

    let mut keys_redo = Vec::new();
    for key in &redo_clusters {
        if let Some(v) = base.clusters.get(key) {
            keys_redo.extend(v.iter().copied());
        }
    }

    if keys_redo.len() < 2 {
        return Ok(base);
    }

    let corr_tmp = submatrix(corr_mat, &keys_redo);
    let mean_redo_tstat = {
        let vals: Vec<f64> =
            redo_clusters.iter().filter_map(|k| cluster_quality.get(k).copied()).collect();
        vals.iter().sum::<f64>() / vals.len() as f64
    };

    let top_state = cluster_kmeans_top(&corr_tmp, repeat)?;
    let mut top_clusters_global = BTreeMap::new();
    for (k, v) in top_state.clusters {
        let mapped: Vec<usize> = v.into_iter().map(|local_idx| keys_redo[local_idx]).collect();
        top_clusters_global.insert(k, mapped);
    }

    let mut kept_clusters = BTreeMap::new();
    for (k, v) in &base.clusters {
        if !redo_clusters.contains(k) {
            kept_clusters.insert(*k, v.clone());
        }
    }

    let improved = improve_clusters(corr_mat, &kept_clusters, &top_clusters_global)?;

    let new_tstat_mean = {
        let mut vals = Vec::new();
        for members in improved.clusters.values() {
            let scores: Vec<f64> =
                members.iter().map(|&idx| improved.silhouette_scores[idx]).collect();
            vals.push(tstat(&scores));
        }
        vals.iter().sum::<f64>() / vals.len() as f64
    };

    Ok(check_improve_clusters(new_tstat_mean, mean_redo_tstat, base, improved))
}

fn improve_clusters(
    corr_mat: &DMatrix<f64>,
    kept_clusters: &BTreeMap<usize, Vec<usize>>,
    top_clusters: &BTreeMap<usize, Vec<usize>>,
) -> Result<ClusterState, OncError> {
    let mut clusters_new: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for members in kept_clusters.values() {
        clusters_new.insert(clusters_new.len(), members.clone());
    }
    for members in top_clusters.values() {
        clusters_new.insert(clusters_new.len(), members.clone());
    }

    let mut new_idx = Vec::new();
    for members in clusters_new.values() {
        new_idx.extend(members.iter().copied());
    }

    let corr_new = submatrix(corr_mat, &new_idx);
    let labels = labels_from_clusters(corr_mat.nrows(), &clusters_new);
    let dist = corr_to_distance(corr_mat);
    let silh_scores_new = silhouette_samples(&dist, &labels);

    Ok(ClusterState {
        ordered_correlation: corr_new,
        clusters: clusters_new,
        silhouette_scores: silh_scores_new,
    })
}

fn cluster_kmeans_base(
    corr_mat: &DMatrix<f64>,
    max_num_clusters: usize,
    repeat: usize,
) -> Result<ClusterState, OncError> {
    let distance = corr_to_distance(corr_mat);

    let mut best_labels: Option<Vec<usize>> = None;
    let mut best_silh: Option<Vec<f64>> = None;

    for rep in 0..repeat {
        for num_clusters in 2..=max_num_clusters {
            let labels = kmeans_labels(
                &distance,
                num_clusters,
                42 + rep as u64 * 131 + num_clusters as u64,
            )?;
            let silh = silhouette_samples(&distance, &labels);

            let stat = tstat(&silh);
            let best_stat = best_silh.as_ref().map_or(f64::NEG_INFINITY, |s| tstat(s));
            // A perfect clustering has zero silhouette variance, so its t-stat is +inf and
            // nothing may replace it. Only a NaN incumbent is replaced unconditionally.
            if best_stat.is_nan() || stat > best_stat {
                best_labels = Some(labels);
                best_silh = Some(silh);
            }
        }
    }

    let labels = best_labels.ok_or(OncError::ClusteringFailed)?;
    let silh = best_silh.ok_or(OncError::ClusteringFailed)?;

    let mut new_idx: Vec<usize> = (0..labels.len()).collect();
    new_idx.sort_by_key(|&i| labels[i]);

    let corr1 = submatrix(corr_mat, &new_idx);

    let mut raw_clusters: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (idx, lbl) in labels.iter().copied().enumerate() {
        raw_clusters.entry(lbl).or_default().push(idx);
    }

    let mut clusters = BTreeMap::new();
    for members in raw_clusters.values() {
        clusters.insert(clusters.len(), members.clone());
    }

    Ok(ClusterState { ordered_correlation: corr1, clusters, silhouette_scores: silh })
}

fn tstat(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64;
    let std = var.sqrt();
    if std <= 1e-12 {
        if mean > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    } else {
        mean / std
    }
}

fn labels_from_clusters(n: usize, clusters: &BTreeMap<usize, Vec<usize>>) -> Vec<usize> {
    let mut labels = vec![0usize; n];
    for (label, members) in clusters {
        for &idx in members {
            labels[idx] = *label;
        }
    }
    labels
}

fn submatrix(m: &DMatrix<f64>, idx: &[usize]) -> DMatrix<f64> {
    let n = idx.len();
    let mut out = DMatrix::zeros(n, n);
    for (i, &ri) in idx.iter().enumerate() {
        for (j, &cj) in idx.iter().enumerate() {
            out[(i, j)] = m[(ri, cj)];
        }
    }
    out
}

fn corr_to_distance(corr: &DMatrix<f64>) -> DMatrix<f64> {
    let n = corr.nrows();
    let mut distance = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let c = corr[(i, j)].clamp(-1.0, 1.0);
            distance[(i, j)] = ((1.0 - c) / 2.0).sqrt();
        }
    }
    distance
}

fn kmeans_labels(data: &DMatrix<f64>, k: usize, seed: u64) -> Result<Vec<usize>, OncError> {
    let n = data.nrows();
    let d = data.ncols();
    if k < 2 || k > n {
        return Err(OncError::ClusteringFailed);
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(&mut rng);

    let mut centroids = DMatrix::<f64>::zeros(k, d);
    for c in 0..k {
        let src = idx[c];
        for j in 0..d {
            centroids[(c, j)] = data[(src, j)];
        }
    }

    let mut labels = vec![0usize; n];
    let mut changed = true;

    for _ in 0..100 {
        if !changed {
            break;
        }
        changed = false;

        for i in 0..n {
            let mut best_c = 0usize;
            let mut best_dist = f64::INFINITY;
            for c in 0..k {
                let mut s = 0.0;
                for j in 0..d {
                    let diff = data[(i, j)] - centroids[(c, j)];
                    s += diff * diff;
                }
                if s < best_dist {
                    best_dist = s;
                    best_c = c;
                }
            }
            if labels[i] != best_c {
                labels[i] = best_c;
                changed = true;
            }
        }

        let mut sums = DMatrix::<f64>::zeros(k, d);
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = labels[i];
            counts[c] += 1;
            for j in 0..d {
                sums[(c, j)] += data[(i, j)];
            }
        }

        for c in 0..k {
            if counts[c] == 0 {
                let repl = idx[c % n];
                for j in 0..d {
                    centroids[(c, j)] = data[(repl, j)];
                }
            } else {
                let inv = 1.0 / counts[c] as f64;
                for j in 0..d {
                    centroids[(c, j)] = sums[(c, j)] * inv;
                }
            }
        }
    }

    Ok(labels)
}

fn silhouette_samples(data: &DMatrix<f64>, labels: &[usize]) -> Vec<f64> {
    let n = data.nrows();
    let mut by_cluster: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, lbl) in labels.iter().copied().enumerate() {
        by_cluster.entry(lbl).or_default().push(i);
    }

    let mut pairwise = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let mut s = 0.0;
            for c in 0..data.ncols() {
                let diff = data[(i, c)] - data[(j, c)];
                s += diff * diff;
            }
            let d = s.sqrt();
            pairwise[(i, j)] = d;
            pairwise[(j, i)] = d;
        }
    }

    let mut scores = vec![0.0; n];
    for i in 0..n {
        let own = labels[i];
        let own_members = &by_cluster[&own];

        // A point alone in its cluster has no intra-cluster distance; its silhouette is 0 by
        // definition (Rousseeuw 1987; scikit-learn does the same). Scoring it (b - 0) / b = 1
        // makes "every point its own cluster" look like the perfect clustering.
        if own_members.len() <= 1 {
            continue;
        }

        let a = own_members.iter().filter(|&&j| j != i).map(|&j| pairwise[(i, j)]).sum::<f64>()
            / (own_members.len() - 1) as f64;

        let mut b = f64::INFINITY;
        for (cluster, members) in &by_cluster {
            if *cluster == own || members.is_empty() {
                continue;
            }
            let mut s = 0.0;
            for &j in members {
                s += pairwise[(i, j)];
            }
            let mean = s / members.len() as f64;
            if mean < b {
                b = mean;
            }
        }

        scores[i] = if !b.is_finite() || (a == 0.0 && b == 0.0) { 0.0 } else { (b - a) / a.max(b) };
    }

    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silhouette_of_a_singleton_cluster_is_zero() {
        // Points 0 and 1 sit together, point 2 is alone and far away.
        let data = DMatrix::from_row_slice(3, 1, &[0.0, 0.1, 10.0]);
        let scores = silhouette_samples(&data, &[0, 0, 1]);
        assert!(scores[0] > 0.9 && scores[1] > 0.9);
        assert_eq!(scores[2], 0.0);
    }

    #[test]
    fn a_perfect_clustering_is_not_replaced_by_a_later_candidate() {
        // Two identical pairs: k = 2 is perfect (every silhouette is 1, zero variance, t-stat
        // +inf). The search goes on to try k = 3 and must keep k = 2.
        let corr = DMatrix::from_row_slice(
            4,
            4,
            &[1.0, 0.9, 0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.0, 0.0, 0.9, 1.0],
        );
        let state = cluster_kmeans_base(&corr, 3, 3).unwrap();
        assert_eq!(state.clusters.len(), 2);
    }
}
