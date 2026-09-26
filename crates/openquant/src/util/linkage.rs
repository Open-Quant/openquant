//! Hierarchical tree building shared by [`crate::hrp`] (single linkage) and [`crate::hcaa`]
//! (single, complete, average or Ward linkage).
//!
//! Both allocators start from a correlation distance matrix (HRP's `sqrt((1 - rho) / 2)`,
//! HCAA's `sqrt(2 (1 - rho))`; one is twice the other, which does not change the tree under any
//! of these linkages, whose updates are all homogeneous of degree one) and either cluster on it
//! directly or on the Euclidean distance between its columns (AFML §16.4.1, Snippet 16.4).

use nalgebra::DMatrix;

/// `d~_ij = sqrt(sum_n (d_ni - d_nj)^2)`: the Euclidean distance between columns `i` and `j`.
/// The diagonal of `d` is taken as exactly 0 (rounding in a covariance-to-correlation step can
/// leave `rho_ii` a few ulps below 1).
pub(crate) fn distance_of_distances(distances: &DMatrix<f64>) -> DMatrix<f64> {
    let mut d = distances.clone_owned();
    d.fill_diagonal(0.0);
    let n = d.nrows();
    let mut out = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in i + 1..n {
            let s: f64 = (0..n).map(|k| (d[(k, i)] - d[(k, j)]).powi(2)).sum();
            out[(i, j)] = s.sqrt();
            out[(j, i)] = out[(i, j)];
        }
    }
    out
}

/// How the distance from a merged cluster to every other cluster is computed (the
/// Lance–Williams update). The formulas are scipy's (`scipy.cluster.hierarchy.linkage`), written
/// for a merge of clusters `x` and `y` (sizes `n_x`, `n_y`) seen from a third cluster `i`:
///
/// | method | `d(x ∪ y, i)` |
/// | --- | --- |
/// | `Single` | `min(d_xi, d_yi)` |
/// | `Complete` | `max(d_xi, d_yi)` |
/// | `Average` | `(n_x d_xi + n_y d_yi) / (n_x + n_y)` (UPGMA) |
/// | `Ward` | `sqrt(((n_i + n_x) d_xi² + (n_i + n_y) d_yi² − n_i d_xy²) / (n_x + n_y + n_i))` |
///
/// `Ward` is scipy's `method="ward"` (R's `ward.D2`): the recurrence is applied to the distances
/// as given. When they are Euclidean distances between points it is exactly Ward's
/// minimum-variance criterion on those points, and the merge height is
/// `sqrt(2 n_x n_y / (n_x + n_y)) · ||c_x − c_y||` for centroids `c`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Linkage {
    Single,
    Complete,
    Average,
    Ward,
}

impl Linkage {
    fn update(self, d_xi: f64, d_yi: f64, d_xy: f64, n_x: f64, n_y: f64, n_i: f64) -> f64 {
        match self {
            Self::Single => d_xi.min(d_yi),
            Self::Complete => d_xi.max(d_yi),
            Self::Average => (n_x * d_xi + n_y * d_yi) / (n_x + n_y),
            Self::Ward => {
                let t = 1.0 / (n_x + n_y + n_i);
                ((n_i + n_x) * t * d_xi * d_xi + (n_i + n_y) * t * d_yi * d_yi
                    - n_i * t * d_xy * d_xy)
                    .max(0.0)
                    .sqrt()
            }
        }
    }
}

/// Single-linkage merges; see [`linkage_children`].
pub(crate) fn single_linkage_children(distance: &DMatrix<f64>) -> Vec<[usize; 2]> {
    linkage_children(distance, Linkage::Single)
}

/// Agglomerative merges on a square distance matrix, in scipy's linkage convention: row `k`
/// merges the two listed nodes (smaller id first) into node `n + k`; ids below `n` are assets.
///
/// Each step merges the closest pair of current clusters and updates the distances to the new
/// cluster with `method`'s Lance–Williams formula. All four methods are reducible, so the merge
/// heights never decrease and the rows come out in scipy's order (sorted by height). Distances
/// within `1e-12` are ties, broken by the smaller pair of ids.
pub(crate) fn linkage_children(distance: &DMatrix<f64>, method: Linkage) -> Vec<[usize; 2]> {
    let n = distance.nrows();
    let mut d = distance.clone_owned();
    // Slot `s` holds a current cluster (node id and size) or, once merged away, None.
    let mut slots: Vec<Option<(usize, f64)>> = (0..n).map(|i| Some((i, 1.0))).collect();
    let mut children = Vec::with_capacity(n.saturating_sub(1));
    let eps = 1e-12;

    for step in 0..n.saturating_sub(1) {
        let mut best: Option<(usize, usize, f64, (usize, usize))> = None;
        for a in 0..n {
            let Some((id_a, _)) = slots[a] else { continue };
            for b in a + 1..n {
                let Some((id_b, _)) = slots[b] else { continue };
                let dist = d[(a, b)];
                let ids = (id_a.min(id_b), id_a.max(id_b));
                let better = match best {
                    None => true,
                    Some((_, _, best_d, best_ids)) => {
                        dist + eps < best_d || ((dist - best_d).abs() <= eps && ids < best_ids)
                    }
                };
                if better {
                    best = Some((a, b, dist, ids));
                }
            }
        }
        let Some((x, y, d_xy, ids)) = best else { break };
        let (Some((_, n_x)), Some((_, n_y))) = (slots[x], slots[y]) else { break };
        for i in 0..n {
            let Some((_, n_i)) = slots[i] else { continue };
            if i == x || i == y {
                continue;
            }
            let v = method.update(d[(x, i)], d[(y, i)], d_xy, n_x, n_y, n_i);
            d[(x, i)] = v;
            d[(i, x)] = v;
        }
        children.push([ids.0, ids.1]);
        slots[x] = Some((n + step, n_x + n_y));
        slots[y] = None;
    }
    children
}

/// The leaves under `curr_index` in dendrogram order (left subtree first), AFML Snippet 16.2.
pub(crate) fn quasi_diagonalization(
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
