//! Single-linkage tree building shared by [`crate::hrp`] and [`crate::hcaa`].
//!
//! Both allocators start from a correlation distance matrix (HRP's `sqrt((1 - rho) / 2)`,
//! HCAA's `sqrt(2 (1 - rho))`; one is twice the other, which does not change a single-linkage
//! tree) and either cluster on it directly or on the Euclidean distance between its columns
//! (AFML §16.4.1, Snippet 16.4).

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

/// Single-linkage merges on a square distance matrix, in scipy's linkage convention: row `k`
/// merges the two listed nodes (smaller id first) into node `n + k`; ids below `n` are assets.
/// Distances within `1e-12` are ties, broken by the smaller pair of ids.
pub(crate) fn single_linkage_children(distance: &DMatrix<f64>) -> Vec<[usize; 2]> {
    #[derive(Clone)]
    struct Cluster {
        id: usize,
        members: Vec<usize>,
    }

    let n = distance.nrows();
    let mut clusters: Vec<Cluster> = (0..n).map(|i| Cluster { id: i, members: vec![i] }).collect();
    let mut next_id = n;
    let mut children = Vec::with_capacity(n.saturating_sub(1));
    let eps = 1e-12;

    while clusters.len() > 1 {
        let mut best_i = 0usize;
        let mut best_j = 1usize;
        let mut best_d = f64::INFINITY;
        let mut best_ids = (
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
                if d + eps < best_d || ((d - best_d).abs() <= eps && ids < best_ids) {
                    best_i = i;
                    best_j = j;
                    best_d = d;
                    best_ids = ids;
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
