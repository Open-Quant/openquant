//! Value tests for `onc`: the silhouette scores ONC reports, checked against scikit-learn.
//! The reference (`tests/fixtures/onc/silhouette_reference.json`) is written by
//! `tests/fixtures/onc/generate.py`; it contains a sample correlation matrix with three planted
//! factor clusters and sklearn's `silhouette_samples` for the planted labelling.
//!
//! `tests/onc.rs` already checks that planted blocks are recovered; nothing checked a silhouette
//! VALUE, and the silhouette t-statistic is what ONC optimises. See
//! `docs/test-sensitivity-audit.md`.

use nalgebra::DMatrix;
use openquant::onc::get_onc_clusters;
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Reference {
    corr: Vec<Vec<f64>>,
    planted_clusters: Vec<Vec<usize>>,
    silhouette: Vec<f64>,
}

fn reference() -> Reference {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onc/silhouette_reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn silhouette_scores_match_sklearn_on_planted_factor_clusters() {
    let r = reference();
    let n = r.corr.len();
    let corr = DMatrix::from_fn(n, n, |i, j| r.corr[i][j]);
    let result = get_onc_clusters(&corr, 10).unwrap();

    let mut got: Vec<Vec<usize>> = result.clusters.values().cloned().collect();
    for members in &mut got {
        members.sort_unstable();
    }
    got.sort();
    assert_eq!(got, r.planted_clusters, "ONC did not recover the planted clusters");

    assert_eq!(result.silhouette_scores.len(), n);
    for (i, (g, w)) in result.silhouette_scores.iter().zip(&r.silhouette).enumerate() {
        // Each score is built from 15 square roots and sums of at most 15 O(1) terms on both
        // sides: relative rounding ~15 * 2.2e-16. 1e-12 is two decades above that.
        assert!((g - w).abs() < 1e-12, "silhouette[{i}] = {g}, sklearn {w}");
    }

    // ordered_correlation is the input correlation re-indexed cluster by cluster
    let order: Vec<usize> = result.clusters.values().flatten().copied().collect();
    for (a, &i) in order.iter().enumerate() {
        for (b, &j) in order.iter().enumerate() {
            assert_eq!(result.ordered_correlation[(a, b)], r.corr[i][j], "ordered[{a},{b}]");
        }
    }
}
