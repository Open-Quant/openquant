use csv::ReaderBuilder;
use nalgebra::DMatrix;
use openquant::onc::{check_improve_clusters, get_onc_clusters, get_onc_clusters_with_seed};
use std::collections::BTreeMap;
use std::path::Path;

fn load_breast_cancer_correlation() -> DMatrix<f64> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/onc/breast_cancer.csv");
    let mut rdr = ReaderBuilder::new().has_headers(false).flexible(true).from_path(path).unwrap();

    let mut rows: Vec<Vec<f64>> = Vec::new();
    for (i, rec) in rdr.records().enumerate() {
        let row = rec.unwrap();
        if i == 0 {
            continue;
        }
        if row.len() < 30 {
            continue;
        }
        let vals: Vec<f64> = row.iter().take(30).map(|v| v.parse::<f64>().unwrap()).collect();
        rows.push(vals);
    }

    let nrows = rows.len();
    let ncols = rows[0].len();
    let mut means = vec![0.0; ncols];
    for r in &rows {
        for c in 0..ncols {
            means[c] += r[c];
        }
    }
    for m in &mut means {
        *m /= nrows as f64;
    }

    let mut std = vec![0.0; ncols];
    for c in 0..ncols {
        let mut s = 0.0;
        for r in &rows {
            let d = r[c] - means[c];
            s += d * d;
        }
        std[c] = (s / (nrows as f64 - 1.0)).sqrt();
    }

    let mut corr = DMatrix::zeros(ncols, ncols);
    for i in 0..ncols {
        for j in 0..ncols {
            let mut s = 0.0;
            for r in &rows {
                s += (r[i] - means[i]) * (r[j] - means[j]);
            }
            let cov = s / (nrows as f64 - 1.0);
            corr[(i, j)] = cov / (std[i] * std[j]);
        }
    }
    corr
}

fn breast_cancer_reference() -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onc/breast_cancer_reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn index_sets(value: &serde_json::Value) -> Vec<Vec<usize>> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect())
        .collect()
}

fn sorted_clusters(clusters: &BTreeMap<usize, Vec<usize>>) -> Vec<Vec<usize>> {
    let mut sets: Vec<Vec<usize>> = clusters
        .values()
        .map(|m| {
            let mut m = m.clone();
            m.sort_unstable();
            m
        })
        .collect();
    sets.sort();
    sets
}

/// ONC on real data is a random search (see the module docs): on the breast-cancer features
/// the partition it returns depends on the random stream. What does not depend on it, in
/// scikit-learn's ONC (tests/fixtures/onc/generate_breast_cancer.py) and here, is that every
/// run keeps each of the eight finest groups whole, i.e. returns one of their coarsenings, and
/// finds the clusters that every reference run finds. Checked under several seeds, so that a
/// change of random stream (a rand upgrade, #218/#219) cannot break it. With the random-point
/// initialisation this replaced, most streams gave a two-cluster partition that splits the
/// groups.
#[test]
fn test_get_onc_clusters() {
    let corr = load_breast_cancer_correlation();
    let reference = breast_cancer_reference();
    let finest = index_sets(&reference["finest_partition"]);
    let stable = index_sets(&reference["stable_clusters"]);
    let min_clusters = reference["min_clusters"].as_u64().unwrap() as usize;

    for seed in 0..3 {
        let result = get_onc_clusters_with_seed(&corr, 50, seed).unwrap();
        let found = sorted_clusters(&result.clusters);
        assert!(found.len() >= min_clusters, "seed {seed}: {found:?}");
        for group in &finest {
            assert!(
                found.iter().any(|c| group.iter().all(|i| c.contains(i))),
                "seed {seed}: group {group:?} split in {found:?}"
            );
        }
        for cluster in &stable {
            assert!(found.contains(cluster), "seed {seed}: missing {cluster:?} in {found:?}");
        }
    }
}

/// #107: the re-clustering step (MLAM Snippet 4.2) must keep its partition when that scores
/// higher. `recluster_reference.json` (tests/fixtures/onc/generate.py) is a two-level factor
/// model whose first pass returns the eight planted sub-groups; re-clustering the ones below
/// average gives a partition with a higher mean cluster t-stat, so that is what ONC must
/// return, under any seed. With the comparison inverted it returned the first pass.
#[test]
fn test_onc_keeps_the_better_partition_after_reclustering() {
    #[derive(serde::Deserialize)]
    struct Reference {
        corr: Vec<Vec<f64>>,
        planted_clusters: Vec<Vec<usize>>,
        silhouette: Vec<f64>,
    }
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onc/recluster_reference.json");
    let r: Reference = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let n = r.corr.len();
    let corr = DMatrix::from_fn(n, n, |i, j| r.corr[i][j]);
    let planted: BTreeMap<usize, Vec<usize>> =
        r.planted_clusters.iter().cloned().enumerate().collect();
    let first_pass = mean_cluster_tstat(&planted, &r.silhouette);

    for seed in 0..4 {
        let result = get_onc_clusters_with_seed(&corr, 10, seed).unwrap();
        let quality = mean_cluster_tstat(&result.clusters, &result.silhouette_scores);
        assert!(
            quality > first_pass * 1.5,
            "seed {seed}: mean cluster t-stat {quality}, first pass {first_pass}"
        );
        assert!(result.clusters.len() < r.planted_clusters.len(), "seed {seed}");
    }
}

#[test]
fn test_check_redo_condition() {
    // The re-clustered partition replaces the old one only if it scores higher (MLAM Snippet
    // 4.2: `if newTstatMean <= redoTstatMean: return old`).
    assert_eq!((1, 2, 3), check_improve_clusters(2.0, 3.0, (1, 2, 3), (4, 5, 6)));
    assert_eq!((1, 2, 3), check_improve_clusters(3.0, 3.0, (1, 2, 3), (4, 5, 6)));
    assert_eq!((4, 5, 6), check_improve_clusters(3.0, 2.0, (1, 2, 3), (4, 5, 6)));
}

fn mean_cluster_tstat(clusters: &BTreeMap<usize, Vec<usize>>, silhouette: &[f64]) -> f64 {
    let tstats: Vec<f64> = clusters
        .values()
        .map(|members| {
            let s: Vec<f64> = members.iter().map(|&i| silhouette[i]).collect();
            let mean = s.iter().sum::<f64>() / s.len() as f64;
            let sd = (s.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / s.len() as f64).sqrt();
            // As in onc.rs: a zero-variance cluster scores +inf if its silhouettes are positive.
            match (sd <= 1e-12, mean > 0.0) {
                (true, true) => f64::INFINITY,
                (true, false) => 0.0,
                _ => mean / sd,
            }
        })
        .collect();
    tstats.iter().sum::<f64>() / tstats.len() as f64
}

/// Correlation matrix with `sizes.len()` planted blocks: `within` inside a block, `between`
/// across blocks, 1 on the diagonal.
fn block_correlation(sizes: &[usize], within: f64, between: f64) -> DMatrix<f64> {
    let n: usize = sizes.iter().sum();
    let mut block_of = Vec::with_capacity(n);
    for (b, size) in sizes.iter().enumerate() {
        block_of.extend(std::iter::repeat_n(b, *size));
    }
    DMatrix::from_fn(n, n, |i, j| {
        if i == j {
            1.0
        } else if block_of[i] == block_of[j] {
            within
        } else {
            between
        }
    })
}

#[test]
fn test_onc_recovers_planted_blocks() {
    // Includes n = 30, the size the library used to special-case, and a perfectly separable
    // case where every silhouette score is identical (zero variance, infinite t-stat). Every
    // seed must find the blocks: with six equal blocks, merging them three and three also
    // gives every item the same silhouette, and before #218 the seeds that met that
    // partition first returned it.
    for (sizes, seed) in
        [vec![4, 4], vec![3, 5, 4], vec![10, 10, 10], vec![6, 9, 7, 8], vec![5, 5, 5, 5, 5, 5]]
            .into_iter()
            .flat_map(|sizes| (0..6).map(move |seed| (sizes.clone(), seed)))
    {
        let corr = block_correlation(&sizes, 0.9, 0.05);
        let result = get_onc_clusters_with_seed(&corr, 10, seed).unwrap();

        let mut got: Vec<Vec<usize>> = result.clusters.values().cloned().collect();
        for members in &mut got {
            members.sort_unstable();
        }
        got.sort();

        let mut expected = Vec::new();
        let mut start = 0;
        for size in &sizes {
            expected.push((start..start + size).collect::<Vec<usize>>());
            start += size;
        }
        assert_eq!(got, expected, "block sizes {sizes:?}, seed {seed}");
    }
}

/// #185 item 14: identical rows are the one case with a single cluster. There is nothing to
/// separate, so every k-means point lands on the first of several identical centroids.
#[test]
fn identical_rows_come_back_as_one_cluster() {
    for n in [2, 3, 6] {
        let corr = DMatrix::from_element(n, n, 1.0);
        let result = get_onc_clusters(&corr, 3).unwrap();
        assert_eq!(result.clusters.len(), 1, "n = {n}");
        assert_eq!(result.clusters[&0], (0..n).collect::<Vec<_>>());
        assert_eq!(result.silhouette_scores, vec![0.0; n]);
    }
    // Duplicates inside otherwise distinct structure still split into at least two clusters.
    let block = |i: usize| i / 3;
    let corr = DMatrix::from_fn(6, 6, |i, j| if block(i) == block(j) { 1.0 } else { 0.2 });
    let result = get_onc_clusters(&corr, 3).unwrap();
    let mut found: Vec<Vec<usize>> = result.clusters.values().cloned().collect();
    found.sort();
    assert_eq!(found, vec![vec![0, 1, 2], vec![3, 4, 5]]);
}
