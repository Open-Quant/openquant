use csv::ReaderBuilder;
use nalgebra::DMatrix;
use openquant::onc::{check_improve_clusters, get_onc_clusters};
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

fn contains_cluster(clusters: &BTreeMap<usize, Vec<usize>>, expected: &[usize]) -> bool {
    let mut sorted_expected = expected.to_vec();
    sorted_expected.sort_unstable();
    clusters.values().any(|members| {
        let mut sorted_members = members.clone();
        sorted_members.sort_unstable();
        sorted_members == sorted_expected
    })
}

#[test]
fn test_get_onc_clusters() {
    let corr = load_breast_cancer_correlation();
    let result = get_onc_clusters(&corr, 50).unwrap();

    assert!(result.clusters.len() >= 5);
    // The clusters that ONC (MLAM snippets 4.1-4.2, scikit-learn KMeans) finds under every seed:
    // tests/fixtures/onc/generate_breast_cancer.py.
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onc/breast_cancer_reference.json");
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for cluster in reference["stable_clusters"].as_array().unwrap() {
        let members: Vec<usize> =
            cluster.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        assert!(
            contains_cluster(&result.clusters, &members),
            "missing {members:?}; got {:?}",
            result.clusters
        );
    }
}

#[test]
fn test_check_redo_condition() {
    assert_eq!((4, 5, 6), check_improve_clusters(2.0, 3.0, (1, 2, 3), (4, 5, 6)));
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
    // case where every silhouette score is identical (zero variance, infinite t-stat).
    for sizes in
        [vec![4, 4], vec![3, 5, 4], vec![10, 10, 10], vec![6, 9, 7, 8], vec![5, 5, 5, 5, 5, 5]]
    {
        let corr = block_correlation(&sizes, 0.9, 0.05);
        let result = get_onc_clusters(&corr, 10).unwrap();

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
        assert_eq!(got, expected, "block sizes {sizes:?}");
    }
}
