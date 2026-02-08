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
    assert!(contains_cluster(&result.clusters, &[11, 14, 18]));
    assert!(contains_cluster(&result.clusters, &[0, 2, 3, 10, 12, 13, 20, 22, 23]));
    assert!(contains_cluster(&result.clusters, &[5, 6, 7, 25, 26, 27]));
}

#[test]
fn test_check_redo_condition() {
    assert_eq!((4, 5, 6), check_improve_clusters(2.0, 3.0, (1, 2, 3), (4, 5, 6)));
}
