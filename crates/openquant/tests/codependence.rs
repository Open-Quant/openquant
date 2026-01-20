use openquant::codependence::{
    absolute_angular_distance, angular_distance, distance_correlation, get_mutual_info,
    get_optimal_number_of_bins, squared_angular_distance, variation_of_information_score,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct CodependenceRow {
    x: f64,
    y_1: f64,
    y_2: f64,
}

fn load_series() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/codependence/random_state_42.csv");
    let mut reader = csv::Reader::from_path(fixture_path).expect("fixture csv");
    let mut x = Vec::new();
    let mut y_1 = Vec::new();
    let mut y_2 = Vec::new();

    for result in reader.deserialize::<CodependenceRow>() {
        let row = result.expect("valid row");
        x.push(row.x);
        y_1.push(row.y_1);
        y_2.push(row.y_2);
    }

    (x, y_1, y_2)
}

fn corrcoef(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    cov / (var_x * var_y).sqrt()
}

#[test]
fn test_correlations() {
    let (x, y_1, y_2) = load_series();

    let angular_dist = angular_distance(&x, &y_1).expect("angular distance");
    let sq_angular_dist = squared_angular_distance(&x, &y_1).expect("sq angular distance");
    let abs_angular_dist = absolute_angular_distance(&x, &y_1).expect("abs angular distance");
    let dist_corr = distance_correlation(&x, &y_1).expect("distance correlation");

    assert!((angular_dist - 0.6703650607372927).abs() < 1e-6);
    assert!((abs_angular_dist - 0.6703650607372927).abs() < 1e-6);
    assert!((sq_angular_dist - 0.7034750294490113).abs() < 1e-6);
    assert!((dist_corr - 0.529291364408913).abs() < 1e-6);

    let dist_corr_y_2 = distance_correlation(&x, &y_2).expect("distance correlation y2");
    assert!((dist_corr_y_2 - 0.5216239463593741).abs() < 1e-6);
}

#[test]
fn test_information_metrics() {
    let (x, y_1, _) = load_series();

    let mut_info = get_mutual_info(&x, &y_1, None, false).expect("mutual info");
    let mut_info_norm = get_mutual_info(&x, &y_1, None, true).expect("mutual info norm");
    let mut_info_bins = get_mutual_info(&x, &y_1, Some(10), false).expect("mutual info bins");

    assert!((mut_info - 0.5228688725834145).abs() < 1e-6);
    assert!((mut_info_norm - 0.6409642333987833).abs() < 1e-6);
    assert!((mut_info_bins - 0.6264238716396385).abs() < 1e-6);

    let info_var = variation_of_information_score(&x, &y_1, None, false)
        .expect("information variation");
    let info_var_norm = variation_of_information_score(&x, &y_1, None, true)
        .expect("information variation norm");
    let info_var_bins = variation_of_information_score(&x, &y_1, Some(10), false)
        .expect("information variation bins");

    assert!((info_var - 1.425767548566149).abs() < 1e-6);
    assert!((info_var_norm - 0.7316744843171117).abs() < 1e-6);
    assert!((info_var_bins - 1.4184909443978817).abs() < 1e-6);
}

#[test]
fn test_number_of_bins() {
    let (x, y_1, _) = load_series();
    let n_bins_x = get_optimal_number_of_bins(x.len(), None).expect("n bins x");
    let corr = corrcoef(&x, &y_1);
    let n_bins_x_y = get_optimal_number_of_bins(x.len(), Some(corr)).expect("n bins x y");

    assert_eq!(n_bins_x, 15);
    assert_eq!(n_bins_x_y, 9);
}
