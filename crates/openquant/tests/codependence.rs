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

/// Values computed from the published definitions by
/// tests/fixtures/codependence/generate.py (MLAM ch. 3 snippets 3.1-3.3; Szekely, Rizzo &
/// Bakirov 2007), independently of this crate.
#[derive(Debug, Deserialize)]
struct Reference {
    angular_distance_x_y1: f64,
    absolute_angular_distance_x_y1: f64,
    squared_angular_distance_x_y1: f64,
    distance_correlation_x_y1: f64,
    distance_correlation_x_y2: f64,
    optimal_bins_univariate: usize,
    optimal_bins_bivariate_x_y1: usize,
    mutual_info_x_y1: f64,
    mutual_info_normalised_x_y1: f64,
    mutual_info_x_y1_10_bins: f64,
    variation_of_information_x_y1: f64,
    variation_of_information_normalised_x_y1: f64,
    variation_of_information_x_y1_10_bins: f64,
}

fn load_reference() -> Reference {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/codependence/reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("reference json"))
        .expect("valid reference json")
}

/// The reference performs the same operations up to summation order (and the same histogram
/// bin assignment), so only rounding differences remain.
const TOL: f64 = 1e-12;

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() < TOL, "{what}: got {actual}, expected {expected}");
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
    let r = load_reference();

    assert_close(
        angular_distance(&x, &y_1).expect("angular distance"),
        r.angular_distance_x_y1,
        "angular distance",
    );
    assert_close(
        absolute_angular_distance(&x, &y_1).expect("abs angular distance"),
        r.absolute_angular_distance_x_y1,
        "absolute angular distance",
    );
    assert_close(
        squared_angular_distance(&x, &y_1).expect("sq angular distance"),
        r.squared_angular_distance_x_y1,
        "squared angular distance",
    );
    assert_close(
        distance_correlation(&x, &y_1).expect("distance correlation"),
        r.distance_correlation_x_y1,
        "distance correlation x-y1",
    );
    assert_close(
        distance_correlation(&x, &y_2).expect("distance correlation y2"),
        r.distance_correlation_x_y2,
        "distance correlation x-y2",
    );
}

#[test]
fn test_information_metrics() {
    let (x, y_1, _) = load_series();
    let r = load_reference();

    assert_close(
        get_mutual_info(&x, &y_1, None, false).expect("mutual info"),
        r.mutual_info_x_y1,
        "mutual info",
    );
    assert_close(
        get_mutual_info(&x, &y_1, None, true).expect("mutual info norm"),
        r.mutual_info_normalised_x_y1,
        "normalised mutual info",
    );
    assert_close(
        get_mutual_info(&x, &y_1, Some(10), false).expect("mutual info bins"),
        r.mutual_info_x_y1_10_bins,
        "mutual info, 10 bins",
    );

    assert_close(
        variation_of_information_score(&x, &y_1, None, false).expect("information variation"),
        r.variation_of_information_x_y1,
        "variation of information",
    );
    assert_close(
        variation_of_information_score(&x, &y_1, None, true).expect("information variation norm"),
        r.variation_of_information_normalised_x_y1,
        "normalised variation of information",
    );
    assert_close(
        variation_of_information_score(&x, &y_1, Some(10), false)
            .expect("information variation bins"),
        r.variation_of_information_x_y1_10_bins,
        "variation of information, 10 bins",
    );
}

#[test]
fn test_number_of_bins() {
    let (x, y_1, _) = load_series();
    let r = load_reference();
    let n_bins_x = get_optimal_number_of_bins(x.len(), None).expect("n bins x");
    let corr = corrcoef(&x, &y_1);
    let n_bins_x_y = get_optimal_number_of_bins(x.len(), Some(corr)).expect("n bins x y");

    assert_eq!(n_bins_x, r.optimal_bins_univariate);
    assert_eq!(n_bins_x_y, r.optimal_bins_bivariate_x_y1);
}

/// A correlation of -1 used to reach the bivariate bin formula, divide by zero, and ask for
/// isize::MAX bins, which panicked with "capacity overflow" inside the histogram.
#[test]
fn perfectly_anticorrelated_series_do_not_panic() {
    use openquant::codependence::{
        get_mutual_info, get_optimal_number_of_bins, variation_of_information_score,
    };

    assert_eq!(
        get_optimal_number_of_bins(1000, Some(-1.0)).unwrap(),
        get_optimal_number_of_bins(1000, Some(1.0)).unwrap()
    );

    let x: Vec<f64> = (0..50).map(f64::from).collect();
    let mirrored: Vec<f64> = x.iter().map(|v| -v).collect();
    // A deterministic one-to-one relationship is maximal dependence either way round.
    assert!((get_mutual_info(&x, &mirrored, None, true).unwrap() - 1.0).abs() < 1e-12);
    assert!(variation_of_information_score(&x, &mirrored, None, true).unwrap().abs() < 1e-12);
}
