//! Value tests for HRP. Every expected number here comes from outside this library: either a
//! closed form derived in the comment, or `tests/fixtures/hrp/reference.json`, which is written
//! by `tests/fixtures/hrp/generate.py` (a numpy/scipy transcription of AFML snippets 16.1-16.4).
//!
//! The pre-existing `tests/hrp.rs` only asserts "non-negative and sums to one"; see
//! `docs/test-sensitivity-audit.md`.

use csv::ReaderBuilder;
use nalgebra::DMatrix;
use openquant::hrp::{HierarchicalRiskParity, HrpDistance, HrpError};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Absolute tolerance on a weight in [0, 1] when comparing against the numpy reference.
///
/// Two sources of disagreement exist, both rounding-sized:
/// * the library adds `f64::EPSILON` (2.2e-16) to `lv + rv` in every bisection; cluster variances
///   of daily returns are ~1e-5, so that perturbs each split factor by ~2e-11 relative, and a
///   weight is a product of at most ceil(log2(23)) = 5 such factors: ~1e-10;
/// * numpy sums the 2140-row covariance pairwise, the library sums it sequentially: ~1e-13.
///
/// 1e-9 is one decade above that bound and eight decades below the ~1e-1 scale of a weight.
const REF_TOL: f64 = 1e-9;

/// One tree and its weights. The top level of a case is the `Correlation` tree; the
/// `distance_of_distances` object is the book's Snippet 16.4 tree.
#[derive(Deserialize, Clone)]
struct Variant {
    link: Vec<[usize; 2]>,
    order: Vec<usize>,
    weights: Vec<f64>,
}

#[derive(Deserialize)]
struct Case {
    link: Vec<[usize; 2]>,
    order: Vec<usize>,
    weights: Vec<f64>,
    #[serde(default)]
    cov: Vec<Vec<f64>>,
    distance_of_distances: Variant,
}

impl Case {
    fn variant(&self, distance: HrpDistance) -> Variant {
        match distance {
            HrpDistance::Correlation => Variant {
                link: self.link.clone(),
                order: self.order.clone(),
                weights: self.weights.clone(),
            },
            HrpDistance::DistanceOfDistances => self.distance_of_distances.clone(),
        }
    }
}

const BOTH: [HrpDistance; 2] = [HrpDistance::Correlation, HrpDistance::DistanceOfDistances];

/// Tree (merge list), leaf order and weights all match the reference for `hrp.distance`.
fn assert_matches(hrp: &HierarchicalRiskParity, case: &Case, what: &str) {
    let want = case.variant(hrp.distance);
    assert_eq!(hrp.clusters, want.link, "{what} {:?}: tree", hrp.distance);
    assert_eq!(hrp.ordered_indices, want.order, "{what} {:?}: leaf order", hrp.distance);
    assert_close(&hrp.weights, &want.weights, REF_TOL, what);
}

fn reference() -> HashMap<String, Case> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/hrp/reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn load_prices_and_names() -> (DMatrix<f64>, Vec<String>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/portfolio_optimization/stock_prices.csv");
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path).unwrap();
    let names: Vec<String> = rdr.headers().unwrap().iter().skip(1).map(str::to_string).collect();
    let mut flat = Vec::new();
    let mut nrows = 0;
    for rec in rdr.records() {
        flat.extend(rec.unwrap().iter().skip(1).map(|x| x.parse::<f64>().unwrap()));
        nrows += 1;
    }
    (DMatrix::from_row_slice(nrows, names.len(), &flat), names)
}

fn names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("a{i}")).collect()
}

fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol, "{what}: weight[{i}] = {g}, reference {w}");
    }
}

#[test]
fn weights_match_independent_reference_on_price_fixture() {
    let (prices, names) = load_prices_and_names();
    let case = &reference()["stock_prices"];
    for distance in BOTH {
        let mut hrp = HierarchicalRiskParity::with_distance(distance);
        hrp.allocate(&names, Some(&prices), None, None, None, false).unwrap();
        assert_matches(&hrp, case, "stock_prices");
    }
}

#[test]
fn weights_match_independent_reference_on_random_covariance() {
    let case = &reference()["random_cov_8"];
    let n = case.cov.len();
    let cov = DMatrix::from_fn(n, n, |i, j| case.cov[i][j]);
    for distance in BOTH {
        let mut hrp = HierarchicalRiskParity::with_distance(distance);
        hrp.allocate(&names(n), None, None, Some(&cov), None, false).unwrap();
        assert_matches(&hrp, case, "random_cov_8");
    }
}

/// `use_shrinkage = true` multiplies every off-diagonal covariance by 0.9. The reference applies
/// that rule in numpy and then runs the independent HRP. (mlfinlab uses OAS shrinkage instead;
/// that difference is a documented design choice of this library, not what this test is about.)
#[test]
fn shrunk_weights_match_independent_reference() {
    let (prices, names) = load_prices_and_names();
    let reference = reference();
    let case = &reference["stock_prices_shrunk"];
    for distance in BOTH {
        let mut hrp = HierarchicalRiskParity::with_distance(distance);
        hrp.allocate(&names, Some(&prices), None, None, None, true).unwrap();
        assert_matches(&hrp, case, "stock_prices_shrunk");

        // and shrinkage must actually change the answer, by far more than the tolerance
        let plain = reference["stock_prices"].variant(distance).weights;
        let moved = hrp.weights.iter().zip(&plain).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(moved > 1e-4, "{distance:?}: shrinkage moved the weights by only {moved}");
    }
}

/// Weekly resampling keeps rows 4, 9, 14, ... of the price matrix. The reference does exactly
/// that in numpy (`prices[4::5]`) before running the independent HRP.
#[test]
fn weekly_resampled_weights_match_independent_reference() {
    let (prices, names) = load_prices_and_names();
    let case = &reference()["stock_prices_weekly"];
    for distance in BOTH {
        let mut hrp = HierarchicalRiskParity::with_distance(distance);
        hrp.allocate(&names, Some(&prices), None, None, Some("W"), false).unwrap();
        assert_matches(&hrp, case, "stock_prices_weekly");
    }
}

/// Same claim without any reference file: asking the library to resample weekly must equal
/// handing it the weekly rows directly.
#[test]
fn weekly_resampling_equals_allocating_on_every_fifth_row() {
    let (prices, names) = load_prices_and_names();
    let kept: Vec<usize> = (4..prices.nrows()).step_by(5).collect();
    let weekly = DMatrix::from_fn(kept.len(), prices.ncols(), |r, c| prices[(kept[r], c)]);

    let mut resampled = HierarchicalRiskParity::new();
    resampled.allocate(&names, Some(&prices), None, None, Some("W"), false).unwrap();
    let mut direct = HierarchicalRiskParity::new();
    direct.allocate(&names, Some(&weekly), None, None, None, false).unwrap();

    assert_eq!(resampled.ordered_indices, direct.ordered_indices);
    // identical arithmetic on identical inputs: bitwise equality is the right expectation
    assert_eq!(resampled.weights, direct.weights);
}

/// Two assets, any correlation. The only bisection is {0} | {1}, a single-asset cluster has
/// variance v_i, so alpha = 1 - v0 / (v0 + v1) and
///     w = ( v1 / (v0 + v1),  v0 / (v0 + v1) ).
/// With v0 = 0.04, v1 = 0.01: w = (0.2, 0.8). The covariance term does not enter.
#[test]
fn two_assets_closed_form() {
    for cov01 in [-0.015, 0.0, 0.004, 0.019] {
        let cov = DMatrix::from_row_slice(2, 2, &[0.04, cov01, cov01, 0.01]);
        let mut hrp = HierarchicalRiskParity::new();
        hrp.allocate(&names(2), None, None, Some(&cov), None, false).unwrap();
        // EPSILON in the denominator perturbs alpha by 2.2e-16 / 0.05 = 4.4e-15 relative.
        assert_close(&hrp.weights, &[0.2, 0.8], 1e-13, "two assets");
    }
}

/// Uncorrelated assets. For a diagonal covariance the inverse-variance portfolio of a cluster C
/// has variance 1 / S_C with S_C = sum_{i in C} 1/v_i, so a bisection L | R gives
///     alpha = 1 - (1/S_L) / (1/S_L + 1/S_R) = S_L / (S_L + S_R),
/// and multiplying down the tree telescopes to w_i = (1/v_i) / sum_j (1/v_j): HRP equals the
/// inverse-variance portfolio, whatever leaf order the (fully tied) clustering produces.
#[test]
fn uncorrelated_assets_get_inverse_variance_weights() {
    let mut rng = StdRng::seed_from_u64(40);
    for n in [2usize, 3, 5, 8, 13] {
        let vars: Vec<f64> = (0..n).map(|_| rng.gen_range(0.01..4.0)).collect();
        let cov = DMatrix::from_fn(n, n, |i, j| if i == j { vars[i] } else { 0.0 });
        let inv_sum: f64 = vars.iter().map(|v| 1.0 / v).sum();
        let want: Vec<f64> = vars.iter().map(|v| (1.0 / v) / inv_sum).collect();

        let mut hrp = HierarchicalRiskParity::new();
        hrp.allocate(&names(n), None, None, Some(&cov), None, false).unwrap();
        // <= 4 bisection levels, each off by EPSILON / (lv + rv) <= 2.2e-16 / 0.0025 ~ 1e-13.
        assert_close(&hrp.weights, &want, 1e-12, "diagonal covariance");
    }
}

/// HRP is invariant to rescaling the covariance (units of return): correlations, hence the tree,
/// are unchanged and every alpha is a ratio of variances.
#[test]
fn weights_are_invariant_to_covariance_scale() {
    let case = &reference()["random_cov_8"];
    let n = case.cov.len();
    for (distance, scale) in BOTH.into_iter().flat_map(|d| [(d, 1e-4), (d, 7.0), (d, 2.5e3)]) {
        let want = case.variant(distance);
        let cov = DMatrix::from_fn(n, n, |i, j| case.cov[i][j] * scale);
        let mut hrp = HierarchicalRiskParity::with_distance(distance);
        hrp.allocate(&names(n), None, None, Some(&cov), None, false).unwrap();
        assert_eq!(hrp.ordered_indices, want.order);
        // The EPSILON term is not scale-free: at scale 1e-4 variances are ~1e-4, so it costs
        // 2.2e-16 / 1e-4 ~ 2e-12 per level over 3 levels.
        assert_close(&hrp.weights, &want.weights, 1e-10, "scaled covariance");
    }
}

/// Prices, returns and covariance inputs describe the same problem and must give the same
/// weights (the library uses simple returns and the ddof=1 sample covariance throughout).
#[test]
fn prices_returns_and_covariance_inputs_agree() {
    let (prices, names) = load_prices_and_names();
    let returns = DMatrix::from_fn(prices.nrows() - 1, prices.ncols(), |r, c| {
        prices[(r + 1, c)] / prices[(r, c)] - 1.0
    });
    let mut from_prices = HierarchicalRiskParity::new();
    from_prices.allocate(&names, Some(&prices), None, None, None, false).unwrap();
    let mut from_returns = HierarchicalRiskParity::new();
    from_returns.allocate(&names, None, Some(&returns), None, None, false).unwrap();
    assert_eq!(from_prices.weights, from_returns.weights);
    let want = reference()["stock_prices"].variant(HrpDistance::default()).weights;
    assert_close(&from_returns.weights, &want, REF_TOL, "returns");
}

/// The two distances build different trees on every fixture case (the reference says so, and
/// the library agrees), so the tests above pin two distinct code paths.
#[test]
fn the_two_distances_build_different_trees_on_the_fixtures() {
    for (name, case) in &reference() {
        assert_ne!(case.link, case.distance_of_distances.link, "{name}: reference trees agree");
    }
    let (prices, names) = load_prices_and_names();
    let mut pairwise = HierarchicalRiskParity::with_distance(HrpDistance::Correlation);
    pairwise.allocate(&names, Some(&prices), None, None, None, false).unwrap();
    let mut book = HierarchicalRiskParity::with_distance(HrpDistance::DistanceOfDistances);
    book.allocate(&names, Some(&prices), None, None, None, false).unwrap();
    assert_ne!(pairwise.clusters, book.clusters);
}

/// `new()` clusters on the default distance, and the names parse case-insensitively.
#[test]
fn default_distance_and_parsing() {
    assert_eq!(HierarchicalRiskParity::new().distance, HrpDistance::default());
    assert_eq!("correlation".parse(), Ok(HrpDistance::Correlation));
    assert_eq!("Distance_Of_Distances".parse(), Ok(HrpDistance::DistanceOfDistances));
    assert_eq!(
        "euclidean".parse::<HrpDistance>(),
        Err(HrpError::UnknownDistance("euclidean".to_string()))
    );
}

/// Two assets: d~_01 = sqrt(d_01^2 + d_01^2) = sqrt(2) d_01, so both distances give the one
/// possible tree and the same weights.
#[test]
fn two_assets_do_not_depend_on_the_distance() {
    let cov = DMatrix::from_row_slice(2, 2, &[0.04, 0.012, 0.012, 0.01]);
    let mut a = HierarchicalRiskParity::with_distance(HrpDistance::Correlation);
    a.allocate(&names(2), None, None, Some(&cov), None, false).unwrap();
    let mut b = HierarchicalRiskParity::with_distance(HrpDistance::DistanceOfDistances);
    b.allocate(&names(2), None, None, Some(&cov), None, false).unwrap();
    assert_eq!(a.clusters, b.clusters);
    assert_eq!(a.weights, b.weights);
}
