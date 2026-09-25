//! Value tests for `hcaa`. Expected numbers are closed forms worked out in the comments from the
//! definition of each allocation metric; none of them was read off the library's output.
//!
//! The pre-existing `tests/hcaa.rs` only asserts "non-negative and sums to one"; see
//! `docs/test-sensitivity-audit.md`.
//!
//! What the library implements (and what these tests therefore pin): single-linkage clustering
//! on the correlation distance, leaf order by quasi-diagonalisation, then weight handed down the
//! dendrogram, each of the top `optimal_num_clusters - 1` merges splitting it between its two
//! children by the chosen metric, with inverse-variance weights inside a side when its risk is
//! measured and inside each cluster below the cut (equal weights for `equal_weighting`).

use csv::ReaderBuilder;
use nalgebra::DMatrix;
use openquant::hcaa::{HcaaError, HierarchicalClusteringAssetAllocation};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::Path;

fn names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("a{i}")).collect()
}

fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol, "{what}: weight[{i}] = {g}, expected {w}");
    }
}

/// Every split factor has `f64::EPSILON` added to its denominator. With denominators >= 0.05 in
/// these hand-sized cases that is <= 4.4e-15 relative per level and there are <= 2 levels, so
/// 1e-13 is a rounding-level tolerance.
const TOL: f64 = 1e-13;

fn two_asset_cov(cov01: f64) -> DMatrix<f64> {
    DMatrix::from_row_slice(2, 2, &[0.04, cov01, cov01, 0.01])
}

fn allocate_cov(
    cov: &DMatrix<f64>,
    metric: &str,
    expected_returns: Option<&[f64]>,
) -> HierarchicalClusteringAssetAllocation {
    let mut hcaa = HierarchicalClusteringAssetAllocation::new("mean");
    hcaa.allocate(
        &names(cov.nrows()),
        None,
        None,
        Some(cov),
        expected_returns,
        metric,
        0.05,
        None,
        None,
    )
    .unwrap();
    hcaa
}

/// Two assets, variances v = (0.04, 0.01). The only split is {0} | {1}.
///   minimum_variance:            alpha = 1 - v0/(v0+v1) = 1 - 0.04/0.05 = 0.2  -> (0.2, 0.8)
///   minimum_standard_deviation:  alpha = 1 - s0/(s0+s1) = 1 - 0.2/0.3  = 1/3  -> (1/3, 2/3)
///   equal_weighting:             alpha = 1/2                                   -> (0.5, 0.5)
/// None of these depends on the covariance term.
#[test]
fn two_assets_risk_metrics_closed_form() {
    for cov01 in [-0.01, 0.0, 0.015] {
        let cov = two_asset_cov(cov01);
        assert_close(&allocate_cov(&cov, "minimum_variance", None).weights, &[0.2, 0.8], TOL, "mv");
        assert_close(
            &allocate_cov(&cov, "minimum_standard_deviation", None).weights,
            &[1.0 / 3.0, 2.0 / 3.0],
            TOL,
            "msd",
        );
        assert_close(&allocate_cov(&cov, "equal_weighting", None).weights, &[0.5, 0.5], TOL, "ew");
    }
}

/// Sharpe allocation gives the left half the share SR_L / (SR_L + SR_R).
/// v = (0.04, 0.01) so s = (0.2, 0.1); mu = (0.06, 0.09) gives SR = (0.3, 0.9), hence
/// alpha = 0.3 / 1.2 = 0.25 and w = (0.25, 0.75).
#[test]
fn two_assets_sharpe_ratio_closed_form() {
    let cov = two_asset_cov(0.003);
    let hcaa = allocate_cov(&cov, "sharpe_ratio", Some(&[0.06, 0.09]));
    assert_close(&hcaa.weights, &[0.25, 0.75], TOL, "sharpe");
}

/// Expected shortfall at the 5% level of ten observations is the single worst return under every
/// common quantile convention (nearest-rank, lower, or linear interpolation all leave exactly one
/// observation at or below the 5% quantile). Worst returns: asset 0 -> -0.04, asset 1 -> -0.01.
///   alpha = 1 - ES_0 / (ES_0 + ES_1) = 1 - 0.04 / 0.05 = 0.2  ->  w = (0.2, 0.8)
#[test]
fn two_assets_expected_shortfall_closed_form() {
    let r0 = [0.010, -0.040, 0.020, 0.005, -0.010, 0.015, 0.000, 0.030, -0.020, 0.010];
    let r1 = [0.004, 0.002, -0.010, 0.001, 0.003, -0.002, 0.005, -0.004, 0.006, 0.000];
    let returns = DMatrix::from_fn(10, 2, |r, c| if c == 0 { r0[r] } else { r1[r] });
    let mut hcaa = HierarchicalClusteringAssetAllocation::new("mean");
    hcaa.allocate(
        &names(2),
        None,
        Some(&returns),
        None,
        None,
        "expected_shortfall",
        0.05,
        None,
        None,
    )
    .unwrap();
    assert_close(&hcaa.weights, &[0.2, 0.8], TOL, "expected shortfall");
}

/// Conditional drawdown at risk at the 5% level of five wealth observations is the maximum
/// drawdown (again under any quantile convention: only the maximum lies at or above the 95%
/// quantile of five points).
///   asset 0 returns ( 0.10, -0.20, 0.05, 0.10): wealth 1, 1.1, 0.88, 0.924, 1.0164
///       peak 1.1, trough 0.88  ->  max drawdown = 0.22 / 1.1 = 0.2
///   asset 1 returns ( 0.02, -0.05, 0.01, 0.03): wealth 1, 1.02, 0.969, 0.97869, 1.0080507
///       peak 1.02, trough 0.969 -> max drawdown = 0.051 / 1.02 = 0.05
///   alpha = 1 - 0.2 / 0.25 = 0.2  ->  w = (0.2, 0.8)
#[test]
fn two_assets_conditional_drawdown_closed_form() {
    let r0 = [0.10, -0.20, 0.05, 0.10];
    let r1 = [0.02, -0.05, 0.01, 0.03];
    let returns = DMatrix::from_fn(4, 2, |r, c| if c == 0 { r0[r] } else { r1[r] });
    let mut hcaa = HierarchicalClusteringAssetAllocation::new("mean");
    hcaa.allocate(
        &names(2),
        None,
        Some(&returns),
        None,
        None,
        "conditional_drawdown_risk",
        0.05,
        None,
        None,
    )
    .unwrap();
    // wealth is a product of four (1 + r) factors: a few ulps on top of the EPSILON term
    assert_close(&hcaa.weights, &[0.2, 0.8], 1e-12, "conditional drawdown");
}

/// Four assets in two tight pairs: corr(0,1) = 0.9, corr(2,3) = 0.8, every cross pair 0.1.
/// Single linkage merges (0,1), then (2,3), then the two pairs, so the leaf order is 0,1,2,3 and
/// the first bisection is {0,1} | {2,3}.
///
/// With s = (0.2, 0.3, 0.1, 0.4):
///   inside {0,1}: inverse-variance weights u = (1/0.04, 1/0.09) / (1/0.04 + 1/0.09)
///                 cluster variance V_L = u' S u
///   inside {2,3}: likewise, V_R
///   top split:    alpha = 1 - V_L / (V_L + V_R)
///   second level: {0}|{1} gives 1 - 0.04/0.13 to asset 0; {2}|{3} gives 1 - 0.01/0.17 to asset 2
/// The expected weights are assembled below from exactly those formulas.
#[test]
fn four_assets_two_clusters_minimum_variance_hand_worked() {
    let s = [0.2, 0.3, 0.1, 0.4];
    let corr = |i: usize, j: usize| match (i.min(j), i.max(j)) {
        (a, b) if a == b => 1.0,
        (0, 1) => 0.9,
        (2, 3) => 0.8,
        _ => 0.1,
    };
    let cov = DMatrix::from_fn(4, 4, |i, j| corr(i, j) * s[i] * s[j]);

    let pair_var = |a: usize, b: usize| {
        let (ia, ib) = (1.0 / cov[(a, a)], 1.0 / cov[(b, b)]);
        let (ua, ub) = (ia / (ia + ib), ib / (ia + ib));
        ua * ua * cov[(a, a)] + 2.0 * ua * ub * cov[(a, b)] + ub * ub * cov[(b, b)]
    };
    let (v_l, v_r): (f64, f64) = (pair_var(0, 1), pair_var(2, 3));
    let alpha = 1.0 - v_l / (v_l + v_r);
    let a01 = 1.0 - 0.04 / (0.04 + 0.09);
    let a23 = 1.0 - 0.01 / (0.01 + 0.16);
    let want = [alpha * a01, alpha * (1.0 - a01), (1.0 - alpha) * a23, (1.0 - alpha) * (1.0 - a23)];
    // sanity of the hand calculation itself: V_L = 0.0506982..., V_R = 0.0129550...
    //   {0,1}: u = (9/13, 4/13):  (81*0.04 + 72*0.054 + 16*0.09) / 169 = 8.568 / 169
    //   {2,3}: u = (16/17, 1/17): (256*0.01 + 32*0.032 + 1*0.16) / 289 = 3.744 / 289
    assert!((v_l - 8.568 / 169.0).abs() < 1e-15, "{v_l}");
    assert!((v_r - 3.744 / 289.0).abs() < 1e-15, "{v_r}");

    let hcaa = allocate_cov(&cov, "minimum_variance", None);
    assert_eq!(hcaa.ordered_indices, vec![0, 1, 2, 3]);
    assert_close(&hcaa.weights, &want, TOL, "four assets");
}

/// Same diagonal-covariance argument as for HRP: with no correlation, minimum-variance bisection
/// telescopes to the inverse-variance portfolio w_i = (1/v_i) / sum_j 1/v_j, and
/// minimum-standard-deviation bisection does NOT (so the two metrics must differ).
#[test]
fn uncorrelated_assets_minimum_variance_is_inverse_variance() {
    let mut rng = StdRng::seed_from_u64(41);
    for n in [2usize, 3, 6, 11] {
        let vars: Vec<f64> = (0..n).map(|_| rng.gen_range(0.01..4.0)).collect();
        let cov = DMatrix::from_fn(n, n, |i, j| if i == j { vars[i] } else { 0.0 });
        let inv_sum: f64 = vars.iter().map(|v| 1.0 / v).sum();
        let want: Vec<f64> = vars.iter().map(|v| (1.0 / v) / inv_sum).collect();
        let got = allocate_cov(&cov, "minimum_variance", None).weights;
        // <= 4 levels, each off by EPSILON / (lv + rv) <= 2.2e-16 / 0.0025 ~ 1e-13
        assert_close(&got, &want, 1e-12, "diagonal covariance");

        if n > 2 {
            let msd = allocate_cov(&cov, "minimum_standard_deviation", None).weights;
            let gap = msd.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(gap > 1e-3, "min-std weights coincide with inverse-variance (gap {gap})");
        }
    }
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

/// Asking for weekly resampling must equal handing over rows 4, 9, 14, ... directly.
#[test]
fn weekly_resampling_equals_allocating_on_every_fifth_row() {
    let (prices, names) = load_prices_and_names();
    let kept: Vec<usize> = (4..prices.nrows()).step_by(5).collect();
    let weekly = DMatrix::from_fn(kept.len(), prices.ncols(), |r, c| prices[(kept[r], c)]);

    let run = |p: &DMatrix<f64>, resample: Option<&str>| {
        let mut hcaa = HierarchicalClusteringAssetAllocation::new("mean");
        hcaa.allocate(&names, Some(p), None, None, None, "minimum_variance", 0.05, None, resample)
            .unwrap();
        hcaa
    };
    let resampled = run(&prices, Some("W"));
    let direct = run(&weekly, None);
    assert_eq!(resampled.ordered_indices, direct.ordered_indices);
    assert_eq!(resampled.weights, direct.weights);
}

fn allocate_cov_k(
    cov: &DMatrix<f64>,
    metric: &str,
    optimal_num_clusters: Option<usize>,
) -> Result<HierarchicalClusteringAssetAllocation, HcaaError> {
    let mut hcaa = HierarchicalClusteringAssetAllocation::new("mean");
    hcaa.allocate(
        &names(cov.nrows()),
        None,
        None,
        Some(cov),
        None,
        metric,
        0.05,
        optimal_num_clusters,
        None,
    )?;
    Ok(hcaa)
}

/// Five assets: {0,1,2} (corr(0,1) = 0.9, corr(0,2) = corr(1,2) = 0.7) and {3,4} (corr 0.5), no
/// correlation across. Single linkage merges (0,1), then 2 into it, then (3,4), then the two
/// groups, so the leaf order is 2,0,1,3,4 and the tree's top split is {2,0,1} | {3,4}. The
/// midpoint of the leaf list is {2,0} | {1,3,4} instead.
///   along the tree:   {2,0,1} 1/2 -> 2: 1/4, {0,1}: 1/8 each;  {3,4} 1/2 -> 1/4 each
///   at the midpoint:  {2,0} 1/2 -> 1/4 each;  {1,3,4} 1/2 -> 1: 1/4, {3,4}: 1/8 each  (#108)
#[test]
fn bisection_follows_the_tree_not_the_midpoint_of_the_leaf_list() {
    let corr = |i: usize, j: usize| match (i.min(j), i.max(j)) {
        (a, b) if a == b => 1.0,
        (0, 1) => 0.9,
        (0, 2) | (1, 2) => 0.7,
        (3, 4) => 0.5,
        _ => 0.0,
    };
    let cov = DMatrix::from_fn(5, 5, |i, j| 0.04 * corr(i, j));
    let hcaa = allocate_cov_k(&cov, "equal_weighting", None).unwrap();
    assert_eq!(hcaa.ordered_indices, vec![2, 0, 1, 3, 4]);
    assert_close(&hcaa.weights, &[0.125, 0.125, 0.25, 0.25, 0.25], TOL, "tree bisection");
}

/// The four assets of `four_assets_two_clusters_minimum_variance_hand_worked`, with the
/// minimum-standard-deviation metric. Cut into two clusters, {0,1} and {2,3}, only the top split
/// uses the metric; inside each cluster the weights are inverse-variance:
///   top split:  alpha = 1 - sqrt(V_L) / (sqrt(V_L) + sqrt(V_R))
///   {0,1}:      u = (9/13, 4/13)     {2,3}: u = (16/17, 1/17)
/// With no cut, the second level splits {0}|{1} by standard deviation instead:
///   1 - 0.2 / (0.2 + 0.3) = 0.6 to asset 0, not 9/13. So the cluster count changes the weights.
#[test]
fn optimal_num_clusters_cuts_the_tree() {
    let s = [0.2, 0.3, 0.1, 0.4];
    let corr = |i: usize, j: usize| match (i.min(j), i.max(j)) {
        (a, b) if a == b => 1.0,
        (0, 1) => 0.9,
        (2, 3) => 0.8,
        _ => 0.1,
    };
    let cov = DMatrix::from_fn(4, 4, |i, j| corr(i, j) * s[i] * s[j]);
    let (sd_l, sd_r) = ((8.568_f64 / 169.0).sqrt(), (3.744_f64 / 289.0).sqrt());
    let alpha = 1.0 - sd_l / (sd_l + sd_r);

    let two = allocate_cov_k(&cov, "minimum_standard_deviation", Some(2)).unwrap();
    let want_two = [
        alpha * 9.0 / 13.0,
        alpha * 4.0 / 13.0,
        (1.0 - alpha) * 16.0 / 17.0,
        (1.0 - alpha) * 1.0 / 17.0,
    ];
    assert_close(&two.weights, &want_two, TOL, "two clusters");

    let (a01, a23) = (1.0 - 0.2 / 0.5, 1.0 - 0.1 / 0.5);
    let want_four =
        [alpha * a01, alpha * (1.0 - a01), (1.0 - alpha) * a23, (1.0 - alpha) * (1.0 - a23)];
    assert_close(
        &allocate_cov_k(&cov, "minimum_standard_deviation", Some(4)).unwrap().weights,
        &want_four,
        TOL,
        "four clusters",
    );
    assert_close(
        &allocate_cov_k(&cov, "minimum_standard_deviation", None).unwrap().weights,
        &want_four,
        TOL,
        "no cut",
    );

    // One cluster: no split at all, so equal weighting is 1/4 each.
    let one = allocate_cov_k(&cov, "equal_weighting", Some(1)).unwrap();
    assert_close(&one.weights, &[0.25; 4], TOL, "one cluster");
}

#[test]
fn optimal_num_clusters_must_be_between_one_and_the_number_of_assets() {
    let cov = DMatrix::from_row_slice(2, 2, &[0.04, 0.0, 0.0, 0.01]);
    for k in [0, 3] {
        let err = allocate_cov_k(&cov, "equal_weighting", Some(k)).unwrap_err();
        assert_eq!(err, HcaaError::InvalidNumClusters { requested: k, assets: 2 });
    }
}
