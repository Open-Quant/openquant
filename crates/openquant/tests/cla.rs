use chrono::NaiveDate;
use csv::ReaderBuilder;
use nalgebra::DMatrix;
use openquant::cla::{
    covariance, AssetPrices, AssetPricesInput, ClaError, ReturnsEstimation, WeightBounds, CLA,
};
use std::path::Path;

fn load_asset_prices() -> AssetPrices {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/portfolio_optimization/stock_prices.csv");
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path).unwrap();
    let mut index: Vec<NaiveDate> = Vec::new();
    let mut data: Vec<Vec<f64>> = Vec::new();
    for result in rdr.records() {
        let rec = result.unwrap();
        let date = NaiveDate::parse_from_str(&rec[0], "%Y-%m-%d").unwrap();
        index.push(date);
        let mut row = Vec::new();
        for field in rec.iter().skip(1) {
            row.push(field.parse::<f64>().unwrap());
        }
        data.push(row);
    }
    let rows = data.len();
    let cols = data[0].len();
    let flat: Vec<f64> = data.into_iter().flat_map(|r| r.into_iter()).collect();
    let matrix = DMatrix::from_row_slice(rows, cols, &flat);
    AssetPrices::new(matrix, index)
}

fn assert_weights_basic(weights: &[f64], expect_nonnegative: bool) {
    let sum: f64 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    if expect_nonnegative {
        assert!(weights.iter().all(|w| *w >= 0.0));
    }
}

#[test]
fn test_cla_with_mean_returns() {
    let mut prices = load_asset_prices();
    let cols = prices.data.ncols();
    for r in 1..10 {
        for c in 0..cols {
            prices.data[(r, c)] = 40.0;
        }
    }
    for r in 11..20 {
        for c in 0..cols {
            prices.data[(r, c)] = 50.0;
        }
    }
    for r in 21..prices.data.nrows() {
        for c in 0..cols {
            prices.data[(r, c)] = 100.0;
        }
    }

    let mut cla = CLA::new(WeightBounds::Tuple(0.0, 1.0), "mean");
    cla.allocate(Some(AssetPricesInput::Prices(&prices)), None, None, None, None).unwrap();
    for turning_point in cla.weights.iter() {
        let cleaned: Vec<f64> =
            turning_point.iter().map(|w| if *w <= 1e-15 { 0.0 } else { *w }).collect();
        assert_eq!(cleaned.len(), cols);
        assert_weights_basic(&cleaned, true);
    }
}

#[test]
fn test_cla_with_weight_bounds_as_lists() {
    let prices = load_asset_prices();
    let n = prices.data.ncols();
    let mut cla = CLA::new(WeightBounds::Lists(vec![0.0; n], vec![1.0; n]), "mean");
    cla.allocate(Some(AssetPricesInput::Prices(&prices)), None, None, None, None).unwrap();
    for turning_point in cla.weights.iter() {
        let cleaned: Vec<f64> =
            turning_point.iter().map(|w| if *w <= 1e-15 { 0.0 } else { *w }).collect();
        assert_eq!(cleaned.len(), n);
        assert_weights_basic(&cleaned, true);
    }
}

#[test]
fn test_cla_with_exponential_returns() {
    let prices = load_asset_prices();
    let mut cla = CLA::new(WeightBounds::Tuple(0.0, 1.0), "exponential");
    cla.allocate(Some(AssetPricesInput::Prices(&prices)), None, None, None, None).unwrap();
    for turning_point in cla.weights.iter() {
        let cleaned: Vec<f64> =
            turning_point.iter().map(|w| if *w <= 1e-15 { 0.0 } else { *w }).collect();
        assert_eq!(cleaned.len(), prices.data.ncols());
        assert_weights_basic(&cleaned, true);
    }
}

#[test]
fn test_cla_max_sharpe() {
    let prices = load_asset_prices();
    let mut cla = CLA::new(WeightBounds::Tuple(0.0, 1.0), "mean");
    cla.allocate(Some(AssetPricesInput::Prices(&prices)), None, None, None, Some("max_sharpe"))
        .unwrap();
    let weights = &cla.weights[0];
    assert!(weights.iter().all(|w| *w >= -1e-12));
    assert_eq!(weights.len(), prices.data.ncols());
    assert_weights_basic(weights, false);
}

#[test]
fn test_cla_min_volatility() {
    let prices = load_asset_prices();
    let mut cla = CLA::new(WeightBounds::Tuple(0.0, 1.0), "mean");
    cla.allocate(Some(AssetPricesInput::Prices(&prices)), None, None, None, Some("min_volatility"))
        .unwrap();
    let weights = &cla.weights[0];
    assert_eq!(weights.len(), prices.data.ncols());
    assert_weights_basic(weights, true);
}

#[test]
fn test_cla_efficient_frontier() {
    let prices = load_asset_prices();
    let mut cla = CLA::new(WeightBounds::Tuple(0.0, 1.0), "mean");
    cla.allocate(
        Some(AssetPricesInput::Prices(&prices)),
        None,
        None,
        None,
        Some("efficient_frontier"),
    )
    .unwrap();
    assert_eq!(cla.efficient_frontier_means.len(), cla.efficient_frontier_sigma.len());
    assert_eq!(cla.efficient_frontier_sigma.len(), cla.weights.len());
    assert!(cla.efficient_frontier_sigma.last().unwrap() <= &cla.efficient_frontier_sigma[0]);
    assert!(cla.efficient_frontier_means.last().unwrap() <= &cla.efficient_frontier_means[0]);
}

#[test]
fn test_expected_returns_equals_means() {
    let mut prices = load_asset_prices();
    let cols = prices.data.ncols();
    for r in 0..prices.data.nrows() {
        for c in 0..cols {
            prices.data[(r, c)] = 0.023_206_53;
        }
    }
    let mut cla = CLA::new(WeightBounds::Tuple(0.0, 1.0), "mean");
    cla._initialise(&prices.data, Some("B"), None, None).unwrap();
    let last = cla.expected_returns[(cla.expected_returns.nrows() - 1, 0)];
    assert!((last - 1e-5).abs() < 1e-12);
}

#[test]
fn test_purge_excess() {
    let prices = load_asset_prices();
    let mut cla = CLA::new(WeightBounds::Tuple(0.0, 1.0), "mean");
    cla.allocate(
        Some(AssetPricesInput::Prices(&prices)),
        None,
        None,
        None,
        Some("cla_turning_points"),
    )
    .unwrap();
    let mut repeated = Vec::new();
    for _ in 0..100 {
        repeated.extend(cla.weights.clone());
    }
    cla.weights = repeated;
    let err = cla._purge_num_err(1e-18).unwrap_err();
    assert_eq!(err, ClaError::IndexError);
}

#[test]
fn test_flag_true_for_purge_num_err() {
    let mut cla = CLA {
        weights: vec![vec![1.0]],
        lower_bounds: vec![100.0],
        upper_bounds: vec![1.0],
        lambdas: vec![0.0],
        gammas: vec![0.0],
        free_weights: vec![vec![]],
        ..CLA::default()
    };
    cla._purge_num_err(1.0).unwrap();
    assert!(cla.weights.is_empty());
    assert!(cla.lambdas.is_empty());
    assert!(cla.gammas.is_empty());
}

#[test]
fn test_value_error_for_unknown_solution() {
    let prices = load_asset_prices();
    let mut cla = CLA::default();
    let err = cla.allocate(
        Some(AssetPricesInput::Prices(&prices)),
        None,
        None,
        None,
        Some("unknown_string"),
    );
    assert!(matches!(err, Err(ClaError::UnknownSolution(_))));
}

#[test]
fn test_a_bare_price_matrix_gives_the_same_answer_as_dated_prices() {
    // mlfinlab rejects anything that is not a DataFrame. That check means nothing here, and
    // rejecting a matrix made the price path unreachable from Python, so a matrix is accepted.
    let prices = load_asset_prices();
    let mut dated = CLA::default();
    dated.allocate(Some(AssetPricesInput::Prices(&prices)), None, None, None, None).unwrap();
    let mut bare = CLA::default();
    bare.allocate(Some(AssetPricesInput::RawMatrix(&prices.data)), None, None, None, None).unwrap();
    assert_eq!(dated.weights, bare.weights);
    assert!(dated.weights.len() > 1);
}

#[test]
fn test_two_uncorrelated_assets_follow_the_closed_form_critical_line() {
    // Four tests used to stand here. They called `_compute_lambda`, `_compute_w` and
    // `_free_bound_weight`, which were placeholders returning (0.0, 0), zeros and
    // (false, false), and asserted exactly those constants. The algorithm exists now, so this
    // checks it against a case that can be worked by hand.
    //
    // Long-only, mu = (0.10, 0.04), variances (0.04, 0.01), no correlation. The line starts all
    // in asset 0. With both free, w = gamma C^-1 1 + lambda C^-1 mu and the budget give
    // w_1 = 0 at lambda = var_0 / (mu_0 - mu_1) = 0.04 / 0.06, and at lambda = 0 the
    // minimum-variance portfolio is inverse-variance: (0.01, 0.04) / 0.05 = (0.2, 0.8).
    let mu = DMatrix::from_column_slice(2, 1, &[0.10, 0.04]);
    let cov = DMatrix::from_row_slice(2, 2, &[0.04, 0.0, 0.0, 0.01]);
    let mut cla = CLA::new(WeightBounds::Tuple(0.0, 1.0), "mean");
    cla.allocate(None, Some(&mu), Some(&cov), None, Some("cla_turning_points")).unwrap();

    assert_eq!(cla.weights.len(), 3);
    assert_eq!(cla.weights[0], vec![1.0, 0.0]);
    assert!(cla.lambdas[0].is_infinite());
    assert!((cla.weights[1][0] - 1.0).abs() < 1e-12 && cla.weights[1][1].abs() < 1e-12);
    assert!((cla.lambdas[1] - 0.04 / 0.06).abs() < 1e-12, "lambda {}", cla.lambdas[1]);
    assert!((cla.weights[2][0] - 0.2).abs() < 1e-12 && (cla.weights[2][1] - 0.8).abs() < 1e-12);
    assert_eq!(cla.lambdas[2], 0.0);
    assert_eq!(cla.free_weights[2].len(), 2);
}

#[test]
fn test_value_error_for_non_date_index() {
    let prices = load_asset_prices();
    let bad = AssetPrices::new(prices.data.clone(), Vec::new());
    let mut cla = CLA::default();
    let err = cla.allocate(
        Some(AssetPricesInput::Prices(&bad)),
        None,
        None,
        None,
        Some("cla_turning_points"),
    );
    assert!(matches!(err, Err(ClaError::InvalidAssetPrices(_))));
}

#[test]
fn test_value_error_for_unknown_returns() {
    let prices = load_asset_prices();
    let mut cla = CLA::new(WeightBounds::Tuple(0.0, 1.0), "unknown_returns");
    let err = cla.allocate(
        Some(AssetPricesInput::Prices(&prices)),
        None,
        None,
        None,
        Some("cla_turning_points"),
    );
    assert!(matches!(err, Err(ClaError::UnknownReturns(_))));
}

#[test]
fn test_resampling_asset_prices() {
    let prices = load_asset_prices();
    let mut cla = CLA::default();
    cla.allocate(
        Some(AssetPricesInput::Prices(&prices)),
        None,
        None,
        Some("B"),
        Some("min_volatility"),
    )
    .unwrap();
    let weights = &cla.weights[0];
    assert_eq!(weights.len(), prices.data.ncols());
    assert_weights_basic(weights, true);
}

#[test]
fn test_all_inputs_none() {
    let mut cla = CLA::default();
    let err = cla.allocate(None, None, None, None, None);
    assert!(matches!(err, Err(ClaError::MissingInputs)));
}

#[test]
fn test_cla_with_input_as_returns_and_covariance() {
    let prices = load_asset_prices();
    let expected_returns =
        ReturnsEstimation::calculate_mean_historical_returns(&prices.data, None).unwrap();
    let expected = DMatrix::from_column_slice(expected_returns.len(), 1, &expected_returns);
    let returns = ReturnsEstimation::calculate_returns(&prices.data, None).unwrap();
    let cov = covariance(&returns);
    let mut cla = CLA::default();
    cla.allocate(None, Some(&expected), Some(&cov), None, None).unwrap();
    for turning_point in cla.weights.iter() {
        let cleaned: Vec<f64> =
            turning_point.iter().map(|w| if *w <= 1e-15 { 0.0 } else { *w }).collect();
        assert_eq!(cleaned.len(), prices.data.ncols());
        assert_weights_basic(&cleaned, true);
    }
}
