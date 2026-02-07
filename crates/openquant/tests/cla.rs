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
    let matrix = DMatrix::from_vec(rows, cols, flat);
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
fn test_lambda_for_no_bounded_weights() {
    let prices = load_asset_prices();
    let mut cla = CLA::new(WeightBounds::Tuple(0.0, 1.0), "mean");
    cla.allocate(Some(AssetPricesInput::Prices(&prices)), None, None, None, Some("min_volatility"))
        .unwrap();
    let cov = covariance(&prices.data);
    let (x, y) = cla._compute_lambda(&cov, &cov, &cla.expected_returns, None, &[1], &[0]);
    assert!(x.is_finite());
    let _ = y as i64;
}

#[test]
fn test_free_bound_weights() {
    let prices = load_asset_prices();
    let mut cla = CLA::new(WeightBounds::Tuple(0.0, 1.0), "mean");
    cla.allocate(Some(AssetPricesInput::Prices(&prices)), None, None, None, Some("min_volatility"))
        .unwrap();
    let free = vec![1usize; cla.expected_returns.nrows() + 1];
    let (x, y) = cla._free_bound_weight(&free);
    assert!(!x);
    assert!(!y);
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
fn test_lambda_for_zero_matrices() {
    let prices = load_asset_prices();
    let mut cla = CLA::new(WeightBounds::Tuple(0.0, 1.0), "mean");
    cla.allocate(Some(AssetPricesInput::Prices(&prices)), None, None, None, Some("min_volatility"))
        .unwrap();
    let mut cov = covariance(&prices.data);
    for v in cov.iter_mut() {
        *v = 0.0;
    }
    let (x, y) = cla._compute_lambda(&cov, &cov, &cla.expected_returns, None, &[1], &[0]);
    assert_eq!(x, 0.0);
    assert_eq!(y, 0);
}

#[test]
fn test_w_for_no_bounded_weights() {
    let prices = load_asset_prices();
    let mut cla = CLA::new(WeightBounds::Tuple(0.0, 1.0), "mean");
    cla.allocate(Some(AssetPricesInput::Prices(&prices)), None, None, None, Some("min_volatility"))
        .unwrap();
    let cov = covariance(&prices.data);
    let (x, y) = cla._compute_w(&cov, &cov, &cla.expected_returns, None);
    assert_eq!(x.len(), cov.nrows());
    assert!(y.is_finite());
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
    let mut cla = CLA::default();
    cla.weights = vec![vec![1.0]];
    cla.lower_bounds = vec![100.0];
    cla.upper_bounds = vec![1.0];
    cla.lambdas = vec![0.0];
    cla.gammas = vec![0.0];
    cla.free_weights = vec![vec![]];
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
fn test_value_error_for_non_dataframe_input() {
    let prices = load_asset_prices();
    let mut cla = CLA::default();
    let err = cla.allocate(
        Some(AssetPricesInput::RawMatrix(&prices.data)),
        None,
        None,
        None,
        Some("cla_turning_points"),
    );
    assert!(matches!(err, Err(ClaError::InvalidAssetPrices(_))));
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
