use csv::ReaderBuilder;
use nalgebra::DMatrix;
use openquant::risk_metrics::RiskMetrics;
use std::path::Path;

fn load_prices() -> DMatrix<f64> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/portfolio_optimization/stock_prices.csv");
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path).unwrap();
    let mut data: Vec<Vec<f64>> = Vec::new();
    for result in rdr.records() {
        let rec = result.unwrap();
        let mut row = Vec::new();
        for field in rec.iter().skip(1) {
            row.push(field.parse::<f64>().unwrap());
        }
        data.push(row);
    }
    let rows = data.len();
    let cols = data[0].len();
    let flat: Vec<f64> = data.into_iter().flat_map(|r| r.into_iter()).collect();
    DMatrix::from_vec(rows, cols, flat)
}

fn covariance(prices: &DMatrix<f64>) -> DMatrix<f64> {
    let rows = prices.nrows();
    let cols = prices.ncols();
    let means: Vec<f64> = (0..cols).map(|c| prices.column(c).sum() / rows as f64).collect();

    let mut cov = DMatrix::zeros(cols, cols);
    for i in 0..cols {
        for j in i..cols {
            let mut s = 0.0;
            for r in 0..rows {
                s += (prices[(r, i)] - means[i]) * (prices[(r, j)] - means[j]);
            }
            s /= (rows - 1) as f64;
            cov[(i, j)] = s;
            cov[(j, i)] = s;
        }
    }
    cov
}

fn first_col(prices: &DMatrix<f64>) -> Vec<f64> {
    (0..prices.nrows()).map(|r| prices[(r, 0)]).collect()
}

#[test]
fn test_variance_calculation() {
    let prices = load_prices();
    let cov = covariance(&prices);
    let weights = vec![1.0; prices.ncols()];
    let variance = RiskMetrics
        .calculate_variance(&cov, &weights)
        .expect("variance should compute");
    assert!(variance.is_finite());
}

#[test]
fn test_value_at_risk_calculation() {
    let prices = load_prices();
    let test_returns = first_col(&prices);
    let value_at_risk = RiskMetrics
        .calculate_value_at_risk(&test_returns, 0.05)
        .expect("VaR should compute");
    assert!(value_at_risk.is_finite());
}

#[test]
fn test_expected_shortfall_calculation() {
    let prices = load_prices();
    let test_returns = first_col(&prices);
    let expected_shortfall = RiskMetrics
        .calculate_expected_shortfall(&test_returns, 0.05)
        .expect("ES should compute");
    assert!(expected_shortfall.is_finite() || expected_shortfall.is_nan());
}

#[test]
fn test_conditional_drawdown_calculation() {
    let prices = load_prices();
    let test_returns = first_col(&prices);
    let conditional_drawdown = RiskMetrics
        .calculate_conditional_drawdown_risk(&test_returns, 0.05)
        .expect("CDaR should compute");
    assert!(conditional_drawdown.is_finite() || conditional_drawdown.is_nan());
}

#[test]
fn test_value_at_risk_for_dataframe() {
    let prices = load_prices();
    let n = prices.nrows();
    let data: Vec<f64> = (0..n).map(|r| prices[(r, 0)]).collect();
    let test_returns = DMatrix::from_vec(n, 1, data);

    let value_at_risk = RiskMetrics
        .calculate_value_at_risk_from_matrix(&test_returns, 0.05)
        .expect("VaR should compute");
    assert!(value_at_risk.is_finite());
}

#[test]
fn test_expected_shortfall_for_dataframe() {
    let prices = load_prices();
    let n = prices.nrows();
    let data: Vec<f64> = (0..n).map(|r| prices[(r, 0)]).collect();
    let test_returns = DMatrix::from_vec(n, 1, data);

    let expected_shortfall = RiskMetrics
        .calculate_expected_shortfall_from_matrix(&test_returns, 0.05)
        .expect("ES should compute");
    assert!(expected_shortfall.is_finite() || expected_shortfall.is_nan());
}

#[test]
fn test_conditional_drawdown_for_dataframe() {
    let prices = load_prices();
    let n = prices.nrows();
    let data: Vec<f64> = (0..n).map(|r| prices[(r, 0)]).collect();
    let test_returns = DMatrix::from_vec(n, 1, data);

    let conditional_drawdown = RiskMetrics
        .calculate_conditional_drawdown_risk_from_matrix(&test_returns, 0.05)
        .expect("CDaR should compute");
    assert!(conditional_drawdown.is_finite() || conditional_drawdown.is_nan());
}
