use nalgebra::DMatrix;
use openquant::filters::{cusum_filter_indices, Threshold};
use openquant::portfolio_optimization::allocate_max_sharpe;
use openquant::risk_metrics::RiskMetrics;

fn main() {
    let close = vec![80.0, 80.2, 80.1, 80.5, 80.3, 80.8, 80.6, 80.9, 81.1, 81.0];
    let events = cusum_filter_indices(&close, Threshold::Scalar(0.001));
    assert!(!events.is_empty());

    let prices = DMatrix::from_row_slice(
        10,
        3,
        &[
            80.0, 40.0, 25.0, 80.1, 40.2, 25.1, 80.3, 40.1, 25.2, 80.2, 40.4, 25.1, 80.6, 40.5,
            25.3, 80.7, 40.6, 25.4, 80.8, 40.8, 25.5, 81.0, 40.9, 25.6, 81.2, 41.0, 25.7, 81.1,
            41.1, 25.8,
        ],
    );
    let alloc = allocate_max_sharpe(&prices, 0.0, None, None).expect("max sharpe alloc");
    let sum_w: f64 = alloc.weights.iter().sum();
    assert!((sum_w - 1.0).abs() < 1e-6);

    let ret = vec![0.0, 0.002, -0.001, 0.004, -0.002, 0.003, 0.001, -0.0005, 0.0025];
    let rm = RiskMetrics::default();
    let _ = rm.calculate_value_at_risk(&ret, 0.05).expect("var");
    let _ = rm.calculate_expected_shortfall(&ret, 0.05).expect("es");
    let _ = rm.calculate_conditional_drawdown_risk(&ret, 0.05).expect("cdar");

    println!("rust notebook smoke: ok");
}
