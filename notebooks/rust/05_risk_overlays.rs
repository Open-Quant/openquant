use openquant::risk_metrics::RiskMetrics;

let returns = vec![0.0, 0.01, -0.008, 0.015, -0.006, 0.011, -0.002];
let rm = RiskMetrics::default();
let var95 = rm.calculate_value_at_risk(&returns, 0.05).unwrap();
let es95 = rm.calculate_expected_shortfall(&returns, 0.05).unwrap();
// CDaR reads drawdowns off the equity curve, and takes the upper-tail level (0.95).
let equity: Vec<f64> = returns.iter().scan(1.0, |w, r| { *w *= 1.0 + r; Some(*w) }).collect();
let cdar95 = rm.calculate_conditional_drawdown_risk(&equity, 0.95).unwrap();
(var95, es95, cdar95)
