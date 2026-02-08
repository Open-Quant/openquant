use openquant::risk_metrics::RiskMetrics;

let returns = vec![0.0, 0.01, -0.008, 0.015, -0.006, 0.011, -0.002];
let rm = RiskMetrics::default();
let var95 = rm.calculate_value_at_risk(&returns, 0.05).unwrap();
let es95 = rm.calculate_expected_shortfall(&returns, 0.05).unwrap();
let cdar95 = rm.calculate_conditional_drawdown_risk(&returns, 0.05).unwrap();
(var95, es95, cdar95)
