//! Value tests for `risk_metrics`, all hand-worked from the definitions in the comments.
//!
//! Every assertion in the pre-existing `tests/risk_metrics.rs` is `is_finite() || is_nan()`,
//! which no numeric result can fail (replacing the quantile with the constant 1234.5 passes).
//! See `docs/test-sensitivity-audit.md`.
//!
//! Definitions (the library follows mlfinlab's `RiskMetrics`):
//!   VaR(q)  = q-quantile of the sample with "higher" interpolation: sorted[ceil(q * (n - 1))]
//!   ES(q)   = mean of the observations strictly below VaR(q)
//!   CDaR(q) = with d_t = running_max(x)_t - x_t and m_t = running_max(d)_t:
//!             mean of the m_t strictly above the "higher" q-quantile of m

use nalgebra::DMatrix;
use openquant::risk_metrics::RiskMetrics;

/// Eleven observations so that q * (n - 1) = 10 q is an integer at multiples of 0.1.
/// Sorted: -0.05 -0.04 -0.03 -0.02 -0.01 0.00 0.01 0.02 0.03 0.04 0.05
const RETURNS: [f64; 11] = [-0.05, 0.02, -0.01, 0.03, -0.03, 0.01, 0.00, 0.04, -0.02, 0.05, -0.04];

/// w = (0.5, 0.3, 0.2), S = [[0.04, 0.006, 0.002], [0.006, 0.09, 0.012], [0.002, 0.012, 0.01]]
///   diagonal terms:  0.25*0.04 + 0.09*0.09 + 0.04*0.01          = 0.0185
///   cross terms:     2*(0.15*0.006 + 0.10*0.002 + 0.06*0.012)   = 0.00364
///   w'Sw = 0.02214
#[test]
fn portfolio_variance_hand_worked() {
    let cov = DMatrix::from_row_slice(
        3,
        3,
        &[0.04, 0.006, 0.002, 0.006, 0.09, 0.012, 0.002, 0.012, 0.01],
    );
    let v = RiskMetrics.calculate_variance(&cov, &[0.5, 0.3, 0.2]).unwrap();
    // nine products of O(1e-2) numbers (ulp ~ 3e-18 each): 1e-15 is rounding, not slack
    assert!((v - 0.02214).abs() < 1e-15, "{v}");
}

/// q = 0.20: position 2.0 -> sorted[2] = -0.03
/// q = 0.25: position 2.5 -> "higher" takes sorted[3] = -0.02 (a "lower"/floor rule gives -0.03)
/// q = 0.00 and 1.00 are the minimum and the maximum.
#[test]
fn value_at_risk_uses_higher_quantile_hand_worked() {
    let rm = RiskMetrics;
    assert_eq!(rm.calculate_value_at_risk(&RETURNS, 0.20).unwrap(), -0.03);
    assert_eq!(rm.calculate_value_at_risk(&RETURNS, 0.25).unwrap(), -0.02);
    assert_eq!(rm.calculate_value_at_risk(&RETURNS, 0.0).unwrap(), -0.05);
    assert_eq!(rm.calculate_value_at_risk(&RETURNS, 1.0).unwrap(), 0.05);
}

/// q = 0.20: VaR = -0.03, observations strictly below it are -0.05 and -0.04: ES = -0.045.
/// q = 0.00: VaR is the minimum, nothing lies strictly below it: NaN (mlfinlab: mean of empty).
#[test]
fn expected_shortfall_is_mean_strictly_below_var_hand_worked() {
    let rm = RiskMetrics;
    let es = rm.calculate_expected_shortfall(&RETURNS, 0.20).unwrap();
    assert!((es - (-0.045)).abs() < 1e-15, "{es}");
    assert!(rm.calculate_expected_shortfall(&RETURNS, 0.0).unwrap().is_nan());
}

/// x = (1, 3, 2, 5, 1, 4)
///   running max   1 3 3 5 5 5
///   drawdown d    0 0 1 0 4 1
///   running max m 0 0 1 1 4 4          sorted m: 0 0 1 1 4 4
/// q = 0.6: position 3.0 -> threshold 1; m above 1 are (4, 4): CDaR = 4
/// q = 0.7: position 3.5 -> "higher" takes sorted[4] = 4; nothing above 4: NaN
///          (a floor rule would give threshold 1 and CDaR 4 instead)
#[test]
fn conditional_drawdown_risk_hand_worked() {
    let rm = RiskMetrics;
    let x = [1.0, 3.0, 2.0, 5.0, 1.0, 4.0];
    assert_eq!(rm.calculate_conditional_drawdown_risk(&x, 0.6).unwrap(), 4.0);
    assert!(rm.calculate_conditional_drawdown_risk(&x, 0.7).unwrap().is_nan());
}

/// The `_from_matrix` variants read the first column only.
#[test]
fn matrix_variants_use_first_column() {
    let rm = RiskMetrics;
    let m = DMatrix::from_fn(RETURNS.len(), 2, |r, c| if c == 0 { RETURNS[r] } else { 9.0 });
    assert_eq!(rm.calculate_value_at_risk_from_matrix(&m, 0.25).unwrap(), -0.02);
    let es = rm.calculate_expected_shortfall_from_matrix(&m, 0.20).unwrap();
    assert!((es - (-0.045)).abs() < 1e-15, "{es}");
}
