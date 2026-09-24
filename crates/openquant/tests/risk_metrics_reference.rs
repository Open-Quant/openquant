//! Value tests for `risk_metrics`, all hand-worked from the definitions in the comments.
//!
//! Every assertion in the pre-existing `tests/risk_metrics.rs` is `is_finite() || is_nan()`,
//! which no numeric result can fail (replacing the quantile with the constant 1234.5 passes).
//! See `docs/test-sensitivity-audit.md`.
//!
//! Definitions (VaR and ES follow mlfinlab's `RiskMetrics`; CDaR deliberately does not):
//!   VaR(q)  = q-quantile of the sample with "higher" interpolation: sorted[ceil(q * (n - 1))]
//!   ES(q)   = mean of the observations strictly below VaR(q)
//!   CDaR(q) = with d_t = running_max(x)_t - x_t the drawdown of a cumulative series x:
//!             mean of the d_t at or above the "higher" q-quantile of d (Chekhlov, Uryasev and
//!             Zabarankin, 2005). q is the upper-tail level: 0.95 averages the worst 5%.
//!             mlfinlab instead averages the running maximum of d strictly above its quantile,
//!             which returns NaN or a non-tail number (#102); these expectations differ from
//!             mlfinlab on purpose.

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
///   drawdown d    0 0 1 0 4 1          sorted d: 0 0 0 1 1 4
/// q = 0.6: position 3.0 -> threshold 1; d at or above 1 are (1, 1, 4): CDaR = 2
/// q = 0.9: position 4.5 -> "higher" takes sorted[5] = 4; the tail is (4): CDaR = 4
/// q = 0.0: every d is in the tail: CDaR = mean(d) = 1
/// Before #102 was fixed, q = 0.6 gave 4 and q = 0.7 gave NaN.
#[test]
fn conditional_drawdown_risk_hand_worked() {
    let rm = RiskMetrics;
    let x = [1.0, 3.0, 2.0, 5.0, 1.0, 4.0];
    assert_eq!(rm.calculate_conditional_drawdown_risk(&x, 0.6).unwrap(), 2.0);
    assert_eq!(rm.calculate_conditional_drawdown_risk(&x, 0.7).unwrap(), 2.0);
    assert_eq!(rm.calculate_conditional_drawdown_risk(&x, 0.9).unwrap(), 4.0);
    assert_eq!(rm.calculate_conditional_drawdown_risk(&x, 0.0).unwrap(), 1.0);
    assert_eq!(rm.calculate_conditional_drawdown_risk(&x, 1.0).unwrap(), 4.0);
}

/// A V: equity falls from 100 to 80 one point a day, then climbs back to 100 (41 points).
///   drawdown d: 0 1 2 ... 20 19 ... 1 0     sorted d: 0 0 1 1 ... 19 19 20
/// q = 0.95: position ceil(0.95 * 40) = 38 -> sorted[38] = 19; the tail is (19, 19, 20):
///           CDaR = 58 / 3, between the 95th-percentile drawdown (19) and the maximum (20).
/// The old code averaged running_max(d), which sits at 20 for the last 21 days, so its
/// quantile was 20, nothing lay strictly above it, and it returned NaN (#102).
#[test]
fn conditional_drawdown_risk_on_a_v_shaped_equity_curve() {
    let equity: Vec<f64> = (0..=20).chain((0..20).rev()).map(|k| 100.0 - k as f64).collect();
    assert_eq!(equity.len(), 41);
    let cdar = RiskMetrics.calculate_conditional_drawdown_risk(&equity, 0.95).unwrap();
    assert_eq!(cdar, 58.0 / 3.0);
}

/// On a long pseudo-random equity curve CDaR(0.95) is finite and lies between the
/// 95th-percentile drawdown and the maximum drawdown (the acceptance criterion of #102).
#[test]
fn conditional_drawdown_risk_is_bounded_by_quantile_and_maximum_drawdown() {
    // xorshift64 returns in [-0.02, 0.02) with a small drift, compounded into an equity curve
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut equity = Vec::with_capacity(1000);
    let mut level = 1.0;
    for _ in 0..1000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u = (state >> 11) as f64 / (1u64 << 53) as f64;
        level *= 1.0 + 0.0002 + 0.04 * (u - 0.5);
        equity.push(level);
    }

    let mut peak = f64::NEG_INFINITY;
    let mut drawdown: Vec<f64> = equity
        .iter()
        .map(|&v| {
            peak = peak.max(v);
            peak - v
        })
        .collect();
    drawdown.sort_by(|a, b| a.total_cmp(b));
    let max_dd = drawdown[drawdown.len() - 1];
    let q95 = drawdown[(0.95 * (drawdown.len() - 1) as f64).ceil() as usize];
    assert!(max_dd > q95, "the curve must have a drawdown tail: {q95} vs {max_dd}");

    let cdar = RiskMetrics.calculate_conditional_drawdown_risk(&equity, 0.95).unwrap();
    assert!(cdar.is_finite(), "{cdar}");
    assert!(cdar >= q95 && cdar <= max_dd, "{q95} <= {cdar} <= {max_dd}");
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
