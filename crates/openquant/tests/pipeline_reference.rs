//! Hand-worked value test for the strategy path of `pipeline::run_mid_frequency_pipeline`.
//!
//! The two inline tests in `src/pipeline.rs` assert lengths and "weights sum to one". Trading on
//! the signal of the SAME bar (look-ahead) instead of the previous bar passes them, as does
//! replacing the equity curve by a constant. See `docs/test-sensitivity-audit.md`.
//!
//! The portfolio stage (`allocate_max_sharpe`) is deliberately not asserted here: that module is
//! being rebuilt under issue #76.

use chrono::{Duration, NaiveDate};
use nalgebra::DMatrix;
use openquant::pipeline::{
    run_mid_frequency_pipeline, ResearchPipelineConfig, ResearchPipelineInput,
};

/// close:        100   110   110    99    99   108.9
/// simple ret:        +0.10   0   -0.10    0   +0.10
/// log ret:          +0.0953  0  -0.1054   0  +0.0953
///
/// Symmetric CUSUM filter on log returns with h = 0.05 (AFML snippet 2.4): the up-move at bar 1
/// exceeds h -> event, reset; the down-move at bar 3 -> event, reset; the up-move at bar 5 ->
/// event. Events = [1, 3, 5].
///
/// Bet size from probability (AFML 10.3, two classes): z = (p - 1/2) / sqrt(p (1 - p)),
/// m = side * (2 Phi(z) - 1). With p = 0.9: z = 0.4 / 0.3 = 1.3333, Phi(1.3333) = 0.9088 (normal
/// table), m = 0.8176, discretised to steps of 0.1 -> 0.8.
///   bar 1: p = 0.9, side +1 -> +0.8       bar 3: p = 0.9, side -1 -> -0.8
///   bar 5: p = 0.5          ->  0.0       (z = 0)
/// The position is held until the next event: timeline = (0, 0.8, 0.8, -0.8, -0.8, 0).
///
/// A signal decided at the close of bar t can only earn the return from t to t+1:
///   strategy return over (t-1, t] = signal[t-1] * simple_return[t]
///   = (0 * 0.1,  0.8 * 0,  0.8 * -0.1,  -0.8 * 0,  -0.8 * 0.1) = (0, 0, -0.08, 0, -0.08)
///   equity = (1, 1, 1, 0.92, 0.92, 0.8464)
/// (With look-ahead the strategy would instead EARN 0.08 on bars 1, 3 and 5.)
#[test]
fn strategy_path_hand_worked_without_look_ahead() {
    let t0 = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..6).map(|d| t0 + Duration::days(d)).collect();
    let close = [100.0, 110.0, 110.0, 99.0, 99.0, 108.9];
    let probs = [0.5, 0.9, 0.5, 0.9, 0.5, 0.5];
    let sides = [1.0, 1.0, 1.0, -1.0, 1.0, 1.0];
    let asset_names: Vec<String> = ["A", "B", "C"].iter().map(|s| s.to_string()).collect();
    let asset_prices = DMatrix::from_row_slice(
        6,
        3,
        &[
            100.0, 100.0, 100.0, 100.2, 99.8, 100.1, 100.3, 99.9, 100.2, 100.4, 100.2, 100.1,
            100.3, 100.0, 100.3, 100.6, 100.1, 100.4,
        ],
    );
    let input = ResearchPipelineInput {
        timestamps: &timestamps,
        close: &close,
        model_probabilities: &probs,
        model_sides: Some(&sides),
        asset_prices: &asset_prices,
        asset_names: &asset_names,
    };
    let config = ResearchPipelineConfig { cusum_threshold: 0.05, ..Default::default() };
    let out = run_mid_frequency_pipeline(input, &config).unwrap();

    assert_eq!(out.events.indices, vec![1, 3, 5]);

    // step_size 0.1 rounding produces multiples of 0.1 (e.g. 8.0 * 0.1), not decimal literals:
    // compare to within a few ulps.
    let close_to = |got: &[f64], want: &[f64], what: &str| {
        assert_eq!(got.len(), want.len(), "{what}");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < 1e-12, "{what}[{i}] = {g}, expected {w}");
        }
    };
    close_to(&out.signals.event_signal, &[0.8, -0.8, 0.0], "event_signal");
    close_to(&out.signals.timeline_signal, &[0.0, 0.8, 0.8, -0.8, -0.8, 0.0], "timeline_signal");
    close_to(&out.backtest.strategy_returns, &[0.0, 0.0, -0.08, 0.0, -0.08], "strategy_returns");
    close_to(&out.backtest.equity_curve, &[1.0, 1.0, 1.0, 0.92, 0.92, 0.8464], "equity_curve");
}
