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

/// #185 item 12: `risk_free_rate` is annual in both stages, and `periods_per_year` (not a
/// hardcoded 252) annualises both.
#[test]
fn risk_free_rate_is_annual_and_periods_per_year_annualises_both_stages() {
    use openquant::backtest_statistics::sharpe_ratio;
    use openquant::portfolio_optimization::allocate_max_sharpe;

    let t0 = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..8).map(|d| t0 + Duration::days(d)).collect();
    let close = [100.0, 102.0, 101.0, 104.0, 103.0, 106.0, 104.0, 108.0];
    let probs = [0.9; 8];
    let asset_names: Vec<String> = ["A", "B"].iter().map(|s| s.to_string()).collect();
    let asset_prices = DMatrix::from_row_slice(
        8,
        2,
        &[
            100.0, 100.0, 101.0, 100.5, 101.5, 101.2, 102.8, 101.0, 103.0, 102.5, 104.1, 102.2,
            104.0, 103.4, 105.6, 103.9,
        ],
    );
    let run = |risk_free_rate: f64, periods_per_year: f64| {
        let input = ResearchPipelineInput {
            timestamps: &timestamps,
            close: &close,
            model_probabilities: &probs,
            model_sides: None,
            asset_prices: &asset_prices,
            asset_names: &asset_names,
        };
        let config = ResearchPipelineConfig {
            cusum_threshold: 0.005,
            risk_free_rate,
            periods_per_year,
            ..Default::default()
        };
        run_mid_frequency_pipeline(input, &config).unwrap()
    };

    // Annual risk-free rate: 2.52% a year is 0.01% a bar at 252 bars a year. The old code
    // subtracted the annual rate from every bar's return.
    let out = run(0.0252, 252.0);
    let direct = sharpe_ratio(&out.backtest.strategy_returns, 252.0, 0.0001);
    assert!((out.risk.realized_sharpe - direct).abs() < 1e-12);
    let alloc = allocate_max_sharpe(&asset_prices, 0.0252, None, None).unwrap();
    assert!((out.portfolio.portfolio_sharpe - alloc.portfolio_sharpe).abs() < 1e-12);

    // One-minute bars: 252 * 390 a year. Sharpe ratios scale by sqrt(390) at a zero rate,
    // returns by 390, and the weights don't move.
    let daily = run(0.0, 252.0);
    let minute = run(0.0, 252.0 * 390.0);
    let root = 390f64.sqrt();
    let (d, m) = (&daily.portfolio, &minute.portfolio);
    assert!((minute.risk.realized_sharpe - daily.risk.realized_sharpe * root).abs() < 1e-9);
    assert!((m.portfolio_sharpe - d.portfolio_sharpe * root).abs() < 1e-9);
    assert!((m.portfolio_return - d.portfolio_return * 390.0).abs() < 1e-9);
    assert!((m.portfolio_risk - d.portfolio_risk * root).abs() < 1e-9);
    for (a, b) in m.weights.iter().zip(&d.weights) {
        assert!((a - b).abs() < 1e-9);
    }

    // A non-zero annual rate at a non-default frequency: the portfolio Sharpe ratio is still
    // (return - rate) / risk in annual units, and realized_sharpe uses rate / periods a bar.
    let p = 252.0 * 390.0;
    let out = run(0.05, p);
    let pf = &out.portfolio;
    assert!(((pf.portfolio_return - 0.05) / pf.portfolio_risk - pf.portfolio_sharpe).abs() < 1e-9);
    let direct = sharpe_ratio(&out.backtest.strategy_returns, p, 0.05 / p);
    assert!((out.risk.realized_sharpe - direct).abs() < 1e-12);
}

#[test]
fn periods_per_year_must_be_positive_and_finite() {
    let t0 = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..4).map(|d| t0 + Duration::days(d)).collect();
    let close = [100.0, 102.0, 100.0, 102.0];
    let probs = [0.9; 4];
    let asset_names = vec!["A".to_string()];
    let asset_prices = DMatrix::from_column_slice(4, 1, &[100.0, 101.0, 102.0, 103.0]);
    for bad in [0.0, -252.0, f64::NAN, f64::INFINITY] {
        let input = ResearchPipelineInput {
            timestamps: &timestamps,
            close: &close,
            model_probabilities: &probs,
            model_sides: None,
            asset_prices: &asset_prices,
            asset_names: &asset_names,
        };
        let config = ResearchPipelineConfig { periods_per_year: bad, ..Default::default() };
        assert!(matches!(
            run_mid_frequency_pipeline(input, &config),
            Err(openquant::pipeline::PipelineError::InvalidParameter(_))
        ));
    }
}
