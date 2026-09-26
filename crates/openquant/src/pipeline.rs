//! End-to-end mid-frequency research pipeline: CUSUM events, bet sizing, a max-Sharpe
//! portfolio, tail-risk metrics and a single-asset backtest in one call.
//!
//! [`run_mid_frequency_pipeline`] chains existing modules rather than implementing a new
//! method, so it has no AFML snippet of its own. Its stages follow the book:
//!
//! 1. **Events** — the symmetric CUSUM filter on `close` (AFML §2.5.2.1, Snippet 2.4; see
//!    [`crate::filters::cusum_filter_indices`]).
//! 2. **Signals** — the model probability and side at each event are turned into a bet size
//!    `2 Phi(z) - 1` (AFML §10.3, Snippet 10.1; [`crate::bet_sizing::get_signal`]) and rounded to
//!    multiples of `step_size` (Snippet 10.3; [`crate::bet_sizing::discrete_signal`]). The size
//!    set at an event is held on every bar until the next event.
//! 3. **Portfolio** — max-Sharpe mean-variance weights of `asset_prices`
//!    ([`crate::portfolio_optimization::allocate_max_sharpe`]; Markowitz, 1952, not AFML).
//! 4. **Risk** — historical VaR and expected shortfall of the strategy's per-bar returns and
//!    conditional drawdown at risk of its equity curve ([`crate::risk_metrics::RiskMetrics`]),
//!    plus the annualised Sharpe ratio (AFML §14.7.1).
//! 5. **Backtest** — the equity curve, drawdowns and time under water (AFML Snippet 14.4;
//!    [`crate::backtest_statistics::drawdown_and_time_under_water`]).
//!
//! No triple-barrier labelling, meta-labelling or model fitting happens here: the model's
//! probabilities and sides are inputs, one per bar.
//!
//! # Conventions
//!
//! - `timestamps`, `close`, `model_probabilities` and `model_sides` are aligned, one entry per
//!   bar, oldest first. `close` must be positive prices. `model_probabilities[i]` is the
//!   probability of the predicted class and `model_sides[i]` its side (`+1`/`-1`), both known
//!   at the close of bar `i`; without sides every bet is long.
//! - The signal is applied with a one-bar lag: the strategy's simple return over bar `i` is
//!   `timeline_signal[i - 1] * (close[i] / close[i - 1] - 1)`, so an event at bar `i` trades
//!   from bar `i + 1`. There is one strategy return per bar after the first, and the equity
//!   curve starts at 1 and compounds them.
//! - `asset_prices` is rows = observations (oldest first), columns = assets, in the order of
//!   `asset_names`. Its expected returns and covariance are annualised with 252 periods a
//!   year, so `risk_free_rate` is an annual rate there. The portfolio stage is independent of
//!   the backtest: its weights are reported, not traded, and its rows need not match
//!   `timestamps`.
//! - `confidence_level` is the lower-tail probability for VaR and expected shortfall (0.05
//!   looks at the worst 5% of per-bar returns); CDaR is computed at the upper-tail level
//!   `1 - confidence_level`. All three are per-bar quantities, not annualised.
//! - `realized_sharpe` annualises per-bar returns with 252 bars a year, whatever the bar
//!   spacing, and subtracts `risk_free_rate` per bar.
//! - The [`LeakageChecks`] are structural, not tests of the data: misaligned inputs are
//!   rejected with an error rather than flagged, CUSUM events are always in increasing order,
//!   and `has_forward_look_bias` is always `false` because the one-bar lag above is built in.
//!   Whether `model_probabilities` themselves were fitted without look-ahead is the caller's
//!   responsibility.
//!
//! # Example
//!
//! Four 2% moves (up, up, down, down) each trip a 1% CUSUM filter. With probability 0.9 the
//! bet size is `2 Phi(1.333) - 1 = 0.818`, which a step of 1 rounds to a full bet, long for the
//! first two events and short for the last two. The one-bar lag means the first down move is
//! still held long.
//!
//! ```
//! use chrono::NaiveDate;
//! use nalgebra::DMatrix;
//! use openquant::pipeline::{
//!     run_mid_frequency_pipeline, ResearchPipelineConfig, ResearchPipelineInput,
//! };
//!
//! let timestamps: Vec<_> = (1..=5)
//!     .map(|d| NaiveDate::from_ymd_opt(2024, 1, d).unwrap().and_hms_opt(16, 0, 0).unwrap())
//!     .collect();
//! let close = [100.0, 102.0, 104.04, 101.9592, 99.920016];
//! let probabilities = [0.9; 5];
//! let sides = [1.0, 1.0, 1.0, -1.0, -1.0];
//! let asset_prices = DMatrix::from_column_slice(5, 1, &[100.0, 101.0, 102.0, 103.0, 104.0]);
//! let asset_names = vec!["A".to_string()];
//! let input = ResearchPipelineInput {
//!     timestamps: &timestamps,
//!     close: &close,
//!     model_probabilities: &probabilities,
//!     model_sides: Some(&sides),
//!     asset_prices: &asset_prices,
//!     asset_names: &asset_names,
//! };
//! let config = ResearchPipelineConfig {
//!     cusum_threshold: 0.01,
//!     step_size: 1.0,
//!     ..ResearchPipelineConfig::default()
//! };
//! let out = run_mid_frequency_pipeline(input, &config)?;
//!
//! assert_eq!(out.events.indices, [1, 2, 3, 4]);
//! assert_eq!(out.signals.event_signal, [1.0, 1.0, -1.0, -1.0]);
//! assert_eq!(out.signals.timeline_signal, [0.0, 1.0, 1.0, -1.0, -1.0]);
//! // Returns are 0 (flat), +2% (long up), -2% (long down, the lag), +2% (short down).
//! let expected = [0.0, 0.02, -0.02, 0.02];
//! for (r, e) in out.backtest.strategy_returns.iter().zip(expected) {
//!     assert!((r - e).abs() < 1e-12);
//! }
//! assert!((out.backtest.equity_curve[4] - 1.02 * 0.98 * 1.02).abs() < 1e-12);
//! // One 2% drawdown from the high at bar 2, under water for the last two days.
//! assert!((out.backtest.drawdowns[0] - 0.02).abs() < 1e-12);
//! assert!((out.backtest.time_under_water_years[0] - 2.0 / 365.25).abs() < 1e-12);
//! // The 5% "higher" quantile of [-0.02, 0, 0.02, 0.02] is 0; the returns below it average -0.02.
//! assert!(out.risk.value_at_risk.abs() < 1e-12);
//! assert!((out.risk.expected_shortfall + 0.02).abs() < 1e-12);
//! // Mean 0.005, sample std 0.01915: 0.2611 per bar, 4.145 annualised with 252 bars.
//! assert!((out.risk.realized_sharpe - 4.1451).abs() < 1e-4);
//! // A single asset takes the whole portfolio.
//! assert!((out.portfolio.weights[0] - 1.0).abs() < 1e-9);
//! assert!(!out.leakage_checks.has_forward_look_bias);
//! # Ok::<(), openquant::pipeline::PipelineError>(())
//! ```
#![deny(missing_docs)]

use chrono::NaiveDateTime;
use nalgebra::DMatrix;

use crate::backtest_statistics::{drawdown_and_time_under_water, sharpe_ratio};
use crate::bet_sizing::{discrete_signal, get_signal};
use crate::filters::{cusum_filter_indices, Threshold};
use crate::portfolio_optimization::allocate_max_sharpe;
use crate::risk_metrics::{RiskMetrics, RiskMetricsError};

/// Parameters of [`run_mid_frequency_pipeline`].
///
/// The [`Default`] is `cusum_threshold = 0.001`, `num_classes = 2`, `step_size = 0.1`,
/// `risk_free_rate = 0.0`, `confidence_level = 0.05`.
#[derive(Debug, Clone)]
pub struct ResearchPipelineConfig {
    /// CUSUM filter threshold `h` on cumulative log returns of `close` (AFML Snippet 2.4), e.g.
    /// 0.001 for 0.1%. Must be > 0; a NaN or infinite threshold is not rejected and simply
    /// produces no events ([`PipelineError::NoEvents`]).
    pub cusum_threshold: f64,
    /// Number of classes `K` of the model behind `model_probabilities`; the bet size is 0 at
    /// probability `1/K` (AFML Snippet 10.1). Must be >= 2.
    pub num_classes: usize,
    /// Bet sizes are rounded to multiples of this step and clamped to `[-1, 1]` (AFML Snippet
    /// 10.3). A step <= 0 is not rejected and leaves the sizes unrounded.
    pub step_size: f64,
    /// Risk-free rate used twice, in two units: as an **annual** rate by the max-Sharpe
    /// allocation (whose returns are annualised with 252 periods) and as a **per-bar** rate by
    /// `realized_sharpe`. Only 0 (the default) means the same thing in both.
    pub risk_free_rate: f64,
    /// Lower-tail probability for VaR and expected shortfall, in `[0, 1]` (0.05 = worst 5% of
    /// per-bar returns). CDaR uses `1 - confidence_level`.
    pub confidence_level: f64,
}

impl Default for ResearchPipelineConfig {
    fn default() -> Self {
        Self {
            cusum_threshold: 0.001,
            num_classes: 2,
            step_size: 0.1,
            risk_free_rate: 0.0,
            confidence_level: 0.05,
        }
    }
}

/// Borrowed inputs of [`run_mid_frequency_pipeline`].
///
/// `timestamps`, `close`, `model_probabilities` and `model_sides` are aligned per bar, oldest
/// first, and must have the same length.
#[derive(Debug, Clone)]
pub struct ResearchPipelineInput<'a> {
    /// Bar timestamps in increasing order; used for event timestamps and time under water.
    /// Their order is not checked.
    pub timestamps: &'a [NaiveDateTime],
    /// Positive closing prices of the traded instrument, one per bar.
    pub close: &'a [f64],
    /// Probability of the predicted class at each bar, in `[0, 1]`, known at that bar's close.
    /// Only the values at CUSUM events are used. The range is not validated.
    pub model_probabilities: &'a [f64],
    /// Side of the prediction at each bar (typically `+1`/`-1`); `None` means always long.
    pub model_sides: Option<&'a [f64]>,
    /// Prices for the portfolio stage: rows = observations (oldest first, at least 2),
    /// columns = assets (at least 1). Not aligned with `timestamps`.
    pub asset_prices: &'a DMatrix<f64>,
    /// One name per column of `asset_prices`, copied to [`PortfolioStage::asset_names`].
    pub asset_names: &'a [String],
}

/// CUSUM events and the model outputs sampled at them, one entry per event.
#[derive(Debug, Clone)]
pub struct EventSelectionStage {
    /// 0-based bar positions of the events, in increasing order.
    pub indices: Vec<usize>,
    /// `timestamps[i]` for each event position `i`.
    pub timestamps: Vec<NaiveDateTime>,
    /// `model_probabilities[i]` for each event position `i`.
    pub probabilities: Vec<f64>,
    /// `model_sides[i]` for each event position `i`, or `1.0` when no sides were given.
    pub sides: Vec<f64>,
}

/// Bet sizes at events and forward-filled onto the bar timeline.
#[derive(Debug, Clone)]
pub struct SignalStage {
    /// Discretised bet size in `[-1, 1]` at each event (side times size, rounded to
    /// `step_size`).
    pub event_signal: Vec<f64>,
    /// Bet size on every bar: 0 before the first event, then the size of the latest event at or
    /// before the bar. Same length as `close`.
    pub timeline_signal: Vec<f64>,
}

/// Max-Sharpe mean-variance allocation of `asset_prices` (annualised with 252 periods).
///
/// Reported only: these weights are not used by the backtest.
#[derive(Debug, Clone)]
pub struct PortfolioStage {
    /// Asset names, in column order of `asset_prices`.
    pub asset_names: Vec<String>,
    /// Portfolio weights, one per asset, summing to 1.
    pub weights: Vec<f64>,
    /// Annualised portfolio volatility.
    pub portfolio_risk: f64,
    /// Annualised expected portfolio return.
    pub portfolio_return: f64,
    /// Annualised Sharpe ratio of the portfolio, net of `risk_free_rate` (annual).
    pub portfolio_sharpe: f64,
}

/// Tail-risk and performance metrics of the strategy's per-bar returns.
#[derive(Debug, Clone)]
pub struct RiskStage {
    /// Historical VaR: the "higher" `confidence_level`-quantile of the per-bar strategy
    /// returns, as a signed return (negative is a loss), not annualised.
    pub value_at_risk: f64,
    /// Mean of the per-bar returns strictly below `value_at_risk`; NaN when none are.
    pub expected_shortfall: f64,
    /// Conditional drawdown at risk of the equity curve at level `1 - confidence_level`, in
    /// equity units (the curve starts at 1).
    pub conditional_drawdown_risk: f64,
    /// Sharpe ratio of the per-bar returns annualised with 252 bars a year (AFML §14.7.1);
    /// NaN with fewer than two returns, infinite or NaN when they are constant.
    pub realized_sharpe: f64,
}

/// Single-asset backtest of `timeline_signal` on `close`.
#[derive(Debug, Clone)]
pub struct BacktestStage {
    /// A copy of the input timestamps, aligned with `equity_curve`.
    pub timestamps: Vec<NaiveDateTime>,
    /// Simple strategy return over each bar after the first,
    /// `timeline_signal[i - 1] * (close[i] / close[i - 1] - 1)`; one shorter than `close`.
    pub strategy_returns: Vec<f64>,
    /// Compounded equity, starting at 1 on the first bar; same length as `close`.
    pub equity_curve: Vec<f64>,
    /// Relative drawdown `1 - trough / peak` for each high-water mark that was followed by a
    /// dip (AFML Snippet 14.4).
    pub drawdowns: Vec<f64>,
    /// Time under water in years (365.25 days) for each entry of `drawdowns`, from its
    /// high-water mark to the next one with a drawdown or the end of the series.
    pub time_under_water_years: Vec<f64>,
}

/// Structural leakage guards of the run.
///
/// These describe the pipeline's construction rather than test the data: an output only
/// exists when the inputs passed validation, so the flags take fixed values in practice.
#[derive(Debug, Clone)]
pub struct LeakageChecks {
    /// Always `true`: misaligned inputs are rejected with [`PipelineError::LengthMismatch`]
    /// instead of producing an output.
    pub inputs_aligned: bool,
    /// Whether the event positions are non-decreasing; always `true` for CUSUM events.
    pub event_indices_sorted: bool,
    /// Always `false`: signals are applied with a one-bar lag. It does not detect look-ahead
    /// in the caller's `model_probabilities` or `model_sides`.
    pub has_forward_look_bias: bool,
}

/// Output of [`run_mid_frequency_pipeline`], one field per stage.
#[derive(Debug, Clone)]
pub struct ResearchPipelineOutput {
    /// CUSUM events and the model outputs at them.
    pub events: EventSelectionStage,
    /// Bet sizes at events and on the bar timeline.
    pub signals: SignalStage,
    /// Max-Sharpe allocation of `asset_prices`.
    pub portfolio: PortfolioStage,
    /// VaR, expected shortfall, CDaR and Sharpe ratio of the strategy.
    pub risk: RiskStage,
    /// Strategy returns, equity curve, drawdowns and time under water.
    pub backtest: BacktestStage,
    /// Structural leakage guards.
    pub leakage_checks: LeakageChecks,
}

/// Errors returned by [`run_mid_frequency_pipeline`].
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineError {
    /// The named input (`timestamps`, `close` or `model_probabilities`) is empty.
    EmptyInput(&'static str),
    /// Two inputs disagree in length: `(name_a, len_a, name_b, len_b)`.
    LengthMismatch(&'static str, usize, &'static str, usize),
    /// A parameter or the shape of `asset_prices` is invalid; the message names it.
    InvalidParameter(&'static str),
    /// The CUSUM filter found no event, so there is nothing to size or trade.
    NoEvents,
    /// The max-Sharpe allocation failed; carries the `Debug` form of the
    /// [`crate::portfolio_optimization::AllocError`].
    PortfolioAllocation(String),
    /// A risk metric failed. Validation rejects the same conditions first, so this is not
    /// expected in practice.
    Risk(RiskMetricsError),
}

impl core::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyInput(name) => write!(f, "{name} must not be empty"),
            Self::LengthMismatch(a, la, b, lb) => {
                write!(f, "{a}/{b} length mismatch: {la} vs {lb}")
            }
            Self::InvalidParameter(name) => write!(f, "invalid parameter: {name}"),
            Self::NoEvents => write!(f, "event filter produced no events"),
            Self::PortfolioAllocation(msg) => write!(f, "portfolio allocation failed: {msg}"),
            Self::Risk(err) => write!(f, "risk metric failed: {err:?}"),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<RiskMetricsError> for PipelineError {
    fn from(value: RiskMetricsError) -> Self {
        Self::Risk(value)
    }
}

/// Runs the events → signals → portfolio → risk → backtest pipeline on one instrument (see the
/// [module documentation](self) for the stages and conventions, and for a worked example).
///
/// # Errors
///
/// - [`PipelineError::EmptyInput`] if `timestamps`, `close` or `model_probabilities` is empty.
/// - [`PipelineError::LengthMismatch`] if `close` and `timestamps`, `model_probabilities` and
///   `close`, `model_sides` and `close`, or `asset_names` and the columns of `asset_prices`
///   differ in length.
/// - [`PipelineError::InvalidParameter`] if `asset_prices` has fewer than 2 rows or no
///   columns, `cusum_threshold <= 0`, `num_classes < 2`, or `confidence_level` is outside
///   `[0, 1]` (NaN included).
/// - [`PipelineError::NoEvents`] if the CUSUM filter selects no bar (including when `close` has
///   a single bar).
/// - [`PipelineError::PortfolioAllocation`] if the max-Sharpe optimisation on `asset_prices`
///   fails.
/// - [`PipelineError::Risk`] is part of the signature but not reachable after validation.
pub fn run_mid_frequency_pipeline(
    input: ResearchPipelineInput<'_>,
    config: &ResearchPipelineConfig,
) -> Result<ResearchPipelineOutput, PipelineError> {
    validate_input(&input, config)?;

    let event_indices =
        cusum_filter_indices(input.close, Threshold::Scalar(config.cusum_threshold))
            .map_err(|_| PipelineError::InvalidParameter("cusum_threshold"))?;
    if event_indices.is_empty() {
        return Err(PipelineError::NoEvents);
    }

    let event_probabilities: Vec<f64> =
        event_indices.iter().map(|&idx| input.model_probabilities[idx]).collect();

    let event_sides: Vec<f64> = match input.model_sides {
        Some(sides) => event_indices.iter().map(|&idx| sides[idx]).collect(),
        None => vec![1.0; event_indices.len()],
    };

    let raw_signal = get_signal(&event_probabilities, config.num_classes, Some(&event_sides));
    let event_signal = discrete_signal(&raw_signal, config.step_size);
    let timeline_signal = build_signal_timeline(input.close.len(), &event_indices, &event_signal)?;

    let events = EventSelectionStage {
        indices: event_indices.clone(),
        timestamps: event_indices.iter().map(|&idx| input.timestamps[idx]).collect(),
        probabilities: event_probabilities,
        sides: event_sides,
    };

    let signals = SignalStage { event_signal, timeline_signal: timeline_signal.clone() };

    let portfolio_out = allocate_max_sharpe(input.asset_prices, config.risk_free_rate, None, None)
        .map_err(|err| PipelineError::PortfolioAllocation(format!("{err:?}")))?;
    let portfolio = PortfolioStage {
        asset_names: input.asset_names.to_vec(),
        weights: portfolio_out.weights,
        portfolio_risk: portfolio_out.portfolio_risk,
        portfolio_return: portfolio_out.portfolio_return,
        portfolio_sharpe: portfolio_out.portfolio_sharpe,
    };

    let (strategy_returns, equity_curve) =
        compute_strategy_path(input.close, &signals.timeline_signal);
    let risk_metrics = RiskMetrics;
    let value_at_risk =
        risk_metrics.calculate_value_at_risk(&strategy_returns, config.confidence_level)?;
    let expected_shortfall =
        risk_metrics.calculate_expected_shortfall(&strategy_returns, config.confidence_level)?;
    // CDaR needs the cumulative curve, and takes the upper-tail level where VaR and ES take the
    // tail probability.
    let conditional_drawdown_risk = risk_metrics
        .calculate_conditional_drawdown_risk(&equity_curve, 1.0 - config.confidence_level)?;
    let realized_sharpe = if strategy_returns.len() > 1 {
        sharpe_ratio(&strategy_returns, 252.0, config.risk_free_rate)
    } else {
        f64::NAN
    };

    let equity_pairs: Vec<(NaiveDateTime, f64)> =
        input.timestamps.iter().copied().zip(equity_curve.iter().copied()).collect();
    let (drawdowns, time_under_water_years) = drawdown_and_time_under_water(&equity_pairs, false);

    let risk =
        RiskStage { value_at_risk, expected_shortfall, conditional_drawdown_risk, realized_sharpe };
    let backtest = BacktestStage {
        timestamps: input.timestamps.to_vec(),
        strategy_returns,
        equity_curve,
        drawdowns,
        time_under_water_years,
    };
    let leakage_checks = LeakageChecks {
        inputs_aligned: true,
        event_indices_sorted: event_indices.windows(2).all(|w| w[0] <= w[1]),
        has_forward_look_bias: false,
    };

    Ok(ResearchPipelineOutput { events, signals, portfolio, risk, backtest, leakage_checks })
}

fn validate_input(
    input: &ResearchPipelineInput<'_>,
    config: &ResearchPipelineConfig,
) -> Result<(), PipelineError> {
    if input.timestamps.is_empty() {
        return Err(PipelineError::EmptyInput("timestamps"));
    }
    if input.close.is_empty() {
        return Err(PipelineError::EmptyInput("close"));
    }
    if input.model_probabilities.is_empty() {
        return Err(PipelineError::EmptyInput("model_probabilities"));
    }
    if input.close.len() != input.timestamps.len() {
        return Err(PipelineError::LengthMismatch(
            "close",
            input.close.len(),
            "timestamps",
            input.timestamps.len(),
        ));
    }
    if input.model_probabilities.len() != input.close.len() {
        return Err(PipelineError::LengthMismatch(
            "model_probabilities",
            input.model_probabilities.len(),
            "close",
            input.close.len(),
        ));
    }
    if let Some(model_sides) = input.model_sides {
        if model_sides.len() != input.close.len() {
            return Err(PipelineError::LengthMismatch(
                "model_sides",
                model_sides.len(),
                "close",
                input.close.len(),
            ));
        }
    }
    if input.asset_prices.nrows() < 2 {
        return Err(PipelineError::InvalidParameter("asset_prices rows must be >= 2"));
    }
    if input.asset_prices.ncols() == 0 {
        return Err(PipelineError::InvalidParameter("asset_prices columns must be >= 1"));
    }
    if input.asset_names.len() != input.asset_prices.ncols() {
        return Err(PipelineError::LengthMismatch(
            "asset_names",
            input.asset_names.len(),
            "asset_prices.ncols",
            input.asset_prices.ncols(),
        ));
    }
    if config.cusum_threshold <= 0.0 {
        return Err(PipelineError::InvalidParameter("cusum_threshold must be > 0"));
    }
    if config.num_classes < 2 {
        return Err(PipelineError::InvalidParameter("num_classes must be >= 2"));
    }
    if !(0.0..=1.0).contains(&config.confidence_level) {
        return Err(PipelineError::InvalidParameter("confidence_level must be in [0, 1]"));
    }
    Ok(())
}

fn build_signal_timeline(
    total_len: usize,
    event_indices: &[usize],
    event_signal: &[f64],
) -> Result<Vec<f64>, PipelineError> {
    if event_indices.len() != event_signal.len() {
        return Err(PipelineError::LengthMismatch(
            "event_indices",
            event_indices.len(),
            "event_signal",
            event_signal.len(),
        ));
    }
    let mut timeline_signal = vec![0.0; total_len];
    let mut event_pos = 0usize;
    let mut current_signal = 0.0f64;
    for (idx, value) in timeline_signal.iter_mut().enumerate().take(total_len) {
        while event_pos < event_indices.len() && event_indices[event_pos] == idx {
            current_signal = event_signal[event_pos];
            event_pos += 1;
        }
        *value = current_signal;
    }
    Ok(timeline_signal)
}

fn compute_strategy_path(close: &[f64], timeline_signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut strategy_returns = Vec::with_capacity(close.len().saturating_sub(1));
    let mut equity_curve = Vec::with_capacity(close.len());
    let mut equity = 1.0f64;
    equity_curve.push(equity);
    for i in 1..close.len() {
        let close_return = close[i] / close[i - 1] - 1.0;
        let strat_return = timeline_signal[i - 1] * close_return;
        strategy_returns.push(strat_return);
        equity *= 1.0 + strat_return;
        equity_curve.push(equity);
    }
    (strategy_returns, equity_curve)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ts(value: &str) -> NaiveDateTime {
        NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S").expect("valid test ts")
    }

    #[test]
    fn test_pipeline_runs_end_to_end() {
        let timestamps = vec![
            parse_ts("2024-01-01 09:30:00"),
            parse_ts("2024-01-01 09:31:00"),
            parse_ts("2024-01-01 09:32:00"),
            parse_ts("2024-01-01 09:33:00"),
            parse_ts("2024-01-01 09:34:00"),
            parse_ts("2024-01-01 09:35:00"),
            parse_ts("2024-01-01 09:36:00"),
            parse_ts("2024-01-01 09:37:00"),
        ];
        let close = vec![100.0, 100.2, 99.9, 100.4, 100.0, 100.7, 100.4, 100.9];
        let probs = vec![0.55, 0.6, 0.52, 0.48, 0.61, 0.58, 0.63, 0.57];
        let sides = vec![1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0];
        let asset_names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let asset_prices = DMatrix::from_row_slice(
            8,
            3,
            &[
                100.0, 100.0, 100.0, 100.2, 99.8, 100.1, 100.3, 99.9, 100.2, 100.4, 100.2, 100.1,
                100.3, 100.0, 100.3, 100.6, 100.1, 100.4, 100.7, 100.3, 100.5, 100.8, 100.4, 100.6,
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
        let config =
            ResearchPipelineConfig { cusum_threshold: 0.0005, ..ResearchPipelineConfig::default() };

        let out = run_mid_frequency_pipeline(input, &config).expect("pipeline run should succeed");
        assert!(!out.events.indices.is_empty());
        assert_eq!(out.signals.timeline_signal.len(), close.len());
        assert_eq!(out.backtest.strategy_returns.len(), close.len() - 1);
        assert_eq!(out.backtest.equity_curve.len(), close.len());
        assert_eq!(out.portfolio.weights.len(), asset_names.len());
        let total_weight: f64 = out.portfolio.weights.iter().sum();
        assert!((total_weight - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pipeline_rejects_length_mismatch() {
        let timestamps = vec![
            parse_ts("2024-01-01 09:30:00"),
            parse_ts("2024-01-01 09:31:00"),
            parse_ts("2024-01-01 09:32:00"),
        ];
        let close = vec![100.0, 100.1, 100.2];
        let probs = vec![0.55, 0.6];
        let asset_names = vec!["A".to_string()];
        let asset_prices = DMatrix::from_row_slice(3, 1, &[100.0, 100.1, 100.2]);
        let input = ResearchPipelineInput {
            timestamps: &timestamps,
            close: &close,
            model_probabilities: &probs,
            model_sides: None,
            asset_prices: &asset_prices,
            asset_names: &asset_names,
        };

        let err =
            run_mid_frequency_pipeline(input, &ResearchPipelineConfig::default()).unwrap_err();
        assert_eq!(err, PipelineError::LengthMismatch("model_probabilities", 2, "close", 3));
    }
}
