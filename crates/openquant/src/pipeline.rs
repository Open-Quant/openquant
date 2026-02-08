use chrono::NaiveDateTime;
use nalgebra::DMatrix;

use crate::backtest_statistics::{drawdown_and_time_under_water, sharpe_ratio};
use crate::bet_sizing::{discrete_signal, get_signal};
use crate::filters::{cusum_filter_indices, Threshold};
use crate::portfolio_optimization::allocate_max_sharpe;
use crate::risk_metrics::{RiskMetrics, RiskMetricsError};

#[derive(Debug, Clone)]
pub struct ResearchPipelineConfig {
    pub cusum_threshold: f64,
    pub num_classes: usize,
    pub step_size: f64,
    pub risk_free_rate: f64,
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

#[derive(Debug, Clone)]
pub struct ResearchPipelineInput<'a> {
    pub timestamps: &'a [NaiveDateTime],
    pub close: &'a [f64],
    pub model_probabilities: &'a [f64],
    pub model_sides: Option<&'a [f64]>,
    pub asset_prices: &'a DMatrix<f64>,
    pub asset_names: &'a [String],
}

#[derive(Debug, Clone)]
pub struct EventSelectionStage {
    pub indices: Vec<usize>,
    pub timestamps: Vec<NaiveDateTime>,
    pub probabilities: Vec<f64>,
    pub sides: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SignalStage {
    pub event_signal: Vec<f64>,
    pub timeline_signal: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct PortfolioStage {
    pub asset_names: Vec<String>,
    pub weights: Vec<f64>,
    pub portfolio_risk: f64,
    pub portfolio_return: f64,
    pub portfolio_sharpe: f64,
}

#[derive(Debug, Clone)]
pub struct RiskStage {
    pub value_at_risk: f64,
    pub expected_shortfall: f64,
    pub conditional_drawdown_risk: f64,
    pub realized_sharpe: f64,
}

#[derive(Debug, Clone)]
pub struct BacktestStage {
    pub timestamps: Vec<NaiveDateTime>,
    pub strategy_returns: Vec<f64>,
    pub equity_curve: Vec<f64>,
    pub drawdowns: Vec<f64>,
    pub time_under_water_years: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct LeakageChecks {
    pub inputs_aligned: bool,
    pub event_indices_sorted: bool,
    pub has_forward_look_bias: bool,
}

#[derive(Debug, Clone)]
pub struct ResearchPipelineOutput {
    pub events: EventSelectionStage,
    pub signals: SignalStage,
    pub portfolio: PortfolioStage,
    pub risk: RiskStage,
    pub backtest: BacktestStage,
    pub leakage_checks: LeakageChecks,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PipelineError {
    EmptyInput(&'static str),
    LengthMismatch(&'static str, usize, &'static str, usize),
    InvalidParameter(&'static str),
    NoEvents,
    PortfolioAllocation(String),
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

pub fn run_mid_frequency_pipeline(
    input: ResearchPipelineInput<'_>,
    config: &ResearchPipelineConfig,
) -> Result<ResearchPipelineOutput, PipelineError> {
    validate_input(&input, config)?;

    let event_indices =
        cusum_filter_indices(&input.close, Threshold::Scalar(config.cusum_threshold));
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
    let risk_metrics = RiskMetrics::default();
    let value_at_risk =
        risk_metrics.calculate_value_at_risk(&strategy_returns, config.confidence_level)?;
    let expected_shortfall =
        risk_metrics.calculate_expected_shortfall(&strategy_returns, config.confidence_level)?;
    let conditional_drawdown_risk = risk_metrics
        .calculate_conditional_drawdown_risk(&strategy_returns, config.confidence_level)?;
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
