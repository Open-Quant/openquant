use openquant::pipeline::{run_mid_frequency_pipeline, ResearchPipelineConfig, ResearchPipelineInput};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::helpers::{format_naive_datetimes, matrix_from_rows, parse_naive_datetimes, to_py_err};

#[pyfunction(name = "run_mid_frequency_pipeline")]
#[pyo3(signature = (
    timestamps,
    close,
    model_probabilities,
    asset_prices,
    model_sides=None,
    asset_names=None,
    cusum_threshold=0.001,
    num_classes=2,
    step_size=0.1,
    risk_free_rate=0.0,
    confidence_level=0.05
))]
fn pipeline_run_mid_frequency_pipeline(
    py: Python<'_>,
    timestamps: Vec<String>,
    close: Vec<f64>,
    model_probabilities: Vec<f64>,
    asset_prices: Vec<Vec<f64>>,
    model_sides: Option<Vec<f64>>,
    asset_names: Option<Vec<String>>,
    cusum_threshold: f64,
    num_classes: usize,
    step_size: f64,
    risk_free_rate: f64,
    confidence_level: f64,
) -> PyResult<PyObject> {
    let timestamps = parse_naive_datetimes(timestamps)?;
    let asset_prices = matrix_from_rows(asset_prices)?;

    let asset_names = asset_names.unwrap_or_else(|| {
        (0..asset_prices.ncols()).map(|i| format!("asset_{i}")).collect::<Vec<_>>()
    });

    let input = ResearchPipelineInput {
        timestamps: &timestamps,
        close: &close,
        model_probabilities: &model_probabilities,
        model_sides: model_sides.as_deref(),
        asset_prices: &asset_prices,
        asset_names: &asset_names,
    };
    let config = ResearchPipelineConfig {
        cusum_threshold,
        num_classes,
        step_size,
        risk_free_rate,
        confidence_level,
    };
    let out = run_mid_frequency_pipeline(input, &config).map_err(to_py_err)?;

    let root = PyDict::new(py);

    let events = PyDict::new(py);
    events.set_item("indices", out.events.indices)?;
    events.set_item("timestamps", format_naive_datetimes(out.events.timestamps))?;
    events.set_item("probabilities", out.events.probabilities)?;
    events.set_item("sides", out.events.sides)?;
    root.set_item("events", events)?;

    let signals = PyDict::new(py);
    signals.set_item("timestamps", format_naive_datetimes(timestamps.clone()))?;
    signals.set_item("values", out.signals.timeline_signal)?;
    signals.set_item("event_signal", out.signals.event_signal)?;
    root.set_item("signals", signals)?;

    let portfolio = PyDict::new(py);
    portfolio.set_item("asset_names", out.portfolio.asset_names)?;
    portfolio.set_item("weights", out.portfolio.weights)?;
    portfolio.set_item("portfolio_risk", out.portfolio.portfolio_risk)?;
    portfolio.set_item("portfolio_return", out.portfolio.portfolio_return)?;
    portfolio.set_item("portfolio_sharpe", out.portfolio.portfolio_sharpe)?;
    root.set_item("portfolio", portfolio)?;

    let risk = PyDict::new(py);
    risk.set_item("value_at_risk", out.risk.value_at_risk)?;
    risk.set_item("expected_shortfall", out.risk.expected_shortfall)?;
    risk.set_item("conditional_drawdown_risk", out.risk.conditional_drawdown_risk)?;
    risk.set_item("realized_sharpe", out.risk.realized_sharpe)?;
    root.set_item("risk", risk)?;

    let backtest = PyDict::new(py);
    backtest.set_item("timestamps", format_naive_datetimes(out.backtest.timestamps))?;
    backtest.set_item("strategy_returns", out.backtest.strategy_returns)?;
    backtest.set_item("equity_curve", out.backtest.equity_curve)?;
    backtest.set_item("drawdowns", out.backtest.drawdowns)?;
    backtest.set_item("time_under_water_years", out.backtest.time_under_water_years)?;
    root.set_item("backtest", backtest)?;

    let leakage_checks = PyDict::new(py);
    leakage_checks.set_item("inputs_aligned", out.leakage_checks.inputs_aligned)?;
    leakage_checks.set_item("event_indices_sorted", out.leakage_checks.event_indices_sorted)?;
    leakage_checks.set_item("has_forward_look_bias", out.leakage_checks.has_forward_look_bias)?;
    root.set_item("leakage_checks", leakage_checks)?;

    Ok(root.into_pyobject(py).unwrap().into_any().unbind())
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "pipeline")?;
    m.add_function(wrap_pyfunction!(pipeline_run_mid_frequency_pipeline, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("pipeline", m)?;
    Ok(())
}
