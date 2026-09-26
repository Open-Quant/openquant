use openquant::pipeline::{
    run_mid_frequency_pipeline, ResearchPipelineConfig, ResearchPipelineInput,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::helpers::{format_naive_datetimes, matrix_from_rows, parse_naive_datetimes, to_py_err};

/// Run the events, signals, portfolio, risk and backtest research pipeline in one call.
///
/// Chains existing modules on one instrument: (1) the symmetric CUSUM filter on `close`
/// selects events (AFML Snippet 2.4); (2) the model probability and side at each event
/// become a bet size `2 Phi(z) - 1` (AFML Snippet 10.1), rounded to multiples of
/// `step_size` (Snippet 10.3) and held on every bar until the next event; (3) max-Sharpe
/// mean-variance weights of `asset_prices` are computed (Markowitz, not AFML); (4)
/// historical VaR, expected shortfall and conditional drawdown at risk of the strategy are
/// computed, plus an annualised Sharpe ratio (AFML section 14.7.1); (5) the equity curve,
/// drawdowns and time under water are built (AFML Snippet 14.4). No labelling or model
/// fitting happens here; probabilities and sides are inputs, one per bar.
///
/// The signal is applied with a one-bar lag: the strategy return over bar `i` is
/// `signal[i - 1] * (close[i] / close[i - 1] - 1)`. The portfolio stage is independent of
/// the backtest (its weights are reported, not traded) and annualises with 252 periods, so
/// `risk_free_rate` is an annual rate there but a per-bar rate in `realized_sharpe`; only 0
/// means the same in both. `confidence_level` is the lower-tail probability for VaR and
/// expected shortfall (0.05 = worst 5% of per-bar returns); CDaR uses
/// `1 - confidence_level`.
///
/// Parameters
/// ----------
/// timestamps : list[str]
///     Bar timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is
///     accepted), oldest first. Their order is not checked.
/// close : list[float]
///     Positive closing prices of the traded instrument, one per bar.
/// model_probabilities : list[float]
///     Probability of the predicted class at each bar, in `[0, 1]`, known at that bar's
///     close. Only values at CUSUM events are used, but every value must be in `[0, 1]`.
/// asset_prices : list[list[float]]
///     Prices for the portfolio stage, one inner list per observation (oldest first, at
///     least 2 rows, one per entry of `timestamps`) and one column per asset.
/// model_sides : list[float] | None, default None
///     Side of the prediction at each bar (typically +1/-1); None means always long.
/// asset_names : list[str] | None, default None
///     One name per column of `asset_prices`; defaults to `asset_0`, `asset_1`, ...
/// cusum_threshold : float, default 0.001
///     CUSUM threshold on cumulative log returns of `close`; must be finite and > 0.
/// num_classes : int, default 2
///     Number of classes of the model behind `model_probabilities`; must be >= 2.
/// step_size : float, default 0.1
///     Bet sizes are rounded to multiples of this step and clamped to `[-1, 1]`; must be
///     finite and > 0.
/// risk_free_rate : float, default 0.0
///     Annual rate for the max-Sharpe allocation, per-bar rate for `realized_sharpe`.
/// confidence_level : float, default 0.05
///     Lower-tail probability for VaR and expected shortfall, in `[0, 1]`.
///
/// Returns
/// -------
/// dict[str, Any]
///     A dict of stage dicts:
///
///     - `events`: `indices` (0-based bar positions of the CUSUM events), `timestamps`,
///       `probabilities` and `sides` (1.0 when no sides were given) at those events.
///     - `signals`: `timestamps` (the input bar timestamps), `values` (bet size on every
///       bar, 0 before the first event) and `event_signal` (discretised bet size in
///       `[-1, 1]` per event).
///     - `portfolio`: `asset_names`, `weights` (sum to 1), `portfolio_risk`,
///       `portfolio_return` and `portfolio_sharpe` (all annualised).
///     - `risk`: `value_at_risk` (signed per-bar return, negative is a loss),
///       `expected_shortfall` (mean of returns strictly below VaR, NaN when none are),
///       `conditional_drawdown_risk` (in equity units) and `realized_sharpe`
///       (annualised with 252 bars a year).
///     - `backtest`: `timestamps`, `strategy_returns` (one shorter than `close`),
///       `equity_curve` (starts at 1), `drawdowns` and `time_under_water_years` (one per
///       drawdown, in 365.25-day years).
///     - `leakage_checks`: booleans `inputs_aligned`, `event_indices_sorted` and
///       `has_forward_look_bias`. These are structural and take fixed values (True, True,
///       False); they do not detect look-ahead in the caller's probabilities or sides.
///
/// Raises
/// ------
/// ValueError
///     If a value is outside the range given under Parameters (a probability outside
///     `[0, 1]`, `cusum_threshold` or `step_size` not finite and > 0, or `asset_prices`
///     without one row per timestamp); a timestamp does not parse; `asset_prices` is empty
///     or ragged; `timestamps`,
///     `close` or `model_probabilities` is empty; `close` differs in length from
///     `timestamps`, `model_probabilities` or `model_sides`, or `asset_names` from the
///     number of assets; `asset_prices` has fewer than 2 rows, `cusum_threshold <= 0`,
///     `num_classes < 2` or `confidence_level` is outside `[0, 1]`; the CUSUM filter
///     finds no event; or the max-Sharpe optimisation fails.
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
// Python keyword signature.
#[allow(clippy::too_many_arguments)]
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
