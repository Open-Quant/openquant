use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::helpers::{to_py_err, warn_deprecated};

fn ou_params_to_dict(
    py: Python<'_>,
    p: &openquant::synthetic_backtesting::OuProcessParams,
) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("phi", p.phi)?;
    d.set_item("intercept", p.intercept)?;
    d.set_item("equilibrium", p.equilibrium)?;
    d.set_item("sigma", p.sigma)?;
    d.set_item("r_squared", p.r_squared)?;
    d.set_item("stationary", p.stationary)?;
    Ok(d.into_pyobject(py).unwrap().into_any().unbind())
}

fn surface_point_to_dict(
    py: Python<'_>,
    p: &openquant::synthetic_backtesting::RuleSurfacePoint,
) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("profit_taking", p.rule.profit_taking)?;
    d.set_item("stop_loss", p.rule.stop_loss)?;
    d.set_item("sharpe", p.sharpe)?;
    d.set_item("mean_return", p.mean_return)?;
    d.set_item("std_return", p.std_return)?;
    d.set_item("win_rate", p.win_rate)?;
    d.set_item("avg_holding_steps", p.avg_holding_steps)?;
    Ok(d.into_pyobject(py).unwrap().into_any().unbind())
}

fn diagnostics_to_dict(
    py: Python<'_>,
    d_in: &openquant::synthetic_backtesting::StabilityDiagnostics,
) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("no_stable_optimum", d_in.no_stable_optimum)?;
    d.set_item("reason", &d_in.reason)?;
    d.set_item("best_sharpe", d_in.best_sharpe)?;
    d.set_item("median_sharpe", d_in.median_sharpe)?;
    d.set_item("peak_margin", d_in.peak_margin)?;
    d.set_item("surface_std", d_in.surface_std)?;
    d.set_item("estimated_phi", d_in.estimated_phi)?;
    Ok(d.into_pyobject(py).unwrap().into_any().unbind())
}

fn otr_result_to_dict(
    py: Python<'_>,
    r: openquant::synthetic_backtesting::OtrSearchResult,
) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("params", ou_params_to_dict(py, &r.params)?)?;
    let rule = PyDict::new(py);
    rule.set_item("profit_taking", r.best_rule.profit_taking)?;
    rule.set_item("stop_loss", r.best_rule.stop_loss)?;
    d.set_item("best_rule", rule)?;
    d.set_item("best_point", surface_point_to_dict(py, &r.best_point)?)?;
    let surface: Vec<PyObject> =
        r.response_surface.iter().map(|p| surface_point_to_dict(py, p)).collect::<PyResult<_>>()?;
    d.set_item("response_surface", surface)?;
    d.set_item("diagnostics", diagnostics_to_dict(py, &r.diagnostics)?)?;
    Ok(d.into_pyobject(py).unwrap().into_any().unbind())
}

/// Fit a discrete Ornstein-Uhlenbeck (AR(1)) process to a price series.
///
/// AFML §13.5.1, step 1. OLS of `P_t` on `P_{t-1}` gives
/// `P_t = intercept + phi * P_{t-1} + sigma * eps_t`; `sigma` is the sample standard
/// deviation of the residuals and `equilibrium` is `intercept / (1 - phi)`, or the mean price
/// when `phi` is within `1e-12` of 1.
///
/// Parameters
/// ----------
/// prices : list[float]
///     Price levels, oldest first; at least three, all finite.
///
/// Returns
/// -------
/// dict[str, Any]
///     Keys `phi`, `intercept`, `equilibrium`, `sigma`, `r_squared` (floats) and
///     `stationary` (bool, `abs(phi) < 1`, true even for `phi = 0.999`).
///
/// Raises
/// ------
/// ValueError
///     If there are fewer than three prices, a price is not finite, the lagged prices are
///     constant, or the fit is exact (zero residual deviation).
#[pyfunction(name = "calibrate_ou_params")]
fn sbt_calibrate_ou_params(py: Python<'_>, prices: Vec<f64>) -> PyResult<PyObject> {
    let params =
        openquant::synthetic_backtesting::calibrate_ou_params(&prices).map_err(to_py_err)?;
    ou_params_to_dict(py, &params)
}

/// Simulate O-U price paths from given process parameters.
///
/// AFML §13.5.1, step 2. Each path starts at `initial_price` and evolves as
/// `P_t = intercept + phi * P_{t-1} + sigma * eps_t` with standard-normal innovations from an
/// RNG seeded with `seed`, so the same seed reproduces the same paths. Only `intercept`, `phi`
/// and `sigma` drive the simulation; when building parameters by hand keep
/// `intercept = (1 - phi) * equilibrium`. Pass `initial_price`, `n_paths`, `horizon` and
/// `seed` by keyword; their defaults are those of `run_synthetic_otr_workflow`.
///
/// Parameters
/// ----------
/// phi : float
///     Autoregressive coefficient.
/// intercept : float
///     Regression intercept.
/// equilibrium : float
///     Long-run mean; must be finite but is not used by the simulation.
/// sigma : float
///     Innovation standard deviation in price units; must be >= 0.
/// r_squared : float | None, default None
///     Deprecated: passing it emits a `DeprecationWarning`. It is a diagnostic of the
///     calibration regression, not a parameter of the process, so it cannot affect the paths
///     (AFML §13.5.1 simulates from the fitted coefficients and residual deviation alone).
/// stationary : bool | None, default None
///     Deprecated: passing it emits a `DeprecationWarning`. It is `abs(phi) < 1`, which `phi`
///     already determines. To simulate a calibrated fit, pass its `phi`, `intercept`,
///     `equilibrium` and `sigma` only.
/// initial_price : float, default 100.0
///     Entry price of every path.
/// n_paths : int, default 1000
///     Number of paths; must be > 0.
/// horizon : int, default 252
///     Points per path, including the entry; must be >= 2.
/// seed : int, default 42
///     Seed of the simulation RNG.
///
/// Returns
/// -------
/// list[list[float]]
///     `n_paths` paths of `horizon` prices each, each starting at `initial_price`.
///
/// Raises
/// ------
/// ValueError
///     If `initial_price` is not finite, `n_paths` is 0, `horizon < 2`, `phi`, `intercept`,
///     `equilibrium` or `sigma` is not finite, or `sigma` is negative.
#[pyfunction(name = "generate_ou_paths")]
#[pyo3(signature = (
    phi,
    intercept,
    equilibrium,
    sigma,
    r_squared=None,
    stationary=None,
    initial_price=100.0,
    n_paths=1000,
    horizon=252,
    seed=42
))]
// Python keyword signature.
#[allow(clippy::too_many_arguments)]
fn sbt_generate_ou_paths(
    py: Python<'_>,
    phi: f64,
    intercept: f64,
    equilibrium: f64,
    sigma: f64,
    r_squared: Option<f64>,
    stationary: Option<bool>,
    initial_price: f64,
    n_paths: usize,
    horizon: usize,
    seed: u64,
) -> PyResult<Vec<Vec<f64>>> {
    if r_squared.is_some() || stationary.is_some() {
        warn_deprecated(
            py,
            "generate_ou_paths: r_squared and stationary are deprecated and have no effect on \
             the simulation; pass phi, intercept, equilibrium and sigma, and initial_price, \
             n_paths, horizon and seed by keyword",
        )?;
    }
    let params = openquant::synthetic_backtesting::OuProcessParams {
        phi,
        intercept,
        equilibrium,
        sigma,
        // Neither is read by the simulation.
        r_squared: r_squared.unwrap_or(f64::NAN),
        stationary: stationary.unwrap_or(phi.abs() < 1.0),
    };
    openquant::synthetic_backtesting::generate_ou_paths(
        params,
        initial_price,
        n_paths,
        horizon,
        seed,
    )
    .map_err(to_py_err)
}

/// Evaluate one profit-taking/stop-loss rule on every path.
///
/// AFML §13.5.1, step 3. Each path is a long entry at its first price; the trade exits at the
/// first step (up to `min(max_holding_steps, len(path) - 1)`) whose PnL is `>= profit_taking`
/// or `<= -stop_loss`, otherwise at that last step. Barriers are in price units from the
/// entry. The Sharpe ratio is `mean_return / std_return * sqrt(annualization_factor)` over
/// per-trade PnL, regardless of holding time (0 when the deviation is 0).
///
/// Parameters
/// ----------
/// paths : list[list[float]]
///     Price paths, e.g. from `generate_ou_paths`; each needs at least two finite points.
/// profit_taking : float
///     Profit-taking width in price units; must be > 0.
/// stop_loss : float
///     Stop-loss width in price units, as a positive number; must be > 0.
/// max_holding_steps : int
///     Maximum holding period in steps; must be > 0.
/// annualization_factor : float
///     Multiplier whose square root scales the Sharpe ratio (1.0 for per trade); finite, > 0.
///
/// Returns
/// -------
/// dict[str, Any]
///     Keys `profit_taking`, `stop_loss`, `sharpe`, `mean_return` (mean per-trade PnL),
///     `std_return` (sample standard deviation of PnL), `win_rate` (share of trades with
///     positive PnL) and `avg_holding_steps`, all floats.
///
/// Raises
/// ------
/// ValueError
///     If `paths` is empty, a barrier is not positive, `max_holding_steps` is 0,
///     `annualization_factor` is not finite and positive, a path has fewer than two points, or
///     a path contains a non-finite value.
#[pyfunction(name = "evaluate_rule_on_paths")]
fn sbt_evaluate_rule_on_paths(
    py: Python<'_>,
    paths: Vec<Vec<f64>>,
    profit_taking: f64,
    stop_loss: f64,
    max_holding_steps: usize,
    annualization_factor: f64,
) -> PyResult<PyObject> {
    let rule = openquant::synthetic_backtesting::TradingRule { profit_taking, stop_loss };
    let result = openquant::synthetic_backtesting::evaluate_rule_on_paths(
        &paths,
        rule,
        max_holding_steps,
        annualization_factor,
    )
    .map_err(to_py_err)?;
    surface_point_to_dict(py, &result)
}

/// Flag a PT/SL response surface that lacks a stable optimum (the flattening of AFML §13.6).
///
/// The verdict is true when the process is near a random walk
/// (`abs(estimated_phi) >= random_walk_phi_threshold`) and the peak is weak
/// (`best - median Sharpe < min_peak_margin`) or the surface flat (Sharpe standard deviation
/// `< min_surface_std`), or when the best Sharpe is below `min_best_sharpe` and the peak is
/// weak. The thresholds are this library's heuristics, not AFML's; the Rust defaults are
/// 0.97, 0.20, 0.10 and 0.30.
///
/// Parameters
/// ----------
/// response_surface : list[tuple[float, float, float, float, float, float, float]]
///     One `(profit_taking, stop_loss, sharpe, mean_return, std_return, win_rate,
///     avg_holding_steps)` row per grid point; only `sharpe` enters the diagnosis.
/// estimated_phi : float
///     The AR(1) coefficient of the process, e.g. from `calibrate_ou_params`.
/// random_walk_phi_threshold : float
///     `abs(phi)` at or above this counts as near a random walk.
/// min_peak_margin : float
///     Best-minus-median Sharpe below this is a weak peak.
/// min_surface_std : float
///     Sharpe standard deviation below this is a flat surface.
/// min_best_sharpe : float
///     Best Sharpe below this is a weak best rule.
///
/// Returns
/// -------
/// dict[str, Any]
///     Keys `no_stable_optimum` (bool), `reason` (str), `best_sharpe`, `median_sharpe`,
///     `peak_margin`, `surface_std` and `estimated_phi` (floats).
///
/// Raises
/// ------
/// ValueError
///     If `response_surface` is empty.
#[pyfunction(name = "detect_no_stable_optimum")]
fn sbt_detect_no_stable_optimum(
    py: Python<'_>,
    response_surface: Vec<(f64, f64, f64, f64, f64, f64, f64)>,
    estimated_phi: f64,
    random_walk_phi_threshold: f64,
    min_peak_margin: f64,
    min_surface_std: f64,
    min_best_sharpe: f64,
) -> PyResult<PyObject> {
    let surface: Vec<openquant::synthetic_backtesting::RuleSurfacePoint> = response_surface
        .into_iter()
        .map(|(pt, sl, sharpe, mean_ret, std_ret, win_rate, avg_hold)| {
            openquant::synthetic_backtesting::RuleSurfacePoint {
                rule: openquant::synthetic_backtesting::TradingRule {
                    profit_taking: pt,
                    stop_loss: sl,
                },
                sharpe,
                mean_return: mean_ret,
                std_return: std_ret,
                win_rate,
                avg_holding_steps: avg_hold,
            }
        })
        .collect();
    let criteria = openquant::synthetic_backtesting::StabilityCriteria {
        random_walk_phi_threshold,
        min_peak_margin,
        min_surface_std,
        min_best_sharpe,
    };
    let result = openquant::synthetic_backtesting::detect_no_stable_optimum(
        &surface,
        estimated_phi,
        criteria,
    )
    .map_err(to_py_err)?;
    diagnostics_to_dict(py, &result)
}

/// End-to-end optimal-trading-rule search on synthetic O-U paths.
///
/// AFML §13.4-13.5 (Snippets 13.1-13.2): calibrates an O-U process on `historical_prices`
/// (`calibrate_ou_params`), simulates `n_paths` paths from `initial_price`
/// (`generate_ou_paths`), evaluates every `(profit_taking, stop_loss)` pair of the grids
/// (`evaluate_rule_on_paths`), and diagnoses the surface (`detect_no_stable_optimum`, with
/// the calibrated `phi`). Barriers are in price units; every rule is a long entry.
///
/// Parameters
/// ----------
/// historical_prices : list[float]
///     Price levels to calibrate on, oldest first; at least three, all finite.
/// initial_price : float, default 100.0
///     Entry price of every simulated path.
/// n_paths : int, default 1000
///     Number of simulated paths.
/// horizon : int, default 252
///     Points per path, including the entry; must be >= 2.
/// seed : int, default 42
///     Seed of the simulation RNG.
/// profit_taking_grid : list[float] | None, default None
///     Profit-taking widths to try, each finite and > 0. `None` uses `0.25, 0.5, ..., 5.0`.
/// stop_loss_grid : list[float] | None, default None
///     Stop-loss widths to try, as positive numbers (distance below the entry), each finite
///     and > 0. `None` uses `0.25, 0.5, ..., 5.0`.
/// max_holding_steps : int, default 252
///     Maximum holding period in steps (capped by the path length).
/// annualization_factor : float, default 252.0
///     Multiplier whose square root scales each rule's Sharpe ratio.
/// random_walk_phi_threshold : float | None, default None
///     See `detect_no_stable_optimum`; `None` uses the Rust default 0.97.
/// min_peak_margin : float | None, default None
///     See `detect_no_stable_optimum`; `None` uses the Rust default 0.20.
/// min_surface_std : float | None, default None
///     See `detect_no_stable_optimum`; `None` uses the Rust default 0.10.
/// min_best_sharpe : float | None, default None
///     See `detect_no_stable_optimum`; `None` uses the Rust default 0.30.
///
/// Returns
/// -------
/// dict[str, Any]
///     Keys `params` (dict as returned by `calibrate_ou_params`), `best_rule` (dict with
///     `profit_taking` and `stop_loss`), `best_point` (dict as returned by
///     `evaluate_rule_on_paths`), `response_surface` (list of such dicts, one per grid point,
///     sorted by Sharpe then mean return, best first) and `diagnostics` (dict as returned by
///     `detect_no_stable_optimum`).
///
/// Raises
/// ------
/// ValueError
///     If a grid is empty or has a value that is not finite and positive, or calibration,
///     simulation or evaluation rejects its input (see `calibrate_ou_params`,
///     `generate_ou_paths` and `evaluate_rule_on_paths`).
#[pyfunction(name = "run_synthetic_otr_workflow")]
#[pyo3(signature = (
    historical_prices,
    initial_price=100.0,
    n_paths=1000,
    horizon=252,
    seed=42,
    profit_taking_grid=None,
    stop_loss_grid=None,
    max_holding_steps=252,
    annualization_factor=252.0,
    random_walk_phi_threshold=None,
    min_peak_margin=None,
    min_surface_std=None,
    min_best_sharpe=None
))]
// Python keyword signature.
#[allow(clippy::too_many_arguments)]
fn sbt_run_synthetic_otr_workflow(
    py: Python<'_>,
    historical_prices: Vec<f64>,
    initial_price: f64,
    n_paths: usize,
    horizon: usize,
    seed: u64,
    profit_taking_grid: Option<Vec<f64>>,
    stop_loss_grid: Option<Vec<f64>>,
    max_holding_steps: usize,
    annualization_factor: f64,
    random_walk_phi_threshold: Option<f64>,
    min_peak_margin: Option<f64>,
    min_surface_std: Option<f64>,
    min_best_sharpe: Option<f64>,
) -> PyResult<PyObject> {
    // Omitted thresholds fall back to the Rust defaults, so the two surfaces cannot drift.
    let defaults = openquant::synthetic_backtesting::StabilityCriteria::default();
    let config = openquant::synthetic_backtesting::SyntheticBacktestConfig {
        initial_price,
        n_paths,
        horizon,
        seed,
        profit_taking_grid: profit_taking_grid
            .unwrap_or_else(|| (1..=20).map(|i| i as f64 * 0.25).collect()),
        stop_loss_grid: stop_loss_grid
            .unwrap_or_else(|| (1..=20).map(|i| i as f64 * 0.25).collect()),
        max_holding_steps,
        annualization_factor,
        stability_criteria: openquant::synthetic_backtesting::StabilityCriteria {
            random_walk_phi_threshold: random_walk_phi_threshold
                .unwrap_or(defaults.random_walk_phi_threshold),
            min_peak_margin: min_peak_margin.unwrap_or(defaults.min_peak_margin),
            min_surface_std: min_surface_std.unwrap_or(defaults.min_surface_std),
            min_best_sharpe: min_best_sharpe.unwrap_or(defaults.min_best_sharpe),
        },
    };
    let result =
        openquant::synthetic_backtesting::run_synthetic_otr_workflow(&historical_prices, &config)
            .map_err(to_py_err)?;
    otr_result_to_dict(py, result)
}

/// Search a PT/SL grid on given paths and diagnose the response surface.
///
/// AFML §13.5.1 (Snippets 13.1-13.2). Evaluates every `(profit_taking, stop_loss)` pair of
/// the grids with `evaluate_rule_on_paths`, sorts the surface by Sharpe ratio (then mean
/// return), and runs `detect_no_stable_optimum` with `phi` as the estimated coefficient. The
/// process parameters are otherwise only echoed back in the result's `params`.
///
/// Parameters
/// ----------
/// phi : float
///     AR(1) coefficient used by the stability diagnosis.
/// intercept : float
///     Echoed in `params`.
/// equilibrium : float
///     Echoed in `params`.
/// sigma : float
///     Echoed in `params`.
/// r_squared : float
///     Echoed in `params`.
/// stationary : bool
///     Echoed in `params`.
/// paths : list[list[float]]
///     Price paths, e.g. from `generate_ou_paths`; each needs at least two finite points.
/// profit_taking_grid : list[float]
///     Profit-taking widths to try, in price units, each > 0.
/// stop_loss_grid : list[float]
///     Stop-loss widths to try, as positive numbers in price units, each > 0.
/// max_holding_steps : int
///     Maximum holding period in steps; must be > 0.
/// annualization_factor : float
///     Multiplier whose square root scales each rule's Sharpe ratio; finite and > 0.
/// random_walk_phi_threshold : float
///     See `detect_no_stable_optimum`.
/// min_peak_margin : float
///     See `detect_no_stable_optimum`.
/// min_surface_std : float
///     See `detect_no_stable_optimum`.
/// min_best_sharpe : float
///     See `detect_no_stable_optimum`.
///
/// Returns
/// -------
/// dict[str, Any]
///     Keys `params`, `best_rule`, `best_point`, `response_surface` and `diagnostics`, as for
///     `run_synthetic_otr_workflow`.
///
/// Raises
/// ------
/// ValueError
///     If either grid is empty, a grid value is not positive, or `evaluate_rule_on_paths`
///     rejects its input.
#[pyfunction(name = "search_optimal_trading_rule")]
// Python keyword signature.
#[allow(clippy::too_many_arguments)]
fn sbt_search_optimal_trading_rule(
    py: Python<'_>,
    phi: f64,
    intercept: f64,
    equilibrium: f64,
    sigma: f64,
    r_squared: f64,
    stationary: bool,
    paths: Vec<Vec<f64>>,
    profit_taking_grid: Vec<f64>,
    stop_loss_grid: Vec<f64>,
    max_holding_steps: usize,
    annualization_factor: f64,
    random_walk_phi_threshold: f64,
    min_peak_margin: f64,
    min_surface_std: f64,
    min_best_sharpe: f64,
) -> PyResult<PyObject> {
    let params = openquant::synthetic_backtesting::OuProcessParams {
        phi,
        intercept,
        equilibrium,
        sigma,
        r_squared,
        stationary,
    };
    let criteria = openquant::synthetic_backtesting::StabilityCriteria {
        random_walk_phi_threshold,
        min_peak_margin,
        min_surface_std,
        min_best_sharpe,
    };
    let result = openquant::synthetic_backtesting::search_optimal_trading_rule(
        params,
        &paths,
        &profit_taking_grid,
        &stop_loss_grid,
        max_holding_steps,
        annualization_factor,
        criteria,
    )
    .map_err(to_py_err)?;
    otr_result_to_dict(py, result)
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "synthetic_bt")?;
    m.add_function(wrap_pyfunction!(sbt_calibrate_ou_params, &m)?)?;
    m.add_function(wrap_pyfunction!(sbt_generate_ou_paths, &m)?)?;
    m.add_function(wrap_pyfunction!(sbt_evaluate_rule_on_paths, &m)?)?;
    m.add_function(wrap_pyfunction!(sbt_detect_no_stable_optimum, &m)?)?;
    m.add_function(wrap_pyfunction!(sbt_run_synthetic_otr_workflow, &m)?)?;
    m.add_function(wrap_pyfunction!(sbt_search_optimal_trading_rule, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("synthetic_bt", m)?;
    Ok(())
}
