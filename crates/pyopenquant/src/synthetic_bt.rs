use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::helpers::to_py_err;

fn ou_params_to_dict(py: Python<'_>, p: &openquant::synthetic_backtesting::OuProcessParams) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("phi", p.phi)?;
    d.set_item("intercept", p.intercept)?;
    d.set_item("equilibrium", p.equilibrium)?;
    d.set_item("sigma", p.sigma)?;
    d.set_item("r_squared", p.r_squared)?;
    d.set_item("stationary", p.stationary)?;
    Ok(d.into_pyobject(py).unwrap().into_any().unbind())
}

fn surface_point_to_dict(py: Python<'_>, p: &openquant::synthetic_backtesting::RuleSurfacePoint) -> PyResult<PyObject> {
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

fn diagnostics_to_dict(py: Python<'_>, d_in: &openquant::synthetic_backtesting::StabilityDiagnostics) -> PyResult<PyObject> {
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

fn otr_result_to_dict(py: Python<'_>, r: openquant::synthetic_backtesting::OtrSearchResult) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("params", ou_params_to_dict(py, &r.params)?)?;
    let rule = PyDict::new(py);
    rule.set_item("profit_taking", r.best_rule.profit_taking)?;
    rule.set_item("stop_loss", r.best_rule.stop_loss)?;
    d.set_item("best_rule", rule)?;
    d.set_item("best_point", surface_point_to_dict(py, &r.best_point)?)?;
    let surface: Vec<PyObject> = r.response_surface.iter().map(|p| surface_point_to_dict(py, p)).collect::<PyResult<_>>()?;
    d.set_item("response_surface", surface)?;
    d.set_item("diagnostics", diagnostics_to_dict(py, &r.diagnostics)?)?;
    Ok(d.into_pyobject(py).unwrap().into_any().unbind())
}

#[pyfunction(name = "calibrate_ou_params")]
fn sbt_calibrate_ou_params(py: Python<'_>, prices: Vec<f64>) -> PyResult<PyObject> {
    let params = openquant::synthetic_backtesting::calibrate_ou_params(&prices).map_err(to_py_err)?;
    ou_params_to_dict(py, &params)
}

#[pyfunction(name = "generate_ou_paths")]
fn sbt_generate_ou_paths(
    phi: f64,
    intercept: f64,
    equilibrium: f64,
    sigma: f64,
    r_squared: f64,
    stationary: bool,
    initial_price: f64,
    n_paths: usize,
    horizon: usize,
    seed: u64,
) -> PyResult<Vec<Vec<f64>>> {
    let params = openquant::synthetic_backtesting::OuProcessParams {
        phi, intercept, equilibrium, sigma, r_squared, stationary,
    };
    openquant::synthetic_backtesting::generate_ou_paths(params, initial_price, n_paths, horizon, seed)
        .map_err(to_py_err)
}

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
        &paths, rule, max_holding_steps, annualization_factor,
    )
    .map_err(to_py_err)?;
    surface_point_to_dict(py, &result)
}

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
                rule: openquant::synthetic_backtesting::TradingRule { profit_taking: pt, stop_loss: sl },
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
    let result = openquant::synthetic_backtesting::detect_no_stable_optimum(&surface, estimated_phi, criteria)
        .map_err(to_py_err)?;
    diagnostics_to_dict(py, &result)
}

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
    random_walk_phi_threshold=0.99,
    min_peak_margin=0.1,
    min_surface_std=0.05,
    min_best_sharpe=0.0
))]
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
    random_walk_phi_threshold: f64,
    min_peak_margin: f64,
    min_surface_std: f64,
    min_best_sharpe: f64,
) -> PyResult<PyObject> {
    let config = openquant::synthetic_backtesting::SyntheticBacktestConfig {
        initial_price,
        n_paths,
        horizon,
        seed,
        profit_taking_grid: profit_taking_grid.unwrap_or_else(|| {
            (1..=20).map(|i| i as f64 * 0.25).collect()
        }),
        stop_loss_grid: stop_loss_grid.unwrap_or_else(|| {
            (1..=20).map(|i| i as f64 * -0.25).collect()
        }),
        max_holding_steps,
        annualization_factor,
        stability_criteria: openquant::synthetic_backtesting::StabilityCriteria {
            random_walk_phi_threshold,
            min_peak_margin,
            min_surface_std,
            min_best_sharpe,
        },
    };
    let result = openquant::synthetic_backtesting::run_synthetic_otr_workflow(&historical_prices, &config)
        .map_err(to_py_err)?;
    otr_result_to_dict(py, result)
}

#[pyfunction(name = "search_optimal_trading_rule")]
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
        phi, intercept, equilibrium, sigma, r_squared, stationary,
    };
    let criteria = openquant::synthetic_backtesting::StabilityCriteria {
        random_walk_phi_threshold,
        min_peak_margin,
        min_surface_std,
        min_best_sharpe,
    };
    let result = openquant::synthetic_backtesting::search_optimal_trading_rule(
        params, &paths, &profit_taking_grid, &stop_loss_grid,
        max_holding_steps, annualization_factor, criteria,
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
