//! Synthetic-data backtesting utilities aligned to AFML Chapter 13.
//!
//! This module focuses on optimal trading-rule (OTR) search over profit-taking
//! and stop-loss corridors by:
//! 1) calibrating an AR(1)/discrete O-U process from historical prices,
//! 2) generating many synthetic paths under that calibrated process,
//! 3) evaluating a PT/SL mesh on those paths, and
//! 4) detecting when the response surface lacks a stable optimum.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OuProcessParams {
    pub phi: f64,
    pub intercept: f64,
    pub equilibrium: f64,
    pub sigma: f64,
    pub r_squared: f64,
    pub stationary: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TradingRule {
    pub profit_taking: f64,
    pub stop_loss: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuleSurfacePoint {
    pub rule: TradingRule,
    pub sharpe: f64,
    pub mean_return: f64,
    pub std_return: f64,
    pub win_rate: f64,
    pub avg_holding_steps: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StabilityCriteria {
    pub random_walk_phi_threshold: f64,
    pub min_peak_margin: f64,
    pub min_surface_std: f64,
    pub min_best_sharpe: f64,
}

impl Default for StabilityCriteria {
    fn default() -> Self {
        Self {
            random_walk_phi_threshold: 0.97,
            min_peak_margin: 0.20,
            min_surface_std: 0.10,
            min_best_sharpe: 0.30,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StabilityDiagnostics {
    pub no_stable_optimum: bool,
    pub reason: String,
    pub best_sharpe: f64,
    pub median_sharpe: f64,
    pub peak_margin: f64,
    pub surface_std: f64,
    pub estimated_phi: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OtrSearchResult {
    pub params: OuProcessParams,
    pub best_rule: TradingRule,
    pub best_point: RuleSurfacePoint,
    pub response_surface: Vec<RuleSurfacePoint>,
    pub diagnostics: StabilityDiagnostics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticBacktestConfig {
    pub initial_price: f64,
    pub n_paths: usize,
    pub horizon: usize,
    pub seed: u64,
    pub profit_taking_grid: Vec<f64>,
    pub stop_loss_grid: Vec<f64>,
    pub max_holding_steps: usize,
    pub annualization_factor: f64,
    pub stability_criteria: StabilityCriteria,
}

pub fn calibrate_ou_params(prices: &[f64]) -> Result<OuProcessParams, String> {
    if prices.len() < 3 {
        return Err("prices must include at least 3 observations".to_string());
    }
    if prices.iter().any(|p| !p.is_finite()) {
        return Err("prices must be finite".to_string());
    }

    let n = prices.len() - 1;
    let x = &prices[..n];
    let y = &prices[1..];

    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;

    let mut var_x = 0.0;
    let mut cov_xy = 0.0;
    for i in 0..n {
        let dx = x[i] - mean_x;
        var_x += dx * dx;
        cov_xy += dx * (y[i] - mean_y);
    }
    if var_x <= 0.0 {
        return Err("cannot calibrate O-U from constant price series".to_string());
    }

    let phi = cov_xy / var_x;
    let intercept = mean_y - phi * mean_x;
    let denom = 1.0 - phi;
    let equilibrium = if denom.abs() > 1e-12 { intercept / denom } else { mean_y };

    let mut residuals = Vec::with_capacity(n);
    for i in 0..n {
        let fitted = intercept + phi * x[i];
        residuals.push(y[i] - fitted);
    }

    let sigma = std_dev(&residuals);
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err("estimated innovation sigma must be positive".to_string());
    }

    let ss_res = residuals.iter().map(|e| e * e).sum::<f64>();
    let ss_tot = y
        .iter()
        .map(|v| {
            let d = *v - mean_y;
            d * d
        })
        .sum::<f64>();
    let r_squared = if ss_tot > 0.0 { (1.0 - ss_res / ss_tot).clamp(-1.0, 1.0) } else { 0.0 };

    Ok(OuProcessParams {
        phi,
        intercept,
        equilibrium,
        sigma,
        r_squared,
        stationary: phi.abs() < 1.0,
    })
}

pub fn generate_ou_paths(
    params: OuProcessParams,
    initial_price: f64,
    n_paths: usize,
    horizon: usize,
    seed: u64,
) -> Result<Vec<Vec<f64>>, String> {
    if !initial_price.is_finite() {
        return Err("initial_price must be finite".to_string());
    }
    if n_paths == 0 || horizon < 2 {
        return Err("n_paths must be > 0 and horizon must be >= 2".to_string());
    }
    if !params.phi.is_finite()
        || !params.intercept.is_finite()
        || !params.equilibrium.is_finite()
        || !params.sigma.is_finite()
    {
        return Err("O-U parameters must be finite".to_string());
    }
    if params.sigma < 0.0 {
        return Err("sigma must be non-negative".to_string());
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let noise = StandardNormal;
    let mut out = Vec::with_capacity(n_paths);

    for _ in 0..n_paths {
        let mut path = Vec::with_capacity(horizon);
        path.push(initial_price);
        for _ in 1..horizon {
            let prev = path[path.len() - 1];
            let eps: f64 = noise.sample(&mut rng);
            let next = params.intercept + params.phi * prev + params.sigma * eps;
            path.push(next);
        }
        out.push(path);
    }

    Ok(out)
}

pub fn evaluate_rule_on_paths(
    paths: &[Vec<f64>],
    rule: TradingRule,
    max_holding_steps: usize,
    annualization_factor: f64,
) -> Result<RuleSurfacePoint, String> {
    if paths.is_empty() {
        return Err("paths cannot be empty".to_string());
    }
    if rule.profit_taking <= 0.0 || rule.stop_loss <= 0.0 {
        return Err("profit_taking and stop_loss must be > 0".to_string());
    }
    if max_holding_steps == 0 {
        return Err("max_holding_steps must be > 0".to_string());
    }
    if annualization_factor <= 0.0 || !annualization_factor.is_finite() {
        return Err("annualization_factor must be finite and > 0".to_string());
    }

    let mut terminal_returns = Vec::with_capacity(paths.len());
    let mut holding_steps = Vec::with_capacity(paths.len());

    for path in paths {
        if path.len() < 2 {
            return Err("every path must have at least 2 points".to_string());
        }
        if path.iter().any(|p| !p.is_finite()) {
            return Err("paths must contain only finite values".to_string());
        }

        let entry = path[0];
        let max_step = max_holding_steps.min(path.len() - 1);

        let mut exited = false;
        let mut ret = path[max_step] - entry;
        let mut hold = max_step;

        for (step, px) in path.iter().enumerate().take(max_step + 1).skip(1) {
            let pnl = *px - entry;
            if pnl >= rule.profit_taking || pnl <= -rule.stop_loss {
                exited = true;
                ret = pnl;
                hold = step;
                break;
            }
        }

        if !exited {
            hold = max_step;
        }
        terminal_returns.push(ret);
        holding_steps.push(hold as f64);
    }

    let mean_return = terminal_returns.iter().sum::<f64>() / terminal_returns.len() as f64;
    let std_return = std_dev(&terminal_returns);
    let sharpe =
        if std_return > 0.0 { mean_return / std_return * annualization_factor.sqrt() } else { 0.0 };
    let wins = terminal_returns.iter().filter(|r| **r > 0.0).count() as f64;
    let win_rate = wins / terminal_returns.len() as f64;
    let avg_holding_steps = holding_steps.iter().sum::<f64>() / holding_steps.len() as f64;

    Ok(RuleSurfacePoint { rule, sharpe, mean_return, std_return, win_rate, avg_holding_steps })
}

pub fn detect_no_stable_optimum(
    response_surface: &[RuleSurfacePoint],
    estimated_phi: f64,
    criteria: StabilityCriteria,
) -> Result<StabilityDiagnostics, String> {
    if response_surface.is_empty() {
        return Err("response_surface cannot be empty".to_string());
    }

    let mut sharpes = response_surface.iter().map(|p| p.sharpe).collect::<Vec<_>>();
    sharpes.sort_by(|a, b| a.total_cmp(b));

    let best_sharpe = *sharpes.last().ok_or_else(|| "no sharpe values".to_string())?;
    let median_sharpe = median_sorted(&sharpes);
    let peak_margin = best_sharpe - median_sharpe;
    let surface_std = std_dev(&sharpes);

    let near_random_walk = estimated_phi.abs() >= criteria.random_walk_phi_threshold;
    let weak_peak = peak_margin < criteria.min_peak_margin;
    let flat_surface = surface_std < criteria.min_surface_std;
    let weak_best = best_sharpe < criteria.min_best_sharpe;
    let no_stable_optimum =
        (near_random_walk && (weak_peak || flat_surface)) || (weak_best && weak_peak);

    let reason = if no_stable_optimum {
        if near_random_walk {
            "no stable optimum: estimated process is near random-walk and the PT/SL surface is weakly structured"
                .to_string()
        } else {
            "no stable optimum: best rule has weak edge relative to the response surface"
                .to_string()
        }
    } else {
        "stable optimum detected".to_string()
    };

    Ok(StabilityDiagnostics {
        no_stable_optimum,
        reason,
        best_sharpe,
        median_sharpe,
        peak_margin,
        surface_std,
        estimated_phi,
    })
}

pub fn search_optimal_trading_rule(
    params: OuProcessParams,
    paths: &[Vec<f64>],
    profit_taking_grid: &[f64],
    stop_loss_grid: &[f64],
    max_holding_steps: usize,
    annualization_factor: f64,
    stability_criteria: StabilityCriteria,
) -> Result<OtrSearchResult, String> {
    if profit_taking_grid.is_empty() || stop_loss_grid.is_empty() {
        return Err("profit_taking_grid and stop_loss_grid must be non-empty".to_string());
    }

    let mut response_surface = Vec::with_capacity(profit_taking_grid.len() * stop_loss_grid.len());
    for &pt in profit_taking_grid {
        for &sl in stop_loss_grid {
            let point = evaluate_rule_on_paths(
                paths,
                TradingRule { profit_taking: pt, stop_loss: sl },
                max_holding_steps,
                annualization_factor,
            )?;
            response_surface.push(point);
        }
    }

    response_surface.sort_by(|a, b| {
        b.sharpe.total_cmp(&a.sharpe).then_with(|| b.mean_return.total_cmp(&a.mean_return))
    });

    let best_point =
        response_surface.first().cloned().ok_or_else(|| "response surface is empty".to_string())?;
    let best_rule = best_point.rule;
    let diagnostics = detect_no_stable_optimum(&response_surface, params.phi, stability_criteria)?;

    Ok(OtrSearchResult { params, best_rule, best_point, response_surface, diagnostics })
}

pub fn run_synthetic_otr_workflow(
    historical_prices: &[f64],
    config: &SyntheticBacktestConfig,
) -> Result<OtrSearchResult, String> {
    if config.profit_taking_grid.iter().any(|v| *v <= 0.0 || !v.is_finite()) {
        return Err("profit_taking_grid values must be finite and > 0".to_string());
    }
    if config.stop_loss_grid.iter().any(|v| *v <= 0.0 || !v.is_finite()) {
        return Err("stop_loss_grid values must be finite and > 0".to_string());
    }

    let params = calibrate_ou_params(historical_prices)?;
    let paths = generate_ou_paths(
        params,
        config.initial_price,
        config.n_paths,
        config.horizon,
        config.seed,
    )?;
    search_optimal_trading_rule(
        params,
        &paths,
        &config.profit_taking_grid,
        &config.stop_loss_grid,
        config.max_holding_steps,
        config.annualization_factor,
        config.stability_criteria,
    )
}

fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut ss = 0.0;
    for v in values {
        let d = *v - mean;
        ss += d * d;
    }
    (ss / (values.len() as f64 - 1.0)).sqrt()
}

fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}
