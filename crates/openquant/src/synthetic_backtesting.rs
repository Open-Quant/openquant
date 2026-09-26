//! Synthetic-data backtesting utilities aligned to AFML Chapter 13.
//!
//! This module focuses on optimal trading-rule (OTR) search over profit-taking
//! and stop-loss corridors by:
//! 1) calibrating an AR(1)/discrete O-U process from historical prices,
//! 2) generating many synthetic paths under that calibrated process,
//! 3) evaluating a PT/SL mesh on those paths, and
//! 4) detecting when the response surface lacks a stable optimum.
//!
//! This is AFML §13.4–13.5 (Snippets 13.1–13.2): instead of picking barriers on the one
//! historical path, which overfits, fit a discrete Ornstein–Uhlenbeck process
//! `P_t = intercept + phi P_{t-1} + sigma eps_t`, simulate many paths, and read the whole
//! profit-taking/stop-loss response surface. §13.6's result is that the surface flattens as
//! `phi` approaches 1, which [`detect_no_stable_optimum`] flags with this library's own
//! heuristic thresholds (not AFML's).
//!
//! Conventions:
//!
//! - Prices are levels, oldest first. Barriers are in **price units** (not multiples of
//!   `sigma`), measured from the entry price; a barrier is touched when PnL reaches it
//!   (`>=`).
//! - Every rule is a **long** entry at the first price of each path; negate the series for a
//!   short.
//! - A rule's Sharpe ratio is the mean over the standard deviation of per-trade PnL, times
//!   `sqrt(annualization_factor)`, regardless of holding time.
//! - Simulation uses a [`StdRng`] seeded by the caller, so results are reproducible.
//!
//! ```
//! use openquant::synthetic_backtesting::{
//!     evaluate_rule_on_paths, generate_ou_paths, OuProcessParams, TradingRule,
//! };
//!
//! # fn main() -> Result<(), openquant::synthetic_backtesting::SyntheticBacktestError> {
//! // phi = 0.9 around 100 (intercept = (1 - phi) * equilibrium), entered three points below.
//! let params = OuProcessParams {
//!     phi: 0.9,
//!     intercept: 10.0,
//!     equilibrium: 100.0,
//!     sigma: 1.0,
//!     r_squared: 0.0,
//!     stationary: true,
//! };
//! let paths = generate_ou_paths(params, 97.0, 2_000, 60, 11)?;
//! assert_eq!(paths, generate_ou_paths(params, 97.0, 2_000, 60, 11)?);
//! assert!(paths.iter().all(|p| p.len() == 60 && p[0] == 97.0));
//!
//! // Expecting reversion, a wide stop beats a tight one.
//! let wide = TradingRule { profit_taking: 2.0, stop_loss: 8.0 };
//! let tight = TradingRule { profit_taking: 2.0, stop_loss: 0.5 };
//! let wide = evaluate_rule_on_paths(&paths, wide, 59, 1.0)?;
//! let tight = evaluate_rule_on_paths(&paths, tight, 59, 1.0)?;
//! assert!(wide.sharpe > tight.sharpe && wide.win_rate > 0.95);
//! # Ok(())
//! # }
//! ```

use crate::util::stats;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

/// Errors returned by the synthetic-backtesting functions.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SyntheticBacktestError {
    /// Fewer than three prices to calibrate on.
    #[error("prices must include at least 3 observations")]
    TooFewPrices,
    /// The named input violates its requirement.
    #[error("{name} must be {requirement}")]
    Invalid {
        /// Input name.
        name: &'static str,
        /// What it must satisfy.
        requirement: &'static str,
    },
    /// The lagged prices are constant, so `phi` is undefined.
    #[error("cannot calibrate O-U from constant price series")]
    ConstantPrices,
    /// The regression residuals have zero (or non-finite) deviation.
    #[error("estimated innovation sigma must be positive")]
    NonPositiveInnovationSigma,
    /// `n_paths` is zero or `horizon` is below 2.
    #[error("n_paths must be > 0 and horizon must be >= 2")]
    InvalidPathShape,
    /// An O-U parameter is not finite.
    #[error("O-U parameters must be finite")]
    NonFiniteOuParameters,
    /// The named input is empty.
    #[error("{0} cannot be empty")]
    Empty(&'static str),
    /// A barrier width is not positive.
    #[error("profit_taking and stop_loss must be > 0")]
    NonPositiveBarriers,
    /// `annualization_factor` is not finite and positive.
    #[error("annualization_factor must be finite and > 0")]
    InvalidAnnualizationFactor,
    /// A path has fewer than two points.
    #[error("every path must have at least 2 points")]
    PathTooShort,
    /// A path contains a non-finite value.
    #[error("paths must contain only finite values")]
    NonFinitePaths,
    /// No Sharpe values to diagnose (internal consistency check).
    #[error("no sharpe values")]
    NoSharpeValues,
    /// A barrier grid is empty.
    #[error("profit_taking_grid and stop_loss_grid must be non-empty")]
    EmptyGrid,
    /// The response surface is empty (internal consistency check).
    #[error("response surface is empty")]
    EmptyResponseSurface,
    /// A profit-taking grid value is not finite and positive.
    #[error("profit_taking_grid values must be finite and > 0")]
    InvalidProfitTakingGrid,
    /// A stop-loss grid value is not finite and positive.
    #[error("stop_loss_grid values must be finite and > 0")]
    InvalidStopLossGrid,
}

/// Parameters of the discrete O-U (AR(1)) process `P_t = intercept + phi P_{t-1} + sigma eps`.
///
/// [`generate_ou_paths`] simulates from `intercept`, `phi` and `sigma` only; when building by
/// hand keep `intercept = (1 - phi) * equilibrium`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OuProcessParams {
    /// Autoregressive coefficient; the speed of reversion is `1 - phi`.
    pub phi: f64,
    /// Regression intercept.
    pub intercept: f64,
    /// Long-run mean `intercept / (1 - phi)` (the mean of the prices when `phi` is 1).
    pub equilibrium: f64,
    /// Standard deviation of the innovations, in price units.
    pub sigma: f64,
    /// R-squared of the AR(1) regression (high for any persistent series).
    pub r_squared: f64,
    /// `|phi| < 1`; true even for `phi = 0.999`.
    pub stationary: bool,
}

/// A profit-taking / stop-loss pair, both positive widths in price units from the entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TradingRule {
    /// Exit when PnL reaches `+profit_taking`.
    pub profit_taking: f64,
    /// Exit when PnL reaches `-stop_loss`.
    pub stop_loss: f64,
}

/// A trading rule's performance across the simulated paths.
#[derive(Debug, Clone, PartialEq)]
pub struct RuleSurfacePoint {
    /// The rule evaluated.
    pub rule: TradingRule,
    /// `mean_return / std_return * sqrt(annualization_factor)`, or 0 when the deviation is 0.
    pub sharpe: f64,
    /// Mean per-trade PnL in price units.
    pub mean_return: f64,
    /// Sample standard deviation of per-trade PnL.
    pub std_return: f64,
    /// Share of trades with positive PnL.
    pub win_rate: f64,
    /// Mean number of steps held.
    pub avg_holding_steps: f64,
}

/// Thresholds for [`detect_no_stable_optimum`]. These are this library's heuristics, not
/// AFML's; the defaults are `phi >= 0.97`, margin 0.20, surface deviation 0.10 and best
/// Sharpe 0.30 (the Python binding uses different defaults).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StabilityCriteria {
    /// `|phi|` at or above this counts as near a random walk.
    pub random_walk_phi_threshold: f64,
    /// Best-minus-median Sharpe below this is a weak peak.
    pub min_peak_margin: f64,
    /// Surface Sharpe deviation below this is a flat surface.
    pub min_surface_std: f64,
    /// Best Sharpe below this is a weak best rule.
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

/// Result of [`detect_no_stable_optimum`].
#[derive(Debug, Clone, PartialEq)]
pub struct StabilityDiagnostics {
    /// Whether the surface lacks a stable optimum under the criteria.
    pub no_stable_optimum: bool,
    /// Human-readable explanation of the verdict.
    pub reason: String,
    /// Highest Sharpe ratio on the surface.
    pub best_sharpe: f64,
    /// Median Sharpe ratio on the surface.
    pub median_sharpe: f64,
    /// `best_sharpe - median_sharpe`.
    pub peak_margin: f64,
    /// Sample standard deviation of the surface's Sharpe ratios.
    pub surface_std: f64,
    /// The `phi` the diagnosis used.
    pub estimated_phi: f64,
}

/// Result of an optimal-trading-rule search.
#[derive(Debug, Clone, PartialEq)]
pub struct OtrSearchResult {
    /// The process parameters the paths came from.
    pub params: OuProcessParams,
    /// The rule with the highest Sharpe ratio (ties broken by mean return).
    pub best_rule: TradingRule,
    /// That rule's surface point.
    pub best_point: RuleSurfacePoint,
    /// Every grid point, sorted by Sharpe ratio (then mean return), best first.
    pub response_surface: Vec<RuleSurfacePoint>,
    /// Stability diagnosis of the surface.
    pub diagnostics: StabilityDiagnostics,
}

/// Configuration of [`run_synthetic_otr_workflow`].
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticBacktestConfig {
    /// Entry price of every simulated path (the forecast relative to equilibrium).
    pub initial_price: f64,
    /// Number of simulated paths.
    pub n_paths: usize,
    /// Points per path, including the entry.
    pub horizon: usize,
    /// Seed of the simulation RNG.
    pub seed: u64,
    /// Profit-taking widths to try, in price units.
    pub profit_taking_grid: Vec<f64>,
    /// Stop-loss widths to try, in price units.
    pub stop_loss_grid: Vec<f64>,
    /// Maximum holding period in steps (capped by the path length).
    pub max_holding_steps: usize,
    /// Multiplier whose square root scales the per-trade Sharpe ratio (1.0 for per trade).
    pub annualization_factor: f64,
    /// Thresholds for the stability diagnosis.
    pub stability_criteria: StabilityCriteria,
}

/// Fits the discrete O-U process to a price series by OLS of `P_t` on `P_{t-1}` (AFML
/// §13.5.1, step 1).
///
/// `sigma` is the sample standard deviation of the residuals; `equilibrium` is
/// `intercept / (1 - phi)`, or the mean price when `phi` is within `1e-12` of 1.
///
/// # Errors
///
/// - [`SyntheticBacktestError::TooFewPrices`] for fewer than three prices.
/// - [`SyntheticBacktestError::Invalid`] if a price is not finite.
/// - [`SyntheticBacktestError::ConstantPrices`] if the lagged prices are constant.
/// - [`SyntheticBacktestError::NonPositiveInnovationSigma`] if the fit is exact (zero
///   residual deviation).
pub fn calibrate_ou_params(prices: &[f64]) -> Result<OuProcessParams, SyntheticBacktestError> {
    if prices.len() < 3 {
        return Err(SyntheticBacktestError::TooFewPrices);
    }
    if prices.iter().any(|p| !p.is_finite()) {
        return Err(SyntheticBacktestError::Invalid { name: "prices", requirement: "finite" });
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
        return Err(SyntheticBacktestError::ConstantPrices);
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
        return Err(SyntheticBacktestError::NonPositiveInnovationSigma);
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

/// Simulates `n_paths` O-U paths of `horizon` points each, starting at `initial_price`
/// (AFML §13.5.1, step 2).
///
/// Uses `intercept`, `phi` and `sigma` from `params` with standard-normal innovations from a
/// [`StdRng`] seeded with `seed`; the same seed reproduces the same paths.
///
/// # Errors
///
/// - [`SyntheticBacktestError::Invalid`] if `initial_price` is not finite or `sigma` is
///   negative.
/// - [`SyntheticBacktestError::InvalidPathShape`] if `n_paths` is zero or `horizon < 2`.
/// - [`SyntheticBacktestError::NonFiniteOuParameters`] if a parameter is not finite.
pub fn generate_ou_paths(
    params: OuProcessParams,
    initial_price: f64,
    n_paths: usize,
    horizon: usize,
    seed: u64,
) -> Result<Vec<Vec<f64>>, SyntheticBacktestError> {
    if !initial_price.is_finite() {
        return Err(SyntheticBacktestError::Invalid {
            name: "initial_price",
            requirement: "finite",
        });
    }
    if n_paths == 0 || horizon < 2 {
        return Err(SyntheticBacktestError::InvalidPathShape);
    }
    if !params.phi.is_finite()
        || !params.intercept.is_finite()
        || !params.equilibrium.is_finite()
        || !params.sigma.is_finite()
    {
        return Err(SyntheticBacktestError::NonFiniteOuParameters);
    }
    if params.sigma < 0.0 {
        return Err(SyntheticBacktestError::Invalid { name: "sigma", requirement: "non-negative" });
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

/// Evaluates one profit-taking/stop-loss rule on every path (AFML §13.5.1, step 3).
///
/// Each path is a long entry at its first price; the trade exits at the first step (up to
/// `min(max_holding_steps, len - 1)`) whose PnL is `>= profit_taking` or `<= -stop_loss`,
/// otherwise at that last step. Returns the mean, deviation, Sharpe ratio and win rate of the
/// per-trade PnL (price units) and the mean holding time.
///
/// # Errors
///
/// - [`SyntheticBacktestError::Empty`] if `paths` is empty.
/// - [`SyntheticBacktestError::NonPositiveBarriers`] if a barrier width is not positive.
/// - [`SyntheticBacktestError::Invalid`] if `max_holding_steps` is zero.
/// - [`SyntheticBacktestError::InvalidAnnualizationFactor`] if the factor is not finite and
///   positive.
/// - [`SyntheticBacktestError::PathTooShort`] if a path has fewer than two points.
/// - [`SyntheticBacktestError::NonFinitePaths`] if a path contains a non-finite value.
pub fn evaluate_rule_on_paths(
    paths: &[Vec<f64>],
    rule: TradingRule,
    max_holding_steps: usize,
    annualization_factor: f64,
) -> Result<RuleSurfacePoint, SyntheticBacktestError> {
    if paths.is_empty() {
        return Err(SyntheticBacktestError::Empty("paths"));
    }
    if rule.profit_taking <= 0.0 || rule.stop_loss <= 0.0 {
        return Err(SyntheticBacktestError::NonPositiveBarriers);
    }
    if max_holding_steps == 0 {
        return Err(SyntheticBacktestError::Invalid {
            name: "max_holding_steps",
            requirement: "> 0",
        });
    }
    if annualization_factor <= 0.0 || !annualization_factor.is_finite() {
        return Err(SyntheticBacktestError::InvalidAnnualizationFactor);
    }

    let mut terminal_returns = Vec::with_capacity(paths.len());
    let mut holding_steps = Vec::with_capacity(paths.len());

    for path in paths {
        if path.len() < 2 {
            return Err(SyntheticBacktestError::PathTooShort);
        }
        if path.iter().any(|p| !p.is_finite()) {
            return Err(SyntheticBacktestError::NonFinitePaths);
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

/// Flags a response surface without a stable optimum (the flattening of AFML §13.6).
///
/// The verdict is true when the process is near a random walk (`|phi| >=
/// random_walk_phi_threshold`) and the peak is weak or the surface flat, or when the best
/// Sharpe ratio is weak and so is the peak. The thresholds are heuristics; look at the
/// surface too.
///
/// # Errors
///
/// [`SyntheticBacktestError::Empty`] if `response_surface` is empty.
pub fn detect_no_stable_optimum(
    response_surface: &[RuleSurfacePoint],
    estimated_phi: f64,
    criteria: StabilityCriteria,
) -> Result<StabilityDiagnostics, SyntheticBacktestError> {
    if response_surface.is_empty() {
        return Err(SyntheticBacktestError::Empty("response_surface"));
    }

    let mut sharpes = response_surface.iter().map(|p| p.sharpe).collect::<Vec<_>>();
    sharpes.sort_by(|a, b| a.total_cmp(b));

    let best_sharpe = *sharpes.last().ok_or(SyntheticBacktestError::NoSharpeValues)?;
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

/// Evaluates every `(profit_taking, stop_loss)` combination of the grids on `paths` and
/// diagnoses the resulting surface (AFML §13.5.1, Snippets 13.1–13.2).
///
/// # Errors
///
/// - [`SyntheticBacktestError::EmptyGrid`] if either grid is empty.
/// - Any [`evaluate_rule_on_paths`] error (including non-positive grid values, reported as
///   [`SyntheticBacktestError::NonPositiveBarriers`]).
pub fn search_optimal_trading_rule(
    params: OuProcessParams,
    paths: &[Vec<f64>],
    profit_taking_grid: &[f64],
    stop_loss_grid: &[f64],
    max_holding_steps: usize,
    annualization_factor: f64,
    stability_criteria: StabilityCriteria,
) -> Result<OtrSearchResult, SyntheticBacktestError> {
    if profit_taking_grid.is_empty() || stop_loss_grid.is_empty() {
        return Err(SyntheticBacktestError::EmptyGrid);
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
        response_surface.first().cloned().ok_or(SyntheticBacktestError::EmptyResponseSurface)?;
    let best_rule = best_point.rule;
    let diagnostics = detect_no_stable_optimum(&response_surface, params.phi, stability_criteria)?;

    Ok(OtrSearchResult { params, best_rule, best_point, response_surface, diagnostics })
}

/// End-to-end optimal-trading-rule search: calibrate on `historical_prices`, simulate, and
/// search the configured grid (AFML §13.5).
///
/// # Errors
///
/// - [`SyntheticBacktestError::InvalidProfitTakingGrid`] or
///   [`SyntheticBacktestError::InvalidStopLossGrid`] if a grid value is not finite and
///   positive.
/// - Any [`calibrate_ou_params`], [`generate_ou_paths`] or [`search_optimal_trading_rule`]
///   error.
pub fn run_synthetic_otr_workflow(
    historical_prices: &[f64],
    config: &SyntheticBacktestConfig,
) -> Result<OtrSearchResult, SyntheticBacktestError> {
    if config.profit_taking_grid.iter().any(|v| *v <= 0.0 || !v.is_finite()) {
        return Err(SyntheticBacktestError::InvalidProfitTakingGrid);
    }
    if config.stop_loss_grid.iter().any(|v| *v <= 0.0 || !v.is_finite()) {
        return Err(SyntheticBacktestError::InvalidStopLossGrid);
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

/// Sample standard deviation (ddof = 1), 0 with fewer than two values.
fn std_dev(values: &[f64]) -> f64 {
    stats::std_dev(values, 1).unwrap_or(0.0)
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
