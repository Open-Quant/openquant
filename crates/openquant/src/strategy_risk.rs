//! Strategy-risk diagnostics aligned to AFML Chapter 15.
//!
//! This module models trade outcomes as a binary process to quantify:
//! - Sharpe-vs-precision/frequency relations under symmetric and asymmetric payouts,
//! - implied precision/frequency needed to hit a Sharpe target, and
//! - probability that a strategy fails to achieve a Sharpe target.
//!
//! The focus is strategy viability risk, not holdings/portfolio variance risk.
//!
//! A strategy makes `n` independent bets a year; each wins `pi_plus` with probability `p`
//! (the precision) and otherwise returns `pi_minus`. Its annualised Sharpe ratio is
//! `(2p - 1) / (2 sqrt(p (1 - p))) * sqrt(n)` for symmetric payouts of `+-pi` (§15.2,
//! Snippet 15.1) and `((pi_plus - pi_minus) p + pi_minus) / ((pi_plus - pi_minus)
//! sqrt(p (1 - p))) * sqrt(n)` for asymmetric ones (§15.3, Snippets 15.2–15.3). The
//! `implied_*` functions invert these for a target Sharpe ratio, and
//! [`estimate_strategy_failure_probability`] bootstraps the precision of a bet record to
//! estimate the probability of missing the target (§15.4, Snippets 15.4–15.5).
//!
//! Conventions: `precision` is in `[0, 1]`; `annual_bet_frequency` is bets per year;
//! payouts are per-bet returns in the same units; Sharpe ratios are annualised. Bets are
//! assumed independent and identically distributed, so overlapping bets overstate `n`.
//!
//! Payouts need only `pi_plus > pi_minus`: the formulas are the mean over the standard
//! deviation of a two-valued bet, which is defined whatever the signs. In §15.4 the payouts
//! are the mean outcome above zero and the mean outcome at or below it, so there
//! `pi_minus <= 0 < pi_plus` by construction. Target Sharpe ratios must be positive, and a
//! strategy whose mean payoff is not positive cannot reach one at any frequency; the
//! `implied_frequency_*` functions return [`StrategyRiskError::NoValidRoot`] for it.
//!
//! ```
//! use openquant::strategy_risk::{
//!     implied_frequency_symmetric, implied_precision_asymmetric, sharpe_asymmetric,
//!     sharpe_symmetric, AsymmetricPayout,
//! };
//!
//! # fn main() -> Result<(), openquant::strategy_risk::StrategyRiskError> {
//! // 55% precision, daily bets.
//! assert!((sharpe_symmetric(0.55, 260.0)? - 1.6206).abs() < 1e-4);
//! // Reaching a Sharpe ratio of 2 at that precision takes 396 bets a year.
//! assert!((implied_frequency_symmetric(0.55, 2.0)? - 396.0).abs() < 1e-6);
//!
//! // Winning 1% and losing 2% needs 72% precision for a Sharpe ratio of 2 at 260 bets a year.
//! let payout = AsymmetricPayout { pi_plus: 0.01, pi_minus: -0.02 };
//! let needed = implied_precision_asymmetric(2.0, 260.0, payout)?;
//! assert!((needed - 0.7222).abs() < 1e-4);
//! assert!((sharpe_asymmetric(needed, 260.0, payout)? - 2.0).abs() < 1e-6);
//! # Ok(())
//! # }
//! ```

use crate::util::stats;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use statrs::distribution::{ContinuousCDF, Normal};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
/// Errors returned by the strategy-risk functions.
pub enum StrategyRiskError {
    /// The named input is empty.
    #[error("{0} must not be empty")]
    EmptyInput(&'static str),
    /// An input is out of its domain; the message names it or the violated condition.
    #[error("invalid input: {0}")]
    InvalidInput(&'static str),
    /// An inversion has no admissible solution for these parameters.
    #[error("no valid root: {0}")]
    NoValidRoot(&'static str),
}

/// Per-bet payouts of a binary strategy; `pi_plus` must exceed `pi_minus`. Neither sign is
/// required (see the [module docs](self)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AsymmetricPayout {
    /// Return of a winning bet.
    pub pi_plus: f64,
    /// Return of a losing bet (typically negative).
    pub pi_minus: f64,
}

/// Parameters of [`estimate_strategy_failure_probability`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrategyRiskConfig {
    /// Length of the bet record in years (sets the annual bet frequency).
    pub years_elapsed: f64,
    /// Annualised Sharpe ratio the strategy must reach.
    pub target_sharpe: f64,
    /// Horizon over which the investor judges the strategy, in years (sets the bootstrap
    /// sample size).
    pub investor_horizon_years: f64,
    /// Number of bootstrap resamples.
    pub bootstrap_iterations: usize,
    /// Seed of the bootstrap RNG.
    pub seed: u64,
    /// KDE bandwidth; `None` uses Silverman's rule.
    pub kde_bandwidth: Option<f64>,
}

/// Result of [`estimate_strategy_failure_probability`].
#[derive(Debug, Clone, PartialEq)]
pub struct StrategyRiskReport {
    /// Mean winning and mean losing outcome of the record.
    pub payout: AsymmetricPayout,
    /// Bets per year: record length over `years_elapsed`.
    pub annual_bet_frequency: f64,
    /// Precision `p*` needed to reach the target Sharpe ratio.
    pub implied_precision_threshold: f64,
    /// Mean of the bootstrapped precisions.
    pub bootstrap_precision_mean: f64,
    /// Sample standard deviation of the bootstrapped precisions.
    pub bootstrap_precision_std: f64,
    /// Share of bootstrapped precisions at or below `p*`.
    pub empirical_failure_probability: f64,
    /// Gaussian-KDE estimate of `P[p <= p*]` from the bootstrapped precisions.
    pub kde_failure_probability: f64,
    /// Every bootstrapped precision.
    pub bootstrap_precision_samples: Vec<f64>,
}

/// Annualised Sharpe ratio of a strategy with symmetric payouts (AFML §15.2, Snippet 15.1):
/// `(2p - 1) / (2 sqrt(p (1 - p))) * sqrt(n)`. The payout size cancels.
///
/// # Errors
///
/// [`StrategyRiskError::InvalidInput`] if `precision` is not strictly inside `(0, 1)` or
/// `annual_bet_frequency` is not finite and positive.
pub fn sharpe_symmetric(
    precision: f64,
    annual_bet_frequency: f64,
) -> Result<f64, StrategyRiskError> {
    validate_precision(precision)?;
    validate_positive("annual_bet_frequency", annual_bet_frequency)?;

    let denom = 2.0 * (precision * (1.0 - precision)).sqrt();
    if denom <= 0.0 {
        return Err(StrategyRiskError::InvalidInput("precision must be strictly between 0 and 1"));
    }
    Ok((2.0 * precision - 1.0) / denom * annual_bet_frequency.sqrt())
}

/// Precision needed for a symmetric-payout strategy with `annual_bet_frequency` bets a year to
/// reach `target_sharpe`: `(1 + theta / sqrt(theta^2 + n)) / 2` (AFML §15.2).
///
/// # Errors
///
/// - [`StrategyRiskError::InvalidInput`] if `annual_bet_frequency` is not finite and
///   positive, or `target_sharpe` is not finite and positive.
/// - [`StrategyRiskError::NoValidRoot`] if the result falls outside `[0, 1]`.
pub fn implied_precision_symmetric(
    target_sharpe: f64,
    annual_bet_frequency: f64,
) -> Result<f64, StrategyRiskError> {
    validate_positive("annual_bet_frequency", annual_bet_frequency)?;
    if target_sharpe <= 0.0 || !target_sharpe.is_finite() {
        return Err(StrategyRiskError::InvalidInput("target_sharpe must be finite and > 0"));
    }

    let root = target_sharpe / (target_sharpe * target_sharpe + annual_bet_frequency).sqrt();
    let p = 0.5 * (1.0 + root);
    if !(0.0..=1.0).contains(&p) {
        return Err(StrategyRiskError::NoValidRoot(
            "implied symmetric precision is outside [0, 1]",
        ));
    }
    Ok(p)
}

/// Bets per year needed for a symmetric-payout strategy with `precision` to reach
/// `target_sharpe`: `4 theta^2 p (1 - p) / (2p - 1)^2` (AFML §15.2).
///
/// The formula squares the Sharpe ratio, so on its own it gives precision `0.5 - x` the
/// frequency of `0.5 + x`. Below 0.5 the Sharpe ratio is negative at every frequency, so a
/// positive target is unattainable and this returns an error instead.
///
/// # Errors
///
/// - [`StrategyRiskError::InvalidInput`] if `precision` is outside `[0, 1]` or within
///   `1e-12` of 0.5, `target_sharpe` is not finite and positive, or the implied frequency is
///   not positive (precision 1).
/// - [`StrategyRiskError::NoValidRoot`] if `precision` is below 0.5.
pub fn implied_frequency_symmetric(
    precision: f64,
    target_sharpe: f64,
) -> Result<f64, StrategyRiskError> {
    validate_precision(precision)?;
    if target_sharpe <= 0.0 || !target_sharpe.is_finite() {
        return Err(StrategyRiskError::InvalidInput("target_sharpe must be finite and > 0"));
    }
    let edge = 2.0 * precision - 1.0;
    if edge.abs() < 1e-12 {
        return Err(StrategyRiskError::InvalidInput(
            "precision too close to 0.5 to imply finite frequency for positive target Sharpe",
        ));
    }
    if edge < 0.0 {
        return Err(StrategyRiskError::NoValidRoot(
            "precision below 0.5 gives a negative Sharpe ratio at every frequency",
        ));
    }
    let n = target_sharpe * target_sharpe * 4.0 * precision * (1.0 - precision) / (edge * edge);
    validate_positive("implied_frequency", n)?;
    Ok(n)
}

/// Annualised Sharpe ratio of a strategy with asymmetric payouts (AFML §15.3,
/// Snippet 15.2): `(d p + pi_minus) / (|d| sqrt(p (1 - p))) * sqrt(n)` with
/// `d = pi_plus - pi_minus`.
///
/// # Errors
///
/// [`StrategyRiskError::InvalidInput`] if `precision` is not strictly inside `(0, 1)`,
/// `annual_bet_frequency` is not finite and positive, or the payouts are not finite with
/// `pi_plus > pi_minus`.
pub fn sharpe_asymmetric(
    precision: f64,
    annual_bet_frequency: f64,
    payout: AsymmetricPayout,
) -> Result<f64, StrategyRiskError> {
    validate_precision(precision)?;
    validate_positive("annual_bet_frequency", annual_bet_frequency)?;
    validate_payout(payout)?;

    let d = payout.pi_plus - payout.pi_minus;
    let mu = d * precision + payout.pi_minus;
    let sigma = d.abs() * (precision * (1.0 - precision)).sqrt();
    if sigma <= 0.0 || !sigma.is_finite() {
        return Err(StrategyRiskError::InvalidInput("asymmetric payout variance must be positive"));
    }
    Ok(mu / sigma * annual_bet_frequency.sqrt())
}

/// Smallest precision in `[0, 1]` at which an asymmetric-payout strategy reaches
/// `target_sharpe` (AFML §15.3, Snippet 15.3), found as a root of the quadratic in `p`.
///
/// # Errors
///
/// - [`StrategyRiskError::InvalidInput`] if `annual_bet_frequency` or `target_sharpe` is not
///   finite and positive, or the payouts are not finite with `pi_plus > pi_minus`.
/// - [`StrategyRiskError::NoValidRoot`] if the quadratic has no real root, or no root in
///   `[0, 1]` reaches the target.
pub fn implied_precision_asymmetric(
    target_sharpe: f64,
    annual_bet_frequency: f64,
    payout: AsymmetricPayout,
) -> Result<f64, StrategyRiskError> {
    validate_positive("annual_bet_frequency", annual_bet_frequency)?;
    if target_sharpe <= 0.0 || !target_sharpe.is_finite() {
        return Err(StrategyRiskError::InvalidInput("target_sharpe must be finite and > 0"));
    }
    validate_payout(payout)?;

    let d = payout.pi_plus - payout.pi_minus;
    let n = annual_bet_frequency;
    let theta2 = target_sharpe * target_sharpe;

    let a = (n + theta2) * d * d;
    let b = (2.0 * n * payout.pi_minus - theta2 * d) * d;
    let c = n * payout.pi_minus * payout.pi_minus;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 || !disc.is_finite() {
        return Err(StrategyRiskError::NoValidRoot(
            "no real implied precision for these parameters",
        ));
    }

    let sqrt_disc = disc.sqrt();
    let r1 = (-b + sqrt_disc) / (2.0 * a);
    let r2 = (-b - sqrt_disc) / (2.0 * a);

    let mut candidates = Vec::new();
    for p in [r1, r2] {
        if (0.0..=1.0).contains(&p) {
            let model_sr = sharpe_asymmetric(p, annual_bet_frequency, payout)?;
            if (model_sr - target_sharpe).abs() < 1e-6 || model_sr >= target_sharpe - 1e-6 {
                candidates.push(p);
            }
        }
    }
    candidates.sort_by(|a, b| a.total_cmp(b));
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

    candidates
        .first()
        .copied()
        .ok_or(StrategyRiskError::NoValidRoot("no admissible implied precision root in [0, 1]"))
}

/// Bets per year needed for an asymmetric-payout strategy with `precision` to reach
/// `target_sharpe`: `theta^2 d^2 p (1 - p) / (d p + pi_minus)^2` (AFML §15.3).
///
/// The formula squares the Sharpe ratio, so on its own it gives a mean payoff `-mu` the
/// frequency of `+mu`. When the mean payoff `d p + pi_minus` is negative the Sharpe ratio is
/// negative at every frequency, so a positive target is unattainable and this returns an
/// error instead (before #168 it returned that mirrored frequency).
///
/// # Errors
///
/// - [`StrategyRiskError::InvalidInput`] if `precision` is outside `[0, 1]`, `target_sharpe`
///   is not finite and positive, the payouts are invalid, or the implied frequency is not
///   positive (precision 0 or 1).
/// - [`StrategyRiskError::NoValidRoot`] if the mean payoff is negative or within `1e-12` of
///   zero.
pub fn implied_frequency_asymmetric(
    precision: f64,
    target_sharpe: f64,
    payout: AsymmetricPayout,
) -> Result<f64, StrategyRiskError> {
    validate_precision(precision)?;
    if target_sharpe <= 0.0 || !target_sharpe.is_finite() {
        return Err(StrategyRiskError::InvalidInput("target_sharpe must be finite and > 0"));
    }
    validate_payout(payout)?;

    let d = payout.pi_plus - payout.pi_minus;
    let mu = d * precision + payout.pi_minus;
    if mu.abs() < 1e-12 {
        return Err(StrategyRiskError::NoValidRoot(
            "mean payoff is near zero; implied frequency is not finite",
        ));
    }
    if mu < 0.0 {
        return Err(StrategyRiskError::NoValidRoot(
            "mean payoff is negative, so the Sharpe ratio is negative at every frequency",
        ));
    }
    let n = target_sharpe * target_sharpe * d * d * precision * (1.0 - precision) / (mu * mu);
    validate_positive("implied_frequency", n)?;
    Ok(n)
}

/// Probability that a strategy misses its target Sharpe ratio (AFML §15.4, Snippets
/// 15.4–15.5).
///
/// From a record of per-bet outcomes (`> 0` is a win; zero counts as a loss): estimates the
/// payouts as the mean win and mean loss (so `pi_minus <= 0 < pi_plus`), the frequency as `len / years_elapsed`, and the
/// required precision `p*` with [`implied_precision_asymmetric`]. It then bootstraps the
/// precision `bootstrap_iterations` times from samples of
/// `floor(frequency * investor_horizon_years)` bets (at least 1), drawn with replacement by a
/// [`StdRng`] seeded with `cfg.seed`, and reports the share of samples at or below `p*`
/// (empirical) and a Gaussian-KDE estimate of the same probability. Deterministic for a given
/// seed. The bootstrap assumes precision stays what it was; it does not price edge decay.
///
/// # Errors
///
/// - [`StrategyRiskError::EmptyInput`] if `bet_outcomes` is empty.
/// - [`StrategyRiskError::InvalidInput`] if an outcome is not finite; `years_elapsed`,
///   `target_sharpe`, `investor_horizon_years` or `kde_bandwidth` is not finite and positive;
///   `bootstrap_iterations` is zero; or the record lacks either a win or a loss.
/// - [`StrategyRiskError::NoValidRoot`] from [`implied_precision_asymmetric`] when no
///   precision reaches the target.
///
/// ```
/// use openquant::strategy_risk::{estimate_strategy_failure_probability, StrategyRiskConfig};
///
/// # fn main() -> Result<(), openquant::strategy_risk::StrategyRiskError> {
/// // Two years of bets: 60% win 1%, 40% lose 1%.
/// let outcomes: Vec<f64> = (0..500).map(|i| if i % 5 < 3 { 0.01 } else { -0.01 }).collect();
/// let cfg = StrategyRiskConfig {
///     years_elapsed: 2.0,
///     target_sharpe: 1.0,
///     investor_horizon_years: 1.0,
///     bootstrap_iterations: 1_000,
///     seed: 7,
///     kde_bandwidth: None,
/// };
/// let report = estimate_strategy_failure_probability(&outcomes, cfg)?;
/// assert_eq!(report.annual_bet_frequency, 250.0);
/// // Symmetric payouts: p* = (1 + 1 / sqrt(1 + 250)) / 2.
/// let p_star = 0.5 * (1.0 + 1.0 / 251f64.sqrt());
/// assert!((report.implied_precision_threshold - p_star).abs() < 1e-9);
/// // 60% precision is about 2.2 standard errors above p* = 0.53 over a 250-bet year.
/// let failure = report.empirical_failure_probability;
/// assert!(failure > 0.0 && failure < 0.05);
/// # Ok(())
/// # }
/// ```
pub fn estimate_strategy_failure_probability(
    bet_outcomes: &[f64],
    cfg: StrategyRiskConfig,
) -> Result<StrategyRiskReport, StrategyRiskError> {
    if bet_outcomes.is_empty() {
        return Err(StrategyRiskError::EmptyInput("bet_outcomes"));
    }
    if bet_outcomes.iter().any(|v| !v.is_finite()) {
        return Err(StrategyRiskError::InvalidInput(
            "bet_outcomes must contain only finite values",
        ));
    }
    validate_positive("years_elapsed", cfg.years_elapsed)?;
    validate_positive("target_sharpe", cfg.target_sharpe)?;
    validate_positive("investor_horizon_years", cfg.investor_horizon_years)?;
    if cfg.bootstrap_iterations == 0 {
        return Err(StrategyRiskError::InvalidInput("bootstrap_iterations must be > 0"));
    }
    if let Some(h) = cfg.kde_bandwidth {
        validate_positive("kde_bandwidth", h)?;
    }

    let neg: Vec<f64> = bet_outcomes.iter().copied().filter(|v| *v <= 0.0).collect();
    let pos: Vec<f64> = bet_outcomes.iter().copied().filter(|v| *v > 0.0).collect();
    if neg.is_empty() || pos.is_empty() {
        return Err(StrategyRiskError::InvalidInput(
            "bet_outcomes must include at least one winning and one losing bet",
        ));
    }

    let payout = AsymmetricPayout { pi_plus: mean(&pos), pi_minus: mean(&neg) };
    validate_payout(payout)?;

    let n = bet_outcomes.len() as f64 / cfg.years_elapsed;
    validate_positive("annual_bet_frequency", n)?;
    let p_star = implied_precision_asymmetric(cfg.target_sharpe, n, payout)?;

    let bootstrap_draw_size = ((n * cfg.investor_horizon_years).floor() as usize).max(1);
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut p_samples = Vec::with_capacity(cfg.bootstrap_iterations);

    for _ in 0..cfg.bootstrap_iterations {
        let mut wins = 0usize;
        for _ in 0..bootstrap_draw_size {
            let idx = rng.gen_range(0..bet_outcomes.len());
            if bet_outcomes[idx] > 0.0 {
                wins += 1;
            }
        }
        p_samples.push(wins as f64 / bootstrap_draw_size as f64);
    }

    let sample_mean = mean(&p_samples);
    let sample_std = std_dev(&p_samples);
    let empirical_failure_probability =
        p_samples.iter().filter(|p| **p <= p_star).count() as f64 / p_samples.len() as f64;

    let bandwidth = cfg.kde_bandwidth.unwrap_or_else(|| silverman_bandwidth(&p_samples));
    let kde_failure_probability = kde_cdf(p_star, &p_samples, bandwidth)?;

    Ok(StrategyRiskReport {
        payout,
        annual_bet_frequency: n,
        implied_precision_threshold: p_star,
        bootstrap_precision_mean: sample_mean,
        bootstrap_precision_std: sample_std,
        empirical_failure_probability,
        kde_failure_probability,
        bootstrap_precision_samples: p_samples,
    })
}

fn validate_payout(payout: AsymmetricPayout) -> Result<(), StrategyRiskError> {
    if !payout.pi_plus.is_finite() || !payout.pi_minus.is_finite() {
        return Err(StrategyRiskError::InvalidInput("payout values must be finite"));
    }
    if payout.pi_plus <= payout.pi_minus {
        return Err(StrategyRiskError::InvalidInput("pi_plus must be greater than pi_minus"));
    }
    Ok(())
}

fn validate_positive(name: &'static str, value: f64) -> Result<(), StrategyRiskError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(StrategyRiskError::InvalidInput(name));
    }
    Ok(())
}

fn validate_precision(precision: f64) -> Result<(), StrategyRiskError> {
    if !precision.is_finite() || !(0.0..=1.0).contains(&precision) {
        return Err(StrategyRiskError::InvalidInput("precision must be finite and in [0, 1]"));
    }
    Ok(())
}

/// Mean of `values`, `NaN` when empty.
fn mean(values: &[f64]) -> f64 {
    stats::mean(values).unwrap_or(f64::NAN)
}

/// Sample standard deviation (ddof = 1), 0 with fewer than two values.
fn std_dev(values: &[f64]) -> f64 {
    stats::std_dev(values, 1).unwrap_or(0.0)
}

fn silverman_bandwidth(samples: &[f64]) -> f64 {
    let sigma = std_dev(samples);
    let n = samples.len().max(2) as f64;
    let raw = 1.06 * sigma * n.powf(-0.2);
    if raw.is_finite() && raw > 1e-6 {
        raw
    } else {
        1e-3
    }
}

fn kde_cdf(x: f64, samples: &[f64], bandwidth: f64) -> Result<f64, StrategyRiskError> {
    if samples.is_empty() {
        return Err(StrategyRiskError::EmptyInput("samples"));
    }
    validate_positive("bandwidth", bandwidth)?;
    let normal = Normal::new(0.0, 1.0)
        .map_err(|_| StrategyRiskError::InvalidInput("failed to construct standard normal"))?;
    let cdf = samples.iter().map(|s| normal.cdf((x - *s) / bandwidth)).sum::<f64>()
        / samples.len() as f64;
    Ok(cdf.clamp(0.0, 1.0))
}
