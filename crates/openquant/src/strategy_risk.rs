//! Strategy-risk diagnostics aligned to AFML Chapter 15.
//!
//! This module models trade outcomes as a binary process to quantify:
//! - Sharpe-vs-precision/frequency relations under symmetric and asymmetric payouts,
//! - implied precision/frequency needed to hit a Sharpe target, and
//! - probability that a strategy fails to achieve a Sharpe target.
//!
//! The focus is strategy viability risk, not holdings/portfolio variance risk.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use statrs::distribution::{ContinuousCDF, Normal};

#[derive(Debug, Clone, PartialEq)]
pub enum StrategyRiskError {
    EmptyInput(&'static str),
    InvalidInput(&'static str),
    NoValidRoot(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AsymmetricPayout {
    pub pi_plus: f64,
    pub pi_minus: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrategyRiskConfig {
    pub years_elapsed: f64,
    pub target_sharpe: f64,
    pub investor_horizon_years: f64,
    pub bootstrap_iterations: usize,
    pub seed: u64,
    pub kde_bandwidth: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StrategyRiskReport {
    pub payout: AsymmetricPayout,
    pub annual_bet_frequency: f64,
    pub implied_precision_threshold: f64,
    pub bootstrap_precision_mean: f64,
    pub bootstrap_precision_std: f64,
    pub empirical_failure_probability: f64,
    pub kde_failure_probability: f64,
    pub bootstrap_precision_samples: Vec<f64>,
}

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
    let n = target_sharpe * target_sharpe * 4.0 * precision * (1.0 - precision) / (edge * edge);
    validate_positive("implied_frequency", n)?;
    Ok(n)
}

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
    let n = target_sharpe * target_sharpe * d * d * precision * (1.0 - precision) / (mu * mu);
    validate_positive("implied_frequency", n)?;
    Ok(n)
}

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

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mu = mean(values);
    let var = values
        .iter()
        .map(|v| {
            let d = *v - mu;
            d * d
        })
        .sum::<f64>()
        / (values.len() as f64 - 1.0);
    var.sqrt()
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
