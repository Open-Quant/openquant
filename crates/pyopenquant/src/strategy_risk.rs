use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::helpers::to_py_err;

/// Annualised Sharpe ratio of a binary strategy with symmetric payouts.
///
/// AFML §15.2, Snippet 15.1: `(2p - 1) / (2 sqrt(p (1 - p))) * sqrt(n)` for `n` independent
/// bets a year that each win `+pi` with probability `p` and lose `-pi` otherwise. The payout
/// size cancels.
///
/// Parameters
/// ----------
/// precision : float
///     Probability `p` that a bet wins, strictly between 0 and 1.
/// annual_bet_frequency : float
///     Bets per year `n`; finite and > 0.
///
/// Returns
/// -------
/// float
///     The annualised Sharpe ratio.
///
/// Raises
/// ------
/// ValueError
///     If `precision` is not strictly inside `(0, 1)` or `annual_bet_frequency` is not finite
///     and positive.
#[pyfunction(name = "sharpe_symmetric")]
fn sr_sharpe_symmetric(precision: f64, annual_bet_frequency: f64) -> PyResult<f64> {
    openquant::strategy_risk::sharpe_symmetric(precision, annual_bet_frequency).map_err(to_py_err)
}

/// Precision needed for a symmetric-payout strategy to reach a target Sharpe ratio.
///
/// AFML §15.2: `(1 + theta / sqrt(theta^2 + n)) / 2` for target Sharpe `theta` and `n` bets a
/// year; the inverse of `sharpe_symmetric`.
///
/// Parameters
/// ----------
/// target_sharpe : float
///     Annualised Sharpe ratio to reach; finite and > 0.
/// annual_bet_frequency : float
///     Bets per year; finite and > 0.
///
/// Returns
/// -------
/// float
///     The required precision, in `(0.5, 1)`.
///
/// Raises
/// ------
/// ValueError
///     If `annual_bet_frequency` or `target_sharpe` is not finite and positive.
#[pyfunction(name = "implied_precision_symmetric")]
fn sr_implied_precision_symmetric(target_sharpe: f64, annual_bet_frequency: f64) -> PyResult<f64> {
    openquant::strategy_risk::implied_precision_symmetric(target_sharpe, annual_bet_frequency)
        .map_err(to_py_err)
}

/// Bets per year needed for a symmetric-payout strategy to reach a target Sharpe ratio.
///
/// AFML §15.2: `4 theta^2 p (1 - p) / (2p - 1)^2` for precision `p` and target Sharpe
/// `theta`; the inverse of `sharpe_symmetric`. A precision below 0.5 gives a positive
/// frequency even though its Sharpe ratio is negative; check `p > 0.5` first.
///
/// Parameters
/// ----------
/// precision : float
///     Probability that a bet wins, in `[0, 1]`.
/// target_sharpe : float
///     Annualised Sharpe ratio to reach; finite and > 0.
///
/// Returns
/// -------
/// float
///     The required number of bets per year.
///
/// Raises
/// ------
/// ValueError
///     If `precision` is outside `[0, 1]`, within `1e-12` of 0.5, or exactly 0 or 1 (zero
///     implied frequency), or `target_sharpe` is not finite and positive.
#[pyfunction(name = "implied_frequency_symmetric")]
fn sr_implied_frequency_symmetric(precision: f64, target_sharpe: f64) -> PyResult<f64> {
    openquant::strategy_risk::implied_frequency_symmetric(precision, target_sharpe)
        .map_err(to_py_err)
}

/// Annualised Sharpe ratio of a binary strategy with asymmetric payouts.
///
/// AFML §15.3, Snippet 15.2: `(d p + pi_minus) / (|d| sqrt(p (1 - p))) * sqrt(n)` with
/// `d = pi_plus - pi_minus`, for `n` independent bets a year that return `pi_plus` with
/// probability `p` and `pi_minus` otherwise.
///
/// Parameters
/// ----------
/// precision : float
///     Probability `p` that a bet wins, strictly between 0 and 1.
/// annual_bet_frequency : float
///     Bets per year `n`; finite and > 0.
/// pi_plus : float
///     Return of a winning bet.
/// pi_minus : float
///     Return of a losing bet (typically negative); must be below `pi_plus`.
///
/// Returns
/// -------
/// float
///     The annualised Sharpe ratio.
///
/// Raises
/// ------
/// ValueError
///     If `precision` is not strictly inside `(0, 1)`, `annual_bet_frequency` is not finite
///     and positive, or the payouts are not finite with `pi_plus > pi_minus`.
#[pyfunction(name = "sharpe_asymmetric")]
fn sr_sharpe_asymmetric(
    precision: f64,
    annual_bet_frequency: f64,
    pi_plus: f64,
    pi_minus: f64,
) -> PyResult<f64> {
    let payout = openquant::strategy_risk::AsymmetricPayout { pi_plus, pi_minus };
    openquant::strategy_risk::sharpe_asymmetric(precision, annual_bet_frequency, payout)
        .map_err(to_py_err)
}

/// Smallest precision at which an asymmetric-payout strategy reaches a target Sharpe ratio.
///
/// AFML §15.3, Snippet 15.3: solves the quadratic in `p` obtained by inverting
/// `sharpe_asymmetric` and returns the smallest root in `[0, 1]` whose Sharpe ratio reaches
/// `target_sharpe`.
///
/// Parameters
/// ----------
/// target_sharpe : float
///     Annualised Sharpe ratio to reach; finite and > 0.
/// annual_bet_frequency : float
///     Bets per year; finite and > 0.
/// pi_plus : float
///     Return of a winning bet.
/// pi_minus : float
///     Return of a losing bet (typically negative); must be below `pi_plus`.
///
/// Returns
/// -------
/// float
///     The required precision, in `[0, 1]`.
///
/// Raises
/// ------
/// ValueError
///     If `annual_bet_frequency` or `target_sharpe` is not finite and positive, the payouts
///     are not finite with `pi_plus > pi_minus`, or no real root in `[0, 1]` reaches the
///     target.
#[pyfunction(name = "implied_precision_asymmetric")]
fn sr_implied_precision_asymmetric(
    target_sharpe: f64,
    annual_bet_frequency: f64,
    pi_plus: f64,
    pi_minus: f64,
) -> PyResult<f64> {
    let payout = openquant::strategy_risk::AsymmetricPayout { pi_plus, pi_minus };
    openquant::strategy_risk::implied_precision_asymmetric(
        target_sharpe,
        annual_bet_frequency,
        payout,
    )
    .map_err(to_py_err)
}

/// Bets per year needed for an asymmetric-payout strategy to reach a target Sharpe ratio.
///
/// AFML §15.3: `theta^2 d^2 p (1 - p) / (d p + pi_minus)^2` with `d = pi_plus - pi_minus`.
/// A strategy whose mean payoff `d p + pi_minus` is negative gets a positive frequency from
/// this formula even though its Sharpe ratio is negative; check the sign of the mean payoff
/// first.
///
/// Parameters
/// ----------
/// precision : float
///     Probability that a bet wins, in `[0, 1]`.
/// target_sharpe : float
///     Annualised Sharpe ratio to reach; finite and > 0.
/// pi_plus : float
///     Return of a winning bet.
/// pi_minus : float
///     Return of a losing bet (typically negative); must be below `pi_plus`.
///
/// Returns
/// -------
/// float
///     The required number of bets per year.
///
/// Raises
/// ------
/// ValueError
///     If `precision` is outside `[0, 1]` or exactly 0 or 1 (zero implied frequency),
///     `target_sharpe` is not finite and positive, the payouts are not finite with
///     `pi_plus > pi_minus`, or the mean payoff is within `1e-12` of zero.
#[pyfunction(name = "implied_frequency_asymmetric")]
fn sr_implied_frequency_asymmetric(
    precision: f64,
    target_sharpe: f64,
    pi_plus: f64,
    pi_minus: f64,
) -> PyResult<f64> {
    let payout = openquant::strategy_risk::AsymmetricPayout { pi_plus, pi_minus };
    openquant::strategy_risk::implied_frequency_asymmetric(precision, target_sharpe, payout)
        .map_err(to_py_err)
}

/// Probability that a strategy misses its target Sharpe ratio, by bootstrapping its precision.
///
/// AFML §15.4, Snippets 15.4-15.5. From a record of per-bet outcomes (`> 0` is a win; zero
/// counts as a loss), estimates the payouts as the mean win and mean loss, the frequency as
/// `len(bet_outcomes) / years_elapsed`, and the required precision `p*` with
/// `implied_precision_asymmetric`. It then bootstraps the precision `bootstrap_iterations`
/// times from samples of `floor(frequency * investor_horizon_years)` bets (at least 1), drawn
/// with replacement by an RNG seeded with `seed`, and reports the share of samples at or
/// below `p*` and a Gaussian-KDE estimate of the same probability. Deterministic for a given
/// seed. Bets are assumed i.i.d.; the bootstrap does not price edge decay.
///
/// Parameters
/// ----------
/// bet_outcomes : list[float]
///     Per-bet returns, all finite, with at least one win and one loss.
/// years_elapsed : float
///     Length of the record in years; finite and > 0.
/// target_sharpe : float
///     Annualised Sharpe ratio the strategy must reach; finite and > 0.
/// investor_horizon_years : float
///     Horizon over which the investor judges the strategy, in years (sets the bootstrap
///     sample size); finite and > 0.
/// bootstrap_iterations : int, default 1000
///     Number of bootstrap resamples; must be > 0.
/// seed : int, default 42
///     Seed of the bootstrap RNG.
/// kde_bandwidth : float | None, default None
///     KDE bandwidth; finite and > 0. `None` uses Silverman's rule.
///
/// Returns
/// -------
/// dict[str, Any]
///     Keys `pi_plus` and `pi_minus` (mean win and mean loss), `annual_bet_frequency`,
///     `implied_precision_threshold` (`p*`), `bootstrap_precision_mean`,
///     `bootstrap_precision_std` (sample standard deviation), `empirical_failure_probability`
///     (share of samples `<= p*`), `kde_failure_probability` (all floats) and
///     `bootstrap_precision_samples` (list[float], every bootstrapped precision).
///
/// Raises
/// ------
/// ValueError
///     If `bet_outcomes` is empty, has a non-finite value, or lacks either a win or a loss;
///     `years_elapsed`, `target_sharpe`, `investor_horizon_years` or `kde_bandwidth` is not
///     finite and positive; `bootstrap_iterations` is 0; or no precision reaches the target.
#[pyfunction(name = "estimate_strategy_failure_probability")]
#[pyo3(signature = (
    bet_outcomes,
    years_elapsed,
    target_sharpe,
    investor_horizon_years,
    bootstrap_iterations=1000,
    seed=42,
    kde_bandwidth=None
))]
// Python keyword signature.
#[allow(clippy::too_many_arguments)]
fn sr_estimate_strategy_failure_probability(
    py: Python<'_>,
    bet_outcomes: Vec<f64>,
    years_elapsed: f64,
    target_sharpe: f64,
    investor_horizon_years: f64,
    bootstrap_iterations: usize,
    seed: u64,
    kde_bandwidth: Option<f64>,
) -> PyResult<PyObject> {
    let cfg = openquant::strategy_risk::StrategyRiskConfig {
        years_elapsed,
        target_sharpe,
        investor_horizon_years,
        bootstrap_iterations,
        seed,
        kde_bandwidth,
    };
    let report =
        openquant::strategy_risk::estimate_strategy_failure_probability(&bet_outcomes, cfg)
            .map_err(to_py_err)?;

    let d = PyDict::new(py);
    d.set_item("pi_plus", report.payout.pi_plus)?;
    d.set_item("pi_minus", report.payout.pi_minus)?;
    d.set_item("annual_bet_frequency", report.annual_bet_frequency)?;
    d.set_item("implied_precision_threshold", report.implied_precision_threshold)?;
    d.set_item("bootstrap_precision_mean", report.bootstrap_precision_mean)?;
    d.set_item("bootstrap_precision_std", report.bootstrap_precision_std)?;
    d.set_item("empirical_failure_probability", report.empirical_failure_probability)?;
    d.set_item("kde_failure_probability", report.kde_failure_probability)?;
    d.set_item("bootstrap_precision_samples", report.bootstrap_precision_samples)?;
    Ok(d.into_pyobject(py).unwrap().into_any().unbind())
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "strategy_risk")?;
    m.add_function(wrap_pyfunction!(sr_sharpe_symmetric, &m)?)?;
    m.add_function(wrap_pyfunction!(sr_implied_precision_symmetric, &m)?)?;
    m.add_function(wrap_pyfunction!(sr_implied_frequency_symmetric, &m)?)?;
    m.add_function(wrap_pyfunction!(sr_sharpe_asymmetric, &m)?)?;
    m.add_function(wrap_pyfunction!(sr_implied_precision_asymmetric, &m)?)?;
    m.add_function(wrap_pyfunction!(sr_implied_frequency_asymmetric, &m)?)?;
    m.add_function(wrap_pyfunction!(sr_estimate_strategy_failure_probability, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("strategy_risk", m)?;
    Ok(())
}
