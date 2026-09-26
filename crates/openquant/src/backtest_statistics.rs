//! Backtest statistics (AFML chapter 14): Sharpe ratios with their uncertainty, plus
//! drawdown, concentration and holding-period statistics.
//!
//! - [`sharpe_ratio`] and [`information_ratio`] annualise a per-period mean over a sample
//!   standard deviation (§14.7.1).
//! - [`probabilistic_sharpe_ratio`] (PSR, §14.7.2), [`deflated_sharpe_ratio`] (DSR, §14.7.3)
//!   and [`minimum_track_record_length`] put the estimation and selection uncertainty back
//!   (Bailey and López de Prado, 2012 and 2014).
//! - [`drawdown_and_time_under_water`] (Snippet 14.4), [`bets_concentration`] and
//!   [`all_bets_concentration`] (Snippet 14.3), [`average_holding_period`] (Snippet 14.2)
//!   and [`timing_of_flattening_and_flips`] (Snippet 14.1) describe the path.
//!
//! Conventions that nothing in the signatures enforces:
//!
//! - PSR, DSR and MinTRL take a **per-period** (not annualised) Sharpe ratio and **raw**
//!   kurtosis (3 for a normal distribution), not excess kurtosis.
//! - `risk_free_rate` and `benchmark` are per-period values, in the same units as the
//!   returns.
//! - Timestamped inputs are assumed sorted in increasing time.
//!
//! ```
//! use openquant::backtest_statistics::{
//!     deflated_sharpe_ratio, minimum_track_record_length, probabilistic_sharpe_ratio,
//! };
//!
//! # fn main() -> Result<(), openquant::util::InputError> {
//! // Per-period Sharpe 0.1 over 500 normal returns.
//! let psr = probabilistic_sharpe_ratio(0.1, 0.0, 500, 0.0, 3.0);
//! assert!((psr - 0.9871).abs() < 1e-4);
//!
//! // Expected maximum Sharpe of 50 skill-less trials whose Sharpe ratios have std 0.05.
//! let hurdle = deflated_sharpe_ratio(0.1, &[0.05, 50.0], 500, 0.0, 3.0, true, true)?;
//! assert!((hurdle - 0.1138).abs() < 1e-4);
//!
//! let min_trl = minimum_track_record_length(0.1, 0.0, 0.0, 3.0, 0.05)?;
//! assert!((min_trl - 272.9).abs() < 0.1);
//! # Ok(())
//! # }
//! ```
#![deny(missing_docs)]

use crate::util::InputError;
use chrono::NaiveDateTime;
use statrs::distribution::{ContinuousCDF, Normal};

const EULER_GAMMA: f64 = 0.5772156649015329_f64;

/// Returns the timestamps at which a position was closed or reversed (AFML Snippet 14.1).
///
/// A flattening is a bar where the position goes from non-zero to zero; a flip is a bar
/// where the position changes sign. The result is sorted, deduplicated, and always ends with
/// the last timestamp of the input (so an open position is treated as closed there). An
/// empty input gives an empty result.
///
/// `target_positions` is `(timestamp, position)` pairs in increasing time.
pub fn timing_of_flattening_and_flips(
    target_positions: &[(NaiveDateTime, f64)],
) -> Vec<NaiveDateTime> {
    let mut flattenings = Vec::new();
    let mut flips = Vec::new();
    for i in 1..target_positions.len() {
        let prev = target_positions[i - 1].1;
        let curr = target_positions[i].1;
        if curr == 0.0 && prev != 0.0 {
            flattenings.push(target_positions[i].0);
        }
        let mult = curr * prev;
        if mult < 0.0 {
            flips.push(target_positions[i].0);
        }
    }
    let mut res = flattenings;
    res.extend(flips);
    res.sort();
    res.dedup();
    if let Some(last) = target_positions.last()
        && !res.contains(&last.0)
    {
        res.push(last.0);
    }
    res
}

/// Average holding period of a position series, in days (AFML Snippet 14.2).
///
/// Tracks a size-weighted average entry time as the position grows, and records a holding
/// time each time it shrinks or flips, weighted by the size that was exited. Returns the
/// weighted mean of those holding times in days (86,400 seconds), or `None` if the input is
/// empty or the position is never reduced.
///
/// `target_positions` is `(timestamp, position)` pairs in increasing time.
pub fn average_holding_period(target_positions: &[(NaiveDateTime, f64)]) -> Option<f64> {
    if target_positions.is_empty() {
        return None;
    }
    let mut holding: Vec<(f64, f64)> = Vec::new(); // (holding_time_days, weight)
    let mut entry_time = 0.0;
    let time_since_start: Vec<f64> = target_positions
        .iter()
        .map(|(ts, _)| (*ts - target_positions[0].0).num_seconds() as f64 / 86_400.0)
        .collect();
    let mut position_diff: Vec<f64> = target_positions.iter().map(|(_, v)| *v).collect();
    for i in (1..position_diff.len()).rev() {
        position_diff[i] -= position_diff[i - 1];
    }
    for i in 1..target_positions.len() {
        let prev_pos = target_positions[i - 1].1;
        let diff = position_diff[i];
        let curr_pos = target_positions[i].1;
        if diff * prev_pos >= 0.0 && curr_pos != 0.0 {
            entry_time = (entry_time * prev_pos + time_since_start[i] * diff) / curr_pos;
        }
        if diff * prev_pos < 0.0 {
            let hold_time = time_since_start[i] - entry_time;
            if curr_pos * prev_pos < 0.0 {
                let weight = prev_pos.abs();
                holding.push((hold_time, weight));
                entry_time = time_since_start[i];
            } else {
                let weight = diff.abs();
                holding.push((hold_time, weight));
            }
        }
    }
    let total_w: f64 = holding.iter().map(|(_, w)| *w).sum();
    if total_w > 0.0 {
        let num: f64 = holding.iter().map(|(h, w)| h * w).sum();
        Some(num / total_w)
    } else {
        None
    }
}

/// Normalised Herfindahl-Hirschman concentration of a set of returns (AFML Snippet 14.3).
///
/// Each return's share of the total is `r_i / sum(r)`; the index is
/// `(HHI - 1/n) / (1 - 1/n)`: 0 when every bet contributed equally and 1 when one bet made
/// everything. Returns `None` for two or fewer returns or when they sum to zero. Mixing signs
/// makes the shares meaningless; pass one sign at a time, as [`all_bets_concentration`] does.
///
/// ```
/// use openquant::backtest_statistics::bets_concentration;
///
/// assert!(bets_concentration(&[1.0, 1.0, 1.0, 1.0]).unwrap().abs() < 1e-12);
/// assert!(bets_concentration(&[97.0, 1.0, 1.0, 1.0]).unwrap() > 0.9);
/// ```
pub fn bets_concentration(returns: &[f64]) -> Option<f64> {
    if returns.len() <= 2 {
        return None;
    }
    let sum: f64 = returns.iter().sum();
    if sum == 0.0 {
        return None;
    }
    let weights: Vec<f64> = returns.iter().map(|r| r / sum).collect();
    let hhi: f64 = weights.iter().map(|w| w * w).sum();
    let n = returns.len() as f64;
    let adj = (hhi - 1.0 / n) / (1.0 - 1.0 / n);
    Some(adj)
}

/// Concentration of positive returns, negative returns, and bets over time (AFML Snippet 14.3).
///
/// Returns `(positive, negative, time)`, each from [`bets_concentration`]: over the
/// non-negative returns, over the negative returns, and over the number of bets per
/// **calendar day** from the first to the last date (days without bets count as zero).
/// AFML and mlfinlab group the time component by month, so the time index here is higher on
/// data with gaps; compare it only with itself.
///
/// `returns` is `(timestamp, return)` pairs in increasing time; the first and last entries
/// define the day range.
pub fn all_bets_concentration(
    returns: &[(NaiveDateTime, f64)],
) -> (Option<f64>, Option<f64>, Option<f64>) {
    let positives: Vec<f64> = returns.iter().filter(|(_, r)| *r >= 0.0).map(|(_, r)| *r).collect();
    let negatives: Vec<f64> = returns.iter().filter(|(_, r)| *r < 0.0).map(|(_, r)| *r).collect();
    let pos = bets_concentration(&positives);
    let neg = bets_concentration(&negatives);
    // time grouping by day including gaps between first and last date (zeros matter)
    let mut per_day: std::collections::HashMap<chrono::NaiveDate, usize> =
        std::collections::HashMap::new();
    for (ts, _) in returns {
        *per_day.entry(ts.date()).or_insert(0) += 1;
    }
    let time = if returns.is_empty() {
        None
    } else {
        let start = returns.first().unwrap().0.date();
        let end = returns.last().unwrap().0.date();
        let mut counts: Vec<f64> = Vec::new();
        let mut day = start;
        while day <= end {
            let cnt = per_day.get(&day).copied().unwrap_or(0) as f64;
            counts.push(cnt);
            day = day.succ_opt().unwrap();
        }
        bets_concentration(&counts)
    };
    (pos, neg, time)
}

/// Drawdowns and time under water of a cumulative series (AFML Snippet 14.4).
///
/// Despite the parameter name, `returns` is a **cumulative** series (equity, NAV or
/// cumulative PnL) as `(timestamp, value)` pairs in increasing time. For every high-water
/// mark that was followed by a dip, returns the drawdown and the time under water:
///
/// - with `dollars = false` the drawdown is `1 - trough / peak` (needs a positive series);
///   with `dollars = true` it is `peak - trough`;
/// - the time under water is in years (365.25 days) from that high-water mark to the next
///   high-water mark that itself had a drawdown, or to the end of the series. It is not the
///   time to recovery: new highs without a dip after them extend it. This reproduces
///   Snippet 14.4 and mlfinlab.
///
/// Both vectors have one entry per such high-water mark. An empty input gives two empty
/// vectors.
pub fn drawdown_and_time_under_water(
    returns: &[(NaiveDateTime, f64)],
    dollars: bool,
) -> (Vec<f64>, Vec<f64>) {
    if returns.is_empty() {
        return (Vec::new(), Vec::new());
    }
    // Track high-water-mark segments and their minima (mirrors pandas grouping in Python version)
    let mut hwms: Vec<f64> = vec![returns[0].1];
    let mut hwm_times: Vec<NaiveDateTime> = vec![returns[0].0];
    let mut segment_min: Vec<f64> = vec![returns[0].1];

    for &(ts, val) in returns.iter().skip(1) {
        let current_hwm = *hwms.last().unwrap();
        if val > current_hwm {
            // start new HWM segment
            hwms.push(val);
            hwm_times.push(ts);
            segment_min.push(val);
        } else {
            let last_min = segment_min.last_mut().unwrap();
            if val < *last_min {
                *last_min = val;
            }
        }
    }

    // Compute drawdowns only for segments that actually dipped below HWM
    let mut drawdowns = Vec::new();
    let mut dd_times = Vec::new();
    for i in 0..hwms.len() {
        if segment_min[i] < hwms[i] {
            let dd =
                if dollars { hwms[i] - segment_min[i] } else { 1.0 - segment_min[i] / hwms[i] };
            drawdowns.push(dd);
            dd_times.push(hwm_times[i]);
        }
    }

    // Time under water between consecutive HWMs that had drawdowns, plus last interval to series end
    let mut tuw = Vec::new();
    for i in 0..dd_times.len() {
        let start = dd_times[i];
        let end = if i + 1 < dd_times.len() { dd_times[i + 1] } else { returns.last().unwrap().0 };
        let years = (end - start).num_seconds() as f64 / (365.25 * 24.0 * 3600.0);
        tuw.push(years);
    }

    (drawdowns, tuw)
}

/// Annualised Sharpe ratio of per-period returns (AFML §14.7.1).
///
/// Computes `(mean - risk_free_rate) / std * sqrt(entries_per_year)` with the sample standard
/// deviation (ddof = 1). `risk_free_rate` is per period, in the same units as `returns`;
/// `entries_per_year` is the number of return periods per year (252 for daily). Nothing is
/// validated: an empty slice or a single return gives `NaN`, constant returns give infinity.
pub fn sharpe_ratio(returns: &[f64], entries_per_year: f64, risk_free_rate: f64) -> f64 {
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let var =
        returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() as f64 - 1.0);
    let std = var.sqrt();
    ((mean - risk_free_rate) / std) * entries_per_year.sqrt()
}

/// Annualised information ratio against a constant per-period benchmark return.
///
/// Equivalent to [`sharpe_ratio`] on `returns - benchmark` with a zero risk-free rate.
pub fn information_ratio(returns: &[f64], benchmark: f64, entries_per_year: f64) -> f64 {
    let excess: Vec<f64> = returns.iter().map(|r| r - benchmark).collect();
    sharpe_ratio(&excess, entries_per_year, 0.0)
}

/// Probabilistic Sharpe ratio: the probability that the true Sharpe ratio exceeds
/// `benchmark_sr` (AFML §14.7.2; Bailey and López de Prado, 2012).
///
/// `observed_sr` and `benchmark_sr` are **per-period** Sharpe ratios estimated over
/// `number_of_returns` observations; `skewness` is the returns' skewness and `kurtosis` their
/// **raw** kurtosis (3 for normal returns). Passing an annualised Sharpe ratio silently
/// overstates the confidence. Returns a probability in `[0, 1]`.
pub fn probabilistic_sharpe_ratio(
    observed_sr: f64,
    benchmark_sr: f64,
    number_of_returns: usize,
    skewness: f64,
    kurtosis: f64,
) -> f64 {
    let z = ((observed_sr - benchmark_sr) * (number_of_returns as f64 - 1.0).sqrt())
        / (1.0 - skewness * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr * observed_sr)
            .sqrt();
    let norm = Normal::new(0.0, 1.0).unwrap();
    norm.cdf(z)
}

/// Deflated Sharpe ratio: the PSR measured against the Sharpe ratio expected from the best of
/// `N` skill-less trials (AFML §14.7.3; Bailey and López de Prado, 2014).
///
/// The benchmark is `SR_0 = sigma_SR * [(1 - g) Z^-1(1 - 1/N) + g Z^-1(1 - 1/(N e))]`, with
/// `g` the Euler–Mascheroni constant. `sr_estimates` supplies the trials in one of two forms:
///
/// - `estimates_param = false`: every trial's per-period Sharpe ratio; `sigma_SR` is their
///   population standard deviation and `N` their count;
/// - `estimates_param = true`: `[sigma_SR, N]`.
///
/// With `benchmark_out = true` returns `SR_0`; otherwise returns
/// [`probabilistic_sharpe_ratio`]`(observed_sr, SR_0, ...)`. Units follow
/// [`probabilistic_sharpe_ratio`]: per-period Sharpe ratios, raw kurtosis. The formula
/// assumes independent trials; correlated variations of one idea overstate `N`.
///
/// # Errors
///
/// - [`InputError::TooShort`] if `sr_estimates` has fewer than two values.
/// - [`InputError::OutOfRange`] if `estimates_param` is true and the number of trials
///   `sr_estimates[1]` is `NaN` or not above 1.
pub fn deflated_sharpe_ratio(
    observed_sr: f64,
    sr_estimates: &[f64],
    number_of_returns: usize,
    skewness: f64,
    kurtosis: f64,
    estimates_param: bool,
    benchmark_out: bool,
) -> Result<f64, InputError> {
    // Both forms need two values: (std, number of trials), or at least two trial Sharpe ratios.
    if sr_estimates.len() < 2 {
        return Err(InputError::TooShort { name: "sr_estimates", len: sr_estimates.len(), min: 2 });
    }
    let benchmark_sr = if estimates_param {
        let sd = sr_estimates[0];
        let n = sr_estimates[1];
        // The expected maximum of n trials is only defined for more than one trial.
        if n.is_nan() || n <= 1.0 {
            return Err(InputError::OutOfRange {
                name: "sr_estimates[1]",
                value: n,
                expected: "a number of trials above 1",
            });
        }
        let norm = Normal::new(0.0, 1.0).unwrap();
        sd * ((1.0 - EULER_GAMMA) * norm.inverse_cdf(1.0 - 1.0 / n)
            + EULER_GAMMA * norm.inverse_cdf(1.0 - 1.0 / n * (-1.0f64).exp()))
    } else {
        let sd = {
            let mean = sr_estimates.iter().sum::<f64>() / sr_estimates.len() as f64;
            (sr_estimates.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / sr_estimates.len() as f64)
                .sqrt()
        };
        let n = sr_estimates.len() as f64;
        let norm = Normal::new(0.0, 1.0).unwrap();
        sd * ((1.0 - EULER_GAMMA) * norm.inverse_cdf(1.0 - 1.0 / n)
            + EULER_GAMMA * norm.inverse_cdf(1.0 - 1.0 / n * (-1.0f64).exp()))
    };

    if benchmark_out {
        return Ok(benchmark_sr);
    }

    Ok(probabilistic_sharpe_ratio(observed_sr, benchmark_sr, number_of_returns, skewness, kurtosis))
}

/// Minimum number of observations for the PSR against `benchmark_sr` to reach `1 - alpha`
/// (AFML §14.7.2; Bailey and López de Prado, 2012).
///
/// Units follow [`probabilistic_sharpe_ratio`]: per-period Sharpe ratios, raw kurtosis. The
/// result is a number of return periods. It is only meaningful when
/// `observed_sr > benchmark_sr`: the difference is squared, so an underperforming strategy
/// gets a finite positive answer and an equal one gets infinity.
///
/// # Errors
///
/// [`InputError::OutOfRange`] if `alpha` is outside `[0, 1]`.
pub fn minimum_track_record_length(
    observed_sr: f64,
    benchmark_sr: f64,
    skewness: f64,
    kurtosis: f64,
    alpha: f64,
) -> Result<f64, InputError> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(InputError::OutOfRange {
            name: "alpha",
            value: alpha,
            expected: "a significance level in [0, 1]",
        });
    }
    let norm = Normal::new(0.0, 1.0).unwrap();
    let z = norm.inverse_cdf(1.0 - alpha);
    Ok(1.0
        + (1.0 - skewness * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr * observed_sr)
            * (z / (observed_sr - benchmark_sr)).powi(2))
}
