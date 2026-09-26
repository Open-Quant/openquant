//! Volatility estimators: AFML's daily volatility target and three range-based estimators.
//!
//! - [`get_daily_vol`]: the exponentially weighted standard deviation of daily returns that
//!   AFML uses to scale triple-barrier widths (§3.3, Snippet 3.1).
//! - [`get_parkinson_vol`], [`get_garman_class_vol`], [`get_yang_zhang_vol`]: rolling
//!   estimators from bar open, high, low and close, which extract more information per bar
//!   than close-to-close returns.
//!
//! Conventions:
//!
//! - Inputs are ordered oldest first; prices are levels (not log prices) and must be positive.
//! - Every estimator returns volatility **per bar** (per day for [`get_daily_vol`]), as a
//!   standard deviation of returns, not annualised.
//! - The rolling estimators return one value per input bar, `NaN` until the first full
//!   window, and `NaN` for any window containing a `NaN` input.
//!
//! ```
//! use openquant::util::volatility::{get_parkinson_vol, get_yang_zhang_vol};
//!
//! let open = [100.0, 101.0, 100.5, 102.0, 101.5];
//! let high = [101.5, 102.0, 101.8, 103.0, 102.4];
//! let low = [99.5, 100.2, 99.9, 101.1, 100.8];
//! let close = [101.0, 100.6, 101.7, 101.4, 102.0];
//!
//! let parkinson = get_parkinson_vol(&high, &low, 3)?;
//! assert!(parkinson[..2].iter().all(|v| v.is_nan()));
//! assert!(parkinson[2] > 0.0 && parkinson[2] < 0.02);
//!
//! // Yang-Zhang needs the previous close, so its first `window` values are NaN.
//! let yz = get_yang_zhang_vol(&open, &high, &low, &close, 3)?;
//! assert!(yz[..3].iter().all(|v| v.is_nan()));
//! assert!(yz[3] > 0.0);
//! # Ok::<(), openquant::util::InputError>(())
//! ```

use super::input_error::same_length;
use super::InputError;
use chrono::{Duration, NaiveDateTime};

/// Daily volatility: the exponentially weighted standard deviation of daily returns (AFML
/// Snippet 3.1).
///
/// `close` is a timestamped price series, oldest first. For each bar the return is
/// `price / previous - 1`, where `previous` is the price of the last bar strictly before one
/// day earlier; bars with no such earlier bar are skipped, so the output starts about a day
/// into the series and carries each remaining bar's timestamp. The returns are then smoothed
/// with pandas' `ewm(span=lookback).std()` (`adjust=True`, bias-corrected), so the first
/// returned value is `NaN`. The result is in return units per day, not annualised, and is
/// the usual target for [`crate::labeling`]'s barrier widths. Mirrors mlfinlab's
/// `get_daily_vol`.
///
/// Returns an empty vector when `close` has fewer than two bars or `lookback == 0`.
///
/// ```
/// use chrono::{Duration, NaiveDate};
/// use openquant::util::volatility::get_daily_vol;
///
/// let t0 = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap();
/// let close: Vec<_> =
///     (0..10).map(|i| (t0 + Duration::days(i), 100.0 * 1.01_f64.powi(i as i32))).collect();
///
/// let vol = get_daily_vol(&close, 5);
/// // Day 0 and day 1 have no bar more than a day before them.
/// assert_eq!(vol.len(), 8);
/// assert_eq!(vol[0].0, close[2].0);
/// assert!(vol[0].1.is_nan());
/// // Every return is the same two-day 2.01%, so the spread is zero.
/// assert!(vol[7].1.abs() < 1e-12);
/// ```
pub fn get_daily_vol(close: &[(NaiveDateTime, f64)], lookback: usize) -> Vec<(NaiveDateTime, f64)> {
    if close.len() < 2 || lookback == 0 {
        return Vec::new();
    }

    // AFML snippet 3.1: the return over (at least) the previous day, then
    // `ewm(span=lookback).std()`. pandas' defaults are `adjust=True, bias=False`: weights
    // (1 - alpha)^k over the whole history and the unbiased weighted variance. The mean and
    // variance are updated incrementally, as pandas does, rather than as
    // `sum(w x^2)/sum(w) - mean^2`, which cancels catastrophically when returns barely vary.
    let decay = 1.0 - 2.0 / (lookback as f64 + 1.0);
    let (mut sum_wt, mut sum_wt2, mut old_wt) = (0.0f64, 0.0f64, 0.0f64);
    let (mut mean, mut var) = (0.0f64, 0.0f64);

    let mut out = Vec::new();
    for (i, &(ts_i, price_i)) in close.iter().enumerate() {
        // `searchsorted(t - 1 day) - 1`: the last bar strictly before one day ago.
        let target_time = ts_i - Duration::days(1);
        let Some(j) = close[..i].iter().rposition(|(ts_j, _)| *ts_j < target_time) else {
            continue;
        };
        let ret = price_i / close[j].1 - 1.0;

        sum_wt *= decay;
        sum_wt2 *= decay * decay;
        old_wt *= decay;

        let old_mean = mean;
        let total = old_wt + 1.0;
        mean = (old_wt * old_mean + ret) / total;
        var = (old_wt * (var + (old_mean - mean).powi(2)) + (ret - mean).powi(2)) / total;

        sum_wt += 1.0;
        sum_wt2 += 1.0;
        old_wt += 1.0;

        // One observation has no sample variance: pandas reports NaN there and so does this,
        // so the result lines up with `ewm().std()` row for row. `get_events` drops a NaN
        // target, exactly as `target[target > min_ret]` does.
        let denom = sum_wt * sum_wt - sum_wt2;
        if denom <= 0.0 {
            out.push((ts_i, f64::NAN));
            continue;
        }
        out.push((ts_i, (var.max(0.0) * sum_wt * sum_wt / denom).sqrt()));
    }

    out
}

/// Parkinson (1980) range-based volatility over a rolling window of `window` bars.
///
/// `sigma² = 1 / (4 ln 2) · mean(ln(H_i / L_i)²)` over the `window` bars ending at each bar.
/// Returns `sigma` per bar (not annualised), one value per input bar, `NaN` for the first
/// `window - 1` bars; `window == 0` gives all `NaN`. It ignores overnight gaps and assumes no
/// drift. Mirrors mlfinlab's `get_parksinson_vol` (upstream misspells "Parkinson"; this crate
/// spells it correctly).
///
/// # Errors
///
/// [`InputError`] if `low` differs in length from `high`.
///
/// ```
/// use openquant::util::volatility::get_parkinson_vol;
///
/// // A constant 1% range: sigma = ln(1.01) / sqrt(4 ln 2) on every full window.
/// let vol = get_parkinson_vol(&[101.0; 4], &[100.0; 4], 2)?;
/// assert!(vol[0].is_nan());
/// let expected = (1.01_f64).ln() / (4.0 * 2.0_f64.ln()).sqrt();
/// assert!((vol[3] - expected).abs() < 1e-12);
/// # Ok::<(), openquant::util::InputError>(())
/// ```
pub fn get_parkinson_vol(high: &[f64], low: &[f64], window: usize) -> Result<Vec<f64>, InputError> {
    same_length("low", low, high.len())?;
    let estimator: Vec<f64> = high
        .iter()
        .zip(low.iter())
        .map(|(&h, &l)| {
            let ret = (h / l).ln();
            (ret * ret) / (4.0 * 2.0f64.ln())
        })
        .collect();
    Ok(rolling_sqrt_mean(&estimator, window))
}

/// Garman-Klass (1980) volatility over a rolling window of `window` bars.
///
/// `sigma² = mean(0.5 ln(H_i / L_i)² - (2 ln 2 - 1) ln(C_i / O_i)²)` over the `window` bars
/// ending at each bar. Returns `sigma` per bar (not annualised), one value per input bar,
/// `NaN` for the first `window - 1` bars; `window == 0` gives all `NaN`. Like Parkinson it
/// ignores overnight gaps. The name keeps mlfinlab's spelling (`get_garman_class_vol`).
///
/// # Errors
///
/// [`InputError`] if `high`, `low` or `close` differs in length from `open`.
///
/// ```
/// use openquant::util::volatility::get_garman_class_vol;
///
/// let vol = get_garman_class_vol(
///     &[100.0, 101.0, 100.5],
///     &[101.5, 102.0, 101.8],
///     &[99.5, 100.2, 99.9],
///     &[101.0, 100.6, 101.7],
///     2,
/// )?;
/// assert!(vol[0].is_nan());
/// assert!(vol[1] > 0.0 && vol[2] > 0.0);
/// # Ok::<(), openquant::util::InputError>(())
/// ```
pub fn get_garman_class_vol(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    window: usize,
) -> Result<Vec<f64>, InputError> {
    same_length("high", high, open.len())?;
    same_length("low", low, open.len())?;
    same_length("close", close, open.len())?;

    let c = 2.0 * 2.0f64.ln() - 1.0;
    let estimator: Vec<f64> = open
        .iter()
        .zip(high.iter())
        .zip(low.iter())
        .zip(close.iter())
        .map(|(((&o, &h), &l), &c_px)| {
            let hl = (h / l).ln();
            let co = (c_px / o).ln();
            0.5 * hl * hl - c * co * co
        })
        .collect();
    Ok(rolling_sqrt_mean(&estimator, window))
}

/// Yang-Zhang volatility estimator (Yang & Zhang 2000) over a rolling window of `window` bars.
///
/// For the `n = window` bars ending at bar `t`:
///
/// ```text
/// sigma^2    = sigma_o^2 + k sigma_c^2 + (1 - k) sigma_rs^2,  k = 0.34 / (1.34 + (n+1)/(n-1))
/// sigma_o^2  = 1/(n-1) sum (o_i - mean(o))^2,  o_i = ln(O_i / C_{i-1})   (overnight)
/// sigma_c^2  = 1/(n-1) sum (c_i - mean(c))^2,  c_i = ln(C_i / O_i)       (open to close)
/// sigma_rs^2 = 1/n sum [ln(H_i/C_i) ln(H_i/O_i) + ln(L_i/C_i) ln(L_i/O_i)]  (Rogers-Satchell)
/// ```
///
/// Returns `sigma` for each bar, per bar (not annualised). The overnight return needs the
/// previous close, so the first `window` values are NaN; a window containing a NaN input gives
/// NaN, and `window < 2` gives all NaN.
///
/// mlfinlab's `get_yang_zhang_vol`, which this function used to mirror, differs in two ways:
/// its close term is `ln(C_i / O_{i-1})` rather than `ln(C_i / O_i)`, and it does not demean
/// `o` and `c` (it also divides the Rogers-Satchell sum by `n - 1`).
///
/// # Errors
///
/// [`InputError`] if `high`, `low` or `close` differs in length from `open`.
pub fn get_yang_zhang_vol(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    window: usize,
) -> Result<Vec<f64>, InputError> {
    same_length("high", high, open.len())?;
    same_length("low", low, open.len())?;
    same_length("close", close, open.len())?;

    let n = open.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if window < 2 {
        return Ok(vec![f64::NAN; n]);
    }

    let w = window as f64;
    let k = 0.34 / (1.34 + (w + 1.0) / (w - 1.0));

    // o_i = ln(O_i / C_{i-1}) needs the previous close, so it starts at bar 1.
    let mut overnight = vec![f64::NAN; n];
    for i in 1..n {
        overnight[i] = (open[i] / close[i - 1]).ln();
    }
    let open_close: Vec<f64> = open.iter().zip(close).map(|(&o, &c)| (c / o).ln()).collect();
    let rogers_satchell: Vec<f64> = (0..n)
        .map(|i| {
            (high[i] / close[i]).ln() * (high[i] / open[i]).ln()
                + (low[i] / close[i]).ln() * (low[i] / open[i]).ln()
        })
        .collect();

    // Two-pass sample variance of each window: the returns' means are small next to their
    // spread, and a running sum of squares would lose the digits the estimator needs.
    // Each window holds `window >= 2` values, so the sample variance exists.
    let sample_var = |x: &[f64]| super::stats::variance(x, 1).unwrap_or(f64::NAN);

    let mut out = vec![f64::NAN; n];
    // The first window with `window` overnight returns ends at bar `window`.
    for (t, out_t) in out.iter_mut().enumerate().skip(window) {
        let bars = t + 1 - window..t + 1;
        let var_o = sample_var(&overnight[bars.clone()]);
        let var_c = sample_var(&open_close[bars.clone()]);
        let var_rs = rogers_satchell[bars].iter().sum::<f64>() / w;
        *out_t = (var_o + k * var_c + (1.0 - k) * var_rs).sqrt();
    }
    Ok(out)
}

fn rolling_sqrt_mean(values: &[f64], window: usize) -> Vec<f64> {
    rolling_sum_with_min_periods(values, window, window)
        .iter()
        .map(|&x| if x.is_nan() { f64::NAN } else { (x / window as f64).sqrt() })
        .collect()
}

fn rolling_sum_with_min_periods(values: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    if window == 0 {
        return vec![f64::NAN; n];
    }

    let mut out = vec![f64::NAN; n];
    let mut sum = 0.0;
    let mut valid = 0usize;

    for i in 0..n {
        let x = values[i];
        if !x.is_nan() {
            sum += x;
            valid += 1;
        }
        if i >= window {
            let old = values[i - window];
            if !old.is_nan() {
                sum -= old;
                valid -= 1;
            }
        }
        if i + 1 >= window && valid >= min_periods {
            out[i] = sum;
        }
    }
    out
}
