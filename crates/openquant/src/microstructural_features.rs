//! Market-microstructure features (AFML Chapter 19) and entropy features (AFML Chapter 18).
//!
//! Most price data carries no record of the spread, of how far an order moved the price, or
//! of whether flow was one-sided. Microstructure theory infers them from what is recorded:
//!
//! - **Spread** (§19.3): [`get_roll_measure`] and [`get_roll_impact`] (Roll, 1984; §19.3.2),
//!   [`get_corwin_schultz_estimator`] (Corwin and Schultz, 2012; §19.3.4, Snippet 19.1) and
//!   the high–low volatility net of that spread, [`get_bekker_parkinson_vol`] (Snippet 19.2).
//! - **Price impact** (§19.4): Kyle's, Amihud's and Hasbrouck's lambdas, each in a
//!   `bar_based` form (a rolling mean of per-bar ratios, sign inferred from the bar's price
//!   change) and a `trades_based` form (a regression through the origin over one bar's
//!   trades with their aggressor flags).
//! - **Order flow** (§19.5): [`get_bvc_buy_volume`] (bulk volume classification) and
//!   [`get_vpin`] (Easley, López de Prado and O'Hara, 2012; §19.5.2).
//! - **Entropy** (Chapter 18): encode a series as a string ([`encode_tick_rule_array`],
//!   [`quantile_mapping`], [`sigma_mapping`], [`encode_array`]; §18.5), then estimate its
//!   entropy in bits per symbol ([`get_shannon_entropy`] §18.2, [`get_plug_in_entropy`]
//!   §18.3, [`get_lempel_ziv_entropy`] and [`get_konto_entropy`] §18.4).
//! - [`MicrostructuralFeaturesGenerator`] streams a trades CSV and emits the trades-based
//!   features and entropies per bar.
//!
//! Conventions:
//!
//! - Inputs are **levels** (prices, volumes, dollar volumes) in time order, oldest first,
//!   except the `trades_based` functions, which take per-trade price changes or log returns.
//! - Rolling functions return a vector as long as the input, with `NaN` until a full window
//!   of valid values is available (and wherever a window contains a `NaN`).
//! - Paired inputs must have the same length; a mismatch is an
//!   [`InputError::LengthMismatch`].
//! - Aggressor flags and tick signs are `+1` (buy / uptick), `-1` (sell / downtick), `0`.
//! - Corwin–Schultz returns a *relative* spread (a fraction of price); Roll returns a spread
//!   in price units.
//!
//! Caveats (see the module docs page for the simulation behind them):
//!
//! - The Roll measure uses `2 sqrt(|cov|)`, following mlfinlab: a trending series with
//!   *positive* autocovariance also reports a "spread".
//! - [`get_vpin`] divides by the current bar's volume, not the window's; it is the
//!   published measure only on volume bars.
//! - Bar-based lambdas are means of ratios, not regressions; one low-volume bar can
//!   dominate a window.
//! - Trades-based Kyle's lambda regresses *transaction* price changes, so it measures impact
//!   plus bid–ask bounce, not permanent impact.
//! - Codebooks from [`quantile_mapping`] and [`sigma_mapping`] assign letters from character
//!   0 upward (including control characters such as NUL); alphabets are capped at 256.
//! - Entropy estimates are biased on short messages; compare messages of equal length and
//!   encoding. The Lempel–Ziv and Kontoyiannis estimators are quadratic or worse in the
//!   message length.
//!
//! ```
//! use openquant::microstructural_features::{
//!     encode_tick_rule_array, get_lempel_ziv_entropy, get_plug_in_entropy, get_shannon_entropy,
//!     get_trades_based_kyle_lambda, get_vpin, vwap,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Regression through the origin of price change on signed volume: sum(xy) / sum(x^2).
//! let lambda =
//!     get_trades_based_kyle_lambda(&[0.2, -0.1, 0.4], &[10.0, 5.0, 20.0], &[1.0, -1.0, 1.0])?;
//! assert!((lambda - 0.02).abs() < 1e-12);
//!
//! // VPIN over 3 bars of equal volume: mean |buys - sells| / volume. NaN until the window fills.
//! let vpin = get_vpin(&[100.0; 5], &[80.0, 20.0, 50.0, 90.0, 10.0], 3)?;
//! assert!(vpin[0].is_nan() && vpin[1].is_nan());
//! assert!((vpin[2] - 0.4).abs() < 1e-12);
//!
//! assert!((vwap(&[1000.0, 2000.0], &[10.0, 10.0])? - 150.0).abs() < 1e-12);
//!
//! // A perfectly periodic message: maximal Shannon entropy, low entropy by every other measure.
//! assert_eq!(encode_tick_rule_array(&[1, 1, -1, 0])?, "aabc");
//! assert_eq!(get_shannon_entropy("abababab"), 1.0);
//! assert_eq!(get_plug_in_entropy("abababab", 2)?, 0.5);
//! assert_eq!(get_lempel_ziv_entropy("abababab"), 0.5);
//! # Ok(())
//! # }
//! ```

use crate::util::input_error::same_length;
use crate::util::stats::{self, quantile_sorted, QuantileMethod};
use crate::util::InputError;
use chrono::NaiveDateTime;
use statrs::distribution::{ContinuousCDF, Normal};

/// Errors returned by the encoding functions and by [`MicrostructuralFeaturesGenerator`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MicrostructuralError {
    /// [`encode_tick_rule_array`] received a value other than `1`, `-1` or `0`.
    #[error("Unknown value for tick rule: {0}")]
    UnknownTickRule(i32),
    /// [`quantile_mapping`] was asked for zero letters or more than 256.
    #[error("num_letters out of range")]
    NumLettersOutOfRange,
    /// [`quantile_mapping`] or [`sigma_mapping`] received an empty array.
    #[error("array must not be empty")]
    EmptyArray,
    /// [`quantile_mapping`], [`sigma_mapping`] or [`encode_array`] received a `NaN` value.
    #[error("array must not contain NaN")]
    NanInArray,
    /// [`sigma_mapping`] received a step that is not a positive finite number (zero,
    /// negative, `NaN` or infinite).
    #[error("step must be positive and finite")]
    NonPositiveStep,
    /// [`encode_array`] or [`MicrostructuralFeaturesGenerator::new_from_csv`] received a
    /// codebook that is empty or has a `NaN` value, so some value would have no letter.
    #[error("invalid codebook: {0}")]
    InvalidCodebook(&'static str),
    /// [`sigma_mapping`] would need more than 256 letters.
    #[error("Length of dictionary exceeds ASCII table")]
    DictionaryTooLong,
    /// The first data row of the trades CSV does not have exactly three columns.
    #[error("Must have only 3 columns in csv: date_time, price, & volume.")]
    WrongColumnCount,
    /// The price in the first data row of the trades CSV is not a number.
    #[error("price column in csv not float.")]
    PriceNotFloat,
    /// The volume in the first data row of the trades CSV is not a number.
    #[error("volume column in csv not int or float.")]
    VolumeNotNumeric,
    /// The timestamp in the first data row of the trades CSV is not in a supported format.
    #[error("column 0 not datetime")]
    TimestampNotDatetime,
    /// A trades CSV row has fewer than three columns; the payload is the column count.
    #[error("expected date_time, price, volume; got {0} columns")]
    ShortRow(usize),
    /// Reading or parsing the trades CSV failed; the payload is the underlying error.
    #[error("{0}")]
    Csv(String),
    /// A feature function rejected its inputs.
    #[error(transparent)]
    Input(#[from] InputError),
}

fn rolling_cov(x: &[f64], y: &[f64], window: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    if window < 2 {
        return out;
    }
    for i in 0..n {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let slice_x = &x[start..=i];
        let slice_y = &y[start..=i];
        let mean_x: f64 = slice_x.iter().sum::<f64>() / window as f64;
        let mean_y: f64 = slice_y.iter().sum::<f64>() / window as f64;
        let mut s = 0.0;
        for j in 0..window {
            s += (slice_x[j] - mean_x) * (slice_y[j] - mean_y);
        }
        out[i] = s / (window as f64 - 1.0);
    }
    out
}

/// Roll's (1984) effective bid–ask spread, `2 sqrt(|cov(Δp_t, Δp_{t−1})|)`, over a rolling
/// window of `close` prices (AFML §19.3.2).
///
/// The covariance is the sample (`n − 1`) covariance of the price changes with their first
/// lag over the last `window` bars; the result is in price units. Roll's formula needs a
/// *negative* autocovariance; following mlfinlab this takes the absolute value, so a
/// trending series with positive autocovariance also reports a "spread". Check the sign
/// yourself on anything that is not a liquid, mean-reverting tick series.
///
/// The first valid value is at index `window + 1`; earlier values, and every value when
/// `window < 2` or `close.len() < 2`, are `NaN`.
///
/// ```
/// use openquant::microstructural_features::get_roll_measure;
///
/// // Bouncing prices: the lag-1 covariance over the last 3 changes is -4/3.
/// let roll = get_roll_measure(&[10.0, 11.0, 10.0, 11.0, 10.0], 3);
/// assert!(roll[..4].iter().all(|v| v.is_nan()));
/// assert!((roll[4] - 2.0 * (4.0f64 / 3.0).sqrt()).abs() < 1e-12);
/// ```
pub fn get_roll_measure(close: &[f64], window: usize) -> Vec<f64> {
    if close.len() < 2 {
        return vec![f64::NAN; close.len()];
    }
    let mut diff = vec![f64::NAN; close.len()];
    for i in 1..close.len() {
        diff[i] = close[i] - close[i - 1];
    }
    let mut diff_lag = vec![f64::NAN; close.len()];
    diff_lag[1..].copy_from_slice(&diff[..diff.len() - 1]);
    let cov = rolling_cov(&diff, &diff_lag, window);
    cov.iter().map(|c| if c.is_nan() { f64::NAN } else { 2.0 * (c.abs()).sqrt() }).collect()
}

/// Roll measure divided by the bar's dollar volume: spread cost per unit of dollar volume
/// (AFML §19.3.2).
///
/// Element-wise [`get_roll_measure`]`(close, window) / dollar_volume`; `NaN` where the Roll
/// measure is `NaN` or the dollar volume is zero.
///
/// # Errors
///
/// [`InputError::LengthMismatch`] if `dollar_volume.len() != close.len()`.
pub fn get_roll_impact(
    close: &[f64],
    dollar_volume: &[f64],
    window: usize,
) -> Result<Vec<f64>, InputError> {
    same_length("dollar_volume", dollar_volume, close.len())?;
    let roll = get_roll_measure(close, window);
    Ok(roll
        .iter()
        .zip(dollar_volume.iter())
        .map(|(r, dv)| if r.is_nan() || *dv == 0.0 { f64::NAN } else { r / dv })
        .collect())
}

fn rolling_max(arr: &[f64], window: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; arr.len()];
    if window == 0 {
        return out;
    }
    for i in 0..arr.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let max_v = arr[start..=i].iter().fold(f64::NEG_INFINITY, |m, v| m.max(*v));
        out[i] = max_v;
    }
    out
}

fn rolling_min(arr: &[f64], window: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; arr.len()];
    if window == 0 {
        return out;
    }
    for i in 0..arr.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let min_v = arr[start..=i].iter().fold(f64::INFINITY, |m, v| m.min(*v));
        out[i] = min_v;
    }
    out
}

fn _get_beta(high: &[f64], low: &[f64], window: usize) -> Vec<f64> {
    let mut ret_sq = vec![f64::NAN; high.len()];
    for i in 0..high.len() {
        if low[i] == 0.0 {
            continue;
        }
        ret_sq[i] = (high[i] / low[i]).ln().powi(2);
    }
    // rolling sum over 2
    let mut two_sum = vec![f64::NAN; ret_sq.len()];
    for i in 1..ret_sq.len() {
        if ret_sq[i].is_nan() || ret_sq[i - 1].is_nan() {
            continue;
        }
        two_sum[i] = ret_sq[i] + ret_sq[i - 1];
    }
    // rolling mean over window
    let mut beta = vec![f64::NAN; ret_sq.len()];
    if window == 0 {
        return beta;
    }
    for i in 0..two_sum.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let slice = &two_sum[start..=i];
        if slice.iter().any(|v| v.is_nan()) {
            continue;
        }
        let mean = slice.iter().sum::<f64>() / window as f64;
        beta[i] = mean;
    }
    beta
}

fn _get_gamma(high: &[f64], low: &[f64]) -> Vec<f64> {
    let high_max = rolling_max(high, 2);
    let low_min = rolling_min(low, 2);
    high_max
        .iter()
        .zip(low_min.iter())
        .map(
            |(h, l)| {
                if h.is_nan() || l.is_nan() || *l == 0.0 {
                    f64::NAN
                } else {
                    (h / l).ln().powi(2)
                }
            },
        )
        .collect()
}

fn _get_alpha(beta: &[f64], gamma: &[f64]) -> Vec<f64> {
    let den = 3.0 - 2.0 * 2.0_f64.sqrt();
    beta.iter()
        .zip(gamma.iter())
        .map(|(b, g)| {
            if b.is_nan() || g.is_nan() {
                return f64::NAN;
            }
            let mut alpha = (2.0_f64.sqrt() - 1.0) * b.sqrt() / den;
            alpha -= (g / den).sqrt();
            if alpha < 0.0 {
                0.0
            } else {
                alpha
            }
        })
        .collect()
}

/// Corwin and Schultz's (2012) *relative* bid–ask spread from bar highs and lows (AFML
/// §19.3.4, Snippet 19.1).
///
/// With `β` the `window`-bar rolling mean of `ln(H_t/L_t)² + ln(H_{t−1}/L_{t−1})²` and `γ`
/// the squared log range of the two-bar high and low, `α = (√(2β) − √β)/(3 − 2√2) −
/// √(γ/(3 − 2√2))` is floored at zero and the spread is `2(e^α − 1)/(1 + e^α)`. Values are
/// `NaN` until `window + 1` bars are available, and wherever a low is zero.
///
/// The estimator goes negative, and is floored at exactly zero, whenever a window's
/// volatility swamps the spread, so its mean over many bars is usable while individual
/// values are not.
///
/// ```
/// use openquant::microstructural_features::get_corwin_schultz_estimator;
///
/// # fn main() -> Result<(), openquant::util::InputError> {
/// // A constant 101/99 range with no volatility is all spread: 2 / 100.
/// let cs = get_corwin_schultz_estimator(&[101.0; 4], &[99.0; 4], 2)?;
/// assert!(cs[0].is_nan() && cs[1].is_nan());
/// assert!((cs[2] - 0.02).abs() < 1e-12 && (cs[3] - 0.02).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// [`InputError::LengthMismatch`] if `low.len() != high.len()`.
pub fn get_corwin_schultz_estimator(
    high: &[f64],
    low: &[f64],
    window: usize,
) -> Result<Vec<f64>, InputError> {
    same_length("low", low, high.len())?;
    let beta = _get_beta(high, low, window);
    let gamma = _get_gamma(high, low);
    let alpha = _get_alpha(&beta, &gamma);
    Ok(alpha
        .iter()
        .map(|a| {
            if a.is_nan() {
                f64::NAN
            } else {
                let ea = a.exp();
                2.0 * (ea - 1.0) / (1.0 + ea)
            }
        })
        .collect())
}

/// Bekker–Parkinson volatility: the high–low (Parkinson) volatility with the Corwin–Schultz
/// spread component removed (AFML §19.3.4, Snippet 19.2).
///
/// Uses the same `β` and `γ` as [`get_corwin_schultz_estimator`]:
/// `σ = (2^{−1/2} − 1)√β / (k₂(3 − 2√2)) + √(γ / (k₂²(3 − 2√2)))` with `k₂ = √(8/π)`, floored
/// at zero. It is a per-bar volatility of log prices, not annualised. Values are `NaN`
/// until `window + 1` bars are available.
///
/// ```
/// use openquant::microstructural_features::get_bekker_parkinson_vol;
///
/// # fn main() -> Result<(), openquant::util::InputError> {
/// // A constant range is all spread, so nothing is left for volatility.
/// let vol = get_bekker_parkinson_vol(&[101.0; 4], &[99.0; 4], 2)?;
/// assert!(vol[1].is_nan());
/// assert!(vol[3].abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// [`InputError::LengthMismatch`] if `low.len() != high.len()`.
pub fn get_bekker_parkinson_vol(
    high: &[f64],
    low: &[f64],
    window: usize,
) -> Result<Vec<f64>, InputError> {
    same_length("low", low, high.len())?;
    let beta = _get_beta(high, low, window);
    let gamma = _get_gamma(high, low);
    let k2 = (8.0 / std::f64::consts::PI).sqrt();
    let den = 3.0 - 2.0 * 2.0_f64.sqrt();
    Ok(beta
        .iter()
        .zip(gamma.iter())
        .map(|(b, g)| {
            if b.is_nan() || g.is_nan() {
                return f64::NAN;
            }
            let mut sigma = (2.0_f64.powf(-0.5) - 1.0) * b.sqrt() / (k2 * den);
            sigma += (g / (k2 * k2 * den)).sqrt();
            if sigma < 0.0 {
                0.0
            } else {
                sigma
            }
        })
        .collect())
}

/// Bar-based Kyle's lambda: the rolling mean over `window` bars of `Δp_t / (V_t · b_t)`
/// (AFML §19.4.1).
///
/// `Δp_t` is the close-to-close change, `V_t` the bar's volume and `b_t` the sign of `Δp_t`,
/// carried forward over unchanged bars. Because the sign comes from the same price change,
/// each ratio equals `|Δp_t| / V_t` and the result is never negative. It is a mean of
/// ratios, not a regression: one bar with tiny volume contributes an enormous ratio and
/// dominates the window. A window containing the first bar or a zero-volume bar is `NaN`,
/// so the first valid value is at index `window`.
///
/// ```
/// use openquant::microstructural_features::get_bar_based_kyle_lambda;
///
/// # fn main() -> Result<(), openquant::util::InputError> {
/// let lambda = get_bar_based_kyle_lambda(&[10.0, 11.0, 10.5, 11.5], &[100.0; 4], 2)?;
/// assert!(lambda[1].is_nan());
/// assert!((lambda[2] - 0.0075).abs() < 1e-12); // mean of 1/100 and 0.5/100
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// [`InputError::LengthMismatch`] if `volume.len() != close.len()`.
pub fn get_bar_based_kyle_lambda(
    close: &[f64],
    volume: &[f64],
    window: usize,
) -> Result<Vec<f64>, InputError> {
    same_length("volume", volume, close.len())?;
    let mut diff = vec![f64::NAN; close.len()];
    for i in 1..close.len() {
        diff[i] = close[i] - close[i - 1];
    }
    let mut sign = vec![f64::NAN; diff.len()];
    for i in 0..diff.len() {
        let s = diff[i].signum();
        sign[i] = if s == 0.0 && i > 0 { sign[i - 1] } else { s };
    }
    let ratio: Vec<f64> = diff
        .iter()
        .zip(volume.iter())
        .zip(sign.iter())
        .map(|((d, v), s)| if *v == 0.0 || s.is_nan() { f64::NAN } else { d / (v * s) })
        .collect();
    let mut out = vec![f64::NAN; close.len()];
    for i in 0..close.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let slice = &ratio[start..=i];
        if slice.iter().any(|v| v.is_nan()) {
            continue;
        }
        out[i] = slice.iter().sum::<f64>() / window as f64;
    }
    Ok(out)
}

/// Bar-based Amihud's lambda: the rolling mean over `window` bars of `|r_t| / DV_t`, with
/// `r_t` the close-to-close log return and `DV_t` the bar's dollar volume (AFML §19.4.2).
///
/// Needs no trade signs. A bar with zero dollar volume contributes 0 to the sum but still
/// counts in the `window` divisor; a window containing the first bar, a zero previous close
/// or a `NaN` dollar volume is `NaN`.
///
/// ```
/// use openquant::microstructural_features::get_bar_based_amihud_lambda;
///
/// # fn main() -> Result<(), openquant::util::InputError> {
/// let lambda = get_bar_based_amihud_lambda(&[100.0, 110.0, 99.0], &[1e6; 3], 2)?;
/// let expected = (1.1f64.ln() + 0.9f64.ln().abs()) / 1e6 / 2.0;
/// assert!((lambda[2] - expected).abs() < 1e-18);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// [`InputError::LengthMismatch`] if `dollar_volume.len() != close.len()`.
pub fn get_bar_based_amihud_lambda(
    close: &[f64],
    dollar_volume: &[f64],
    window: usize,
) -> Result<Vec<f64>, InputError> {
    same_length("dollar_volume", dollar_volume, close.len())?;
    let mut ret_abs = vec![f64::NAN; close.len()];
    for i in 1..close.len() {
        if close[i - 1] == 0.0 {
            continue;
        }
        ret_abs[i] = (close[i] / close[i - 1]).ln().abs();
    }
    let mut out = vec![f64::NAN; close.len()];
    for i in 0..close.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let r = &ret_abs[start..=i];
        let dv = &dollar_volume[start..=i];
        if r.iter().any(|v| v.is_nan()) || dv.iter().any(|v| v.is_nan()) {
            continue;
        }
        let mut sum = 0.0;
        for (a, b) in r.iter().zip(dv.iter()) {
            if *b != 0.0 {
                sum += a / b;
            }
        }
        out[i] = sum / window as f64;
    }
    Ok(out)
}

/// Bar-based Hasbrouck's lambda: the rolling mean over `window` bars of
/// `r_t / (b_t √DV_t)` (AFML §19.4.3).
///
/// `r_t` is the close-to-close log return, `DV_t` the bar's dollar volume and `b_t` the sign
/// of `r_t`, carried forward over unchanged bars; each ratio therefore equals
/// `|r_t| / √DV_t`. A bar with zero dollar volume contributes 0 but still counts in the
/// `window` divisor; a window containing the first bar, a zero previous close or a negative
/// dollar volume is `NaN`.
///
/// ```
/// use openquant::microstructural_features::get_bar_based_hasbrouck_lambda;
///
/// # fn main() -> Result<(), openquant::util::InputError> {
/// let lambda = get_bar_based_hasbrouck_lambda(&[100.0, 110.0, 99.0], &[1e4; 3], 2)?;
/// let expected = (1.1f64.ln() + 0.9f64.ln().abs()) / 100.0 / 2.0;
/// assert!((lambda[2] - expected).abs() < 1e-15);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// [`InputError::LengthMismatch`] if `dollar_volume.len() != close.len()`.
pub fn get_bar_based_hasbrouck_lambda(
    close: &[f64],
    dollar_volume: &[f64],
    window: usize,
) -> Result<Vec<f64>, InputError> {
    same_length("dollar_volume", dollar_volume, close.len())?;
    let mut log_ret = vec![f64::NAN; close.len()];
    for i in 1..close.len() {
        if close[i - 1] == 0.0 {
            continue;
        }
        log_ret[i] = (close[i] / close[i - 1]).ln();
    }
    let mut sign = vec![f64::NAN; log_ret.len()];
    for i in 0..log_ret.len() {
        let s = log_ret[i].signum();
        sign[i] = if s == 0.0 && i > 0 { sign[i - 1] } else { s };
    }
    let signed_sqrt: Vec<f64> = sign
        .iter()
        .zip(dollar_volume.iter())
        .map(|(s, dv)| if s.is_nan() || *dv < 0.0 { f64::NAN } else { s * dv.sqrt() })
        .collect();
    let mut out = vec![f64::NAN; close.len()];
    for i in 0..close.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let lr = &log_ret[start..=i];
        let sdv = &signed_sqrt[start..=i];
        if lr.iter().any(|v| v.is_nan()) || sdv.iter().any(|v| v.is_nan()) {
            continue;
        }
        let mut sum = 0.0;
        for (r, s) in lr.iter().zip(sdv.iter()) {
            if *s != 0.0 {
                sum += r / s;
            }
        }
        out[i] = sum / window as f64;
    }
    Ok(out)
}

/// Trades-based Kyle's lambda: the regression through the origin of per-trade price changes
/// on signed volume, `Σ(Δp · v·a) / Σ(v·a)²` (AFML §19.4.1).
///
/// `price_diff` holds each trade's price change from the previous trade, `volume` its size
/// and `aggressor_flags` its side (`+1` buy, `-1` sell). Returns `NaN` if every signed volume
/// is zero. Because a buy prints at the ask, transaction price changes carry half a spread
/// in the direction of the trade, so the estimate is impact plus bounce; regress mid-price
/// changes if quotes are available. See the module example.
///
/// # Errors
///
/// [`InputError::LengthMismatch`] if `volume` or `aggressor_flags` is not as long as
/// `price_diff`.
pub fn get_trades_based_kyle_lambda(
    price_diff: &[f64],
    volume: &[f64],
    aggressor_flags: &[f64],
) -> Result<f64, InputError> {
    same_length("volume", volume, price_diff.len())?;
    same_length("aggressor_flags", aggressor_flags, price_diff.len())?;
    let signed: Vec<f64> = volume.iter().zip(aggressor_flags.iter()).map(|(v, a)| v * a).collect();
    let num: f64 = signed.iter().zip(price_diff.iter()).map(|(x, y)| x * y).sum();
    let den: f64 = signed.iter().map(|x| x * x).sum();
    Ok(if den == 0.0 { f64::NAN } else { num / den })
}

/// Trades-based Amihud's lambda: the regression through the origin of per-trade absolute log
/// returns on dollar volume, `Σ(DV · |r|) / Σ DV²` (AFML §19.4.2).
///
/// Returns `NaN` if every dollar volume is zero.
///
/// ```
/// use openquant::microstructural_features::get_trades_based_amihud_lambda;
///
/// # fn main() -> Result<(), openquant::util::InputError> {
/// // (100 * 0.01 + 200 * 0.02) / (100^2 + 200^2) = 5 / 50_000
/// let lambda = get_trades_based_amihud_lambda(&[0.01, -0.02], &[100.0, 200.0])?;
/// assert!((lambda - 1e-4).abs() < 1e-15);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// [`InputError::LengthMismatch`] if `dollar_volume.len() != log_ret.len()`.
pub fn get_trades_based_amihud_lambda(
    log_ret: &[f64],
    dollar_volume: &[f64],
) -> Result<f64, InputError> {
    same_length("dollar_volume", dollar_volume, log_ret.len())?;
    let num: f64 = dollar_volume.iter().zip(log_ret.iter()).map(|(x, y)| x * y.abs()).sum();
    let den: f64 = dollar_volume.iter().map(|x| x * x).sum();
    Ok(if den == 0.0 { f64::NAN } else { num / den })
}

/// Trades-based Hasbrouck's lambda: the regression through the origin of per-trade log
/// returns on signed root dollar volume, `Σ(r · a√DV) / Σ(a√DV)²` (AFML §19.4.3).
///
/// `aggressor_flags` are `+1` (buy) or `-1` (sell). Returns `NaN` if every signed root
/// dollar volume is zero, and `NaN` if any dollar volume is negative.
///
/// This regresses the *signed* return, as in AFML. Earlier versions regressed the absolute
/// return, so buys and sells cancelled and the estimate was near zero under balanced flow
/// ([#105](https://github.com/Open-Quant/openquant/issues/105)); results computed before
/// that fix are not comparable.
///
/// ```
/// use openquant::microstructural_features::get_trades_based_hasbrouck_lambda;
///
/// # fn main() -> Result<(), openquant::util::InputError> {
/// // Signed root dollar volume is [10, -20]: (0.1 + 0.4) / (100 + 400).
/// let lambda =
///     get_trades_based_hasbrouck_lambda(&[0.01, -0.02], &[100.0, 400.0], &[1.0, -1.0])?;
/// assert!((lambda - 1e-3).abs() < 1e-15);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// [`InputError::LengthMismatch`] if `dollar_volume` or `aggressor_flags` is not as long as
/// `log_ret`.
pub fn get_trades_based_hasbrouck_lambda(
    log_ret: &[f64],
    dollar_volume: &[f64],
    aggressor_flags: &[f64],
) -> Result<f64, InputError> {
    same_length("dollar_volume", dollar_volume, log_ret.len())?;
    same_length("aggressor_flags", aggressor_flags, log_ret.len())?;
    let signed: Vec<f64> =
        dollar_volume.iter().zip(aggressor_flags.iter()).map(|(v, a)| v.sqrt() * a).collect();
    let num: f64 = signed.iter().zip(log_ret.iter()).map(|(x, y)| x * y).sum();
    let den: f64 = signed.iter().map(|x| x * x).sum();
    Ok(if den == 0.0 { f64::NAN } else { num / den })
}

// Misc helpers
/// Volume-weighted average price, `Σ dollar_volume / Σ volume`; `NaN` if the volumes sum to
/// zero. See the module example.
///
/// # Errors
///
/// [`InputError::LengthMismatch`] if `volume.len() != dollar_volume.len()`.
pub fn vwap(dollar_volume: &[f64], volume: &[f64]) -> Result<f64, InputError> {
    same_length("volume", volume, dollar_volume.len())?;
    let sum_v: f64 = volume.iter().sum();
    if sum_v == 0.0 {
        return Ok(f64::NAN);
    }
    Ok(dollar_volume.iter().sum::<f64>() / sum_v)
}

/// Arithmetic mean of `tick_sizes` (in [`MicrostructuralFeaturesGenerator`], the mean trade
/// size of a bar); `NaN` for an empty slice.
///
/// ```
/// use openquant::microstructural_features::get_avg_tick_size;
///
/// assert_eq!(get_avg_tick_size(&[1.0, 2.0, 6.0]), 3.0);
/// assert!(get_avg_tick_size(&[]).is_nan());
/// ```
pub fn get_avg_tick_size(tick_sizes: &[f64]) -> f64 {
    if tick_sizes.is_empty() {
        return f64::NAN;
    }
    tick_sizes.iter().sum::<f64>() / tick_sizes.len() as f64
}

/// Volume-synchronised probability of informed trading (Easley, López de Prado and O'Hara,
/// 2012; AFML §19.5.2): the rolling mean over `window` bars of `|V_buy − V_sell|`, divided
/// by the **current** bar's volume.
///
/// `V_sell = volume − buy_volume`. Dividing by the current bar's volume rather than the
/// window's makes this the published measure only on **volume bars**, where every bar has
/// the same volume; on time bars a quiet bar inflates it and a busy bar deflates it. Values
/// are `NaN` for the first `window − 1` bars and where the current volume is zero. See the
/// module example.
///
/// # Errors
///
/// [`InputError::LengthMismatch`] if `buy_volume.len() != volume.len()`.
pub fn get_vpin(volume: &[f64], buy_volume: &[f64], window: usize) -> Result<Vec<f64>, InputError> {
    same_length("buy_volume", buy_volume, volume.len())?;
    let sell_volume: Vec<f64> = volume.iter().zip(buy_volume.iter()).map(|(v, b)| v - b).collect();
    let imbalance: Vec<f64> =
        buy_volume.iter().zip(sell_volume.iter()).map(|(b, s)| (b - s).abs()).collect();
    let mut out = vec![f64::NAN; volume.len()];
    for i in 0..volume.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let imb_slice = &imbalance[start..=i];
        let vol = volume[i];
        if vol == 0.0 || imb_slice.iter().any(|v| v.is_nan()) {
            continue;
        }
        let mean_imb = imb_slice.iter().sum::<f64>() / window as f64;
        out[i] = mean_imb / vol;
    }
    Ok(out)
}

/// Bulk volume classification: the buy volume of each bar estimated as
/// `V_t · Φ(Δp_t / σ_Δp)` (AFML §19.5.2, Easley, López de Prado and O'Hara, 2012).
///
/// `Δp_t` is the close-to-close change and `σ_Δp` the sample standard deviation of the last
/// `window` changes (including the current one, floored at `1e-12`); `Φ` is the standard
/// normal CDF. Values are `NaN` until `window` changes are available (index `window`).
/// `window` must be at least 2, the fewest changes a sample standard deviation needs.
///
/// ```
/// use openquant::microstructural_features::get_bvc_buy_volume;
///
/// # fn main() -> Result<(), openquant::util::InputError> {
/// // At the last bar the change is +1 and the 2-change standard deviation is sqrt(2).
/// let buys = get_bvc_buy_volume(&[10.0, 11.0, 10.0, 11.0], &[100.0; 4], 2)?;
/// assert!(buys[1].is_nan());
/// assert!((buys[3] - 76.02499389).abs() < 1e-6); // 100 * Phi(1 / sqrt(2))
/// assert!((buys[2] + buys[3] - 100.0).abs() < 1e-9);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// - [`InputError::LengthMismatch`] if `volume.len() != close.len()`.
/// - [`InputError::OutOfRange`] if `window < 2`.
pub fn get_bvc_buy_volume(
    close: &[f64],
    volume: &[f64],
    window: usize,
) -> Result<Vec<f64>, InputError> {
    same_length("volume", volume, close.len())?;
    if window < 2 {
        return Err(InputError::OutOfRange {
            name: "window",
            value: window as f64,
            expected: "at least 2",
        });
    }
    let mut out = vec![f64::NAN; close.len()];
    let norm = Normal::new(0.0, 1.0).unwrap();
    let mut diff = vec![f64::NAN; close.len()];
    for i in 1..close.len() {
        diff[i] = close[i] - close[i - 1];
    }
    let mut rolling_std = vec![f64::NAN; close.len()];
    for i in 0..close.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let slice = &diff[start..=i];
        if slice.iter().any(|v| v.is_nan()) {
            continue;
        }
        // `slice` has `window >= 2` values, so the sample deviation exists.
        rolling_std[i] = stats::std_dev(slice, 1).unwrap_or(f64::NAN);
    }
    for i in 0..close.len() {
        if diff[i].is_nan() || rolling_std[i].is_nan() {
            continue;
        }
        let z = diff[i] / rolling_std[i].max(1e-12);
        out[i] = volume[i] * norm.cdf(z);
    }
    Ok(out)
}

// Encoding utilities
/// Encodes tick signs as letters, `1 → 'a'`, `-1 → 'b'`, `0 → 'c'` (AFML §18.5). See the
/// module example.
///
/// # Errors
///
/// [`MicrostructuralError::UnknownTickRule`] for any other value.
pub fn encode_tick_rule_array(arr: &[i32]) -> Result<String, MicrostructuralError> {
    let mut s = String::new();
    for v in arr {
        match *v {
            1 => s.push('a'),
            -1 => s.push('b'),
            0 => s.push('c'),
            other => return Err(MicrostructuralError::UnknownTickRule(other)),
        }
    }
    Ok(s)
}

fn ascii_table() -> Vec<char> {
    (0..=255u8).map(char::from).collect()
}

/// Builds a quantile codebook for [`encode_array`] (AFML §18.5): `num_letters` pairs of
/// `(value, letter)`, one per quantile.
///
/// The quantiles are evenly spaced from 0.01 to 1.0 (just 0.01 when `num_letters == 1`);
/// each value is the order statistic at `round(q · (n − 1))`, so bins are roughly equally
/// populated. Letters are characters `0, 1, 2, …` of the 256-character table, starting with
/// control characters such as NUL: fine for the entropy functions, unsafe to print or pass
/// as C text. Duplicate values can appear in the codebook when the data has ties.
///
/// ```
/// use openquant::microstructural_features::{encode_array, quantile_mapping};
///
/// # fn main() -> Result<(), openquant::microstructural_features::MicrostructuralError> {
/// let codebook = quantile_mapping(&[1.0, 2.0, 3.0, 4.0, 5.0], 2)?;
/// assert_eq!(codebook, vec![(1.0, '\u{0}'), (5.0, '\u{1}')]);
/// assert_eq!(encode_array(&[1.2, 4.9], &codebook)?, "\u{0}\u{1}");
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// - [`MicrostructuralError::NumLettersOutOfRange`] if `num_letters` is 0 or above 256.
/// - [`MicrostructuralError::EmptyArray`] if `array` is empty.
/// - [`MicrostructuralError::NanInArray`] if `array` contains `NaN`.
pub fn quantile_mapping(
    array: &[f64],
    num_letters: usize,
) -> Result<Vec<(f64, char)>, MicrostructuralError> {
    if num_letters == 0 || num_letters > 256 {
        return Err(MicrostructuralError::NumLettersOutOfRange);
    }
    if array.is_empty() {
        return Err(MicrostructuralError::EmptyArray);
    }
    if array.iter().any(|v| v.is_nan()) {
        return Err(MicrostructuralError::NanInArray);
    }
    let table = ascii_table();
    let alphabet = &table[..num_letters];
    let mut sorted = array.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut out: Vec<(f64, char)> = Vec::new();
    for (q, letter) in linspace(0.01, 1.0, alphabet.len()).iter().zip(alphabet.iter()) {
        // `sorted` is non-empty (checked above).
        let value = quantile_sorted(&sorted, *q, QuantileMethod::Nearest).unwrap_or(f64::NAN);
        out.push((value, *letter));
    }
    Ok(out)
}

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n as f64 - 1.0);
    (0..n).map(|i| start + step * i as f64).collect()
}

/// Builds a fixed-width codebook for [`encode_array`] (AFML §18.5): values `min, min + step,
/// …` strictly below `max(array)`, lettered from character 0 upward.
///
/// Like [`quantile_mapping`], it rejects an empty array and `NaN` values, and never returns
/// an empty codebook: an array whose values are all equal gives the one entry `(min, '\0')`.
///
/// ```
/// use openquant::microstructural_features::sigma_mapping;
///
/// # fn main() -> Result<(), openquant::microstructural_features::MicrostructuralError> {
/// assert_eq!(sigma_mapping(&[0.0, 0.3, 1.0], 0.5)?, vec![(0.0, '\u{0}'), (0.5, '\u{1}')]);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// - [`MicrostructuralError::NonPositiveStep`] if `step` is not a positive finite number
///   (`NaN` included).
/// - [`MicrostructuralError::EmptyArray`] if `array` is empty.
/// - [`MicrostructuralError::NanInArray`] if `array` contains `NaN`.
/// - [`MicrostructuralError::DictionaryTooLong`] if more than 256 letters would be needed.
pub fn sigma_mapping(array: &[f64], step: f64) -> Result<Vec<(f64, char)>, MicrostructuralError> {
    if !(step > 0.0 && step.is_finite()) {
        return Err(MicrostructuralError::NonPositiveStep);
    }
    if array.is_empty() {
        return Err(MicrostructuralError::EmptyArray);
    }
    if array.iter().any(|v| v.is_nan()) {
        return Err(MicrostructuralError::NanInArray);
    }
    let table = ascii_table();
    let mut out: Vec<(f64, char)> = Vec::new();
    let mut val = array.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = array.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // The minimum always gets a letter, so a constant array has a one-entry codebook.
    loop {
        if out.len() >= table.len() {
            return Err(MicrostructuralError::DictionaryTooLong);
        }
        out.push((val, table[out.len()]));
        val += step;
        if val >= max_val {
            break;
        }
    }
    Ok(out)
}

fn check_codebook(encoding: &[(f64, char)]) -> Result<(), MicrostructuralError> {
    if encoding.is_empty() {
        return Err(MicrostructuralError::InvalidCodebook("the codebook is empty"));
    }
    if encoding.iter().any(|(v, _)| v.is_nan()) {
        return Err(MicrostructuralError::InvalidCodebook("a codebook value is NaN"));
    }
    Ok(())
}

/// The letter of the entry nearest `value`, the first on ties. `encoding` is non-empty and
/// NaN-free, and `value` is not NaN, so there always is one (an infinite `value` still has
/// a nearest entry: the first with the smallest, possibly infinite, distance).
fn find_nearest(enc: &[(f64, char)], value: f64) -> char {
    let mut best = enc[0].1;
    let mut dist = (enc[0].0 - value).abs();
    for (k, c) in &enc[1..] {
        let d = (k - value).abs();
        if d < dist {
            dist = d;
            best = *c;
        }
    }
    best
}

/// Encodes each value as the letter of the nearest codebook value (the first on ties), using
/// a codebook from [`quantile_mapping`] or [`sigma_mapping`] (AFML §18.5).
///
/// The string has exactly one letter per value: nothing is dropped. See [`quantile_mapping`]
/// for an example.
///
/// # Errors
///
/// - [`MicrostructuralError::InvalidCodebook`] if `encoding` is empty or has a `NaN` value.
/// - [`MicrostructuralError::NanInArray`] if `array` contains `NaN`.
pub fn encode_array(
    array: &[f64],
    encoding: &[(f64, char)],
) -> Result<String, MicrostructuralError> {
    check_codebook(encoding)?;
    if array.iter().any(|v| v.is_nan()) {
        return Err(MicrostructuralError::NanInArray);
    }
    Ok(array.iter().map(|v| find_nearest(encoding, *v)).collect())
}

fn parse_datetime(s: &str) -> Result<NaiveDateTime, chrono::ParseError> {
    NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f")
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S"))
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y/%m/%d %H:%M:%S%.f"))
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y/%m/%d %H:%M:%S"))
}

// Entropy helpers
/// Shannon entropy of the message's characters, `−Σ p log₂ p`, in bits per symbol (AFML
/// §18.2).
///
/// It ignores order entirely: `"abababab"` scores 1 bit, the same as a fair coin. An empty
/// message returns (negative) zero. See the module example.
pub fn get_shannon_entropy(message: &str) -> f64 {
    let mut counts = std::collections::HashMap::new();
    for ch in message.chars() {
        *counts.entry(ch).or_insert(0usize) += 1;
    }
    let len = message.chars().count() as f64;
    let mut ent = 0.0;
    for v in counts.values() {
        let freq = *v as f64 / len;
        ent += freq * freq.log2();
    }
    -ent
}

/// Lempel–Ziv complexity: the size of the LZ parsing dictionary divided by the message
/// length (AFML §18.4, Snippet 18.2).
///
/// Lower values mean more repetition. Works on characters, not bytes; an empty message
/// returns 0. The estimate is erratic on short messages and quadratic or worse in length.
/// See the module example.
pub fn get_lempel_ziv_entropy(message: &str) -> f64 {
    if message.is_empty() {
        return 0.0;
    }
    // Chars, not bytes: the encoders emit letters up to U+00FF, which are two bytes in UTF-8.
    let message: Vec<char> = message.chars().collect();
    let mut i = 1usize;
    let mut lib: Vec<&[char]> = vec![&message[0..1]];
    while i < message.len() {
        let mut j = i;
        while j < message.len() {
            let substr = &message[i..=j];
            if !lib.contains(&substr) {
                lib.push(substr);
                break;
            }
            j += 1;
        }
        i = j + 1;
    }
    lib.len() as f64 / message.len() as f64
}

/// Word frequencies. Requires `word_length <= message.len()`.
fn prob_mass_function(message: &[char], word_length: usize) -> Vec<f64> {
    let mut counts: std::collections::HashMap<&[char], usize> = std::collections::HashMap::new();
    for i in word_length..message.len() {
        *counts.entry(&message[i - word_length..i]).or_default() += 1;
    }
    let total = (message.len() - word_length) as f64;
    counts.into_values().map(|count| count as f64 / total).collect()
}

/// Plug-in (maximum-likelihood) entropy rate: the Shannon entropy of overlapping words of
/// `word_length` characters, divided by `word_length`, in bits per symbol (AFML §18.3,
/// Snippet 18.1).
///
/// As in Snippet 18.1, the words are those ending before the last character, so there are
/// `len − word_length` of them and `word_length == len` returns (negative) zero. The estimate
/// is biased downward for long words on short messages (most words are seen once). See the
/// module example.
///
/// # Errors
///
/// [`InputError::OutOfRange`] if `word_length` is 0 or exceeds the message length in
/// characters.
pub fn get_plug_in_entropy(message: &str, word_length: usize) -> Result<f64, InputError> {
    let message: Vec<char> = message.chars().collect();
    if word_length == 0 || word_length > message.len() {
        return Err(InputError::OutOfRange {
            name: "word_length",
            value: word_length as f64,
            expected: "between 1 and the message length",
        });
    }
    let pmf = prob_mass_function(&message, word_length);
    Ok(-pmf.iter().map(|p| p * p.log2()).sum::<f64>() / word_length as f64)
}

fn match_length(message: &[char], start: usize, window: usize) -> usize {
    let mut matched = 0usize;
    let start_window = start.saturating_sub(window);
    for length in 0..window {
        let end1 = start + length + 1;
        if end1 > message.len() {
            break;
        }
        let msg1 = &message[start..end1];
        for j in start_window..start {
            let end0 = j + length + 1;
            if end0 > message.len() {
                continue;
            }
            let msg0 = &message[j..end0];
            if msg0.len() != msg1.len() {
                continue;
            }
            if msg0 == msg1 {
                matched = msg1.len();
                break;
            }
        }
    }
    matched + 1
}

/// Kontoyiannis' entropy-rate estimator from longest-match lengths, in bits per symbol (AFML
/// §18.4, Snippets 18.3–18.4).
///
/// For each point `i`, `L_i` is one plus the length of the longest substring starting at `i`
/// that also starts within the preceding look-back window; the estimate is the mean of
/// `log₂(n + 1) / L_i`. With `window == 0` the window expands (`n = i`, points
/// `1..=len/2`); otherwise the window is first clamped to `w = min(window, len/2)`, as in
/// Snippet 18.4, and `w` sets the points (`w` to `len − w`), the look-back and the
/// `log₂(w + 1)` numerator. Messages shorter than 2 characters return 0.
///
/// ```
/// use openquant::microstructural_features::get_konto_entropy;
///
/// assert_eq!(get_konto_entropy("ab", 0), 1.0);
/// // Points 1 and 2 match 1 and 2 characters back: (log2(2)/2 + log2(3)/3) / 2.
/// let h = get_konto_entropy("aaaa", 0);
/// assert!((h - (0.5 + 3f64.log2() / 3.0) / 2.0).abs() < 1e-12);
/// // A window of 5 on 4 characters is clamped to 2: point 2 matches "aa", log2(3) / 3.
/// assert!((get_konto_entropy("aaaa", 5) - 3f64.log2() / 3.0).abs() < 1e-12);
/// ```
pub fn get_konto_entropy(message: &str, window: usize) -> f64 {
    let message: Vec<char> = message.chars().collect();
    let message = message.as_slice();
    if message.len() < 2 {
        return 0.0;
    }
    // Snippet 18.4 reassigns `window = min(window, len/2)` before using it anywhere.
    let window = window.min(message.len() / 2);
    let points: Vec<usize> = if window == 0 {
        (1..=message.len() / 2).collect()
    } else {
        (window..=message.len() - window).collect()
    };
    let mut sum = 0.0;
    let mut num = 0.0;
    for i in points {
        let l = match_length(message, i, if window == 0 { i } else { window });
        let denom = if window == 0 { (i + 1) as f64 } else { (window + 1) as f64 };
        sum += denom.log2() / l as f64;
        num += 1.0;
    }
    if num == 0.0 {
        0.0
    } else {
        sum / num
    }
}

/// Streams a trades CSV and emits trades-based microstructural features and entropies per
/// bar (AFML Chapters 18–19; a port of mlfinlab's `MicrostructuralFeaturesGenerator`).
///
/// The CSV has a header row and three columns: timestamp (`%Y-%m-%d %H:%M:%S`, or with `/`
/// separators, optionally with fractional seconds), price and volume. Bars are defined by
/// `tick_num_series`, the **cumulative** trade numbers (1-based) at which each bar closes,
/// e.g. the tick numbers of volume or dollar bars built from the same file. Ticks after the
/// last threshold are not read, and a trailing partial bar is not emitted.
///
/// Each emitted bar is a `Vec<f64>` of, in order:
///
/// 1. the closing trade's timestamp in milliseconds since the Unix epoch (read as UTC);
/// 2. mean trade size ([`get_avg_tick_size`]);
/// 3. sum of tick-rule signs;
/// 4. [`vwap`];
/// 5. [`get_trades_based_kyle_lambda`], with tick-rule signs as aggressor flags;
/// 6. [`get_trades_based_amihud_lambda`];
/// 7. [`get_trades_based_hasbrouck_lambda`], with tick-rule signs as aggressor flags.
///
/// These are followed by groups of four entropies — [`get_shannon_entropy`],
/// [`get_plug_in_entropy`] with words of 1 (`NaN` for an empty message),
/// [`get_lempel_ziv_entropy`] and [`get_konto_entropy`] with an expanding window — of the
/// tick-rule message, then (if `volume_encoding` is set) of the encoded trade sizes, then
/// (if `pct_encoding` is set) of the encoded log returns: 11, 15 or 19 values per bar.
///
/// The tick rule signs each trade by its price change, carrying the previous sign over
/// unchanged prices; the very first trade is signed 0. Price changes, log returns and tick
/// signs carry across bar boundaries.
///
/// ```
/// use openquant::microstructural_features::MicrostructuralFeaturesGenerator;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let path = std::env::temp_dir().join(format!("oq_msf_doctest_{}.csv", std::process::id()));
/// std::fs::write(
///     &path,
///     "date_time,price,volume\n\
///      2024-01-02 09:30:00,100.0,10\n\
///      2024-01-02 09:30:01,101.0,5\n\
///      2024-01-02 09:30:02,100.5,5\n\
///      2024-01-02 09:30:03,100.5,10\n",
/// )?;
/// let path = path.to_str().unwrap();
///
/// // Two bars: trades 1-2 and trades 3-4.
/// let mut generator = MicrostructuralFeaturesGenerator::new_from_csv(path, &[2, 4], None, None)?;
/// let bars = generator.get_features_from_csv(path)?;
/// std::fs::remove_file(path)?;
///
/// assert_eq!(bars.len(), 2);
/// assert_eq!(bars[0].len(), 11); // 7 features + 4 tick-rule entropies
/// assert_eq!(bars[0][1], 7.5); // mean trade size
/// assert_eq!(bars[0][2], 1.0); // tick signs 0, +1
/// assert_eq!(bars[1][2], -2.0); // tick signs -1, -1 (unchanged price keeps the sign)
/// assert_eq!(bars[1][3], 100.5); // VWAP
/// # Ok(())
/// # }
/// ```
pub struct MicrostructuralFeaturesGenerator {
    tick_num_iter: std::vec::IntoIter<usize>,
    current_bar_tick: usize,
    price_diff: Vec<f64>,
    trade_size: Vec<f64>,
    tick_rule: Vec<f64>,
    dollar_size: Vec<f64>,
    log_ret: Vec<f64>,
    prev_price: Option<f64>,
    prev_tick_rule: f64,
    volume_encoding: Option<Vec<(f64, char)>>,
    pct_encoding: Option<Vec<(f64, char)>>,
    _entropy_types: Vec<&'static str>,
}

impl MicrostructuralFeaturesGenerator {
    /// Validates the trades CSV at `trades_path` and creates a generator for the bars closing
    /// at the cumulative trade numbers in `tick_num_series`.
    ///
    /// Only the first data row is checked. `volume_encoding` and `pct_encoding` are optional
    /// codebooks (from [`quantile_mapping`] or [`sigma_mapping`]) for trade sizes and log
    /// returns. An empty `tick_num_series`, or a 0 in it, stops bars from being emitted from
    /// that point on.
    ///
    /// # Errors
    ///
    /// - [`MicrostructuralError::Csv`] if the file cannot be opened or its first record read.
    /// - [`MicrostructuralError::WrongColumnCount`] if the first data row does not have three
    ///   columns.
    /// - [`MicrostructuralError::PriceNotFloat`], [`MicrostructuralError::VolumeNotNumeric`]
    ///   or [`MicrostructuralError::TimestampNotDatetime`] if its price, volume or timestamp
    ///   does not parse.
    /// - [`MicrostructuralError::InvalidCodebook`] if a codebook is empty or has a `NaN`.
    pub fn new_from_csv(
        trades_path: &str,
        tick_num_series: &[usize],
        volume_encoding: Option<Vec<(f64, char)>>,
        pct_encoding: Option<Vec<(f64, char)>>,
    ) -> Result<Self, MicrostructuralError> {
        for encoding in volume_encoding.iter().chain(pct_encoding.iter()) {
            check_codebook(encoding)?;
        }
        // validate header
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(trades_path)
            .map_err(|e| MicrostructuralError::Csv(e.to_string()))?;
        if let Some(result) = rdr.records().next() {
            let rec = result.map_err(|e| MicrostructuralError::Csv(e.to_string()))?;
            if rec.len() != 3 {
                return Err(MicrostructuralError::WrongColumnCount);
            }
            rec[1].parse::<f64>().map_err(|_| MicrostructuralError::PriceNotFloat)?;
            rec[2].parse::<f64>().map_err(|_| MicrostructuralError::VolumeNotNumeric)?;
            // Try multiple datetime formats (with/without fractional seconds)
            let _ =
                parse_datetime(&rec[0]).map_err(|_| MicrostructuralError::TimestampNotDatetime)?;
        }
        // Take the first threshold *out of* the iterator. Peeking at it instead leaves
        // it to be served again after the first bar closes, which emits a one-tick bar.
        // The generator owns its thresholds; borrowing would add a lifetime to a public type.
        #[allow(clippy::unnecessary_to_owned)]
        let mut tick_num_iter = tick_num_series.to_vec().into_iter();
        let current_bar_tick = tick_num_iter.next().unwrap_or(0);
        Ok(Self {
            tick_num_iter,
            current_bar_tick,
            price_diff: Vec::new(),
            trade_size: Vec::new(),
            tick_rule: Vec::new(),
            dollar_size: Vec::new(),
            log_ret: Vec::new(),
            prev_price: None,
            prev_tick_rule: 0.0,
            volume_encoding,
            pct_encoding,
            _entropy_types: vec!["shannon", "plug_in", "lempel_ziv", "konto"],
        })
    }

    fn reset_cache(&mut self) {
        self.price_diff.clear();
        self.trade_size.clear();
        self.tick_rule.clear();
        self.dollar_size.clear();
        self.log_ret.clear();
    }

    fn apply_tick_rule(&mut self, price: f64) -> f64 {
        let tick_diff = if let Some(prev) = self.prev_price { price - prev } else { 0.0 };

        if tick_diff != 0.0 {
            let s = tick_diff.signum();
            self.prev_tick_rule = s;
            s
        } else {
            self.prev_tick_rule
        }
    }

    fn get_price_diff(&self, price: f64) -> f64 {
        if let Some(prev) = self.prev_price {
            price - prev
        } else {
            0.0
        }
    }

    fn get_log_ret(&self, price: f64) -> f64 {
        if let Some(prev) = self.prev_price {
            (price / prev).ln()
        } else {
            0.0
        }
    }

    fn encode_entropy_features(&self, message: &str, out: &mut Vec<f64>) {
        out.push(get_shannon_entropy(message));
        // Every bar has at least one tick, so messages are non-empty; NaN is a safety net.
        out.push(get_plug_in_entropy(message, 1).unwrap_or(f64::NAN));
        out.push(get_lempel_ziv_entropy(message));
        out.push(get_konto_entropy(message, 0));
    }

    fn bar_features(&self, date_time: NaiveDateTime) -> Result<Vec<f64>, MicrostructuralError> {
        let mut features = vec![
            date_time.and_utc().timestamp_millis() as f64,
            get_avg_tick_size(&self.trade_size),
            self.tick_rule.iter().sum::<f64>(),
            vwap(&self.dollar_size, &self.trade_size)?,
            get_trades_based_kyle_lambda(&self.price_diff, &self.trade_size, &self.tick_rule)?,
            get_trades_based_amihud_lambda(&self.log_ret, &self.dollar_size)?,
            get_trades_based_hasbrouck_lambda(&self.log_ret, &self.dollar_size, &self.tick_rule)?,
        ];

        let tick_msg =
            encode_tick_rule_array(&self.tick_rule.iter().map(|v| *v as i32).collect::<Vec<_>>())
                .unwrap_or_default();
        self.encode_entropy_features(&tick_msg, &mut features);

        if let Some(enc) = &self.volume_encoding {
            let msg = encode_array(&self.trade_size, enc)?;
            self.encode_entropy_features(&msg, &mut features);
        }
        if let Some(enc) = &self.pct_encoding {
            let msg = encode_array(&self.log_ret, enc)?;
            self.encode_entropy_features(&msg, &mut features);
        }
        Ok(features)
    }

    /// Reads the trades CSV at `trades_path` and returns one feature vector per completed
    /// bar; see the type-level docs for the layout.
    ///
    /// The generator keeps its position in `tick_num_series` and its last price and tick
    /// sign, so it is meant to be called once; create a new generator to reprocess a file.
    ///
    /// # Errors
    ///
    /// - [`MicrostructuralError::Csv`] if the file cannot be opened, a record cannot be read
    ///   (including rows whose column count differs from the header's), or a timestamp,
    ///   price or volume does not parse.
    /// - [`MicrostructuralError::ShortRow`] if a row has fewer than three columns.
    /// - [`MicrostructuralError::Input`] if a feature function rejects its inputs (not
    ///   expected, since the per-bar buffers always have equal lengths).
    /// - [`MicrostructuralError::NanInArray`] if a codebook is set and a trade size or log
    ///   return to encode is `NaN` (a `NaN` volume, or a non-positive price).
    pub fn get_features_from_csv(
        &mut self,
        trades_path: &str,
    ) -> Result<Vec<Vec<f64>>, MicrostructuralError> {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(trades_path)
            .map_err(|e| MicrostructuralError::Csv(e.to_string()))?;
        let mut bars: Vec<Vec<f64>> = Vec::new();
        let mut tick_num = 0usize;
        for rec in rdr.records() {
            let rec = rec.map_err(|e| MicrostructuralError::Csv(e.to_string()))?;
            if rec.len() < 3 {
                return Err(MicrostructuralError::ShortRow(rec.len()));
            }
            let ts =
                parse_datetime(&rec[0]).map_err(|e| MicrostructuralError::Csv(e.to_string()))?;
            let price =
                rec[1].parse::<f64>().map_err(|e| MicrostructuralError::Csv(e.to_string()))?;
            let volume =
                rec[2].parse::<f64>().map_err(|e| MicrostructuralError::Csv(e.to_string()))?;
            let dollar_value = price * volume;
            let signed_tick = self.apply_tick_rule(price);
            tick_num += 1;
            self.price_diff.push(self.get_price_diff(price));
            self.trade_size.push(volume);
            self.tick_rule.push(signed_tick);
            self.dollar_size.push(dollar_value);
            self.log_ret.push(self.get_log_ret(price));
            self.prev_price = Some(price);

            if self.current_bar_tick > 0 && tick_num >= self.current_bar_tick {
                bars.push(self.bar_features(ts)?);
                if let Some(next) = self.tick_num_iter.next() {
                    self.current_bar_tick = next;
                } else {
                    break;
                }
                self.reset_cache();
            }
        }
        Ok(bars)
    }
}
