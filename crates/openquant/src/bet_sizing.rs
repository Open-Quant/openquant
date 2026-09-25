//! Bet sizing (AFML chapter 10): turning a prediction into a position size.
//!
//! Three routes, depending on what the model produces:
//!
//! | You have | Entry point | AFML |
//! | --- | --- | --- |
//! | a class probability (typically from a meta-label model) | [`bet_size_probability`] | §10.3–10.5, Snippets 10.1–10.3 |
//! | a price forecast and the current market price | [`bet_size_dynamic`] | §10.6, Snippet 10.4 |
//! | only the sides and lifespans of the bets | [`bet_size_budget`], [`bet_size_reserve`] | §10.2 |
//!
//! Sizes are fractions of the maximum position in `[-1, 1]`, signed by side. The `func`
//! string arguments select the sizing curve: `"sigmoid"` (`m(x) = x / sqrt(w + x^2)`) or
//! `"power"` (`m(x) = sgn(x) |x|^w`, which needs the price divergence `x` scaled into
//! `[-1, 1]`). Any other name is [`BetSizingError::InvalidFunction`].
//!
//! A bet `(t0, t1)` is live on the half-open interval `[t0, t1)`.
//!
//! ```
//! use openquant::bet_sizing::{get_signal, get_target_pos, get_w, limit_price};
//!
//! # fn main() -> Result<(), openquant::bet_sizing::BetSizingError> {
//! // Probability to size for a two-class model: no edge at 1/K, saturating towards 1.
//! let sizes = get_signal(&[0.5, 0.7, 0.9], 2, None);
//! assert!(sizes[0].abs() < 1e-12);
//! assert!((sizes[1] - 0.3374).abs() < 1e-4 && (sizes[2] - 0.8176).abs() < 1e-4);
//!
//! // Calibrate the sigmoid so a price gap of 2.0 is a 0.9 bet, then size a 50-lot book.
//! let w = get_w(2.0, 0.9, "sigmoid")?;
//! let target = get_target_pos(w, 101.5, 100.0, 50.0, "sigmoid")?;
//! let limit = limit_price(target, 0.0, 101.5, w, 50.0, "sigmoid")?;
//! assert_eq!(target, 42.0);
//! assert!(limit > 100.0 && limit < 101.5);
//! # Ok(())
//! # }
//! ```
#![deny(missing_docs)]

use chrono::NaiveDateTime;
use rand::Rng;
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Errors returned by the bet-sizing functions.
#[derive(Debug, Clone, PartialEq)]
pub enum BetSizingError {
    /// `func` is not `"sigmoid"` or `"power"`.
    InvalidFunction {
        /// Function that received the name.
        context: &'static str,
        /// The unrecognised name.
        func: String,
    },
    /// A power-curve price divergence is outside `[-1, 1]`.
    PriceDivergenceOutOfRange {
        /// The offending divergence.
        value: f64,
    },
    /// A required input is empty.
    EmptyInput(&'static str),
    /// A broadcast input has neither length 1 nor the common length.
    ShapeMismatch {
        /// Input name.
        name: &'static str,
        /// Its length.
        len: usize,
        /// The common length it should broadcast to.
        expected: usize,
    },
    /// Two paired inputs have different lengths.
    LengthMismatch {
        /// Input name.
        name: &'static str,
        /// Its length.
        len: usize,
        /// The length it must match.
        expected: usize,
    },
}

impl fmt::Display for BetSizingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BetSizingError::InvalidFunction { context, func } => {
                write!(f, "invalid {context} function: {func}")
            }
            BetSizingError::PriceDivergenceOutOfRange { value } => {
                write!(f, "price divergence must be in [-1, 1], got {value}")
            }
            BetSizingError::EmptyInput(name) => write!(f, "input '{name}' must be non-empty"),
            BetSizingError::ShapeMismatch { name, len, expected } => {
                write!(f, "input '{name}' has length {len}, expected 1 or {expected} for broadcast")
            }
            BetSizingError::LengthMismatch { name, len, expected } => {
                write!(f, "input '{name}' has length {len}, expected {expected}")
            }
        }
    }
}

impl std::error::Error for BetSizingError {}

/// Sigmoid bet size `x / sqrt(w + x^2)` for price divergence `x = forecast - market`
/// (AFML Snippet 10.4).
///
/// `w_param` is the curve's width (see [`get_w_sigmoid`]); the result is in `(-1, 1)`.
pub fn bet_size_sigmoid(w_param: f64, price_div: f64) -> f64 {
    price_div * (w_param + price_div * price_div).powf(-0.5)
}

/// Power bet size `sgn(x) |x|^w` for a price divergence `x` scaled into `[-1, 1]`.
///
/// # Errors
///
/// [`BetSizingError::PriceDivergenceOutOfRange`] if `price_div` is outside `[-1, 1]`.
pub fn bet_size_power(w_param: f64, price_div: f64) -> Result<f64, BetSizingError> {
    if !(-1.0..=1.0).contains(&price_div) {
        return Err(BetSizingError::PriceDivergenceOutOfRange { value: price_div });
    }
    if price_div == 0.0 {
        return Ok(0.0);
    }
    Ok(price_div.signum() * price_div.abs().powf(w_param))
}

/// Bet size for price divergence `price_div` using the curve named by `func`.
///
/// Dispatches to [`bet_size_sigmoid`] or [`bet_size_power`].
///
/// # Errors
///
/// - [`BetSizingError::InvalidFunction`] if `func` is not `"sigmoid"` or `"power"`.
/// - [`BetSizingError::PriceDivergenceOutOfRange`] from [`bet_size_power`].
pub fn bet_size(w_param: f64, price_div: f64, func: &str) -> Result<f64, BetSizingError> {
    match func {
        "sigmoid" => Ok(bet_size_sigmoid(w_param, price_div)),
        "power" => bet_size_power(w_param, price_div),
        _ => Err(BetSizingError::InvalidFunction { context: "bet size", func: func.to_string() }),
    }
}

/// Inverse of the sigmoid curve: the market price at which the sigmoid bet size equals
/// `m_bet_size` given `forecast_price` (AFML Snippet 10.4).
///
/// Returns `forecast - m sqrt(w / (1 - m^2))`; not finite at `|m_bet_size| = 1`.
pub fn inv_price_sigmoid(forecast_price: f64, w_param: f64, m_bet_size: f64) -> f64 {
    forecast_price - m_bet_size * (w_param / (1.0 - m_bet_size * m_bet_size)).sqrt()
}

/// Inverse of the power curve: the market price at which the power bet size equals
/// `m_bet_size` given `forecast_price`.
pub fn inv_price_power(forecast_price: f64, w_param: f64, m_bet_size: f64) -> f64 {
    if m_bet_size == 0.0 {
        return forecast_price;
    }
    forecast_price - m_bet_size.signum() * m_bet_size.abs().powf(1.0 / w_param)
}

/// Market price implied by bet size `m_bet_size` for the curve named by `func`.
///
/// Dispatches to [`inv_price_sigmoid`] or [`inv_price_power`].
///
/// # Errors
///
/// [`BetSizingError::InvalidFunction`] if `func` is not `"sigmoid"` or `"power"`.
pub fn inv_price(
    forecast_price: f64,
    w_param: f64,
    m_bet_size: f64,
    func: &str,
) -> Result<f64, BetSizingError> {
    match func {
        "sigmoid" => Ok(inv_price_sigmoid(forecast_price, w_param, m_bet_size)),
        "power" => Ok(inv_price_power(forecast_price, w_param, m_bet_size)),
        _ => Err(BetSizingError::InvalidFunction { context: "inv_price", func: func.to_string() }),
    }
}

/// Converts predicted-class probabilities into bet sizes (AFML Snippet 10.1).
///
/// With `p` the probability of the predicted class and `K = num_classes`,
/// `z = (p - 1/K) / sqrt(p (1 - p))` and the size is `2 Phi(z) - 1`, in `[-1, 1]`. When `pred`
/// is given, each size is multiplied by the matching side (typically `+1`/`-1` from a primary
/// model); `pred` is zipped with `prob`, so a shorter `pred` truncates the output.
///
/// A probability below `1/K` gives a negative size, i.e. a bet against the side. A
/// probability of exactly 0 or 1 gives a size of exactly `-1` or `+1`.
pub fn get_signal(prob: &[f64], num_classes: usize, pred: Option<&[f64]>) -> Vec<f64> {
    if prob.is_empty() {
        return Vec::new();
    }
    let norm = Normal::new(0.0, 1.0).unwrap();
    let mapped: Vec<f64> = prob
        .iter()
        .map(|p| {
            let z = (p - 1.0 / num_classes as f64) / (p * (1.0 - p)).sqrt();
            2.0 * norm.cdf(z) - 1.0
        })
        .collect();
    match pred {
        Some(side) => mapped.iter().zip(side.iter()).map(|(m, s)| m * s).collect(),
        None => mapped,
    }
}

/// Rounds sizes to the nearest multiple of `step_size` and clamps them to `[-1, 1]` (AFML
/// Snippet 10.3).
///
/// A `step_size` of zero or less returns the input unchanged.
pub fn discrete_signal(signal0: &[f64], step_size: f64) -> Vec<f64> {
    if step_size <= 0.0 {
        return signal0.to_vec();
    }
    signal0.iter().map(|s| ((s / step_size).round() * step_size).clamp(-1.0, 1.0)).collect()
}

/// Averages the signals that are live at each change point (AFML Snippet 10.2).
///
/// `signal` is `(start, size)` pairs and `t1` the matching end times (zipped, so extra
/// entries in the longer slice are ignored). The evaluation points are the sorted, unique
/// union of all starts and ends; at each, the result is the mean of the signals with
/// `start <= t < end`, or 0 if none is live.
pub fn avg_active_signals(
    signal: &[(NaiveDateTime, f64)],
    t1: &[NaiveDateTime],
) -> Vec<(NaiveDateTime, f64)> {
    let mut t_points: Vec<NaiveDateTime> = t1.to_vec();
    t_points.extend(signal.iter().map(|(ts, _)| *ts));
    t_points.sort();
    t_points.dedup();
    mp_avg_active_signals(signal, t1, &t_points)
}

/// Averages the live signals at each timestamp in `molecule` (the worker behind
/// [`avg_active_signals`]).
///
/// At each `t` in `molecule`, returns the mean size of the signals with
/// `start <= t < end`, or 0 if none is live. `signal` and `t1` are zipped.
pub fn mp_avg_active_signals(
    signal: &[(NaiveDateTime, f64)],
    t1: &[NaiveDateTime],
    molecule: &[NaiveDateTime],
) -> Vec<(NaiveDateTime, f64)> {
    let mut out = Vec::new();
    for loc in molecule {
        let mut sum = 0.0;
        let mut count = 0.0;
        for ((s_ts, s_val), end) in signal.iter().zip(t1.iter()) {
            if *s_ts <= *loc && (*loc < *end) {
                sum += *s_val;
                count += 1.0;
            }
        }
        if count > 0.0 {
            out.push((*loc, sum / count));
        } else {
            out.push((*loc, 0.0));
        }
    }
    out
}

/// Sizes bets from class probabilities: [`get_signal`], then optionally
/// [`avg_active_signals`], then [`discrete_signal`] (AFML §10.3–10.5).
///
/// Each event is `(start, t1, prob, side)`. Without averaging, the result has one
/// `(start, size)` row per event; with `average_active`, one row per change point as in
/// [`avg_active_signals`]. `step_size.abs()` is used as the discretisation step, so zero
/// disables it.
///
/// ```
/// use chrono::NaiveDate;
/// use openquant::bet_sizing::bet_size_probability;
///
/// let t = |h| NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(h, 0, 0).unwrap();
/// let events = vec![(t(9), t(11), 0.7, 1.0), (t(10), t(12), 0.9, 1.0)];
/// let sized = bet_size_probability(&events, 2, 0.1, true);
/// let sizes: Vec<f64> = sized.iter().map(|(_, s)| *s).collect();
/// // 09:00 one long live, 10:00 both, 11:00 only the second, 12:00 flat.
/// assert_eq!(sizes.len(), 4);
/// assert!((sizes[0] - 0.3).abs() < 1e-12 && (sizes[1] - 0.6).abs() < 1e-12);
/// assert!((sizes[2] - 0.8).abs() < 1e-12 && sizes[3] == 0.0);
/// ```
pub fn bet_size_probability(
    events: &[(NaiveDateTime, NaiveDateTime, f64, f64)], // (start, t1, prob, side)
    num_classes: usize,
    step_size: f64,
    average_active: bool,
) -> Vec<(NaiveDateTime, f64)> {
    let prob: Vec<f64> = events.iter().map(|(_, _, p, _)| *p).collect();
    let side: Vec<f64> = events.iter().map(|(_, _, _, s)| *s).collect();
    let signal0 = get_signal(&prob, num_classes, Some(&side));
    let mut signals: Vec<(NaiveDateTime, f64)> =
        events.iter().map(|(ts, _, _, _)| *ts).zip(signal0).collect();
    if average_active {
        let t1: Vec<NaiveDateTime> = events.iter().map(|(_, t1, _, _)| *t1).collect();
        signals = avg_active_signals(&signals, &t1);
    }
    let discretized: Vec<f64> =
        discrete_signal(&signals.iter().map(|(_, v)| *v).collect::<Vec<_>>(), step_size.abs());
    signals.iter().zip(discretized.iter()).map(|(t, v)| (t.0, *v)).collect()
}

/// Broadcasts the four [`bet_size_dynamic`] inputs to a common length and zips them into
/// `(pos, max_pos, market_price, forecast)` rows.
///
/// The common length is the longest input; each input must have that length or length 1.
///
/// # Errors
///
/// - [`BetSizingError::EmptyInput`] if every input is empty.
/// - [`BetSizingError::ShapeMismatch`] if an input has neither length 1 nor the common
///   length (an empty input alongside a non-empty one is also a mismatch).
pub fn confirm_and_cast_to_df(
    pos: &[f64],
    max_pos: &[f64],
    m_p: &[f64],
    f: &[f64],
) -> Result<Vec<(f64, f64, f64, f64)>, BetSizingError> {
    let lengths = [pos.len(), max_pos.len(), m_p.len(), f.len()];
    let target_len = lengths.into_iter().max().unwrap_or(0);
    if target_len == 0 {
        return Err(BetSizingError::EmptyInput("pos/max_pos/m_p/f"));
    }

    fn broadcast(
        values: &[f64],
        len: usize,
        name: &'static str,
    ) -> Result<Vec<f64>, BetSizingError> {
        if values.len() == len {
            Ok(values.to_vec())
        } else if values.len() == 1 {
            Ok(vec![values[0]; len])
        } else {
            Err(BetSizingError::ShapeMismatch { name, len: values.len(), expected: len })
        }
    }

    let pos_v = broadcast(pos, target_len, "pos")?;
    let max_pos_v = broadcast(max_pos, target_len, "max_pos")?;
    let m_p_v = broadcast(m_p, target_len, "m_p")?;
    let f_v = broadcast(f, target_len, "f")?;

    Ok((0..target_len).map(|i| (pos_v[i], max_pos_v[i], m_p_v[i], f_v[i])).collect())
}

/// Calibrates the curve width `w` so that divergence `price_div` gives size `m_bet_size`.
///
/// Dispatches to [`get_w_sigmoid`] or [`get_w_power`].
///
/// # Errors
///
/// - [`BetSizingError::InvalidFunction`] if `func` is not `"sigmoid"` or `"power"`.
/// - [`BetSizingError::PriceDivergenceOutOfRange`] from [`get_w_power`].
pub fn get_w(price_div: f64, m_bet_size: f64, func: &str) -> Result<f64, BetSizingError> {
    match func {
        "sigmoid" => Ok(get_w_sigmoid(price_div, m_bet_size)),
        "power" => get_w_power(price_div, m_bet_size),
        _ => Err(BetSizingError::InvalidFunction { context: "get_w", func: func.to_string() }),
    }
}

/// Target position for forecast `f` and market price `m_p`, out of a maximum `max_pos`,
/// using the curve named by `func` (AFML Snippet 10.4).
///
/// Dispatches to [`get_target_pos_sigmoid`] or [`get_target_pos_power`]. The result is
/// truncated toward zero to a whole number of units.
///
/// # Errors
///
/// - [`BetSizingError::InvalidFunction`] if `func` is not `"sigmoid"` or `"power"`.
/// - [`BetSizingError::PriceDivergenceOutOfRange`] from [`get_target_pos_power`].
pub fn get_target_pos(
    w: f64,
    f: f64,
    m_p: f64,
    max_pos: f64,
    func: &str,
) -> Result<f64, BetSizingError> {
    match func {
        "sigmoid" => Ok(get_target_pos_sigmoid(w, f, m_p, max_pos)),
        "power" => get_target_pos_power(w, f, m_p, max_pos),
        _ => Err(BetSizingError::InvalidFunction {
            context: "get_target_pos",
            func: func.to_string(),
        }),
    }
}

/// Breakeven limit price for moving from position `pos` to target `t_pos`, using the curve
/// named by `func` (AFML Snippet 10.4).
///
/// Dispatches to [`limit_price_sigmoid`] or [`limit_price_power`]; see those for the
/// formula and edge cases.
///
/// # Errors
///
/// [`BetSizingError::InvalidFunction`] if `func` is not `"sigmoid"` or `"power"`.
pub fn limit_price(
    t_pos: f64,
    pos: f64,
    f: f64,
    w: f64,
    max_pos: f64,
    func: &str,
) -> Result<f64, BetSizingError> {
    match func {
        "sigmoid" => Ok(limit_price_sigmoid(t_pos, pos, f, w, max_pos)),
        "power" => Ok(limit_price_power(t_pos, pos, f, w, max_pos)),
        _ => {
            Err(BetSizingError::InvalidFunction { context: "limit_price", func: func.to_string() })
        }
    }
}

/// Sigmoid width `w = x^2 (1/m^2 - 1)` such that divergence `price_div` gives size
/// `m_bet_size` (AFML Snippet 10.4).
pub fn get_w_sigmoid(price_div: f64, m_bet_size: f64) -> f64 {
    (price_div * price_div) * ((1.0 / (m_bet_size * m_bet_size)) - 1.0)
}

/// Power exponent `w = ln(m / sgn(x)) / ln|x|` such that divergence `price_div` gives size
/// `m_bet_size`; negative results are floored at 0.
///
/// # Errors
///
/// [`BetSizingError::PriceDivergenceOutOfRange`] if `price_div` is outside `[-1, 1]`.
pub fn get_w_power(price_div: f64, m_bet_size: f64) -> Result<f64, BetSizingError> {
    if !(-1.0..=1.0).contains(&price_div) {
        return Err(BetSizingError::PriceDivergenceOutOfRange { value: price_div });
    }
    let w_calc = (m_bet_size / price_div.signum()).ln() / price_div.abs().ln();
    if w_calc < 0.0 {
        return Ok(0.0);
    }
    Ok(w_calc)
}

/// Sigmoid target position `trunc(m(forecast - market) * max_pos)` (AFML Snippet 10.4).
pub fn get_target_pos_sigmoid(
    w_param: f64,
    forecast_price: f64,
    market_price: f64,
    max_pos: f64,
) -> f64 {
    (bet_size_sigmoid(w_param, forecast_price - market_price) * max_pos).trunc()
}

/// Power-curve target position `trunc(m(forecast - market) * max_pos)`.
///
/// # Errors
///
/// [`BetSizingError::PriceDivergenceOutOfRange`] if `forecast_price - market_price` is
/// outside `[-1, 1]`.
pub fn get_target_pos_power(
    w_param: f64,
    forecast_price: f64,
    market_price: f64,
    max_pos: f64,
) -> Result<f64, BetSizingError> {
    Ok((bet_size_power(w_param, forecast_price - market_price)? * max_pos).trunc())
}

/// Sigmoid breakeven limit price for moving from `pos` to `t_pos` with forecast `f`, width
/// `w` and maximum position `max_pos` (AFML Snippet 10.4).
///
/// Averages [`inv_price_sigmoid`] over the units `j = |pos + sgn| ..= |t_pos|` (with `sgn` the sign
/// of `t_pos - pos`, both truncated to whole units), each at size `j / max_pos`, and divides
/// by `|t_pos - pos|`.
///
/// Returns `NaN` when the truncated target equals the truncated current position. When
/// `|pos + sgn| > |t_pos|` (for example reducing a long position toward zero) the range is
/// empty and the result is `0.0`, not a price.
pub fn limit_price_sigmoid(t_pos: f64, pos: f64, f: f64, w: f64, max_pos: f64) -> f64 {
    let target = t_pos.trunc() as i64;
    let current = pos.trunc() as i64;
    if target == current {
        return f64::NAN;
    }
    let sgn = (target - current).signum();
    let mut l_p = 0.0;
    let start = (current + sgn).abs();
    let end = target.abs();
    for j in start..=end {
        let m_bet = j as f64 / max_pos;
        l_p += inv_price_sigmoid(f, w, m_bet);
    }
    l_p / (target - current).abs() as f64
}

/// Power-curve breakeven limit price for moving from `pos` to `t_pos` with forecast `f`,
/// exponent `w` and maximum position `max_pos`.
///
/// Averages [`inv_price_power`] over the units `j = |pos + sgn| ..= |t_pos|` (with `sgn` the sign
/// of `t_pos - pos`, both truncated to whole units), each at size `j / max_pos`, and divides
/// by `|t_pos - pos|`.
///
/// Returns `NaN` when the truncated target equals the truncated current position. When
/// `|pos + sgn| > |t_pos|` (for example reducing a long position toward zero) the range is
/// empty and the result is `0.0`, not a price.
pub fn limit_price_power(t_pos: f64, pos: f64, f: f64, w: f64, max_pos: f64) -> f64 {
    let target = t_pos.trunc() as i64;
    let current = pos.trunc() as i64;
    if target == current {
        return f64::NAN;
    }
    let sgn = (target - current).signum();
    let mut l_p = 0.0;
    let start = (current + sgn).abs();
    let end = target.abs();
    for j in start..=end {
        let m_bet = j as f64 / max_pos;
        l_p += inv_price_power(f, w, m_bet);
    }
    l_p / (target - current).abs() as f64
}

/// Dynamic bet sizing from a price forecast (AFML §10.6, Snippet 10.4).
///
/// For each row of current position `pos`, maximum position `max_pos`, market price `m_p`
/// and forecast `f` (broadcast as in [`confirm_and_cast_to_df`]) returns
/// `(bet_size, target_position, limit_price)` using the sigmoid curve.
///
/// The width is fixed at `get_w_sigmoid(10.0, 0.95)`: a divergence of 10 price units gives a
/// size of 0.95, whatever the instrument. For anything else calibrate your own `w` with
/// [`get_w`] and use [`get_target_pos`] and [`limit_price`].
///
/// # Errors
///
/// [`BetSizingError::EmptyInput`] or [`BetSizingError::ShapeMismatch`] from
/// [`confirm_and_cast_to_df`].
///
/// ```
/// use openquant::bet_sizing::bet_size_dynamic;
///
/// # fn main() -> Result<(), openquant::bet_sizing::BetSizingError> {
/// let rows = bet_size_dynamic(&[0.0], &[100.0], &[95.0], &[100.0])?;
/// let (size, target, limit) = rows[0];
/// assert!((size - 0.836).abs() < 1e-3);
/// assert_eq!(target, 83.0);
/// assert!((limit - 98.22).abs() < 1e-2);
/// # Ok(())
/// # }
/// ```
pub fn bet_size_dynamic(
    pos: &[f64],
    max_pos: &[f64],
    m_p: &[f64],
    f: &[f64],
) -> Result<Vec<(f64, f64, f64)>, BetSizingError> {
    let w_param = get_w_sigmoid(10.0, 0.95);
    let rows = confirm_and_cast_to_df(pos, max_pos, m_p, f)?;
    Ok(rows
        .into_iter()
        .map(|(p, m, mp, forecast)| {
            let t_pos = get_target_pos_sigmoid(w_param, forecast, mp, m);
            let l_p = limit_price_sigmoid(t_pos, p, forecast, w_param, m);
            let b = bet_size_sigmoid(w_param, forecast - mp);
            (b, t_pos, l_p)
        })
        .collect())
}

/// Counts the long and short bets live at each bet's start (AFML §10.2).
///
/// `t1` is `(start, end)` per bet and `side` its direction (`> 0` is long, anything else is
/// short). For each bet's start `t`, counts the bets with `start <= t < end`. Returns
/// `(start, active_long, active_short)` in input order.
///
/// # Errors
///
/// [`BetSizingError::LengthMismatch`] if `side` and `t1` differ in length.
pub fn get_concurrent_sides(
    t1: &[(NaiveDateTime, NaiveDateTime)],
    side: &[f64],
) -> Result<Vec<(NaiveDateTime, f64, f64)>, BetSizingError> {
    // returns (index, active_long, active_short)
    if side.len() != t1.len() {
        return Err(BetSizingError::LengthMismatch {
            name: "side",
            len: side.len(),
            expected: t1.len(),
        });
    }
    let mut out = Vec::new();
    for (start, _end) in t1.iter() {
        let mut long = 0.0;
        let mut short = 0.0;
        for (j, (s, e)) in t1.iter().enumerate() {
            if *s <= *start && *e > *start {
                if side[j] > 0.0 {
                    long += 1.0;
                } else {
                    short += 1.0;
                }
            }
        }
        out.push((*start, long, short));
    }
    Ok(out)
}

/// Budgeting bet size `L_t / max L - S_t / max S` from concurrent bet counts (AFML §10.2).
///
/// Uses [`get_concurrent_sides`]; a side with no bets contributes 0. The maxima are taken over
/// the whole input, so computing this over a full backtest and trading on it looks ahead.
///
/// # Errors
///
/// [`BetSizingError::LengthMismatch`] if `side` and `t1` differ in length.
pub fn bet_size_budget(
    t1: &[(NaiveDateTime, NaiveDateTime)],
    side: &[f64],
) -> Result<Vec<(NaiveDateTime, f64)>, BetSizingError> {
    let conc = get_concurrent_sides(t1, side)?;
    let max_long = conc.iter().map(|(_, l, _)| *l).fold(0.0, f64::max);
    let max_short = conc.iter().map(|(_, _, s)| *s).fold(0.0, f64::max);
    Ok(conc
        .iter()
        .map(|(ts, l, s)| {
            let avg_long = if max_long > 0.0 { l / max_long } else { 0.0 };
            let avg_short = if max_short > 0.0 { s / max_short } else { 0.0 };
            (*ts, avg_long - avg_short)
        })
        .collect())
}

/// CDF at `x` of a two-Gaussian mixture `p1 N(mu1, sigma1) + (1 - p1) N(mu2, sigma2)`.
///
/// Standard deviations are floored at `1e-8`.
///
/// # Panics
///
/// Panics if `mu1` or `mu2` is `NaN`.
pub fn cdf_mixture(mu1: f64, mu2: f64, sigma1: f64, sigma2: f64, p1: f64, x: f64) -> f64 {
    let n1 = Normal::new(mu1, sigma1.max(1e-8)).unwrap();
    let n2 = Normal::new(mu2, sigma2.max(1e-8)).unwrap();
    p1 * n1.cdf(x) + (1.0 - p1) * n2.cdf(x)
}

fn normal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let s = sigma.max(1e-8);
    let z = (x - mu) / s;
    (-0.5 * z * z).exp() / (s * (2.0 * std::f64::consts::PI).sqrt())
}

fn fit_two_normal_mixture_em(
    samples: &[f64],
    fit_runs: usize,
    epsilon: f64,
    max_iter: usize,
) -> [f64; 5] {
    debug_assert!(!samples.is_empty(), "callers reject empty input");
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n.max(1.0);
    let std = var.sqrt().max(1e-3);
    let mut rng = rand::thread_rng();

    let mut best_ll = f64::NEG_INFINITY;
    let mut best = [mean - std, mean + std, std, std, 0.5];

    for _ in 0..fit_runs.max(1) {
        let x1 = samples[rng.gen_range(0..samples.len())];
        let x2 = samples[rng.gen_range(0..samples.len())];
        let mut mu1 = x1.min(x2);
        let mut mu2 = x1.max(x2);
        let mut sigma1 = std;
        let mut sigma2 = std;
        let mut p1 = rng.gen_range(0.25..0.75);
        let mut prev_ll = f64::NEG_INFINITY;

        for _ in 0..max_iter.max(1) {
            let mut gammas = Vec::with_capacity(samples.len());
            let mut ll = 0.0;

            for &x in samples {
                let w1 = p1 * normal_pdf(x, mu1, sigma1);
                let w2 = (1.0 - p1) * normal_pdf(x, mu2, sigma2);
                let denom = (w1 + w2).max(1e-16);
                gammas.push(w1 / denom);
                ll += denom.ln();
            }

            let n1 = gammas.iter().sum::<f64>().max(1e-8);
            let n2 = (n - n1).max(1e-8);
            p1 = (n1 / n).clamp(1e-6, 1.0 - 1e-6);

            mu1 = gammas.iter().zip(samples.iter()).map(|(g, x)| g * x).sum::<f64>() / n1;
            mu2 = gammas.iter().zip(samples.iter()).map(|(g, x)| (1.0 - g) * x).sum::<f64>() / n2;

            sigma1 = (gammas
                .iter()
                .zip(samples.iter())
                .map(|(g, x)| g * (x - mu1).powi(2))
                .sum::<f64>()
                / n1)
                .sqrt()
                .max(1e-6);

            sigma2 = (gammas
                .iter()
                .zip(samples.iter())
                .map(|(g, x)| (1.0 - g) * (x - mu2).powi(2))
                .sum::<f64>()
                / n2)
                .sqrt()
                .max(1e-6);

            if (ll - prev_ll).abs() < epsilon {
                prev_ll = ll;
                break;
            }
            prev_ll = ll;
        }

        if prev_ll > best_ll {
            best_ll = prev_ll;
            best = [mu1, mu2, sigma1, sigma2, p1];
        }
    }

    best
}

/// Reserve bet size for net concurrency `c` under a fitted mixture `fit` (AFML §10.2).
///
/// With `F` the mixture CDF ([`cdf_mixture`]) and `fit` = `[mu1, mu2, sigma1, sigma2, p1]`,
/// returns `(F(c) - F(0)) / (1 - F(0))` for `c >= 0` and `(F(c) - F(0)) / F(0)` otherwise.
///
/// # Panics
///
/// Panics if `fit[0]` or `fit[1]` is `NaN`.
pub fn single_bet_size_mixed(c: f64, fit: &[f64; 5]) -> f64 {
    let c0 = cdf_mixture(fit[0], fit[1], fit[2], fit[3], fit[4], 0.0);
    let cdf = cdf_mixture(fit[0], fit[1], fit[2], fit[3], fit[4], c);
    if c >= 0.0 {
        (cdf - c0) / (1.0 - c0)
    } else {
        (cdf - c0) / c0
    }
}

/// Reserve bet sizes under a given mixture fit, with the net concurrency kept in each row.
///
/// Returns [`ReserveBetSizeRow`]s `(start, active_long, active_short, c_t, bet_size)` with
/// `c_t = long - short` and the size from [`single_bet_size_mixed`].
///
/// # Errors
///
/// [`BetSizingError::LengthMismatch`] if `side` and `t1` differ in length.
///
/// # Panics
///
/// Panics if `fit[0]` or `fit[1]` is `NaN`.
pub fn bet_size_reserve_with_fit(
    t1: &[(NaiveDateTime, NaiveDateTime)],
    side: &[f64],
    fit: &[f64; 5],
) -> Result<Vec<ReserveBetSizeRow>, BetSizingError> {
    Ok(get_concurrent_sides(t1, side)?
        .into_iter()
        .map(|(ts, l, s)| {
            let c_t = l - s;
            let b = single_bet_size_mixed(c_t, fit);
            (ts, l, s, c_t, b)
        })
        .collect())
}

/// Reserve bet-size row: `(timestamp, active_long, active_short, c_t, bet_size)`.
pub type ReserveBetSizeRow = (NaiveDateTime, f64, f64, f64, f64);
/// Fitted two-normal mixture parameters `[mu1, mu2, sigma1, sigma2, p1]`.
pub type MixtureParams = [f64; 5];

/// Reserve bet sizing with the mixture fitted to the data (AFML §10.2).
///
/// Computes `c_t = long - short` with [`get_concurrent_sides`], fits a two-Gaussian mixture
/// to it by EM (`fit_runs` random restarts, each stopping when the log-likelihood changes by
/// less than `epsilon` or after `max_iter` iterations), and sizes each bet with
/// [`single_bet_size_mixed`]. AFML fits the mixture with EF3M moment matching instead.
///
/// The EM starts are drawn from the thread RNG, so **results change from run to run**. Pass
/// `return_parameters = true` to get the fitted [`MixtureParams`] back, and reuse them with
/// [`bet_size_reserve`] for reproducible sizes.
///
/// # Errors
///
/// - [`BetSizingError::EmptyInput`] if `t1` is empty.
/// - [`BetSizingError::LengthMismatch`] if `side` and `t1` differ in length.
pub fn bet_size_reserve_full(
    t1: &[(NaiveDateTime, NaiveDateTime)],
    side: &[f64],
    fit_runs: usize,
    epsilon: f64,
    max_iter: usize,
    return_parameters: bool,
) -> Result<(Vec<ReserveBetSizeRow>, Option<MixtureParams>), BetSizingError> {
    if t1.is_empty() {
        return Err(BetSizingError::EmptyInput("t1"));
    }
    let concurrent = get_concurrent_sides(t1, side)?;
    let c_t: Vec<f64> = concurrent.iter().map(|(_, l, s)| l - s).collect();
    let fit = fit_two_normal_mixture_em(&c_t, fit_runs, epsilon, max_iter);
    let events = concurrent
        .into_iter()
        .zip(c_t)
        .map(|((ts, l, s), c)| {
            let b = single_bet_size_mixed(c, &fit);
            (ts, l, s, c, b)
        })
        .collect();
    let params = if return_parameters { Some(fit) } else { None };
    Ok((events, params))
}

/// Reserve bet sizes under a given mixture fit (AFML §10.2).
///
/// Like [`bet_size_reserve_with_fit`] without the `c_t` column: returns
/// `(start, active_long, active_short, bet_size)`. `fit` is `[mu1, mu2, sigma1, sigma2, p1]`,
/// e.g. from [`bet_size_reserve_full`].
///
/// # Errors
///
/// [`BetSizingError::LengthMismatch`] if `side` and `t1` differ in length.
///
/// # Panics
///
/// Panics if `fit[0]` or `fit[1]` is `NaN`.
pub fn bet_size_reserve(
    t1: &[(NaiveDateTime, NaiveDateTime)],
    side: &[f64],
    fit: &[f64; 5],
) -> Result<Vec<(NaiveDateTime, f64, f64, f64)>, BetSizingError> {
    Ok(bet_size_reserve_with_fit(t1, side, fit)?
        .into_iter()
        .map(|(ts, l, s, _c_t, b)| (ts, l, s, b))
        .collect())
}
