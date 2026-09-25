//! Fractional differentiation (AFML chapter 5).
//!
//! Differencing a series `d` times for real-valued `d` keeps part of its memory while
//! (for a large enough `d`) making it stationary: AFML §5.2's stationarity-versus-memory
//! dilemma. The weights come from the binomial expansion of `(1 - B)^d` (Snippet 5.1),
//! applied either on an expanding window ([`frac_diff`], Snippet 5.2) or on a fixed-width
//! window ([`frac_diff_ffd`], Snippet 5.3). The fixed-width form is the one to use for
//! features, because every output is the same function of the same number of lags.
//!
//! Conventions:
//!
//! - Inputs are **levels** (log prices or prices), oldest first. Passing returns
//!   differences them a second time.
//! - Weight vectors are returned **oldest lag first**: the last element is `w_0 = 1` and the
//!   one before it is `-d`.
//! - Leading outputs without enough history are `f64::NAN`; the output has the same length
//!   as the input.
//! - Nothing is validated: a negative `d` integrates instead of differencing, and a `NaN` in
//!   the input poisons every output whose window covers it.
//!
//! ```
//! use openquant::fracdiff::{frac_diff_ffd, get_weights};
//!
//! assert_eq!(get_weights(0.5, 4), vec![-0.0625, -0.125, -0.5, 1.0]);
//!
//! // d = 1 is the ordinary first difference.
//! let series: Vec<f64> = (1..=5).map(f64::from).collect();
//! let diffed = frac_diff_ffd(&series, 1.0, 1e-5);
//! assert!(diffed[0].is_nan());
//! assert_eq!(&diffed[1..], &[1.0, 1.0, 1.0, 1.0]);
//! ```
#![deny(missing_docs)]

/// Returns the first `size` fractional-differencing weights for order `diff_amt` (AFML
/// Snippet 5.1).
///
/// The weights follow `w_0 = 1`, `w_k = -w_{k-1} (d - k + 1) / k` and are returned oldest
/// lag first, so `result[size - 1] == 1.0` and `result[size - 2] == -diff_amt`. Returns an
/// empty vector when `size == 0`.
///
/// ```
/// use openquant::fracdiff::get_weights;
///
/// assert_eq!(get_weights(1.0, 3), vec![0.0, -1.0, 1.0]);
/// ```
pub fn get_weights(diff_amt: f64, size: usize) -> Vec<f64> {
    if size == 0 {
        return Vec::new();
    }
    let mut weights = Vec::with_capacity(size);
    weights.push(1.0);
    for k in 1..size {
        let w = -weights[k - 1] * (diff_amt - k as f64 + 1.0) / k as f64;
        weights.push(w);
    }
    weights.reverse();
    weights
}

/// Returns the fixed-width-window weights for order `diff_amt` (AFML Snippet 5.3).
///
/// Weights are generated with the same recursion as [`get_weights`] until the next one is
/// smaller than `thresh` in absolute value, or until `lim` weights have been produced,
/// whichever comes first. The result is oldest lag first (last element `w_0 = 1`). Returns an
/// empty vector when `lim == 0` and `[1.0]` when `lim == 1`; the result never has more than
/// `lim` elements.
///
/// The cap is what guarantees termination: for a non-integer `diff_amt` the weights never
/// reach zero, so a `thresh <= 0` (or `NaN`) never stops the expansion and the result has
/// exactly `lim` weights. Such a threshold is accepted, not rejected, and simply means "cap
/// only".
///
/// `thresh` is a floor on a single weight's magnitude, not a share of cumulative weight as in
/// [`frac_diff`].
///
/// ```
/// use openquant::fracdiff::get_weights_ffd;
///
/// // d = 1: the expansion ends at an exact zero after two weights.
/// assert_eq!(get_weights_ffd(1.0, 1e-5, 100), vec![-1.0, 1.0]);
/// assert_eq!(get_weights_ffd(0.5, 1e-2, 10_000).len(), 10);
///
/// // A zero threshold never stops the expansion; the cap does.
/// assert_eq!(get_weights_ffd(0.5, 0.0, 1), vec![1.0]);
/// assert_eq!(get_weights_ffd(0.5, 0.0, 4), vec![-0.0625, -0.125, -0.5, 1.0]);
/// ```
pub fn get_weights_ffd(diff_amt: f64, thresh: f64, lim: usize) -> Vec<f64> {
    if lim == 0 {
        return Vec::new();
    }
    let mut weights = vec![1.0];
    while weights.len() < lim {
        let k = weights.len();
        let next = -weights[k - 1] * (diff_amt - k as f64 + 1.0) / k as f64;
        if next.abs() < thresh {
            break;
        }
        weights.push(next);
    }
    weights.reverse();
    weights
}

/// Fractionally differences `series` with an expanding window (AFML Snippet 5.2).
///
/// Output `t` applies the first `t + 1` weights of `(1 - B)^d` to `series[0..=t]`. The
/// leading outputs, where the weights that would apply to missing history exceed `thresh`
/// of the total absolute weight, are `NaN`. `thresh = 1.0` skips nothing (the mlfinlab
/// default); AFML uses 0.01. The growing number of terms introduces a window-induced drift
/// (§5.5.1), which is why [`frac_diff_ffd`] is preferred for features.
///
/// `series` is a level (log price or price), oldest first. The output has the same length.
/// Cost is `O(n^2)`.
///
/// ```
/// use openquant::fracdiff::frac_diff;
///
/// let out = frac_diff(&[1.0, 2.0, 4.0], 1.0, 1.0);
/// assert_eq!(out, vec![1.0, 1.0, 2.0]);
/// ```
pub fn frac_diff(series: &[f64], diff_amt: f64, thresh: f64) -> Vec<f64> {
    let n = series.len();
    if n == 0 {
        return Vec::new();
    }
    let weights = get_weights(diff_amt, n);

    let mut cum = Vec::with_capacity(n);
    let mut s = 0.0;
    for w in &weights {
        s += w.abs();
        cum.push(s);
    }
    let total = *cum.last().unwrap_or(&1.0);
    if total != 0.0 {
        for v in &mut cum {
            *v /= total;
        }
    }
    let skip = cum.iter().filter(|v| **v > thresh).count();

    let mut out = vec![f64::NAN; n];
    for (iloc, slot) in out.iter_mut().enumerate().skip(skip) {
        let w_start = n - (iloc + 1);
        let mut acc = 0.0;
        for j in 0..=iloc {
            acc += weights[w_start + j] * series[j];
        }
        *slot = acc;
    }
    out
}

/// Fractionally differences `series` with a fixed-width window (AFML Snippet 5.3).
///
/// The weights are [`get_weights_ffd`]`(diff_amt, thresh, series.len())`; every output is the
/// dot product of those weights with the matching trailing window, so the first
/// `weights.len() - 1` outputs are `NaN`. A threshold too small for the data yields a window
/// as long as the series and a single non-`NaN` output.
///
/// `series` is a level (log price or price), oldest first. The output has the same length.
/// Each output uses only values at or before its own position. Cost is `O(n * width)`.
///
/// ```
/// use openquant::fracdiff::frac_diff_ffd;
///
/// let series: Vec<f64> = (1..=10).map(f64::from).collect();
/// let diffed = frac_diff_ffd(&series, 1.0, 1e-5);
/// assert!(diffed[0].is_nan());
/// assert!(diffed[1..].iter().all(|v| *v == 1.0));
/// ```
pub fn frac_diff_ffd(series: &[f64], diff_amt: f64, thresh: f64) -> Vec<f64> {
    let n = series.len();
    if n == 0 {
        return Vec::new();
    }
    let weights = get_weights_ffd(diff_amt, thresh, n);
    if weights.is_empty() {
        return vec![f64::NAN; n];
    }
    let width = weights.len() - 1;
    let mut out = vec![f64::NAN; n];
    for (iloc, slot) in out.iter_mut().enumerate().skip(width) {
        let loc0 = iloc - width;
        let mut acc = 0.0;
        for (k, w) in weights.iter().enumerate() {
            acc += *w * series[loc0 + k];
        }
        *slot = acc;
    }
    out
}
