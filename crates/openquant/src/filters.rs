//! Event-sampling filters: the symmetric CUSUM filter (AFML §2.5.2.1, Snippet 2.4) and a
//! rolling z-score filter ported from mlfinlab.
//!
//! A filter decides which bars become events to label. Both filters take a slice of close
//! **prices** (not returns), oldest first, and return either 0-based positions into that
//! slice or the timestamps at those positions.
//!
//! - [`cusum_filter_indices`] accumulates log returns `ln(p_t / p_{t-1})` in separate upward
//!   and downward accumulators floored at zero, and fires when either crosses the threshold
//!   `h` (in log-return units). Only the side that fired is reset.
//! - [`z_score_filter_indices`] fires when the price is at or above its rolling mean plus
//!   `k` rolling sample standard deviations (ddof = 1, matching pandas). It is one-sided and
//!   has no reset; it is not from AFML.
//!
//! Neither filter looks ahead: the decision at bar `t` uses only data up to `t`.
//!
//! ```
//! use openquant::filters::{cusum_filter_indices, Threshold};
//!
//! # fn main() -> Result<(), openquant::filters::FilterError> {
//! let close = vec![100.0, 100.4, 100.9, 101.3, 101.0, 100.2, 99.6, 99.9];
//! // A 1% threshold: one upward event, then one downward.
//! let events = cusum_filter_indices(&close, Threshold::Scalar(0.01))?;
//! assert_eq!(events, vec![3, 5]);
//! # Ok(())
//! # }
//! ```

use chrono::NaiveDateTime;
use std::fmt;

/// Errors returned by the filters in this module.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterError {
    /// A [`Threshold::Dynamic`] vector has no value for bar `index`.
    MissingDynamicThreshold {
        /// Bar position that needed a threshold.
        index: usize,
        /// Length of the dynamic threshold vector.
        available: usize,
    },
    /// An event position has no matching entry in the timestamp slice.
    TimestampIndexOutOfBounds {
        /// Event position (index into `close`).
        index: usize,
        /// Length of the timestamp slice.
        available: usize,
    },
}

impl fmt::Display for FilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterError::MissingDynamicThreshold { index, available } => {
                write!(
                    f,
                    "dynamic threshold missing value at index {index} (available={available})"
                )
            }
            FilterError::TimestampIndexOutOfBounds { index, available } => {
                write!(f, "timestamp index {index} out of bounds for length {available}")
            }
        }
    }
}

impl std::error::Error for FilterError {}

/// CUSUM threshold `h`, in log-return units.
pub enum Threshold {
    /// One threshold for every bar.
    Scalar(f64),
    /// One threshold per bar: bar `t` is compared with element `t` (element 0 is never read,
    /// since the first return is at bar 1). Typically a multiple of a daily-volatility
    /// estimate. Must be at least as long as the price series.
    Dynamic(Vec<f64>),
}

fn threshold_at_checked(threshold: &Threshold, idx: usize) -> Result<f64, FilterError> {
    match threshold {
        Threshold::Scalar(v) => Ok(*v),
        Threshold::Dynamic(v) => v
            .get(idx)
            .copied()
            .ok_or(FilterError::MissingDynamicThreshold { index: idx, available: v.len() }),
    }
}

/// Symmetric CUSUM filter returning event positions (AFML Snippet 2.4).
///
/// For each bar `t >= 1`, with `r_t = ln(close[t] / close[t-1])`:
/// `S+ = max(0, S+ + r_t)`, `S- = min(0, S- + r_t)`. Bar `t` is an event when `S- < -h`
/// (checked first) or `S+ > h`; the accumulator that fired is reset to zero. Returns 0-based
/// positions into `close`, in increasing order. Fewer than two prices yield no events.
///
/// `close` must be positive prices, oldest first. Unlike AFML's snippet, which differences
/// whatever series it is given, this always takes log returns.
///
/// # Errors
///
/// [`FilterError::MissingDynamicThreshold`] if `threshold` is [`Threshold::Dynamic`] and
/// shorter than `close`.
pub fn cusum_filter_indices(
    close: &[f64],
    threshold: Threshold,
) -> Result<Vec<usize>, FilterError> {
    let mut events = Vec::new();
    if close.len() < 2 {
        return Ok(events);
    }

    let mut s_pos = 0.0_f64;
    let mut s_neg = 0.0_f64;

    for i in 1..close.len() {
        let log_ret = (close[i] / close[i - 1]).ln();
        let thresh = threshold_at_checked(&threshold, i)?;

        let pos = s_pos + log_ret;
        let neg = s_neg + log_ret;
        s_pos = pos.max(0.0);
        s_neg = neg.min(0.0);

        if s_neg < -thresh {
            s_neg = 0.0;
            events.push(i);
        } else if s_pos > thresh {
            s_pos = 0.0;
            events.push(i);
        }
    }

    Ok(events)
}

/// Symmetric CUSUM filter returning event timestamps.
///
/// Runs [`cusum_filter_indices`] and maps each event position to `timestamps[position]`.
/// `timestamps` should be aligned with `close`.
///
/// # Errors
///
/// - [`FilterError::MissingDynamicThreshold`] as for [`cusum_filter_indices`].
/// - [`FilterError::TimestampIndexOutOfBounds`] if an event falls beyond the end of
///   `timestamps`.
pub fn cusum_filter_timestamps(
    close: &[f64],
    timestamps: &[NaiveDateTime],
    threshold: Threshold,
) -> Result<Vec<NaiveDateTime>, FilterError> {
    let indices = cusum_filter_indices(close, threshold)?;
    indices
        .into_iter()
        .map(|i| {
            timestamps.get(i).copied().ok_or(FilterError::TimestampIndexOutOfBounds {
                index: i,
                available: timestamps.len(),
            })
        })
        .collect()
}

fn rolling_mean_std(window: &[f64]) -> (f64, f64) {
    let len = window.len() as f64;
    let mean = window.iter().sum::<f64>() / len;
    // sample std (ddof=1) to match pandas default
    let var = if window.len() > 1 {
        window
            .iter()
            .map(|v| {
                let diff = v - mean;
                diff * diff
            })
            .sum::<f64>()
            / (len - 1.0)
    } else {
        0.0
    };
    (mean, var.sqrt())
}

/// Rolling z-score filter returning event positions (ported from mlfinlab, not in AFML).
///
/// Bar `i` is an event when `close[i] >= mean + threshold * std`, where `mean` is the mean of
/// the last `mean_window` prices and `std` the sample standard deviation (ddof = 1) of the
/// last `std_window` prices, both windows including bar `i`. Evaluation starts at bar
/// `max(mean_window, std_window) - 1`. Returns an empty vector if the series is empty, both
/// windows are zero, or the series is shorter than the longer window.
///
/// The filter is one-sided (only upward excursions fire), works on price levels, and has no
/// reset, so consecutive bars above the band are consecutive events.
pub fn z_score_filter_indices(
    close: &[f64],
    mean_window: usize,
    std_window: usize,
    threshold: f64,
) -> Vec<usize> {
    let mut events = Vec::new();
    let n = close.len();
    if n == 0 {
        return events;
    }
    let window = mean_window.max(std_window);
    if window == 0 || n < window {
        return events;
    }

    for i in (window - 1)..n {
        let start_mean = i + 1 - mean_window;
        let start_std = i + 1 - std_window;
        let (mean, _) = rolling_mean_std(&close[start_mean..=i]);
        let (_, std_for_threshold) = rolling_mean_std(&close[start_std..=i]);
        let threshold_val = mean + threshold * std_for_threshold;
        if close[i] >= threshold_val {
            events.push(i);
        }
    }

    events
}

/// Rolling z-score filter returning event timestamps.
///
/// Runs [`z_score_filter_indices`] and maps each event position to `timestamps[position]`.
///
/// # Errors
///
/// [`FilterError::TimestampIndexOutOfBounds`] if an event falls beyond the end of
/// `timestamps`.
pub fn z_score_filter_timestamps(
    close: &[f64],
    timestamps: &[NaiveDateTime],
    mean_window: usize,
    std_window: usize,
    threshold: f64,
) -> Result<Vec<NaiveDateTime>, FilterError> {
    let indices = z_score_filter_indices(close, mean_window, std_window, threshold);
    indices
        .into_iter()
        .map(|i| {
            timestamps.get(i).copied().ok_or(FilterError::TimestampIndexOutOfBounds {
                index: i,
                available: timestamps.len(),
            })
        })
        .collect()
}
