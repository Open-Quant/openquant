//! Filters module (ported from mlfinlab).
//!
//! Mirrors the Python implementations used in tests; behavior is kept intentionally similar
//! (including rolling std with ddof=1).

use chrono::NaiveDateTime;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum FilterError {
    MissingDynamicThreshold { index: usize, available: usize },
    TimestampIndexOutOfBounds { index: usize, available: usize },
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

/// Threshold type for CUSUM filter.
pub enum Threshold {
    Scalar(f64),
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

/// CUSUM filter returning indices of events (0-based positions in the input).
pub fn cusum_filter_indices(close: &[f64], threshold: Threshold) -> Vec<usize> {
    cusum_filter_indices_checked(close, threshold)
        .expect("invalid threshold in cusum_filter_indices")
}

/// CUSUM filter returning indices of events (0-based positions in the input).
pub fn cusum_filter_indices_checked(
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

/// CUSUM filter returning timestamps of events.
pub fn cusum_filter_timestamps(
    close: &[f64],
    timestamps: &[NaiveDateTime],
    threshold: Threshold,
) -> Vec<NaiveDateTime> {
    cusum_filter_timestamps_checked(close, timestamps, threshold)
        .expect("timestamp index out of bounds in cusum_filter_timestamps")
}

/// CUSUM filter returning timestamps of events.
pub fn cusum_filter_timestamps_checked(
    close: &[f64],
    timestamps: &[NaiveDateTime],
    threshold: Threshold,
) -> Result<Vec<NaiveDateTime>, FilterError> {
    let indices = cusum_filter_indices_checked(close, threshold)?;
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

/// Z-score filter returning indices of events.
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

/// Z-score filter returning timestamps of events.
pub fn z_score_filter_timestamps(
    close: &[f64],
    timestamps: &[NaiveDateTime],
    mean_window: usize,
    std_window: usize,
    threshold: f64,
) -> Vec<NaiveDateTime> {
    z_score_filter_timestamps_checked(close, timestamps, mean_window, std_window, threshold)
        .expect("timestamp index out of bounds in z_score_filter_timestamps")
}

/// Z-score filter returning timestamps of events.
pub fn z_score_filter_timestamps_checked(
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
