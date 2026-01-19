//! Filters module (ported from mlfinlab).
//!
//! Mirrors the Python implementations used in tests; behavior is kept intentionally similar
//! (including rolling std with ddof=1).

use chrono::NaiveDateTime;

/// Threshold type for CUSUM filter.
pub enum Threshold {
    Scalar(f64),
    Dynamic(Vec<f64>),
}

fn threshold_at(threshold: &Threshold, idx: usize) -> f64 {
    match threshold {
        Threshold::Scalar(v) => *v,
        Threshold::Dynamic(v) => v
            .get(idx)
            .copied()
            .unwrap_or_else(|| panic!("dynamic threshold missing value at {idx}")),
    }
}

/// CUSUM filter returning indices of events (0-based positions in the input).
pub fn cusum_filter_indices(close: &[f64], threshold: Threshold) -> Vec<usize> {
    let mut events = Vec::new();
    if close.len() < 2 {
        return events;
    }

    let mut s_pos = 0.0_f64;
    let mut s_neg = 0.0_f64;

    for i in 1..close.len() {
        let log_ret = (close[i] / close[i - 1]).ln();
        let thresh = threshold_at(&threshold, i);

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

    events
}

/// CUSUM filter returning timestamps of events.
pub fn cusum_filter_timestamps(
    close: &[f64],
    timestamps: &[NaiveDateTime],
    threshold: Threshold,
) -> Vec<NaiveDateTime> {
    let indices = cusum_filter_indices(close, threshold);
    indices.into_iter().map(|i| timestamps.get(i).copied().expect("timestamp index")).collect()
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
    let indices = z_score_filter_indices(close, mean_window, std_window, threshold);
    indices.into_iter().map(|i| timestamps.get(i).copied().expect("timestamp index")).collect()
}
