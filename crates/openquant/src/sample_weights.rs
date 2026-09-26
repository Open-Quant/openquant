//! Training sample weights for overlapping labels (AFML Chapter 4).
//!
//! Two independent weightings of triple-barrier events, meant to be passed to a learner as
//! `sample_weight` (AFML suggests multiplying them, §4.7):
//! - [`get_weights_by_return`] — return attribution (§4.6, Snippet 4.10): each label is
//!   credited with the absolute sum of the log returns inside its span, each divided by the
//!   number of labels alive at that bar; the weights are rescaled to sum to the number of
//!   labels.
//! - [`get_weights_by_time_decay`] — time decay (§4.7, Snippet 4.11): a piecewise-linear decay
//!   along **cumulative average uniqueness** (not calendar time), with the newest label at
//!   weight 1. These weights are not rescaled.
//!
//! Conventions:
//! - Events are `(start, end, label)` triples; the label (third element) is not used. Spans are
//!   inclusive, `[start, end]`, and matched to `close` by exact timestamp comparison.
//! - `close` is the `(timestamp, price)` bar series the labels were built on, in ascending
//!   time order. Prices, not returns; log returns are computed once over the whole series, so
//!   a label also collects the return arriving at its `start` bar.
//! - Both functions return one `(event start, weight)` pair per event, in input order,
//!   duplicates included.
//! - Concurrency and cumulative uniqueness are computed over the events passed in: compute
//!   weights on the training fold only, or the future's label density leaks into the past.
//! - Class imbalance (§4.8) is a separate correction; neither function looks at the label.
//!
//! ```
//! use chrono::{Duration, NaiveDate};
//! use openquant::sample_weights::{get_weights_by_return, get_weights_by_time_decay};
//!
//! # fn main() -> Result<(), openquant::sample_weights::SampleWeightsError> {
//! let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 30, 0).unwrap();
//! let prices = [100.0, 100.5, 101.5, 101.0, 101.2, 101.1, 103.0, 103.2];
//! let close: Vec<_> =
//!     prices.iter().enumerate().map(|(i, p)| (open + Duration::minutes(i as i64), *p)).collect();
//! let at = |i: usize| close[i].0;
//! // A and B overlap on bars 1-3; C and D stand alone.
//! let events =
//!     vec![(at(0), at(3), 1.0), (at(1), at(3), 1.0), (at(4), at(5), -1.0), (at(6), at(7), 1.0)];
//!
//! let by_return = get_weights_by_return(&events, &close)?;
//! let expected = [0.631794, 0.631794, 0.125670, 2.610742];
//! for ((_, w), e) in by_return.iter().zip(expected) {
//!     assert!((w - e).abs() < 1e-6);
//! }
//!
//! let by_decay = get_weights_by_time_decay(&events, &close, 0.5)?;
//! let expected = [0.60, 0.68, 0.84, 1.00];
//! for ((_, w), e) in by_decay.iter().zip(expected) {
//!     assert!((w - e).abs() < 1e-12);
//! }
//! # Ok(())
//! # }
//! ```
#![deny(missing_docs)]

use chrono::NaiveDateTime;
use itertools::Itertools;
use std::collections::BTreeMap;

/// Errors returned by the sample-weight functions.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SampleWeightsError {
    /// An event's `end` precedes its `start`.
    #[error("event {index} ends before it starts")]
    EndBeforeStart {
        /// Position (in the input slice) of the first offending event.
        index: usize,
    },
    /// [`get_weights_by_time_decay`] was given a `decay` outside AFML's domain `(-1, 1]`
    /// (`NaN` included).
    #[error("decay must be in (-1, 1], got {0}")]
    InvalidDecay(f64),
}

/// Reject events whose end precedes their start; such a span holds no bars to weight.
fn validate_events(
    triple_barrier_events: &[(NaiveDateTime, NaiveDateTime, f64)],
) -> Result<(), SampleWeightsError> {
    match triple_barrier_events.iter().position(|(t_in, t1, _)| t1 < t_in) {
        Some(index) => Err(SampleWeightsError::EndBeforeStart { index }),
        None => Ok(()),
    }
}

/// Compute number of concurrent events for each timestamp.
fn num_concurrent_events(
    close_index: &[NaiveDateTime],
    label_endtime: &[(NaiveDateTime, NaiveDateTime)],
) -> BTreeMap<NaiveDateTime, usize> {
    let mut counts: BTreeMap<NaiveDateTime, usize> = BTreeMap::new();
    for (start, end) in label_endtime {
        for ts in close_index.iter().filter(|ts| **ts >= *start && **ts <= *end) {
            *counts.entry(*ts).or_insert(0) += 1;
        }
    }
    counts
}

/// Average uniqueness from triple barrier events.
#[allow(dead_code)]
fn get_av_uniqueness_from_triple_barrier(
    triple_barrier_events: &[(NaiveDateTime, NaiveDateTime, f64)],
    close_index: &[NaiveDateTime],
) -> BTreeMap<NaiveDateTime, f64> {
    let label_endtime: Vec<_> =
        triple_barrier_events.iter().map(|(t_in, t1, _)| (*t_in, *t1)).collect();
    let num_conc = num_concurrent_events(close_index, &label_endtime);

    // Compute uniqueness per event timestamp
    let mut uniqueness: Vec<(NaiveDateTime, f64)> = Vec::new();
    for (t_in, t1, _) in triple_barrier_events {
        let mut denom = 0.0;
        for ts in close_index.iter().filter(|ts| **ts >= *t_in && **ts <= *t1) {
            if let Some(c) = num_conc.get(ts) {
                denom += 1.0 / (*c as f64);
            }
        }
        uniqueness.push((*t_in, denom));
    }

    // Normalize to weights summing to 1 over ordering
    let mut t_w: BTreeMap<NaiveDateTime, f64> = BTreeMap::new();
    for (ts, uniq) in uniqueness {
        t_w.insert(ts, uniq);
    }
    t_w
}

/// Sample weights by return attribution (AFML §4.6, Snippet 4.10).
///
/// For each event `i`, `w_i = |sum_{t in [start_i, end_i]} r_t / c_t|`, where `r_t` is the log
/// return arriving at bar `t` (`ln(close_t / close_{t-1})`, computed over the whole series, so
/// the return arriving at `start_i` is included) and `c_t` is the number of events whose span
/// contains bar `t`. The weights are then scaled to sum to the number of events (mean 1),
/// unless they are all zero, in which case they are returned unscaled.
///
/// `triple_barrier_events` are `(start, end, label)` with the label ignored; `close` is
/// `(timestamp, price)` in ascending time order with strictly positive prices (a non-positive
/// price yields a non-finite log return and non-finite weights). Returns
/// `(event start, weight)` per event, in input order. An empty `triple_barrier_events` returns
/// an empty vector.
///
/// A single large move (a gap or bad print) can take most of the total weight, and a label
/// over a flat stretch gets a weight near zero.
///
/// # Errors
///
/// [`SampleWeightsError::EndBeforeStart`] if any event's `end` precedes its `start`.
///
/// ```
/// use chrono::{Duration, NaiveDate};
/// use openquant::sample_weights::get_weights_by_return;
///
/// let t0 = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 30, 0).unwrap();
/// let close: Vec<_> = [100.0, 110.0, 99.0]
///     .iter()
///     .enumerate()
///     .map(|(i, p)| (t0 + Duration::minutes(i as i64), *p))
///     .collect();
/// // Two non-overlapping events: |ln(1.1)| and |ln(0.9)|, rescaled to sum to 2.
/// let events = vec![(close[1].0, close[1].0, 1.0), (close[2].0, close[2].0, -1.0)];
/// let w = get_weights_by_return(&events, &close).unwrap();
/// let (a, b) = (1.1f64.ln().abs(), 0.9f64.ln().abs());
/// assert!((w[0].1 - 2.0 * a / (a + b)).abs() < 1e-12);
/// assert!((w[0].1 + w[1].1 - 2.0).abs() < 1e-12);
/// ```
pub fn get_weights_by_return(
    triple_barrier_events: &[(NaiveDateTime, NaiveDateTime, f64)],
    close: &[(NaiveDateTime, f64)],
) -> Result<Vec<(NaiveDateTime, f64)>, SampleWeightsError> {
    if triple_barrier_events.is_empty() {
        return Ok(Vec::new());
    }
    validate_events(triple_barrier_events)?;

    let num_conc = num_concurrent_events(
        &close.iter().map(|(ts, _)| *ts).collect_vec(),
        &triple_barrier_events.iter().map(|(t_in, t1, _)| (*t_in, *t1)).collect_vec(),
    );

    // Snippet 4.10: `ret = log(close).diff()` over the whole series, then the sum of
    // ret / concurrency over [t_in, t_out]. That includes the return arriving at t_in, so the
    // returns are taken from the full series, not restarted inside each window.
    let log_returns: Vec<(NaiveDateTime, f64)> =
        close.windows(2).map(|w| (w[1].0, (w[1].1 / w[0].1).ln())).collect();

    let mut weights: Vec<(NaiveDateTime, f64)> = Vec::new();
    for (t_in, t_out, _) in triple_barrier_events {
        let sum: f64 = log_returns
            .iter()
            .filter(|(ts, _)| *ts >= *t_in && *ts <= *t_out)
            .filter_map(|(ts, ret)| num_conc.get(ts).map(|c| ret / (*c as f64)))
            .sum();
        weights.push((*t_in, sum.abs()));
    }

    // Normalize
    let total: f64 = weights.iter().map(|(_, w)| *w).sum();
    if total > 0.0 {
        let scale = (weights.len() as f64) / total;
        for (_, w) in weights.iter_mut() {
            *w *= scale;
        }
    }
    Ok(weights)
}

/// Sample weights by time decay (AFML §4.7, Snippet 4.11).
///
/// Each event's average uniqueness is the mean of `1 / c_t` over the bars of `close` in its
/// span (`c_t` = number of events containing bar `t`, floored at 1); an event covering no bar
/// has uniqueness 0. With `x_i` the cumulative uniqueness in start order and `X` its total, the
/// weight is `max(0, a + b x_i)` with `b = (1 - decay) / X` for `decay >= 0`,
/// `b = 1 / ((decay + 1) X)` for `decay < 0`, and `a = 1 - b X`, so the newest event has
/// weight 1. AFML's domain is `decay` in `(-1, 1]`: `1` is no decay, `0 < decay < 1` decays
/// linearly toward `decay`, `0` toward 0, and `-1 < decay < 0` zeroes the oldest `-decay`
/// fraction of cumulative uniqueness. The oldest weight *approaches* `decay` rather than
/// equalling it, because the line is anchored at `x = 0`. The
/// weights are not rescaled and do not include return attribution; `close` prices are not
/// used, only its timestamps.
///
/// Returns one `(start, weight)` pair per event, in input order. Cumulative uniqueness is
/// accumulated in start order; events that share a start keep their input order (a stable
/// sort), so each takes its own cumulative position and the later-listed one counts as newer.
///
/// # Errors
///
/// - [`SampleWeightsError::InvalidDecay`] if `decay` is outside `(-1, 1]` or `NaN`.
/// - [`SampleWeightsError::EndBeforeStart`] if any event's `end` precedes its `start`.
///
/// ```
/// use chrono::{Duration, NaiveDate};
/// use openquant::sample_weights::get_weights_by_time_decay;
///
/// let t0 = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 30, 0).unwrap();
/// let close: Vec<_> = (0..4).map(|i| (t0 + Duration::minutes(i), 100.0)).collect();
/// // Four non-overlapping one-bar events: x = 1, 2, 3, 4 and X = 4.
/// let events: Vec<_> = close.iter().map(|(t, _)| (*t, *t, 1.0)).collect();
///
/// // decay = 0: b = 1/4, a = 0.
/// let w = get_weights_by_time_decay(&events, &close, 0.0).unwrap();
/// let got: Vec<f64> = w.iter().map(|(_, w)| *w).collect();
/// assert_eq!(got, vec![0.25, 0.5, 0.75, 1.0]);
///
/// // decay = -0.5: b = 1/2, a = -1; the oldest half is clipped to 0.
/// let w = get_weights_by_time_decay(&events, &close, -0.5).unwrap();
/// let got: Vec<f64> = w.iter().map(|(_, w)| *w).collect();
/// assert_eq!(got, vec![0.0, 0.0, 0.5, 1.0]);
/// ```
pub fn get_weights_by_time_decay(
    triple_barrier_events: &[(NaiveDateTime, NaiveDateTime, f64)],
    close: &[(NaiveDateTime, f64)],
    decay: f64,
) -> Result<Vec<(NaiveDateTime, f64)>, SampleWeightsError> {
    // `decay = -1` divides by zero in the slope below; above 1 the weights grow with age.
    if !(decay > -1.0 && decay <= 1.0) {
        return Err(SampleWeightsError::InvalidDecay(decay));
    }
    validate_events(triple_barrier_events)?;
    let close_index: Vec<NaiveDateTime> = close.iter().map(|(ts, _)| *ts).collect();

    // num concurrent events per bar
    let mut conc: Vec<(NaiveDateTime, usize)> = Vec::new();
    for ts in &close_index {
        let mut count = 0usize;
        for (start, end, _) in triple_barrier_events {
            if *ts >= *start && *ts <= *end {
                count += 1;
            }
        }
        conc.push((*ts, count.max(1)));
    }

    // average uniqueness per event
    let mut av_uniqueness = Vec::new();
    for (start, end, _) in triple_barrier_events {
        let vals: Vec<f64> = conc
            .iter()
            .filter(|(ts, _)| *ts >= *start && *ts <= *end)
            .map(|(_, c)| 1.0 / (*c as f64))
            .collect();
        let avg =
            if vals.is_empty() { 0.0 } else { vals.iter().sum::<f64>() / (vals.len() as f64) };
        av_uniqueness.push(avg);
    }

    // Accumulate in start order. The sort is stable, so events sharing a start keep their input
    // order; carrying the event index (not the timestamp) keeps each of them.
    let mut order: Vec<usize> = (0..triple_barrier_events.len()).collect();
    order.sort_by_key(|&i| triple_barrier_events[i].0);
    let mut decay_w = vec![0.0; triple_barrier_events.len()];
    let mut cum = 0.0;
    for &i in &order {
        cum += av_uniqueness[i];
        decay_w[i] = cum;
    }
    if !order.is_empty() {
        let denom = cum;
        let slope =
            if decay >= 0.0 { (1.0 - decay) / denom } else { 1.0 / ((decay + 1.0) * denom) };
        let constant = 1.0 - slope * denom;
        for w in decay_w.iter_mut() {
            *w = constant + slope * *w;
            if *w < 0.0 {
                *w = 0.0;
            }
        }
    }

    Ok(triple_barrier_events.iter().zip(decay_w).map(|((start, _, _), w)| (*start, w)).collect())
}
