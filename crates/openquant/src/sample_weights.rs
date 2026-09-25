use chrono::NaiveDateTime;
use itertools::Itertools;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SampleWeightsError {
    #[error("event {index} ends before it starts")]
    EndBeforeStart { index: usize },
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

/// Sample weights by return attribution.
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

/// Sample weights by time decay.
///
/// Returns one `(start, weight)` pair per event, in input order. Cumulative uniqueness is
/// accumulated in start order; events that share a start keep their input order (a stable
/// sort), so each takes its own cumulative position and the later-listed one counts as newer.
pub fn get_weights_by_time_decay(
    triple_barrier_events: &[(NaiveDateTime, NaiveDateTime, f64)],
    close: &[(NaiveDateTime, f64)],
    decay: f64,
) -> Result<Vec<(NaiveDateTime, f64)>, SampleWeightsError> {
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
