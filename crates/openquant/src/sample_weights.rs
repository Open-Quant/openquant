use chrono::NaiveDateTime;
use itertools::Itertools;
use std::collections::{BTreeMap, HashMap};

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
fn get_av_uniqueness_from_triple_barrier(
    triple_barrier_events: &[(NaiveDateTime, NaiveDateTime, f64)],
    close_index: &[NaiveDateTime],
) -> BTreeMap<NaiveDateTime, f64> {
    let label_endtime: Vec<_> = triple_barrier_events
        .iter()
        .map(|(t_in, t1, _)| (*t_in, *t1))
        .collect();
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
) -> Result<Vec<(NaiveDateTime, f64)>, String> {
    if triple_barrier_events.is_empty() {
        return Ok(Vec::new());
    }
    // Validate none are NaN
    if triple_barrier_events.iter().any(|(t_in, t1, _)| t_in.and_utc().timestamp() == 0 || t1.and_utc().timestamp() == 0)
    {
        return Err("NaN values in triple_barrier_events, delete nans".into());
    }

    let _close_map: HashMap<NaiveDateTime, f64> = close.iter().cloned().collect();
    let num_conc = num_concurrent_events(
        &close.iter().map(|(ts, _)| *ts).collect_vec(),
        &triple_barrier_events
            .iter()
            .map(|(t_in, t1, _)| (*t_in, *t1))
            .collect_vec(),
    );

    let mut weights: Vec<(NaiveDateTime, f64)> = Vec::new();
    for (t_in, t_out, _) in triple_barrier_events {
        let mut sum: f64 = 0.0;
        let mut last: Option<f64> = None;
        for (ts, price) in close.iter().filter(|(ts, _)| *ts >= *t_in && *ts <= *t_out) {
            if let Some(prev) = last {
                let ret = (price / prev).ln();
                if let Some(c) = num_conc.get(ts) {
                    sum += ret / (*c as f64);
                }
            }
            last = Some(*price);
        }
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
pub fn get_weights_by_time_decay(
    triple_barrier_events: &[(NaiveDateTime, NaiveDateTime, f64)],
    close: &[(NaiveDateTime, f64)],
    decay: f64,
) -> Result<Vec<(NaiveDateTime, f64)>, String> {
    if triple_barrier_events
        .iter()
        .any(|(t_in, t1, _)| t_in.and_utc().timestamp() == 0 || t1.and_utc().timestamp() == 0)
    {
        return Err("NaN values in triple_barrier_events, delete nans".into());
    }
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
        let avg = if vals.is_empty() {
            0.0
        } else {
            vals.iter().sum::<f64>() / (vals.len() as f64)
        };
        av_uniqueness.push((*start, avg));
    }

    // sort by time for decay application
    av_uniqueness.sort_by_key(|(ts, _)| *ts);
    let mut decay_w: Vec<(NaiveDateTime, f64)> = Vec::new();
    let mut cum = 0.0;
    for (ts, val) in &av_uniqueness {
        cum += *val;
        decay_w.push((*ts, cum));
    }
    if let Some((_, last)) = decay_w.last().cloned() {
        let denom = last;
        let slope = if decay >= 0.0 {
            (1.0 - decay) / denom
        } else {
            1.0 / ((decay + 1.0) * denom)
        };
        let constant = 1.0 - slope * denom;
        for (_, w) in decay_w.iter_mut() {
            *w = constant + slope * *w;
            if *w < 0.0 {
                *w = 0.0;
            }
        }
    }

    // Return weights aligned to original event order
    let mut weight_map: HashMap<NaiveDateTime, f64> = decay_w.into_iter().collect();
    let mut out = Vec::new();
    for (start, _, _) in triple_barrier_events {
        if let Some(w) = weight_map.remove(start) {
            out.push((*start, w));
        }
    }
    Ok(out)
}
