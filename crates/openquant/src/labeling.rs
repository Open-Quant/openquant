use chrono::{Duration, NaiveDateTime};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Event {
    pub t1: Option<NaiveDateTime>,
    pub trgt: f64,
    pub side: Option<f64>,
    pub pt: f64,
    pub sl: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct TripleBarrierConfig<'a> {
    pub pt: f64,
    pub sl: f64,
    pub min_ret: f64,
    pub vertical_barrier_times: Option<&'a [(NaiveDateTime, NaiveDateTime)]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LabeledEvent {
    pub timestamp: NaiveDateTime,
    pub ret: f64,
    pub trgt: f64,
    pub label: i8,
    pub side: Option<f64>,
}

/// Add vertical barrier for each event by shifting timestamp forward.
pub fn add_vertical_barrier(
    t_events: &[NaiveDateTime],
    close: &[(NaiveDateTime, f64)],
    num_days: i64,
    num_hours: i64,
    num_minutes: i64,
    num_seconds: i64,
) -> Vec<(NaiveDateTime, NaiveDateTime)> {
    let delta = Duration::days(num_days)
        + Duration::hours(num_hours)
        + Duration::minutes(num_minutes)
        + Duration::seconds(num_seconds);

    let mut out = Vec::new();
    for &start in t_events {
        let target = start + delta;
        if let Some(&(ts, _)) = close.iter().find(|(ts, _)| *ts >= target) {
            out.push((start, ts));
        }
    }
    out
}

fn apply_pt_sl_on_t1(close: &[(NaiveDateTime, f64)], events: &mut [(NaiveDateTime, Event)]) {
    if close.is_empty() {
        return;
    }

    let last_ts = close.last().map(|(ts, _)| *ts).expect("non-empty close");
    let close_index: HashMap<NaiveDateTime, usize> =
        close.iter().enumerate().map(|(i, (ts, _))| (*ts, i)).collect();

    for (loc, ev) in events.iter_mut() {
        let Some(&start_idx) = close_index.get(loc) else {
            continue;
        };
        let end_ts = ev.t1.unwrap_or(last_ts);
        let end_idx = close_index.get(&end_ts).copied().unwrap_or(close.len() - 1);
        if end_idx <= start_idx {
            ev.t1 = Some(end_ts);
            continue;
        }

        let start_price = close[start_idx].1;
        let side = ev.side.unwrap_or(1.0);
        let pt_level = if ev.pt > 0.0 { ev.pt * ev.trgt } else { f64::INFINITY };
        let sl_level = if ev.sl > 0.0 { -ev.sl * ev.trgt } else { f64::NEG_INFINITY };

        let mut first_touch = None;
        for &(ts, price) in &close[(start_idx + 1)..=end_idx] {
            let ret = (price / start_price - 1.0) * side;
            if ret >= pt_level || ret <= sl_level {
                first_touch = Some(ts);
                break;
            }
        }

        let resolved = match (ev.t1, first_touch) {
            (Some(vertical), Some(touched)) => Some(vertical.min(touched)),
            (Some(vertical), None) => Some(vertical),
            (None, Some(touched)) => Some(touched),
            (None, None) => Some(end_ts),
        };
        ev.t1 = resolved;
    }
}

/// Construct triple-barrier events.
pub fn triple_barrier_events(
    close: &[(NaiveDateTime, f64)],
    t_events: &[NaiveDateTime],
    target: &[(NaiveDateTime, f64)],
    config: TripleBarrierConfig<'_>,
    side_prediction: Option<&[(NaiveDateTime, f64)]>,
) -> Vec<(NaiveDateTime, Event)> {
    if close.is_empty() {
        return Vec::new();
    }

    let close_index: HashMap<NaiveDateTime, usize> =
        close.iter().enumerate().map(|(i, (ts, _))| (*ts, i)).collect();
    let target_map: HashMap<NaiveDateTime, f64> = target.iter().copied().collect();
    let side_map: HashMap<NaiveDateTime, f64> =
        side_prediction.unwrap_or(&[]).iter().copied().collect();
    let vbar_map: HashMap<NaiveDateTime, NaiveDateTime> =
        config.vertical_barrier_times.unwrap_or(&[]).iter().copied().collect();

    let mut events = Vec::new();
    for &ts in t_events {
        if !close_index.contains_key(&ts) {
            continue;
        }
        let Some(&trgt) = target_map.get(&ts) else {
            continue;
        };
        if trgt <= config.min_ret {
            continue;
        }

        let side = if side_prediction.is_some() { side_map.get(&ts).copied() } else { None };
        if side_prediction.is_some() && side.is_none() {
            continue;
        }

        events.push((
            ts,
            Event { t1: vbar_map.get(&ts).copied(), trgt, side, pt: config.pt, sl: config.sl },
        ));
    }

    apply_pt_sl_on_t1(close, &mut events);
    events
}

/// Label triple-barrier outcomes.
///
/// Label regime:
/// - `{-1, 0, 1}` when `side` is absent (standard triple-barrier labels)
/// - `{0, 1}` when `side` is present (meta-labeling)
pub fn triple_barrier_labels(
    events: &[(NaiveDateTime, Event)],
    close: &[(NaiveDateTime, f64)],
) -> Vec<LabeledEvent> {
    if close.is_empty() {
        return Vec::new();
    }

    let close_price: HashMap<NaiveDateTime, f64> = close.iter().copied().collect();
    let mut out = Vec::new();
    for (start, ev) in events {
        let t1 = match ev.t1 {
            Some(ts) => ts,
            None => continue,
        };
        let start_price = close_price.get(start).copied();
        let end_price = close_price.get(&t1).copied();
        if let (Some(p0), Some(p1)) = (start_price, end_price) {
            let ret = p1 / p0 - 1.0;
            let mut signed_ret = ret;
            if let Some(side) = ev.side {
                signed_ret *= side;
            }

            let label = if ev.side.is_some() {
                if signed_ret > 0.0 {
                    1
                } else {
                    0
                }
            } else if signed_ret > 0.0 {
                1
            } else if signed_ret < 0.0 {
                -1
            } else {
                0
            };

            out.push(LabeledEvent {
                timestamp: *start,
                ret: signed_ret,
                trgt: ev.trgt,
                label,
                side: ev.side,
            });
        }
    }
    out
}

/// Label outcomes in meta-labeling mode (`{0, 1}` labels).
pub fn meta_labels(
    events: &[(NaiveDateTime, Event)],
    close: &[(NaiveDateTime, f64)],
) -> Vec<LabeledEvent> {
    let with_side: Vec<(NaiveDateTime, Event)> =
        events.iter().filter(|(_, ev)| ev.side.is_some()).cloned().collect();
    triple_barrier_labels(&with_side, close)
}

/// Backward-compatible triple-barrier API.
pub fn get_events(
    close: &[(NaiveDateTime, f64)],
    t_events: &[NaiveDateTime],
    pt_sl: (f64, f64),
    target: &[(NaiveDateTime, f64)],
    min_ret: f64,
    num_threads: usize, // unused, kept for parity
    vertical_barrier_times: Option<&[(NaiveDateTime, NaiveDateTime)]>,
    side_prediction: Option<&[(NaiveDateTime, f64)]>,
) -> Vec<(NaiveDateTime, Event)> {
    let _ = num_threads;
    triple_barrier_events(
        close,
        t_events,
        target,
        TripleBarrierConfig { pt: pt_sl.0, sl: pt_sl.1, min_ret, vertical_barrier_times },
        side_prediction,
    )
}

/// Backward-compatible label API.
pub fn get_bins(
    events: &[(NaiveDateTime, Event)],
    close: &[(NaiveDateTime, f64)],
) -> Vec<(NaiveDateTime, f64, f64, i8, Option<f64>)> {
    triple_barrier_labels(events, close)
        .into_iter()
        .map(|row| (row.timestamp, row.ret, row.trgt, row.label, row.side))
        .collect()
}

/// Drop labels whose frequency is below `min_pct`.
pub fn drop_labels(
    events: &[(NaiveDateTime, f64, f64, i8, Option<f64>)],
    min_pct: f64,
) -> Vec<(NaiveDateTime, f64, f64, i8, Option<f64>)> {
    let mut filtered: Vec<_> = events.to_vec();
    loop {
        let mut counts: std::collections::HashMap<i8, usize> = std::collections::HashMap::new();
        for (_, _, _, bin, _) in &filtered {
            *counts.entry(*bin).or_default() += 1;
        }
        let total = filtered.len() as f64;
        let mut min_label: Option<(i8, f64)> = None;
        for (label, count) in &counts {
            let pct = *count as f64 / total;
            if min_label.map_or(true, |(_, p)| pct < p) {
                min_label = Some((*label, pct));
            }
        }
        if let Some((label, pct)) = min_label {
            if pct <= min_pct && counts.len() >= 3 {
                filtered.retain(|(_, _, _, b, _)| *b != label);
                continue;
            }
        }
        break;
    }
    filtered
}
