//! Triple-barrier labeling and meta-labeling (AFML chapter 3).
//!
//! An event at bar `t0` with target `trgt` (a volatility estimate known at `t0`, e.g. from
//! Snippet 3.1) is resolved by whichever barrier the close path reaches first: a profit
//! target at `pt * trgt`, a stop at `-sl * trgt`, or a vertical (time) barrier (§3.4,
//! Snippet 3.2). The label is the sign of the return at that point (Snippet 3.5); with a
//! side from a primary model it is instead 1 if the side-signed return is positive and 0
//! otherwise (meta-labeling, §3.6, Snippets 3.6–3.7).
//!
//! Conventions:
//!
//! - Prices are `(timestamp, close)` pairs in increasing time. Events, targets, sides and
//!   vertical barriers are joined to them by **exact** timestamp; an event whose timestamp is
//!   not a bar is skipped silently.
//! - Returns are simple, `p_t / p_t0 - 1`, from the event bar's close, multiplied by the side
//!   when one is given (a missing side is taken as `+1` for the barriers).
//! - A barrier is touched only when the return goes **strictly** beyond it, and only closes
//!   are checked. A multiple of zero disables that barrier.
//! - The label at a vertical barrier is the sign of the return there, not 0.
//!
//! ```
//! use chrono::{Duration, NaiveDate};
//! use openquant::labeling::{add_vertical_barrier, get_bins, get_events};
//!
//! let t0 = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
//! let day = |d: i64| t0 + Duration::days(d);
//! let prices = [100.0, 100.5, 101.0, 103.5, 102.0, 101.0, 100.0, 99.0];
//! let close: Vec<_> = prices.iter().enumerate().map(|(i, p)| (day(i as i64), *p)).collect();
//!
//! // One event on day 0, a 2% target, barriers one target wide, a five-day limit.
//! let events = vec![day(0)];
//! let target = vec![(day(0), 0.02)];
//! let vertical = add_vertical_barrier(&events, &close, 5, 0, 0, 0);
//! let found = get_events(&close, &events, (1.0, 1.0), &target, 0.0, 1, Some(&vertical), None);
//! let bins = get_bins(&found, &close);
//!
//! // +1.0% on day 2 is inside the barrier; +3.5% on day 3 is beyond it.
//! assert_eq!(found[0].1.t1, Some(day(3)));
//! let (_, ret, _, label, side) = bins[0];
//! assert!((ret - 0.035).abs() < 1e-12);
//! assert_eq!((label, side), (1, None));
//! ```
#![deny(missing_docs)]

use chrono::{Duration, NaiveDateTime};
use std::collections::HashMap;

/// A triple-barrier event, keyed elsewhere by its start timestamp `t0`.
#[derive(Debug, Clone)]
pub struct Event {
    /// When the event resolved: the first barrier touch or the vertical barrier, whichever
    /// is earlier. `None` when there is no vertical barrier and no horizontal barrier has
    /// been touched yet (the outcome is unknown).
    pub t1: Option<NaiveDateTime>,
    /// Target return (the unit of the horizontal barriers), known at `t0`.
    pub trgt: f64,
    /// Side from a primary model (`+1` long, `-1` short), for meta-labeling.
    pub side: Option<f64>,
    /// Profit-taking multiple of `trgt` (0 disables the barrier).
    pub pt: f64,
    /// Stop-loss multiple of `trgt` (0 disables the barrier).
    pub sl: f64,
}

/// Parameters of [`triple_barrier_events`].
#[derive(Debug, Clone, Copy)]
pub struct TripleBarrierConfig<'a> {
    /// Profit-taking multiple of the target (0 disables it).
    pub pt: f64,
    /// Stop-loss multiple of the target (0 disables it).
    pub sl: f64,
    /// Events whose target is not above this (or is `NaN`) are dropped.
    pub min_ret: f64,
    /// `(event, vertical barrier)` pairs, e.g. from [`add_vertical_barrier`]; events without
    /// one have no time limit.
    pub vertical_barrier_times: Option<&'a [(NaiveDateTime, NaiveDateTime)]>,
}

/// A labelled event.
#[derive(Debug, Clone, PartialEq)]
pub struct LabeledEvent {
    /// Event start `t0`.
    pub timestamp: NaiveDateTime,
    /// Simple return from `t0` to `t1`, multiplied by the side when there is one.
    pub ret: f64,
    /// The event's target.
    pub trgt: f64,
    /// `-1`, `0` or `1` without a side; `0` or `1` with one (meta-label).
    pub label: i8,
    /// The event's side, if any.
    pub side: Option<f64>,
}

/// Vertical (time) barriers: for each event, the first bar at or after the event time plus
/// the given offset (AFML Snippet 3.4).
///
/// Returns `(event, barrier)` pairs. An event too close to the end of `close` to have such a
/// bar gets no pair at all (no shortened barrier). `close` must be in increasing time.
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
            // No bar after t0 to check. A vertical barrier at or before t0 stays as it is; with
            // no vertical barrier (an event on the last bar) the outcome is unknown and t1
            // stays None, as below. Filling in `last_ts` here used to give t1 = t0 and a
            // label of 0 (#162).
            continue;
        }

        let start_price = close[start_idx].1;
        let side = ev.side.unwrap_or(1.0);
        let pt_level = if ev.pt > 0.0 { ev.pt * ev.trgt } else { f64::INFINITY };
        let sl_level = if ev.sl > 0.0 { -ev.sl * ev.trgt } else { f64::NEG_INFINITY };

        let mut first_touch = None;
        for &(ts, price) in &close[(start_idx + 1)..=end_idx] {
            let ret = (price / start_price - 1.0) * side;
            // Snippet 3.2: a barrier is touched when the path goes strictly beyond it.
            if ret > pt_level || ret < sl_level {
                first_touch = Some(ts);
                break;
            }
        }

        let resolved = match (ev.t1, first_touch) {
            (Some(vertical), Some(touched)) => Some(vertical.min(touched)),
            (Some(vertical), None) => Some(vertical),
            (None, Some(touched)) => Some(touched),
            // No vertical barrier and nothing touched: the outcome is not known yet. Snippet 3.2
            // leaves t1 empty; filling in the last bar would label an unresolved event.
            (None, None) => None,
        };
        ev.t1 = resolved;
    }
}

/// Triple-barrier events: resolves each event's end time `t1` (AFML Snippets 3.3 and 3.6).
///
/// Events are dropped when their timestamp is not a bar of `close`, their target is missing,
/// `NaN` or not above `config.min_ret`, or (when `side_prediction` is given) they have no
/// side. For each kept event, `t1` is the first bar whose side-signed simple return from
/// `t0` goes strictly beyond `pt * trgt` or `-sl * trgt`, or the vertical barrier if that is
/// earlier. With no vertical barrier and no touch, `t1` stays `None`; that includes an event
/// on the last bar, which has no later bar to touch. Such events are kept here and skipped
/// by [`triple_barrier_labels`]. Returns `(t0, event)` pairs in `t_events` order.
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
        // `target[target > min_ret]`, which in pandas also drops a NaN target.
        if trgt.is_nan() || trgt <= config.min_ret {
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

/// Labels resolved events by the return from `t0` to `t1` (AFML Snippets 3.5 and 3.7).
///
/// Label regime:
/// - `{-1, 0, 1}` when `side` is absent (standard triple-barrier labels): the sign of the
///   return, 0 only for an exactly zero return;
/// - `{0, 1}` when `side` is present (meta-labeling): 1 if the side-signed return is positive.
///
/// Events with `t1 = None`, or whose `t0` or `t1` is not a bar of `close`, are skipped.
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

/// Meta-labels (`{0, 1}`, AFML §3.6): [`triple_barrier_labels`] restricted to events that
/// carry a side.
pub fn meta_labels(
    events: &[(NaiveDateTime, Event)],
    close: &[(NaiveDateTime, f64)],
) -> Vec<LabeledEvent> {
    let with_side: Vec<(NaiveDateTime, Event)> =
        events.iter().filter(|(_, ev)| ev.side.is_some()).cloned().collect();
    triple_barrier_labels(&with_side, close)
}

/// mlfinlab-compatible form of [`triple_barrier_events`] (AFML Snippet 3.6's `getEvents`).
///
/// `pt_sl` is `(profit multiple, stop multiple)`. `num_threads` is ignored; it is kept so
/// mlfinlab call sites port unchanged.
// Mirrors the mlfinlab `get_events` signature.
#[allow(clippy::too_many_arguments)]
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

/// mlfinlab-compatible form of [`triple_barrier_labels`] (AFML Snippet 3.7's `getBins`),
/// returning `(t0, ret, trgt, label, side)` tuples.
pub fn get_bins(
    events: &[(NaiveDateTime, Event)],
    close: &[(NaiveDateTime, f64)],
) -> Vec<(NaiveDateTime, f64, f64, i8, Option<f64>)> {
    triple_barrier_labels(events, close)
        .into_iter()
        .map(|row| (row.timestamp, row.ret, row.trgt, row.label, row.side))
        .collect()
}

/// Drops under-represented labels (AFML Snippet 3.8).
///
/// Repeatedly removes the rarest label while its share is at most `min_pct` and at least
/// three distinct labels remain. Rows are `(t0, ret, trgt, label, side)` as from
/// [`get_bins`].
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
            if min_label.is_none_or(|(_, p)| pct < p) {
                min_label = Some((*label, pct));
            }
        }
        if let Some((label, pct)) = min_label
            && pct <= min_pct
            && counts.len() >= 3
        {
            filtered.retain(|(_, _, _, b, _)| *b != label);
            continue;
        }
        break;
    }
    filtered
}
