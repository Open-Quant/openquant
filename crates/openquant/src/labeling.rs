use chrono::{Duration, NaiveDateTime};
#[derive(Debug, Clone)]
pub struct Event {
    pub t1: Option<NaiveDateTime>,
    pub trgt: f64,
    pub side: Option<f64>,
    pub pt: f64,
    pub sl: f64,
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

fn apply_pt_sl_on_t1(
    close: &[(NaiveDateTime, f64)],
    events: &[(NaiveDateTime, Event)],
    pt_sl: (f64, f64),
    molecule: &[NaiveDateTime],
) -> Vec<(NaiveDateTime, Option<NaiveDateTime>, Option<NaiveDateTime>)> {
    let mut out = Vec::new();
    for &loc in molecule {
        let ev = events.iter().find(|(ts, _)| *ts == loc).expect("event");
        let side = ev.1.side.unwrap_or(1.0);
        let profit_taking = if pt_sl.0 > 0.0 { pt_sl.0 * ev.1.trgt } else { f64::NAN };
        let stop_loss = if pt_sl.1 > 0.0 { -pt_sl.1 * ev.1.trgt } else { f64::NAN };
        let last_ts = close.last().map(|(ts, _)| *ts).expect("close empty");
        let end_ts = ev.1.t1.unwrap_or(last_ts);

        let start_idx = close.iter().position(|(ts, _)| *ts == loc).expect("loc in close");
        let end_idx = close.iter().position(|(ts, _)| *ts == end_ts).unwrap_or(close.len() - 1);

        let start_price = close[start_idx].1;
        let mut sl_ts: Option<NaiveDateTime> = None;
        let mut pt_ts: Option<NaiveDateTime> = None;

        for &(ts, price) in &close[start_idx..=end_idx] {
            let ret = (price / start_price - 1.0) * side;
            if profit_taking.is_finite() && pt_ts.is_none() && ret > profit_taking {
                pt_ts = Some(ts);
            }
            if stop_loss.is_finite() && sl_ts.is_none() && ret < stop_loss {
                sl_ts = Some(ts);
            }
            if pt_ts.is_some() || sl_ts.is_some() {
                break;
            }
        }

        out.push((loc, sl_ts, pt_ts));
    }
    out
}

/// Get events for triple barrier method.
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
    // map target to t_events and filter
    let mut target_map = Vec::new();
    for &te in t_events {
        if let Some((_, val)) = target.iter().find(|(ts, _)| *ts == te) {
            if *val > min_ret {
                target_map.push((te, *val));
            }
        }
    }

    // vertical barriers map
    let mut vbar_map: Vec<(NaiveDateTime, Option<NaiveDateTime>)> =
        target_map.iter().map(|(ts, _)| (*ts, None)).collect();
    if let Some(vbars) = vertical_barrier_times {
        for (start, end) in vbars {
            if let Some(item) = vbar_map.iter_mut().find(|(ts, _)| ts == start) {
                item.1 = Some(*end);
            }
        }
    }

    // side
    let side_map: Vec<(NaiveDateTime, f64)> = if let Some(side) = side_prediction {
        side.iter().map(|(ts, val)| (*ts, *val)).collect()
    } else {
        target_map.iter().map(|(ts, _)| (*ts, 1.0)).collect()
    };

    let mut events: Vec<(NaiveDateTime, Event)> = target_map
        .iter()
        .map(|(ts, trgt)| {
            let t1 = vbar_map.iter().find(|(t, _)| t == ts).and_then(|(_, v)| *v);
            let side = side_map.iter().find(|(t, _)| t == ts).map(|(_, s)| *s);
            (*ts, Event { t1, trgt: *trgt, side, pt: pt_sl.0, sl: pt_sl.1 })
        })
        .collect();

    // apply barriers
    let molecule: Vec<NaiveDateTime> = events.iter().map(|(ts, _)| *ts).collect();
    let first_touch = apply_pt_sl_on_t1(close, &events, pt_sl, &molecule);
    for (loc, sl_ts, pt_ts) in first_touch {
        if let Some(event) = events.iter_mut().find(|(ts, _)| *ts == loc) {
            let mut candidates = Vec::new();
            if let Some(ts) = event.1.t1 {
                candidates.push(ts);
            }
            if let Some(ts) = sl_ts {
                candidates.push(ts);
            }
            if let Some(ts) = pt_ts {
                candidates.push(ts);
            }
            if let Some(min_ts) = candidates.into_iter().min() {
                event.1.t1 = Some(min_ts);
            }
            if side_prediction.is_none() {
                event.1.side = None;
            }
        }
    }

    events
}

/// Label outcomes given events and close prices.
pub fn get_bins(
    events: &[(NaiveDateTime, Event)],
    close: &[(NaiveDateTime, f64)],
) -> Vec<(NaiveDateTime, f64, f64, i8, Option<f64>)> {
    let mut out = Vec::new();
    for (start, ev) in events {
        let t1 = match ev.t1 {
            Some(ts) => ts,
            None => continue,
        };
        let start_price = close.iter().find(|(ts, _)| *ts == *start).map(|(_, p)| *p);
        let end_price = close.iter().find(|(ts, _)| *ts == t1).map(|(_, p)| *p);
        if let (Some(p0), Some(p1)) = (start_price, end_price) {
            let mut ret = p1.ln() - p0.ln();
            if let Some(side) = ev.side {
                ret *= side;
            }
            let mut bin = barrier_touched(ret, ev.trgt, ev.pt, ev.sl);
            if ev.side.is_some() && ret <= 0.0 {
                bin = 0;
            }
            let ret_norm = ret.exp() - 1.0;
            out.push((*start, ret_norm, ev.trgt, bin, ev.side));
        }
    }
    out
}

fn barrier_touched(ret: f64, trgt: f64, pt: f64, sl: f64) -> i8 {
    let pt_level = pt * trgt;
    let sl_level = -sl * trgt;
    if ret > 0.0 && pt > 0.0 && ret > pt_level {
        1
    } else if ret < 0.0 && sl > 0.0 && ret < sl_level {
        -1
    } else {
        0
    }
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
