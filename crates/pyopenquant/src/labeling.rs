use pyo3::prelude::*;

use crate::helpers::{
    build_labeling_events, format_naive_datetime, pair_timestamps_values, parse_datetime_str,
    parse_naive_datetime, parse_naive_datetimes, parse_vertical_barriers, LabelingEventArgs,
};

/// Python-facing event row: `(timestamp, t1, trgt, side, pt, sl)`.
type EventRow = (String, Option<String>, f64, Option<f64>, f64, f64);

/// Python-facing label row: `(timestamp, ret, trgt, bin, side)`.
type BinRow = (String, f64, f64, i8, Option<f64>);

/// Vertical (time) barriers: the first bar at or after each event time plus an offset.
///
/// AFML Snippet 3.4. The offset is `num_days + num_hours + num_minutes + num_seconds`. An
/// event too close to the end of the series to have such a bar gets no row at all (no
/// shortened barrier). `close_prices` is only checked for length. The result can be passed as
/// `vertical_barrier_times` to the other labeling functions.
///
/// Parameters
/// ----------
/// t_events : list[str]
///     Event timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is accepted).
/// close_timestamps : list[str]
///     Bar timestamps in the same format, in increasing order.
/// close_prices : list[float]
///     Close price of each bar (same length as `close_timestamps`).
/// num_days : int, default 0
///     Days in the offset.
/// num_hours : int, default 0
///     Hours in the offset.
/// num_minutes : int, default 0
///     Minutes in the offset.
/// num_seconds : int, default 0
///     Seconds in the offset.
///
/// Returns
/// -------
/// list[tuple[str, str]]
///     `(event, barrier)` timestamp pairs, in `t_events` order.
///
/// Raises
/// ------
/// ValueError
///     If all four offsets are zero, the timestamps and prices differ in length, or a
///     timestamp does not parse.
#[pyfunction(name = "add_vertical_barrier")]
#[pyo3(signature = (
    t_events,
    close_timestamps,
    close_prices,
    num_days=0,
    num_hours=0,
    num_minutes=0,
    num_seconds=0
))]
fn labeling_add_vertical_barrier(
    t_events: Vec<String>,
    close_timestamps: Vec<String>,
    close_prices: Vec<f64>,
    num_days: i64,
    num_hours: i64,
    num_minutes: i64,
    num_seconds: i64,
) -> PyResult<Vec<(String, String)>> {
    if num_days == 0 && num_hours == 0 && num_minutes == 0 && num_seconds == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "add_vertical_barrier needs a non-zero horizon: pass at least one of \
             num_days, num_hours, num_minutes or num_seconds",
        ));
    }
    let t_events = parse_naive_datetimes(t_events)?;
    let close =
        pair_timestamps_values(close_timestamps, close_prices, "close_timestamps", "close_prices")?;
    let barriers = openquant::labeling::add_vertical_barrier(
        &t_events,
        &close,
        num_days,
        num_hours,
        num_minutes,
        num_seconds,
    );
    Ok(barriers
        .into_iter()
        .map(|(a, b)| (format_naive_datetime(&a), format_naive_datetime(&b)))
        .collect())
}

/// Triple-barrier events: resolve each event's end time `t1`.
///
/// AFML Snippets 3.3 and 3.6. An event at bar `t0` with target `trgt` (a volatility estimate
/// known at `t0`, e.g. from `get_daily_vol`) ends at the first bar whose side-signed simple
/// return from `t0` goes strictly beyond `pt * trgt` or `-sl * trgt` (only closes are checked),
/// or at its vertical barrier if that is earlier. With no vertical barrier and no touch, `t1`
/// is None; that includes an event on the last bar.
///
/// Events, targets, sides and vertical barriers are joined to the bars by exact timestamp.
/// An event is dropped silently when its timestamp is not a bar, its target is missing, NaN or
/// not above `min_ret`, or (when `side_prediction` is given) it has no side.
///
/// Parameters
/// ----------
/// close_timestamps : list[str]
///     Bar timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is accepted),
///     in increasing order.
/// close_prices : list[float]
///     Close price of each bar.
/// t_events : list[str]
///     Event start timestamps, in the same format.
/// target_timestamps : list[str]
///     Timestamps of the target series.
/// target_values : list[float]
///     Target return at each target timestamp (the unit of the horizontal barriers).
/// pt : float, default 1.0
///     Profit-taking multiple of the target; 0 disables the barrier.
/// sl : float, default 1.0
///     Stop-loss multiple of the target; 0 disables the barrier.
/// min_ret : float, default 0.0
///     Events whose target is not above this are dropped.
/// vertical_barrier_times : list[tuple[str, str]] | None, default None
///     `(event, barrier)` pairs, e.g. from `add_vertical_barrier`; events without one have no
///     time limit.
/// side_prediction : list[tuple[str, float]] | None, default None
///     `(timestamp, side)` pairs from a primary model (`+1` long, `-1` short) for
///     meta-labeling. When given, events without a side are dropped.
///
/// Returns
/// -------
/// list[tuple[str, str | None, float, float | None, float, float]]
///     One `(t0, t1, trgt, side, pt, sl)` row per kept event, in `t_events` order.
///
/// Raises
/// ------
/// ValueError
///     If a timestamp list and its value list differ in length, or a timestamp does not
///     parse.
#[pyfunction(name = "triple_barrier_events")]
#[pyo3(signature = (
    close_timestamps,
    close_prices,
    t_events,
    target_timestamps,
    target_values,
    pt=1.0,
    sl=1.0,
    min_ret=0.0,
    vertical_barrier_times=None,
    side_prediction=None
))]
// Python keyword signature.
#[allow(clippy::too_many_arguments)]
fn labeling_triple_barrier_events(
    close_timestamps: Vec<String>,
    close_prices: Vec<f64>,
    t_events: Vec<String>,
    target_timestamps: Vec<String>,
    target_values: Vec<f64>,
    pt: f64,
    sl: f64,
    min_ret: f64,
    vertical_barrier_times: Option<Vec<(String, String)>>,
    side_prediction: Option<Vec<(String, f64)>>,
) -> PyResult<Vec<EventRow>> {
    let (_, events) = build_labeling_events(LabelingEventArgs {
        close_timestamps,
        close_prices,
        t_events,
        target_timestamps,
        target_values,
        pt,
        sl,
        min_ret,
        vertical_barrier_times,
        side_prediction,
    })?;
    Ok(events
        .into_iter()
        .map(|(ts, ev)| {
            (
                format_naive_datetime(&ts),
                ev.t1.map(|v| format_naive_datetime(&v)),
                ev.trgt,
                ev.side,
                ev.pt,
                ev.sl,
            )
        })
        .collect())
}

/// Triple-barrier labels: the sign of the return from `t0` to the resolved `t1`.
///
/// AFML Snippets 3.2 and 3.5. Builds events as `triple_barrier_events` (without a side) and
/// labels each by the simple return `close[t1] / close[t0] - 1`: 1 if positive, -1 if
/// negative, 0 only for an exactly zero return. A vertical-barrier exit is labelled by the
/// sign of the return there, not 0. Events whose `t1` is None (unresolved) are skipped.
///
/// Parameters
/// ----------
/// close_timestamps : list[str]
///     Bar timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is accepted),
///     in increasing order.
/// close_prices : list[float]
///     Close price of each bar.
/// t_events : list[str]
///     Event start timestamps, in the same format.
/// target_timestamps : list[str]
///     Timestamps of the target series.
/// target_values : list[float]
///     Target return at each target timestamp (the unit of the horizontal barriers).
/// pt : float, default 1.0
///     Profit-taking multiple of the target; 0 disables the barrier.
/// sl : float, default 1.0
///     Stop-loss multiple of the target; 0 disables the barrier.
/// min_ret : float, default 0.0
///     Events whose target is not above this (or is NaN) are dropped.
/// vertical_barrier_times : list[tuple[str, str]] | None, default None
///     `(event, barrier)` pairs, e.g. from `add_vertical_barrier`; events without one have no
///     time limit.
///
/// Returns
/// -------
/// list[tuple[str, float, float, int, float | None]]
///     One `(t0, ret, trgt, bin, side)` row per resolved event; `bin` is -1, 0 or 1 and
///     `side` is always None.
///
/// Raises
/// ------
/// ValueError
///     If a timestamp list and its value list differ in length, or a timestamp does not
///     parse.
#[pyfunction(name = "triple_barrier_labels")]
#[pyo3(signature = (
    close_timestamps,
    close_prices,
    t_events,
    target_timestamps,
    target_values,
    pt=1.0,
    sl=1.0,
    min_ret=0.0,
    vertical_barrier_times=None
))]
// Python keyword signature.
#[allow(clippy::too_many_arguments)]
fn labeling_triple_barrier_labels(
    close_timestamps: Vec<String>,
    close_prices: Vec<f64>,
    t_events: Vec<String>,
    target_timestamps: Vec<String>,
    target_values: Vec<f64>,
    pt: f64,
    sl: f64,
    min_ret: f64,
    vertical_barrier_times: Option<Vec<(String, String)>>,
) -> PyResult<Vec<BinRow>> {
    let (close, events) = build_labeling_events(LabelingEventArgs {
        close_timestamps,
        close_prices,
        t_events,
        target_timestamps,
        target_values,
        pt,
        sl,
        min_ret,
        vertical_barrier_times,
        side_prediction: None,
    })?;
    Ok(openquant::labeling::triple_barrier_labels(&events, &close)
        .into_iter()
        .map(|row| (format_naive_datetime(&row.timestamp), row.ret, row.trgt, row.label, row.side))
        .collect())
}

/// Meta-labels: whether a primary model's side would have made money on each event.
///
/// AFML 3.6, Snippets 3.6 and 3.7. Builds events as `triple_barrier_events` with the given
/// sides (the barriers are applied to the side-signed return) and labels each event 1 if the
/// side-signed return from `t0` to `t1` is positive and 0 otherwise. Events without a side, or
/// whose `t1` is None (unresolved), are dropped.
///
/// Parameters
/// ----------
/// close_timestamps : list[str]
///     Bar timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is accepted),
///     in increasing order.
/// close_prices : list[float]
///     Close price of each bar.
/// t_events : list[str]
///     Event start timestamps, in the same format.
/// target_timestamps : list[str]
///     Timestamps of the target series.
/// target_values : list[float]
///     Target return at each target timestamp (the unit of the horizontal barriers).
/// side_prediction : list[tuple[str, float]]
///     `(timestamp, side)` pairs from the primary model (`+1` long, `-1` short).
/// pt : float, default 1.0
///     Profit-taking multiple of the target; 0 disables the barrier.
/// sl : float, default 1.0
///     Stop-loss multiple of the target; 0 disables the barrier.
/// min_ret : float, default 0.0
///     Events whose target is not above this (or is NaN) are dropped.
/// vertical_barrier_times : list[tuple[str, str]] | None, default None
///     `(event, barrier)` pairs, e.g. from `add_vertical_barrier`; events without one have no
///     time limit.
///
/// Returns
/// -------
/// list[tuple[str, float, float, int, float | None]]
///     One `(t0, ret, trgt, bin, side)` row per resolved event; `ret` is the side-signed
///     return and `bin` is 0 or 1.
///
/// Raises
/// ------
/// ValueError
///     If a timestamp list and its value list differ in length, or a timestamp does not
///     parse.
#[pyfunction(name = "meta_labels")]
#[pyo3(signature = (
    close_timestamps,
    close_prices,
    t_events,
    target_timestamps,
    target_values,
    side_prediction,
    pt=1.0,
    sl=1.0,
    min_ret=0.0,
    vertical_barrier_times=None
))]
// Python keyword signature.
#[allow(clippy::too_many_arguments)]
fn labeling_meta_labels(
    close_timestamps: Vec<String>,
    close_prices: Vec<f64>,
    t_events: Vec<String>,
    target_timestamps: Vec<String>,
    target_values: Vec<f64>,
    side_prediction: Vec<(String, f64)>,
    pt: f64,
    sl: f64,
    min_ret: f64,
    vertical_barrier_times: Option<Vec<(String, String)>>,
) -> PyResult<Vec<BinRow>> {
    let (close, events) = build_labeling_events(LabelingEventArgs {
        close_timestamps,
        close_prices,
        t_events,
        target_timestamps,
        target_values,
        pt,
        sl,
        min_ret,
        vertical_barrier_times,
        side_prediction: Some(side_prediction),
    })?;
    Ok(openquant::labeling::meta_labels(&events, &close)
        .into_iter()
        .map(|row| (format_naive_datetime(&row.timestamp), row.ret, row.trgt, row.label, row.side))
        .collect())
}

/// mlfinlab-compatible form of `triple_barrier_events` (AFML Snippet 3.6's `getEvents`).
///
/// Same behaviour as `triple_barrier_events`, with the barrier multiples passed as one
/// `pt_sl` pair and `min_ret` required. `num_threads` is ignored; it is kept so mlfinlab call
/// sites port unchanged.
///
/// Parameters
/// ----------
/// close_timestamps : list[str]
///     Bar timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is accepted),
///     in increasing order.
/// close_prices : list[float]
///     Close price of each bar.
/// t_events : list[str]
///     Event start timestamps, in the same format.
/// pt_sl : tuple[float, float]
///     `(profit-taking, stop-loss)` multiples of the target; 0 disables that barrier.
/// target_timestamps : list[str]
///     Timestamps of the target series.
/// target_values : list[float]
///     Target return at each target timestamp.
/// min_ret : float
///     Events whose target is not above this (or is NaN) are dropped.
/// num_threads : int, default 1
///     Ignored.
/// vertical_barrier_times : list[tuple[str, str]] | None, default None
///     `(event, barrier)` pairs, e.g. from `add_vertical_barrier`; events without one have no
///     time limit.
/// side_prediction : list[tuple[str, float]] | None, default None
///     `(timestamp, side)` pairs from a primary model for meta-labeling; when given, events
///     without a side are dropped.
///
/// Returns
/// -------
/// list[tuple[str, str | None, float, float | None, float, float]]
///     One `(t0, t1, trgt, side, pt, sl)` row per kept event, in `t_events` order.
///
/// Raises
/// ------
/// ValueError
///     If a timestamp list and its value list differ in length, or a timestamp does not
///     parse.
#[pyfunction(name = "get_events")]
#[pyo3(signature = (
    close_timestamps,
    close_prices,
    t_events,
    pt_sl,
    target_timestamps,
    target_values,
    min_ret,
    num_threads=1,
    vertical_barrier_times=None,
    side_prediction=None
))]
// Python keyword signature.
#[allow(clippy::too_many_arguments)]
fn labeling_get_events(
    close_timestamps: Vec<String>,
    close_prices: Vec<f64>,
    t_events: Vec<String>,
    pt_sl: (f64, f64),
    target_timestamps: Vec<String>,
    target_values: Vec<f64>,
    min_ret: f64,
    num_threads: usize,
    vertical_barrier_times: Option<Vec<(String, String)>>,
    side_prediction: Option<Vec<(String, f64)>>,
) -> PyResult<Vec<EventRow>> {
    let close =
        pair_timestamps_values(close_timestamps, close_prices, "close_timestamps", "close_prices")?;
    let t_ev = parse_naive_datetimes(t_events)?;
    let target = pair_timestamps_values(
        target_timestamps,
        target_values,
        "target_timestamps",
        "target_values",
    )?;
    let vbars = parse_vertical_barriers(vertical_barrier_times)?;

    let side_storage: Option<Vec<(chrono::NaiveDateTime, f64)>> =
        if let Some(side) = side_prediction {
            let (timestamps, values): (Vec<String>, Vec<f64>) = side.into_iter().unzip();
            Some(pair_timestamps_values(timestamps, values, "side timestamps", "side values")?)
        } else {
            None
        };

    let events = openquant::labeling::get_events(
        &close,
        &t_ev,
        pt_sl,
        &target,
        min_ret,
        num_threads,
        vbars.as_deref(),
        side_storage.as_deref(),
    );
    Ok(events
        .into_iter()
        .map(|(ts, ev)| {
            (
                format_naive_datetime(&ts),
                ev.t1.map(|v| format_naive_datetime(&v)),
                ev.trgt,
                ev.side,
                ev.pt,
                ev.sl,
            )
        })
        .collect())
}

/// mlfinlab-compatible labelling of resolved events (AFML Snippet 3.7's `getBins`).
///
/// Labels each event by the return from `t0` to `t1`, multiplied by the side when there is
/// one. Without a side the label is the sign of the return (-1, 0 or 1); with a side it is 1
/// if the side-signed return is positive and 0 otherwise (meta-label). Events whose `t1` is
/// None, or whose `t0` or `t1` is not a bar, are skipped.
///
/// Parameters
/// ----------
/// events : list[tuple[str, str | None, float, float | None, float, float]]
///     `(t0, t1, trgt, side, pt, sl)` rows, as returned by `get_events` or
///     `triple_barrier_events`.
/// close_timestamps : list[str]
///     Bar timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is accepted).
/// close_prices : list[float]
///     Close price of each bar.
///
/// Returns
/// -------
/// list[tuple[str, float, float, int, float | None]]
///     One `(t0, ret, trgt, bin, side)` row per labelled event; `ret` is side-signed.
///
/// Raises
/// ------
/// ValueError
///     If the timestamps and prices differ in length, or a timestamp (in `events` or
///     `close_timestamps`) does not parse.
#[pyfunction(name = "get_bins")]
fn labeling_get_bins(
    events: Vec<EventRow>,
    close_timestamps: Vec<String>,
    close_prices: Vec<f64>,
) -> PyResult<Vec<BinRow>> {
    let close =
        pair_timestamps_values(close_timestamps, close_prices, "close_timestamps", "close_prices")?;

    let parsed_events: Vec<(chrono::NaiveDateTime, openquant::labeling::Event)> = events
        .into_iter()
        .map(|(ts_str, t1_str, trgt, side, pt, sl)| {
            let ts = parse_naive_datetime(&ts_str, "datetime")?;
            let t1 = t1_str.map(|s| parse_naive_datetime(&s, "datetime")).transpose()?;
            Ok((ts, openquant::labeling::Event { t1, trgt, side, pt, sl }))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let bins = openquant::labeling::get_bins(&parsed_events, &close);
    Ok(bins
        .into_iter()
        .map(|(ts, ret, trgt, label, side)| (format_naive_datetime(&ts), ret, trgt, label, side))
        .collect())
}

/// Drop under-represented labels.
///
/// AFML Snippet 3.8. Repeatedly removes every row of the rarest label while its share of the
/// rows is at most `min_pct` and at least three distinct labels remain. Rows whose timestamp
/// does not parse are dropped silently rather than raising.
///
/// Parameters
/// ----------
/// events : list[tuple[str, float, float, int, float | None]]
///     `(t0, ret, trgt, bin, side)` rows, as returned by `get_bins`.
/// min_pct : float
///     Minimum share of the rows a label must have to be kept, e.g. 0.05.
///
/// Returns
/// -------
/// list[tuple[str, float, float, int, float | None]]
///     The remaining rows, in input order.
#[pyfunction(name = "drop_labels")]
fn labeling_drop_labels(events: Vec<BinRow>, min_pct: f64) -> Vec<BinRow> {
    let parsed: Vec<(chrono::NaiveDateTime, f64, f64, i8, Option<f64>)> = events
        .into_iter()
        .filter_map(|(ts_str, ret, trgt, label, side)| {
            let ts = parse_datetime_str(&ts_str).ok()?;
            Some((ts, ret, trgt, label, side))
        })
        .collect();

    let result = openquant::labeling::drop_labels(&parsed, min_pct);
    result
        .into_iter()
        .map(|(ts, ret, trgt, label, side)| (format_naive_datetime(&ts), ret, trgt, label, side))
        .collect()
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "labeling")?;
    m.add_function(wrap_pyfunction!(labeling_add_vertical_barrier, &m)?)?;
    m.add_function(wrap_pyfunction!(labeling_triple_barrier_events, &m)?)?;
    m.add_function(wrap_pyfunction!(labeling_triple_barrier_labels, &m)?)?;
    m.add_function(wrap_pyfunction!(labeling_meta_labels, &m)?)?;
    m.add_function(wrap_pyfunction!(labeling_get_events, &m)?)?;
    m.add_function(wrap_pyfunction!(labeling_get_bins, &m)?)?;
    m.add_function(wrap_pyfunction!(labeling_drop_labels, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("labeling", m)?;
    Ok(())
}
