use pyo3::prelude::*;

use crate::helpers::{
    build_labeling_events, pair_timestamps_values,
    parse_naive_datetimes, parse_vertical_barriers,
};

#[pyfunction(name = "add_vertical_barrier")]
fn labeling_add_vertical_barrier(
    t_events: Vec<String>,
    close_timestamps: Vec<String>,
    close_prices: Vec<f64>,
    num_days: i64,
    num_hours: i64,
    num_minutes: i64,
    num_seconds: i64,
) -> PyResult<Vec<(String, String)>> {
    let t_events = parse_naive_datetimes(t_events)?;
    let close = pair_timestamps_values(
        close_timestamps,
        close_prices,
        "close_timestamps",
        "close_prices",
    )?;
    let barriers =
        openquant::labeling::add_vertical_barrier(&t_events, &close, num_days, num_hours, num_minutes, num_seconds);
    Ok(barriers
        .into_iter()
        .map(|(a, b)| {
            (
                a.format("%Y-%m-%d %H:%M:%S").to_string(),
                b.format("%Y-%m-%d %H:%M:%S").to_string(),
            )
        })
        .collect())
}

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
) -> PyResult<Vec<(String, Option<String>, f64, Option<f64>, f64, f64)>> {
    let (_, events) = build_labeling_events(
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
    )?;
    Ok(events
        .into_iter()
        .map(|(ts, ev)| {
            (
                ts.format("%Y-%m-%d %H:%M:%S").to_string(),
                ev.t1.map(|v| v.format("%Y-%m-%d %H:%M:%S").to_string()),
                ev.trgt,
                ev.side,
                ev.pt,
                ev.sl,
            )
        })
        .collect())
}

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
) -> PyResult<Vec<(String, f64, f64, i8, Option<f64>)>> {
    let (close, events) = build_labeling_events(
        close_timestamps,
        close_prices,
        t_events,
        target_timestamps,
        target_values,
        pt,
        sl,
        min_ret,
        vertical_barrier_times,
        None,
    )?;
    Ok(openquant::labeling::triple_barrier_labels(&events, &close)
        .into_iter()
        .map(|row| {
            (
                row.timestamp.format("%Y-%m-%d %H:%M:%S").to_string(),
                row.ret,
                row.trgt,
                row.label,
                row.side,
            )
        })
        .collect())
}

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
) -> PyResult<Vec<(String, f64, f64, i8, Option<f64>)>> {
    let (close, events) = build_labeling_events(
        close_timestamps,
        close_prices,
        t_events,
        target_timestamps,
        target_values,
        pt,
        sl,
        min_ret,
        vertical_barrier_times,
        Some(side_prediction),
    )?;
    Ok(openquant::labeling::meta_labels(&events, &close)
        .into_iter()
        .map(|row| {
            (
                row.timestamp.format("%Y-%m-%d %H:%M:%S").to_string(),
                row.ret,
                row.trgt,
                row.label,
                row.side,
            )
        })
        .collect())
}

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
) -> PyResult<Vec<(String, Option<String>, f64, Option<f64>, f64, f64)>> {
    let close = pair_timestamps_values(
        close_timestamps,
        close_prices,
        "close_timestamps",
        "close_prices",
    )?;
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
                ts.format("%Y-%m-%d %H:%M:%S").to_string(),
                ev.t1.map(|v| v.format("%Y-%m-%d %H:%M:%S").to_string()),
                ev.trgt,
                ev.side,
                ev.pt,
                ev.sl,
            )
        })
        .collect())
}

#[pyfunction(name = "get_bins")]
fn labeling_get_bins(
    events: Vec<(String, Option<String>, f64, Option<f64>, f64, f64)>,
    close_timestamps: Vec<String>,
    close_prices: Vec<f64>,
) -> PyResult<Vec<(String, f64, f64, i8, Option<f64>)>> {
    let close = pair_timestamps_values(
        close_timestamps,
        close_prices,
        "close_timestamps",
        "close_prices",
    )?;

    let parsed_events: Vec<(chrono::NaiveDateTime, openquant::labeling::Event)> = events
        .into_iter()
        .map(|(ts_str, t1_str, trgt, side, pt, sl)| {
            let ts = chrono::NaiveDateTime::parse_from_str(&ts_str, "%Y-%m-%d %H:%M:%S")
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid datetime: {e}")))?;
            let t1 = t1_str
                .map(|s| chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S")
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid datetime: {e}"))))
                .transpose()?;
            Ok((ts, openquant::labeling::Event { t1, trgt, side, pt, sl }))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let bins = openquant::labeling::get_bins(&parsed_events, &close);
    Ok(bins
        .into_iter()
        .map(|(ts, ret, trgt, label, side)| {
            (ts.format("%Y-%m-%d %H:%M:%S").to_string(), ret, trgt, label, side)
        })
        .collect())
}

#[pyfunction(name = "drop_labels")]
fn labeling_drop_labels(
    events: Vec<(String, f64, f64, i8, Option<f64>)>,
    min_pct: f64,
) -> Vec<(String, f64, f64, i8, Option<f64>)> {
    let parsed: Vec<(chrono::NaiveDateTime, f64, f64, i8, Option<f64>)> = events
        .into_iter()
        .filter_map(|(ts_str, ret, trgt, label, side)| {
            let ts = chrono::NaiveDateTime::parse_from_str(&ts_str, "%Y-%m-%d %H:%M:%S").ok()?;
            Some((ts, ret, trgt, label, side))
        })
        .collect();

    let result = openquant::labeling::drop_labels(&parsed, min_pct);
    result
        .into_iter()
        .map(|(ts, ret, trgt, label, side)| {
            (ts.format("%Y-%m-%d %H:%M:%S").to_string(), ret, trgt, label, side)
        })
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
