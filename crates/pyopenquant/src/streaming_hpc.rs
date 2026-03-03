use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::helpers::to_py_err;

#[pyfunction(name = "run_streaming_pipeline")]
fn shpc_run_streaming_pipeline(
    py: Python<'_>,
    events: Vec<(i64, f64, f64, f64, usize)>,
    bucket_volume: f64,
    support_buckets: usize,
    lookback_events: usize,
    vpin_threshold: f64,
    hhi_threshold: f64,
) -> PyResult<PyObject> {
    let stream_events: Vec<openquant::streaming_hpc::StreamEvent> = events
        .into_iter()
        .map(|(ts, price, buy_vol, sell_vol, venue)| openquant::streaming_hpc::StreamEvent {
            timestamp_ns: ts,
            price,
            buy_volume: buy_vol,
            sell_volume: sell_vol,
            venue_id: venue,
        })
        .collect();

    let cfg = openquant::streaming_hpc::StreamingPipelineConfig {
        vpin: openquant::streaming_hpc::VpinConfig { bucket_volume, support_buckets },
        hhi: openquant::streaming_hpc::HhiConfig { lookback_events },
        thresholds: openquant::streaming_hpc::AlertThresholds {
            vpin: vpin_threshold,
            hhi: hhi_threshold,
        },
    };

    let report = openquant::streaming_hpc::run_streaming_pipeline(&stream_events, cfg)
        .map_err(to_py_err)?;

    let d = PyDict::new(py);

    let snapshots: Vec<(i64, f64, Option<f64>, Option<f64>, Option<f64>, bool)> = report
        .snapshots
        .into_iter()
        .map(|s| (s.timestamp_ns, s.price, s.vpin, s.hhi, s.normalized_risk_score, s.is_alert))
        .collect();
    d.set_item("snapshots", snapshots)?;

    let metrics = PyDict::new(py);
    metrics.set_item("processed_events", report.metrics.processed_events)?;
    metrics.set_item("events_per_sec", report.metrics.events_per_sec)?;
    metrics.set_item("avg_event_latency_micros", report.metrics.avg_event_latency_micros)?;
    metrics.set_item("max_event_latency_micros", report.metrics.max_event_latency_micros)?;
    metrics.set_item("runtime_secs", report.metrics.runtime.as_secs_f64())?;
    d.set_item("metrics", metrics)?;
    d.set_item("alert_count", report.alert_count)?;

    Ok(d.into_pyobject(py).unwrap().into_any().unbind())
}

#[pyfunction(name = "generate_synthetic_flash_crash_stream")]
#[pyo3(signature = (events=1000, crash_start_fraction=0.7, calm_venues=3, shock_venue=0))]
fn shpc_generate_synthetic_flash_crash_stream(
    events: usize,
    crash_start_fraction: f64,
    calm_venues: usize,
    shock_venue: usize,
) -> PyResult<Vec<(i64, f64, f64, f64, usize)>> {
    let cfg = openquant::streaming_hpc::SyntheticStreamConfig {
        events,
        crash_start_fraction,
        calm_venues,
        shock_venue,
    };
    let stream = openquant::streaming_hpc::generate_synthetic_flash_crash_stream(cfg)
        .map_err(to_py_err)?;
    Ok(stream
        .into_iter()
        .map(|e| (e.timestamp_ns, e.price, e.buy_volume, e.sell_volume, e.venue_id))
        .collect())
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "streaming_hpc")?;
    m.add_function(wrap_pyfunction!(shpc_run_streaming_pipeline, &m)?)?;
    m.add_function(wrap_pyfunction!(shpc_generate_synthetic_flash_crash_stream, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("streaming_hpc", m)?;
    Ok(())
}
