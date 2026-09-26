use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::helpers::to_py_err;

/// Python-facing stream event: `(timestamp_ns, price, buy_volume, sell_volume, venue_id)`.
type StreamEventRow = (i64, f64, f64, f64, usize);

/// Python-facing snapshot:
/// `(timestamp_ns, price, vpin, hhi, normalized_risk_score, is_alert, vpin_cdf)`.
/// `vpin_cdf` is last so that the first six positions keep their meaning.
type SnapshotRow = (i64, f64, Option<f64>, Option<f64>, Option<f64>, bool, Option<f64>);

/// Run the VPIN/HHI early-warning engine over a stream of events, one snapshot per event.
///
/// AFML Chapter 22 (§22.6.4-22.6.5). VPIN runs on a volume clock: events fill buckets of
/// `bucket_volume` (an event that overflows a bucket is split, keeping its buy/sell ratio),
/// and VPIN is the mean of `|V_buy - V_sell| / bucket_volume` over the last `support_buckets`
/// full buckets. `vpin_cdf_threshold` applies to the empirical CDF of VPIN over the last
/// `cdf_lookback` VPIN values (one per completed bucket, ties counted half), not to raw VPIN;
/// e.g. 0.99 with `cdf_lookback >= 50`. HHI is `sum_v (Q_v / sum_j Q_j)^2`, weighting venues
/// by their share of volume over the last `lookback_events` events. An event alerts when
/// `vpin_cdf >= vpin_cdf_threshold` and `hhi >= hhi_threshold`. Events are processed in input
/// order; timestamps are carried through but neither validated nor used.
///
/// Parameters
/// ----------
/// events : list[tuple[int, float, float, float, int]]
///     `(timestamp_ns, price, buy_volume, sell_volume, venue_id)` rows, e.g. from
///     `generate_synthetic_flash_crash_stream`. Buy and sell volumes are already classified
///     by the caller; `venue_id` is a non-negative int.
/// bucket_volume : float
///     Volume per VPIN bucket, in the units of the event volumes; finite and > 0.
/// support_buckets : int
///     Number of completed buckets in the VPIN mean; must be > 0.
/// lookback_events : int
///     Number of events in the HHI window; must be > 0.
/// vpin_cdf_threshold : float
///     Alert threshold on the CDF of VPIN, in `(0, 1)` and at most `1 - 0.5 / cdf_lookback`
///     (the largest value the CDF can take).
/// hhi_threshold : float
///     Alert threshold on HHI (1 means one venue carries all the volume), in `(0, 1]`.
/// cdf_lookback : int
///     Number of past VPIN values the CDF is taken over; must be >= 2.
///
/// Returns
/// -------
/// dict[str, Any]
///     Keys:
///     - `snapshots`: list of `(timestamp_ns, price, vpin, hhi, normalized_risk_score,
///       is_alert, vpin_cdf)` tuples, one per event in input order. `vpin`, `hhi`, `vpin_cdf`
///       and `normalized_risk_score` (`min(vpin_cdf / vpin_cdf_threshold,
///       hhi / hhi_threshold)`) are `None` until their windows fill, and `is_alert` is
///       `False` then. `vpin_cdf` is last so that the first six positions keep their meaning.
///     - `metrics`: dict with `processed_events` (int), `events_per_sec`,
///       `avg_event_latency_micros`, `max_event_latency_micros` and `runtime_secs` (floats;
///       wall-clock timings that vary between runs).
///     - `alert_count`: number of alerting snapshots (int).
///
/// Raises
/// ------
/// ValueError
///     If a configuration value is out of range (see above), or an event has a price that is
///     not finite and > 0, a negative or non-finite volume, or zero total volume. The run
///     stops at the first invalid event and returns no partial result.
#[pyfunction(name = "run_streaming_pipeline")]
#[allow(clippy::too_many_arguments)]
fn shpc_run_streaming_pipeline(
    py: Python<'_>,
    events: Vec<StreamEventRow>,
    bucket_volume: f64,
    support_buckets: usize,
    lookback_events: usize,
    vpin_cdf_threshold: f64,
    hhi_threshold: f64,
    cdf_lookback: usize,
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
        vpin: openquant::streaming_hpc::VpinConfig { bucket_volume, support_buckets, cdf_lookback },
        hhi: openquant::streaming_hpc::HhiConfig { lookback_events },
        thresholds: openquant::streaming_hpc::AlertThresholds {
            vpin_cdf: vpin_cdf_threshold,
            hhi: hhi_threshold,
        },
    };

    let report =
        openquant::streaming_hpc::run_streaming_pipeline(&stream_events, cfg).map_err(to_py_err)?;

    let d = PyDict::new(py);

    let snapshots: Vec<SnapshotRow> = report
        .snapshots
        .into_iter()
        .map(|s| {
            (
                s.timestamp_ns,
                s.price,
                s.vpin,
                s.hhi,
                s.normalized_risk_score,
                s.is_alert,
                s.vpin_cdf,
            )
        })
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

/// Generate a deterministic two-regime event stream loosely modelled on the 2010 Flash Crash.
///
/// AFML §22.6.4. Event `i` has `timestamp_ns = i * 1_000_000` (1 ms apart). Before the crash
/// (which starts at event `round(events * crash_start_fraction)`), event `i` is on venue
/// `i % calm_venues` with 120 bought and 130 sold, and the price grows by 0.01% per event;
/// from the crash on, every event is on `shock_venue` with 80 bought and 320 sold, and the
/// price falls by 0.25% per event. The price starts from 100 and moves before the first event
/// is emitted. A small stream can round to no calm events or no crash events.
///
/// Parameters
/// ----------
/// events : int, default 1000
///     Number of events; must be > 0.
/// crash_start_fraction : float, default 0.7
///     Fraction of the stream before the crash, strictly between 0 and 1.
/// calm_venues : int, default 3
///     Number of venues the calm flow rotates over (venues `0 .. calm_venues - 1`); > 0.
/// shock_venue : int, default 0
///     Venue that carries all the flow from the crash on; may be one of the calm venues.
///
/// Returns
/// -------
/// list[tuple[int, float, float, float, int]]
///     `(timestamp_ns, price, buy_volume, sell_volume, venue_id)` rows, the input format of
///     `run_streaming_pipeline`.
///
/// Raises
/// ------
/// ValueError
///     If `events` is 0, `crash_start_fraction` is not finite and strictly between 0 and 1,
///     or `calm_venues` is 0.
#[pyfunction(name = "generate_synthetic_flash_crash_stream")]
#[pyo3(signature = (events=1000, crash_start_fraction=0.7, calm_venues=3, shock_venue=0))]
fn shpc_generate_synthetic_flash_crash_stream(
    events: usize,
    crash_start_fraction: f64,
    calm_venues: usize,
    shock_venue: usize,
) -> PyResult<Vec<StreamEventRow>> {
    let cfg = openquant::streaming_hpc::SyntheticStreamConfig {
        events,
        crash_start_fraction,
        calm_venues,
        shock_venue,
    };
    let stream =
        openquant::streaming_hpc::generate_synthetic_flash_crash_stream(cfg).map_err(to_py_err)?;
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
