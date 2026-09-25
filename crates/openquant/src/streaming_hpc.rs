//! AFML Chapter 22: streaming analytics utilities for low-latency early warning.
//!
//! This module emphasizes bounded-memory, incremental updates suitable for
//! near-real-time decision workflows. It includes:
//! - VPIN-like flow-toxicity tracking with rolling volume buckets, and the empirical CDF of
//!   VPIN over a rolling history of past VPIN values,
//! - HHI-style market fragmentation concentration (venue volume shares) over rolling windows,
//! - an event-by-event early-warning pipeline, and
//! - serial/parallel execution helpers for multi-stream workloads.
//!
//! The VPIN alert threshold applies to the CDF of VPIN, not to raw VPIN (AFML §22.6.5; Easley,
//! López de Prado and O'Hara, 2011): alerting at `CDF(VPIN) >= 0.99` means "VPIN is in the top 1%
//! of its own recent history", which adapts to each instrument's baseline level of toxicity.

use crate::hpc_parallel::{run_parallel, HpcParallelConfig, HpcParallelError, ParallelRunReport};
use std::collections::{HashMap, VecDeque};
use std::fmt::{Display, Formatter};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq)]
pub enum StreamingHpcError {
    InvalidConfig(&'static str),
    InvalidEvent(&'static str),
    Parallel(HpcParallelError),
}

impl Display for StreamingHpcError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid streaming HPC config: {msg}"),
            Self::InvalidEvent(msg) => write!(f, "invalid streaming event: {msg}"),
            Self::Parallel(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for StreamingHpcError {}

impl From<HpcParallelError> for StreamingHpcError {
    fn from(value: HpcParallelError) -> Self {
        Self::Parallel(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamEvent {
    pub timestamp_ns: i64,
    pub price: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub venue_id: usize,
}

impl StreamEvent {
    pub fn total_volume(self) -> f64 {
        self.buy_volume + self.sell_volume
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VpinConfig {
    /// Volume in each bucket.
    pub bucket_volume: f64,
    /// Number of completed buckets in rolling VPIN window.
    pub support_buckets: usize,
    /// Number of past VPIN values in the rolling history that VPIN's empirical CDF is taken over.
    ///
    /// One VPIN value is recorded each time a bucket completes (once `support_buckets` buckets
    /// exist), so the history spans the last `cdf_lookback` buckets, i.e.
    /// `cdf_lookback * bucket_volume` units of volume. The CDF is `None` until the history is
    /// full. It includes the current value and ties count half (see [`VpinState::current_cdf`]),
    /// so the largest CDF a value can reach is `1 - 0.5 / cdf_lookback`. Must be at least 2.
    pub cdf_lookback: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HhiConfig {
    /// Number of events in rolling concentration window. Venues are weighted by their share of
    /// the volume traded in these events, not by their number of events.
    pub lookback_events: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlertThresholds {
    /// Threshold on the empirical CDF of VPIN, in (0, 1): 0.99 alerts when VPIN is in the top 1%
    /// of its rolling history. It is a probability, not a raw VPIN level, and must be at most
    /// `1 - 0.5 / vpin.cdf_lookback`, the largest value the CDF can take.
    pub vpin_cdf: f64,
    /// Threshold on the volume-share HHI (1 = one venue carries all the volume).
    pub hhi: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamingPipelineConfig {
    pub vpin: VpinConfig,
    pub hhi: HhiConfig,
    pub thresholds: AlertThresholds,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarlyWarningSnapshot {
    pub timestamp_ns: i64,
    pub price: f64,
    pub vpin: Option<f64>,
    /// Empirical CDF of the current VPIN over its rolling history ([`VpinConfig::cdf_lookback`]).
    pub vpin_cdf: Option<f64>,
    pub hhi: Option<f64>,
    /// `min(vpin_cdf / thresholds.vpin_cdf, hhi / thresholds.hhi)`: the alert condition as one
    /// number for dashboards. It is at least 1 exactly when `is_alert` is true (up to rounding).
    pub normalized_risk_score: Option<f64>,
    pub is_alert: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamingRunMetrics {
    pub processed_events: usize,
    pub events_per_sec: f64,
    pub avg_event_latency_micros: f64,
    pub max_event_latency_micros: f64,
    pub runtime: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamingRunReport {
    pub snapshots: Vec<EarlyWarningSnapshot>,
    pub metrics: StreamingRunMetrics,
    pub alert_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamSummary {
    pub processed_events: usize,
    pub alert_count: usize,
    pub latest_vpin: Option<f64>,
    pub latest_vpin_cdf: Option<f64>,
    pub latest_hhi: Option<f64>,
    pub latest_risk_score: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParallelStreamingReport {
    pub stream_summaries: Vec<StreamSummary>,
    pub parallel_metrics: crate::hpc_parallel::HpcParallelMetrics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VpinState {
    cfg: VpinConfig,
    window: VecDeque<f64>,
    window_sum: f64,
    current_bucket_abs_imbalance: f64,
    current_bucket_volume: f64,
    /// The last `cdf_lookback` VPIN values in arrival order, and the same values sorted.
    history: VecDeque<f64>,
    history_sorted: Vec<f64>,
}

/// VPIN values closer than this are ties for [`VpinState::current_cdf`], so rounding noise in the
/// rolling sum cannot rank one of two equal VPIN values above the other.
const VPIN_CDF_TIE_TOL: f64 = 1e-9;

impl VpinState {
    pub fn new(cfg: VpinConfig) -> Result<Self, StreamingHpcError> {
        if !cfg.bucket_volume.is_finite() || cfg.bucket_volume <= 0.0 {
            return Err(StreamingHpcError::InvalidConfig(
                "vpin.bucket_volume must be finite and > 0",
            ));
        }
        if cfg.support_buckets == 0 {
            return Err(StreamingHpcError::InvalidConfig("vpin.support_buckets must be > 0"));
        }
        if cfg.cdf_lookback < 2 {
            return Err(StreamingHpcError::InvalidConfig("vpin.cdf_lookback must be >= 2"));
        }
        Ok(Self {
            cfg,
            window: VecDeque::with_capacity(cfg.support_buckets),
            window_sum: 0.0,
            current_bucket_abs_imbalance: 0.0,
            current_bucket_volume: 0.0,
            history: VecDeque::with_capacity(cfg.cdf_lookback + 1),
            history_sorted: Vec::with_capacity(cfg.cdf_lookback + 1),
        })
    }

    pub fn update(
        &mut self,
        mut buy_volume: f64,
        sell_volume: f64,
    ) -> Result<Option<f64>, StreamingHpcError> {
        validate_non_negative_finite("buy_volume", buy_volume)?;
        validate_non_negative_finite("sell_volume", sell_volume)?;
        let mut remaining = buy_volume + sell_volume;
        if remaining == 0.0 {
            return Ok(self.current());
        }
        while remaining > 0.0 {
            let capacity = self.cfg.bucket_volume - self.current_bucket_volume;
            let take = remaining.min(capacity);
            if take <= 0.0 {
                break;
            }
            // Preserve buy/sell ratio within partial fill.
            let ratio_buy = if remaining > 0.0 { buy_volume / remaining } else { 0.5 };
            let used_buy = take * ratio_buy;
            let used_sell = take - used_buy;

            self.current_bucket_volume += take;
            self.current_bucket_abs_imbalance += (used_buy - used_sell).abs();

            buy_volume -= used_buy;
            remaining -= take;

            if self.current_bucket_volume >= self.cfg.bucket_volume - 1e-12 {
                let toxicity = self.current_bucket_abs_imbalance / self.cfg.bucket_volume;
                self.window.push_back(toxicity);
                self.window_sum += toxicity;
                if self.window.len() > self.cfg.support_buckets {
                    if let Some(expired) = self.window.pop_front() {
                        self.window_sum -= expired;
                    }
                }
                self.current_bucket_volume = 0.0;
                self.current_bucket_abs_imbalance = 0.0;
                if let Some(vpin) = self.current() {
                    self.record_vpin(vpin);
                }
            }
        }
        Ok(self.current())
    }

    pub fn current(&self) -> Option<f64> {
        if self.window.len() < self.cfg.support_buckets {
            None
        } else {
            Some(self.window_sum / self.window.len() as f64)
        }
    }

    /// Empirical CDF of the current VPIN over the last `cdf_lookback` VPIN values (the current
    /// one included), or `None` until that many have been recorded.
    ///
    /// Ties count half (the mid-distribution function): with `n` values in the history, `b` of
    /// them below the current VPIN and `t` equal to it (itself included), the CDF is
    /// `(b + t / 2) / n`. A history of identical values gives 0.5 rather than 1, so a perfectly
    /// steady stream does not look like its own extreme; a new maximum gives `1 - 0.5 / n`.
    pub fn current_cdf(&self) -> Option<f64> {
        let n = self.history_sorted.len();
        if n < self.cfg.cdf_lookback {
            return None;
        }
        let v = *self.history.back()?;
        let below = self.history_sorted.partition_point(|&x| x < v - VPIN_CDF_TIE_TOL);
        let at_or_below = self.history_sorted.partition_point(|&x| x <= v + VPIN_CDF_TIE_TOL);
        let ties = at_or_below - below;
        Some((below as f64 + 0.5 * ties as f64) / n as f64)
    }

    pub fn completed_buckets(&self) -> usize {
        self.window.len()
    }

    fn record_vpin(&mut self, vpin: f64) {
        self.history.push_back(vpin);
        let at = self.history_sorted.partition_point(|&x| x < vpin);
        self.history_sorted.insert(at, vpin);
        if self.history.len() > self.cfg.cdf_lookback {
            if let Some(expired) = self.history.pop_front() {
                // Values are stored bit for bit, so the expired one is found exactly.
                let at = self.history_sorted.partition_point(|&x| x < expired);
                self.history_sorted.remove(at);
            }
        }
    }
}

/// Herfindahl–Hirschman index of venue concentration over the last `lookback_events` events,
/// each venue weighted by its share of the volume traded in those events:
/// `HHI = sum_v (volume_v / total_volume)^2`.
#[derive(Debug, Clone, PartialEq)]
pub struct HhiState {
    cfg: HhiConfig,
    /// `(venue_id, volume)` of each event in the window.
    window: VecDeque<(usize, f64)>,
    /// Per venue in the window: `(events, volume)`. A venue is dropped when its last event leaves
    /// the window, so rounding in its running volume cannot accumulate.
    venues: HashMap<usize, (usize, f64)>,
}

impl HhiState {
    pub fn new(cfg: HhiConfig) -> Result<Self, StreamingHpcError> {
        if cfg.lookback_events == 0 {
            return Err(StreamingHpcError::InvalidConfig("hhi.lookback_events must be > 0"));
        }
        Ok(Self {
            cfg,
            window: VecDeque::with_capacity(cfg.lookback_events + 1),
            venues: HashMap::new(),
        })
    }

    /// Adds one event that traded `volume` (finite, > 0) on `venue_id`.
    pub fn update(
        &mut self,
        venue_id: usize,
        volume: f64,
    ) -> Result<Option<f64>, StreamingHpcError> {
        if !volume.is_finite() || volume <= 0.0 {
            return Err(StreamingHpcError::InvalidEvent("volume must be finite and > 0"));
        }
        self.window.push_back((venue_id, volume));
        let entry = self.venues.entry(venue_id).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += volume;

        if self.window.len() > self.cfg.lookback_events {
            if let Some((expired, expired_volume)) = self.window.pop_front() {
                if let Some(entry) = self.venues.get_mut(&expired) {
                    if entry.0 <= 1 {
                        self.venues.remove(&expired);
                    } else {
                        entry.0 -= 1;
                        entry.1 -= expired_volume;
                    }
                }
            }
        }
        Ok(self.current())
    }

    pub fn current(&self) -> Option<f64> {
        let n = self.window.len();
        if n < self.cfg.lookback_events || n == 0 {
            return None;
        }
        let total: f64 = self.venues.values().map(|&(_, v)| v).sum();
        if total <= 0.0 {
            return None;
        }
        Some(self.venues.values().map(|&(_, v)| (v / total) * (v / total)).sum())
    }

    pub fn window_len(&self) -> usize {
        self.window.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamingEarlyWarningEngine {
    cfg: StreamingPipelineConfig,
    vpin_state: VpinState,
    hhi_state: HhiState,
}

impl StreamingEarlyWarningEngine {
    pub fn new(cfg: StreamingPipelineConfig) -> Result<Self, StreamingHpcError> {
        let vpin_cdf = cfg.thresholds.vpin_cdf;
        if !vpin_cdf.is_finite() || vpin_cdf <= 0.0 || vpin_cdf >= 1.0 {
            return Err(StreamingHpcError::InvalidConfig("thresholds.vpin_cdf must be in (0, 1)"));
        }
        if !cfg.thresholds.hhi.is_finite() || cfg.thresholds.hhi <= 0.0 {
            return Err(StreamingHpcError::InvalidConfig("thresholds.hhi must be finite and > 0"));
        }
        let vpin_state = VpinState::new(cfg.vpin)?;
        // The largest CDF value is 1 - 0.5 / cdf_lookback; allow for rounding in that expression.
        if vpin_cdf > 1.0 - 0.5 / cfg.vpin.cdf_lookback as f64 + 1e-12 {
            return Err(StreamingHpcError::InvalidConfig(
                "thresholds.vpin_cdf is unreachable: it must be <= 1 - 0.5 / vpin.cdf_lookback",
            ));
        }
        Ok(Self { vpin_state, hhi_state: HhiState::new(cfg.hhi)?, cfg })
    }

    pub fn on_event(
        &mut self,
        event: StreamEvent,
    ) -> Result<EarlyWarningSnapshot, StreamingHpcError> {
        validate_event(event)?;
        let vpin = self.vpin_state.update(event.buy_volume, event.sell_volume)?;
        let vpin_cdf = self.vpin_state.current_cdf();
        let hhi = self.hhi_state.update(event.venue_id, event.total_volume())?;
        let t = self.cfg.thresholds;
        let normalized_risk_score = match (vpin_cdf, hhi) {
            (Some(c), Some(h)) => Some((c / t.vpin_cdf).min(h / t.hhi)),
            _ => None,
        };
        let is_alert = match (vpin_cdf, hhi) {
            (Some(c), Some(h)) => c >= t.vpin_cdf && h >= t.hhi,
            _ => false,
        };
        Ok(EarlyWarningSnapshot {
            timestamp_ns: event.timestamp_ns,
            price: event.price,
            vpin,
            vpin_cdf,
            hhi,
            normalized_risk_score,
            is_alert,
        })
    }
}

pub fn run_streaming_pipeline(
    events: &[StreamEvent],
    cfg: StreamingPipelineConfig,
) -> Result<StreamingRunReport, StreamingHpcError> {
    let mut engine = StreamingEarlyWarningEngine::new(cfg)?;
    let mut snapshots = Vec::with_capacity(events.len());
    let mut alert_count = 0usize;
    let mut total_event_latency = Duration::ZERO;
    let mut max_event_latency = Duration::ZERO;
    let started = Instant::now();

    for event in events {
        let t0 = Instant::now();
        let snapshot = engine.on_event(*event)?;
        let elapsed = t0.elapsed();
        total_event_latency += elapsed;
        if elapsed > max_event_latency {
            max_event_latency = elapsed;
        }
        if snapshot.is_alert {
            alert_count += 1;
        }
        snapshots.push(snapshot);
    }

    let runtime = started.elapsed();
    let processed = events.len();
    let runtime_secs = runtime.as_secs_f64();
    let events_per_sec = if runtime_secs > 0.0 { processed as f64 / runtime_secs } else { 0.0 };
    let avg_event_latency_micros = if processed > 0 {
        total_event_latency.as_secs_f64() * 1_000_000.0 / processed as f64
    } else {
        0.0
    };
    let max_event_latency_micros = max_event_latency.as_secs_f64() * 1_000_000.0;

    Ok(StreamingRunReport {
        snapshots,
        metrics: StreamingRunMetrics {
            processed_events: processed,
            events_per_sec,
            avg_event_latency_micros,
            max_event_latency_micros,
            runtime,
        },
        alert_count,
    })
}

pub fn run_streaming_pipeline_parallel(
    streams: &[Vec<StreamEvent>],
    pipeline_cfg: StreamingPipelineConfig,
    parallel_cfg: HpcParallelConfig,
) -> Result<ParallelStreamingReport, StreamingHpcError> {
    let report: ParallelRunReport<Vec<StreamSummary>> =
        run_parallel(streams, parallel_cfg, |chunk| {
            let mut summaries = Vec::with_capacity(chunk.len());
            for stream in chunk {
                let run = run_streaming_pipeline(stream, pipeline_cfg)
                    .map_err(|err| format!("stream pipeline failed: {err}"))?;
                let last = run.snapshots.last();
                summaries.push(StreamSummary {
                    processed_events: run.metrics.processed_events,
                    alert_count: run.alert_count,
                    latest_vpin: last.and_then(|s| s.vpin),
                    latest_vpin_cdf: last.and_then(|s| s.vpin_cdf),
                    latest_hhi: last.and_then(|s| s.hhi),
                    latest_risk_score: last.and_then(|s| s.normalized_risk_score),
                });
            }
            Ok::<Vec<StreamSummary>, String>(summaries)
        })?;

    let mut stream_summaries = Vec::new();
    for batch in report.outputs {
        stream_summaries.extend(batch);
    }
    Ok(ParallelStreamingReport { stream_summaries, parallel_metrics: report.metrics })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyntheticStreamConfig {
    pub events: usize,
    pub crash_start_fraction: f64,
    pub calm_venues: usize,
    pub shock_venue: usize,
}

pub fn generate_synthetic_flash_crash_stream(
    cfg: SyntheticStreamConfig,
) -> Result<Vec<StreamEvent>, StreamingHpcError> {
    if cfg.events == 0 {
        return Err(StreamingHpcError::InvalidConfig("synthetic events must be > 0"));
    }
    if !cfg.crash_start_fraction.is_finite()
        || cfg.crash_start_fraction <= 0.0
        || cfg.crash_start_fraction >= 1.0
    {
        return Err(StreamingHpcError::InvalidConfig(
            "crash_start_fraction must be finite and in (0, 1)",
        ));
    }
    if cfg.calm_venues == 0 {
        return Err(StreamingHpcError::InvalidConfig("calm_venues must be > 0"));
    }

    let crash_start = (cfg.events as f64 * cfg.crash_start_fraction).round() as usize;
    let mut events = Vec::with_capacity(cfg.events);
    let mut price = 100.0;
    for i in 0..cfg.events {
        let in_shock = i >= crash_start;
        let venue = if in_shock { cfg.shock_venue } else { i % cfg.calm_venues };
        let (buy_volume, sell_volume, drift) = if in_shock {
            // Toxic order flow and downside pressure during flash-crash regime.
            (80.0, 320.0, -0.0025)
        } else {
            (120.0, 130.0, 0.0001)
        };
        price *= 1.0 + drift;
        events.push(StreamEvent {
            timestamp_ns: i as i64 * 1_000_000,
            price,
            buy_volume,
            sell_volume,
            venue_id: venue,
        });
    }
    Ok(events)
}

fn validate_non_negative_finite(name: &'static str, value: f64) -> Result<(), StreamingHpcError> {
    if !value.is_finite() || value < 0.0 {
        return Err(StreamingHpcError::InvalidEvent(name));
    }
    Ok(())
}

fn validate_event(event: StreamEvent) -> Result<(), StreamingHpcError> {
    if !event.price.is_finite() || event.price <= 0.0 {
        return Err(StreamingHpcError::InvalidEvent("price must be finite and > 0"));
    }
    validate_non_negative_finite("buy_volume", event.buy_volume)?;
    validate_non_negative_finite("sell_volume", event.sell_volume)?;
    if event.total_volume() <= 0.0 {
        return Err(StreamingHpcError::InvalidEvent(
            "event must have strictly positive total volume",
        ));
    }
    Ok(())
}
