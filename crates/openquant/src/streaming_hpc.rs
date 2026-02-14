//! AFML Chapter 22: streaming analytics utilities for low-latency early warning.
//!
//! This module emphasizes bounded-memory, incremental updates suitable for
//! near-real-time decision workflows. It includes:
//! - VPIN-like flow-toxicity tracking with rolling volume buckets,
//! - HHI-style market fragmentation concentration tracking over rolling windows,
//! - an event-by-event early-warning pipeline, and
//! - serial/parallel execution helpers for multi-stream workloads.

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HhiConfig {
    /// Number of events in rolling concentration window.
    pub lookback_events: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlertThresholds {
    pub vpin: f64,
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
    pub hhi: Option<f64>,
    /// Simple normalized alert score for operations dashboards.
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
}

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
        Ok(Self {
            cfg,
            window: VecDeque::with_capacity(cfg.support_buckets),
            window_sum: 0.0,
            current_bucket_abs_imbalance: 0.0,
            current_bucket_volume: 0.0,
        })
    }

    pub fn update(
        &mut self,
        mut buy_volume: f64,
        mut sell_volume: f64,
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
            sell_volume -= used_sell;
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

    pub fn completed_buckets(&self) -> usize {
        self.window.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HhiState {
    cfg: HhiConfig,
    window: VecDeque<usize>,
    venue_counts: HashMap<usize, usize>,
    sum_sq_counts: usize,
}

impl HhiState {
    pub fn new(cfg: HhiConfig) -> Result<Self, StreamingHpcError> {
        if cfg.lookback_events == 0 {
            return Err(StreamingHpcError::InvalidConfig("hhi.lookback_events must be > 0"));
        }
        Ok(Self {
            cfg,
            window: VecDeque::with_capacity(cfg.lookback_events),
            venue_counts: HashMap::new(),
            sum_sq_counts: 0,
        })
    }

    pub fn update(&mut self, venue_id: usize) -> Option<f64> {
        self.window.push_back(venue_id);
        let count_before = *self.venue_counts.get(&venue_id).unwrap_or(&0);
        self.sum_sq_counts += 2 * count_before + 1;
        self.venue_counts.insert(venue_id, count_before + 1);

        if self.window.len() > self.cfg.lookback_events {
            if let Some(expired) = self.window.pop_front() {
                let old_count = *self.venue_counts.get(&expired).unwrap_or(&0);
                if old_count > 0 {
                    self.sum_sq_counts = self.sum_sq_counts.saturating_sub(2 * old_count - 1);
                    if old_count == 1 {
                        self.venue_counts.remove(&expired);
                    } else {
                        self.venue_counts.insert(expired, old_count - 1);
                    }
                }
            }
        }
        self.current()
    }

    pub fn current(&self) -> Option<f64> {
        let n = self.window.len();
        if n < self.cfg.lookback_events || n == 0 {
            return None;
        }
        let denom = (n * n) as f64;
        Some(self.sum_sq_counts as f64 / denom)
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
        if !cfg.thresholds.vpin.is_finite() || cfg.thresholds.vpin <= 0.0 {
            return Err(StreamingHpcError::InvalidConfig("thresholds.vpin must be finite and > 0"));
        }
        if !cfg.thresholds.hhi.is_finite() || cfg.thresholds.hhi <= 0.0 {
            return Err(StreamingHpcError::InvalidConfig("thresholds.hhi must be finite and > 0"));
        }
        Ok(Self { vpin_state: VpinState::new(cfg.vpin)?, hhi_state: HhiState::new(cfg.hhi)?, cfg })
    }

    pub fn on_event(
        &mut self,
        event: StreamEvent,
    ) -> Result<EarlyWarningSnapshot, StreamingHpcError> {
        validate_event(event)?;
        let vpin = self.vpin_state.update(event.buy_volume, event.sell_volume)?;
        let hhi = self.hhi_state.update(event.venue_id);
        let normalized_risk_score = match (vpin, hhi) {
            (Some(v), Some(h)) => {
                Some(0.5 * (v / self.cfg.thresholds.vpin + h / self.cfg.thresholds.hhi))
            }
            _ => None,
        };
        let is_alert = match (vpin, hhi) {
            (Some(v), Some(h)) => v >= self.cfg.thresholds.vpin && h >= self.cfg.thresholds.hhi,
            _ => false,
        };
        Ok(EarlyWarningSnapshot {
            timestamp_ns: event.timestamp_ns,
            price: event.price,
            vpin,
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
