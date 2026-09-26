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
//!
//! References: AFML Chapter 22 (H. Simon and K. Wu), §22.6.4 The Flash Crash of 2010 and §22.6.5
//! VPIN calibration; Easley, López de Prado and O'Hara (2011, 2012) for VPIN; Hirschman (1980)
//! for the HHI. There is no AFML code snippet for this chapter.
//!
//! # Conventions
//!
//! - Events are processed in arrival order; `timestamp_ns` is carried through to the snapshot
//!   but neither validated nor used, so out-of-order events are not detected.
//! - Buy and sell volumes are inputs, already signed by the caller (for example with bulk
//!   volume classification); nothing here classifies trades. Volumes are in any consistent
//!   unit (shares, contracts, notional), the same unit as [`VpinConfig::bucket_volume`].
//! - VPIN is on a volume clock: an event that overflows a bucket is split between it and the
//!   next, keeping its buy/sell ratio. VPIN is the mean of `|V_buy - V_sell| / bucket_volume`
//!   over the last `support_buckets` full buckets, in `[0, 1]`.
//! - The three rolling windows run on different clocks: VPIN over `support_buckets` buckets,
//!   its CDF over `cdf_lookback` buckets, HHI over `lookback_events` events.
//! - HHI is `sum_v (Q_v / sum_j Q_j)^2` over venue volumes `Q_v` in its window, in
//!   `[1/K, 1]` for `K` venues.
//! - The alert is `CDF(VPIN) >= thresholds.vpin_cdf && HHI >= thresholds.hhi`. Snapshots
//!   report `None` for any indicator whose window is not yet full, and never alert then.
//! - The CDF ranks a *jump* in VPIN against its recent past: once a high-VPIN plateau fills
//!   the history, the CDF of a constant VPIN falls back toward 0.5 and the alert stops.
//!
//! # Example
//!
//! One bucket of 100 units, a CDF over two VPIN values and an HHI over one event. A balanced
//! first event gives VPIN 0; an all-buy second event gives VPIN 1, which ranks above the 0
//! in its history: CDF `(1 + 0.5) / 2 = 0.75`, the largest the CDF can be with two values.
//! One venue carries all the volume, so HHI is 1, and both thresholds are met.
//!
//! ```
//! use openquant::streaming_hpc::{
//!     AlertThresholds, HhiConfig, StreamEvent, StreamingEarlyWarningEngine,
//!     StreamingPipelineConfig, VpinConfig,
//! };
//!
//! let cfg = StreamingPipelineConfig {
//!     vpin: VpinConfig { bucket_volume: 100.0, support_buckets: 1, cdf_lookback: 2 },
//!     hhi: HhiConfig { lookback_events: 1 },
//!     thresholds: AlertThresholds { vpin_cdf: 0.75, hhi: 0.5 },
//! };
//! let mut engine = StreamingEarlyWarningEngine::new(cfg)?;
//! let event = |t: i64, buy: f64, sell: f64| StreamEvent {
//!     timestamp_ns: t,
//!     price: 100.0,
//!     buy_volume: buy,
//!     sell_volume: sell,
//!     venue_id: 0,
//! };
//!
//! let first = engine.on_event(event(0, 50.0, 50.0))?;
//! assert_eq!(first.vpin, Some(0.0));
//! assert_eq!(first.vpin_cdf, None); // only one VPIN value in the history so far
//! assert!(!first.is_alert);
//!
//! let second = engine.on_event(event(1, 100.0, 0.0))?;
//! assert_eq!(second.vpin, Some(1.0));
//! assert_eq!(second.vpin_cdf, Some(0.75));
//! assert_eq!(second.hhi, Some(1.0));
//! assert_eq!(second.normalized_risk_score, Some(1.0)); // min(0.75 / 0.75, 1.0 / 0.5)
//! assert!(second.is_alert);
//! # Ok::<(), openquant::streaming_hpc::StreamingHpcError>(())
//! ```
#![deny(missing_docs)]

use crate::hpc_parallel::{run_parallel, HpcParallelConfig, HpcParallelError, ParallelRunReport};
use std::collections::{HashMap, VecDeque};
use std::fmt::{Display, Formatter};
use std::time::{Duration, Instant};

/// Errors returned by the streaming indicators, engine and runners.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamingHpcError {
    /// A configuration value is out of range; the message names it and the condition.
    InvalidConfig(&'static str),
    /// An event or volume is invalid; the message names the field or the condition.
    InvalidEvent(&'static str),
    /// The parallel runner failed (see [`run_streaming_pipeline_parallel`]).
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

/// One trade (or aggregated trade print) of the stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamEvent {
    /// Event time in nanoseconds; carried to the snapshot, not validated or used.
    pub timestamp_ns: i64,
    /// Trade price; must be finite and > 0.
    pub price: f64,
    /// Buyer-initiated volume; finite and >= 0.
    pub buy_volume: f64,
    /// Seller-initiated volume; finite and >= 0. Buy plus sell must be > 0.
    pub sell_volume: f64,
    /// Identifier of the venue the volume traded on, for the HHI.
    pub venue_id: usize,
}

impl StreamEvent {
    /// Total volume of the event, `buy_volume + sell_volume`.
    pub fn total_volume(self) -> f64 {
        self.buy_volume + self.sell_volume
    }
}

/// Parameters of the VPIN estimator ([`VpinState`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VpinConfig {
    /// Volume in each bucket, in the units of the event volumes; finite and > 0.
    pub bucket_volume: f64,
    /// Number of completed buckets in rolling VPIN window (`n` in the VPIN mean); > 0.
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

/// Parameters of the venue-concentration HHI ([`HhiState`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HhiConfig {
    /// Number of events in rolling concentration window. Venues are weighted by their share of
    /// the volume traded in these events, not by their number of events. Must be > 0.
    pub lookback_events: usize,
}

/// Alert thresholds of [`StreamingEarlyWarningEngine`]; the alert needs both to be met.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlertThresholds {
    /// Threshold on the empirical CDF of VPIN, in (0, 1): 0.99 alerts when VPIN is in the top 1%
    /// of its rolling history. It is a probability, not a raw VPIN level, and must be at most
    /// `1 - 0.5 / vpin.cdf_lookback`, the largest value the CDF can take.
    pub vpin_cdf: f64,
    /// Threshold on the volume-share HHI (1 = one venue carries all the volume), in `(0, 1]`.
    /// The HHI never exceeds 1, so a higher threshold is rejected rather than never firing.
    pub hhi: f64,
}

/// Full configuration of the early-warning engine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamingPipelineConfig {
    /// VPIN bucket size, window and CDF history.
    pub vpin: VpinConfig,
    /// HHI window.
    pub hhi: HhiConfig,
    /// Alert thresholds on the CDF of VPIN and on HHI.
    pub thresholds: AlertThresholds,
}

/// State of the indicators after one event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarlyWarningSnapshot {
    /// The event's `timestamp_ns`.
    pub timestamp_ns: i64,
    /// The event's price.
    pub price: f64,
    /// VPIN over the last `support_buckets` full buckets, or `None` until that many exist.
    pub vpin: Option<f64>,
    /// Empirical CDF of the current VPIN over its rolling history ([`VpinConfig::cdf_lookback`]).
    pub vpin_cdf: Option<f64>,
    /// Volume-share HHI over the last `lookback_events` events, or `None` until that many.
    pub hhi: Option<f64>,
    /// `min(vpin_cdf / thresholds.vpin_cdf, hhi / thresholds.hhi)`: the alert condition as one
    /// number for dashboards. It is at least 1 exactly when `is_alert` is true (up to rounding).
    /// `None` unless both `vpin_cdf` and `hhi` are available.
    pub normalized_risk_score: Option<f64>,
    /// `vpin_cdf >= thresholds.vpin_cdf && hhi >= thresholds.hhi`; `false` while either is
    /// `None`.
    pub is_alert: bool,
}

/// Wall-clock timing of a [`run_streaming_pipeline`] call.
///
/// These time the `on_event` calls only (not parsing, queueing or I/O) and vary from run to
/// run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamingRunMetrics {
    /// Number of events processed (the length of the input).
    pub processed_events: usize,
    /// `processed_events / runtime`, or 0 when the runtime rounds to zero.
    pub events_per_sec: f64,
    /// Mean wall-clock time of one `on_event` call, in microseconds; 0 for an empty stream.
    pub avg_event_latency_micros: f64,
    /// Longest wall-clock time of one `on_event` call, in microseconds.
    pub max_event_latency_micros: f64,
    /// Wall-clock time of the whole loop.
    pub runtime: Duration,
}

/// Result of [`run_streaming_pipeline`].
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingRunReport {
    /// One snapshot per input event, in input order.
    pub snapshots: Vec<EarlyWarningSnapshot>,
    /// Timing of the run.
    pub metrics: StreamingRunMetrics,
    /// Number of snapshots with `is_alert == true`.
    pub alert_count: usize,
}

/// Summary of one stream in [`run_streaming_pipeline_parallel`]: the alert count and the last
/// snapshot's indicators.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamSummary {
    /// Number of events in the stream.
    pub processed_events: usize,
    /// Number of alerting events in the stream.
    pub alert_count: usize,
    /// VPIN after the last event (`None` if not yet available or the stream is empty).
    pub latest_vpin: Option<f64>,
    /// CDF of VPIN after the last event.
    pub latest_vpin_cdf: Option<f64>,
    /// HHI after the last event.
    pub latest_hhi: Option<f64>,
    /// Normalised risk score after the last event.
    pub latest_risk_score: Option<f64>,
}

/// Result of [`run_streaming_pipeline_parallel`].
#[derive(Debug, Clone, PartialEq)]
pub struct ParallelStreamingReport {
    /// One summary per input stream, in input order.
    pub stream_summaries: Vec<StreamSummary>,
    /// Partitioning and throughput metrics of the parallel run (one atom per stream).
    pub parallel_metrics: crate::hpc_parallel::HpcParallelMetrics,
}

/// Incremental VPIN on a volume clock, with the empirical CDF of VPIN over a rolling history
/// (AFML §22.6.5; Easley, López de Prado and O'Hara, 2012).
///
/// Memory is bounded by `support_buckets + cdf_lookback` values. Each completed bucket costs
/// `O(cdf_lookback)` to update the sorted history; an event that spans many buckets loops
/// once per bucket it fills.
///
/// ```
/// use openquant::streaming_hpc::{VpinConfig, VpinState};
///
/// let mut vpin =
///     VpinState::new(VpinConfig { bucket_volume: 10.0, support_buckets: 2, cdf_lookback: 2 })?;
/// assert_eq!(vpin.update(10.0, 0.0)?, None); // bucket 1: |10 - 0| / 10 = 1, one bucket so far
/// let v = vpin.update(3.0, 7.0)?.unwrap(); // bucket 2: |3 - 7| / 10 = 0.4
/// assert!((v - 0.7).abs() < 1e-12);
/// assert_eq!(vpin.current_cdf(), None); // one VPIN value recorded, two needed
/// let v = vpin.update(5.0, 5.0)?.unwrap(); // bucket 3: 0, window [0.4, 0]
/// assert!((v - 0.2).abs() < 1e-12);
/// assert_eq!(vpin.current_cdf(), Some(0.25)); // 0.2 is the lower of [0.7, 0.2]: 0.5 / 2
/// // 20 units at 3:1 fill two buckets of 7.5 bought and 2.5 sold, each with toxicity 0.5.
/// let v = vpin.update(15.0, 5.0)?.unwrap();
/// assert!((v - 0.5).abs() < 1e-12);
/// assert_eq!(vpin.completed_buckets(), 2);
/// assert_eq!(vpin.current_cdf(), Some(0.75)); // 0.5 is above 0.25 in [0.25, 0.5]
/// # Ok::<(), openquant::streaming_hpc::StreamingHpcError>(())
/// ```
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
    /// Creates an empty estimator.
    ///
    /// # Errors
    ///
    /// [`StreamingHpcError::InvalidConfig`] if `bucket_volume` is not finite and > 0,
    /// `support_buckets` is 0 or `cdf_lookback` is below 2.
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

    /// Adds `buy_volume` and `sell_volume` (finite, >= 0) to the current bucket, completing as
    /// many buckets as they fill, and returns the current VPIN ([`Self::current`]).
    ///
    /// Volume beyond a bucket's capacity spills into the next bucket with the same buy/sell
    /// ratio. Each completed bucket, once `support_buckets` exist, records one VPIN value in
    /// the CDF history. Zero total volume changes nothing.
    ///
    /// The cost is bounded whatever the volume: at most `support_buckets + cdf_lookback + 2`
    /// buckets are processed one by one. Every whole bucket an event fills has the same
    /// toxicity `|buy - sell| / (buy + sell)`, and only the last `support_buckets +
    /// cdf_lookback` completed buckets can still affect the window or the CDF history, so
    /// the earlier ones are skipped.
    ///
    /// # Errors
    ///
    /// [`StreamingHpcError::InvalidEvent`] (`"buy_volume"` or `"sell_volume"`) if either
    /// volume is negative or not finite, or (`"buy_volume + sell_volume must be finite"`) if
    /// their sum overflows to infinity (which used to loop forever); the state is then
    /// unchanged.
    pub fn update(
        &mut self,
        buy_volume: f64,
        sell_volume: f64,
    ) -> Result<Option<f64>, StreamingHpcError> {
        validate_non_negative_finite("buy_volume", buy_volume)?;
        validate_non_negative_finite("sell_volume", sell_volume)?;
        let total = buy_volume + sell_volume;
        if !total.is_finite() {
            return Err(StreamingHpcError::InvalidEvent("buy_volume + sell_volume must be finite"));
        }
        if total == 0.0 {
            return Ok(self.current());
        }
        // The event is spread over buckets at its own buy/sell ratio, so each unit of it adds
        // `imbalance_rate` of absolute imbalance.
        let imbalance_rate = (buy_volume - sell_volume).abs() / total;
        let bucket = self.cfg.bucket_volume;

        // 1. Top up the partly filled bucket.
        let take = total.min(bucket - self.current_bucket_volume);
        self.current_bucket_volume += take;
        self.current_bucket_abs_imbalance += take * imbalance_rate;
        if !self.bucket_is_full() {
            return Ok(self.current());
        }
        self.complete_bucket();
        let mut remaining = (total - take).max(0.0);

        // 2. Whole buckets. Only the last `support_buckets + cdf_lookback` can still be in the
        //    window or have recorded a VPIN that is still in the history.
        let whole = (remaining / bucket).floor();
        let replay = (self.cfg.support_buckets + self.cfg.cdf_lookback) as f64;
        for _ in 0..(whole.min(replay) as usize) {
            self.current_bucket_abs_imbalance = bucket * imbalance_rate;
            self.complete_bucket();
        }
        remaining = (remaining - whole * bucket).max(0.0);

        // 3. The remainder starts a new bucket (or, after rounding, completes one more).
        if remaining > 0.0 {
            self.current_bucket_volume = remaining.min(bucket);
            self.current_bucket_abs_imbalance = self.current_bucket_volume * imbalance_rate;
            if self.bucket_is_full() {
                self.complete_bucket();
            }
        }
        Ok(self.current())
    }

    fn bucket_is_full(&self) -> bool {
        self.current_bucket_volume >= self.cfg.bucket_volume - 1e-12
    }

    /// Closes the current bucket: its toxicity enters the window, the bucket is reset, and the
    /// VPIN (once defined) is recorded in the CDF history.
    fn complete_bucket(&mut self) {
        let toxicity = self.current_bucket_abs_imbalance / self.cfg.bucket_volume;
        self.window.push_back(toxicity);
        self.window_sum += toxicity;
        if self.window.len() > self.cfg.support_buckets
            && let Some(expired) = self.window.pop_front()
        {
            self.window_sum -= expired;
        }
        self.current_bucket_volume = 0.0;
        self.current_bucket_abs_imbalance = 0.0;
        if let Some(vpin) = self.current() {
            self.record_vpin(vpin);
        }
    }

    /// Current VPIN, the mean toxicity of the last `support_buckets` completed buckets, or
    /// `None` until that many buckets have completed. The partly filled bucket is excluded.
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

    /// Number of completed buckets in the VPIN window, at most `support_buckets`.
    pub fn completed_buckets(&self) -> usize {
        self.window.len()
    }

    fn record_vpin(&mut self, vpin: f64) {
        self.history.push_back(vpin);
        let at = self.history_sorted.partition_point(|&x| x < vpin);
        self.history_sorted.insert(at, vpin);
        if self.history.len() > self.cfg.cdf_lookback
            && let Some(expired) = self.history.pop_front()
        {
            // Values are stored bit for bit, so the expired one is found exactly.
            let at = self.history_sorted.partition_point(|&x| x < expired);
            self.history_sorted.remove(at);
        }
    }
}

/// Herfindahl–Hirschman index of venue concentration over the last `lookback_events` events,
/// each venue weighted by its share of the volume traded in those events:
/// `HHI = sum_v (volume_v / total_volume)^2`.
///
/// It is `1/K` when `K` venues share the volume evenly and 1 when one venue has all of it.
/// Each update is `O(venues)`.
///
/// ```
/// use openquant::streaming_hpc::{HhiConfig, HhiState};
///
/// let mut hhi = HhiState::new(HhiConfig { lookback_events: 3 })?;
/// assert_eq!(hhi.update(0, 10.0)?, None);
/// assert_eq!(hhi.update(1, 10.0)?, None);
/// // Venue 0 has 30 of 40 units, venue 1 has 10: 0.75^2 + 0.25^2.
/// assert_eq!(hhi.update(0, 20.0)?, Some(0.625));
/// // The first event leaves the window: venues hold 20, 10 and 40 of 70.
/// let h = hhi.update(2, 40.0)?.unwrap();
/// assert!((h - 21.0 / 49.0).abs() < 1e-12);
/// assert_eq!(hhi.window_len(), 3);
/// # Ok::<(), openquant::streaming_hpc::StreamingHpcError>(())
/// ```
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
    /// Creates an empty HHI window.
    ///
    /// # Errors
    ///
    /// [`StreamingHpcError::InvalidConfig`] if `lookback_events` is 0.
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

    /// Adds one event that traded `volume` (finite, > 0) on `venue_id`, drops the oldest event
    /// once the window is full, and returns the current HHI ([`Self::current`]).
    ///
    /// # Errors
    ///
    /// [`StreamingHpcError::InvalidEvent`] if `volume` is not finite or not > 0; the state is
    /// then unchanged.
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

        if self.window.len() > self.cfg.lookback_events
            && let Some((expired, expired_volume)) = self.window.pop_front()
            && let Some(entry) = self.venues.get_mut(&expired)
        {
            if entry.0 <= 1 {
                self.venues.remove(&expired);
            } else {
                entry.0 -= 1;
                entry.1 -= expired_volume;
            }
        }
        Ok(self.current())
    }

    /// Current volume-share HHI over the window, or `None` until `lookback_events` events have
    /// been added.
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

    /// Number of events in the window, at most `lookback_events`.
    pub fn window_len(&self) -> usize {
        self.window.len()
    }
}

/// Event-by-event early-warning engine: a [`VpinState`] and an [`HhiState`] updated together,
/// with the alert rule of [`AlertThresholds`] (AFML §22.6.4–22.6.5).
///
/// See the [module documentation](self) for a worked example.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingEarlyWarningEngine {
    cfg: StreamingPipelineConfig,
    vpin_state: VpinState,
    hhi_state: HhiState,
}

impl StreamingEarlyWarningEngine {
    /// Creates an engine with empty indicator windows.
    ///
    /// # Errors
    ///
    /// [`StreamingHpcError::InvalidConfig`] if `thresholds.vpin_cdf` is not in `(0, 1)` or is
    /// above `1 - 0.5 / vpin.cdf_lookback` (the largest CDF value), `thresholds.hhi` is not
    /// in `(0, 1]` (the HHI never exceeds 1, so a higher threshold could never trigger), or
    /// the [`VpinConfig`] or [`HhiConfig`] is invalid (as for
    /// [`VpinState::new`] and [`HhiState::new`]).
    pub fn new(cfg: StreamingPipelineConfig) -> Result<Self, StreamingHpcError> {
        let vpin_cdf = cfg.thresholds.vpin_cdf;
        if !vpin_cdf.is_finite() || vpin_cdf <= 0.0 || vpin_cdf >= 1.0 {
            return Err(StreamingHpcError::InvalidConfig("thresholds.vpin_cdf must be in (0, 1)"));
        }
        let hhi = cfg.thresholds.hhi;
        if !(hhi > 0.0 && hhi <= 1.0) {
            return Err(StreamingHpcError::InvalidConfig("thresholds.hhi must be in (0, 1]"));
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

    /// Validates `event`, updates VPIN, its CDF and HHI, and returns the resulting snapshot.
    ///
    /// # Errors
    ///
    /// [`StreamingHpcError::InvalidEvent`] if the price is not finite and > 0, either volume is
    /// negative or not finite, or the total volume is 0. The event is validated before any
    /// state changes, so a rejected event leaves the engine as it was.
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

/// Runs a fresh [`StreamingEarlyWarningEngine`] over `events` in order, timing each call.
///
/// ```
/// use openquant::streaming_hpc::{
///     generate_synthetic_flash_crash_stream, run_streaming_pipeline, AlertThresholds, HhiConfig,
///     StreamingPipelineConfig, SyntheticStreamConfig, VpinConfig,
/// };
///
/// // The docs-site example: a crash at event 700 of 1,000.
/// let events = generate_synthetic_flash_crash_stream(SyntheticStreamConfig {
///     events: 1_000,
///     crash_start_fraction: 0.7,
///     calm_venues: 4,
///     shock_venue: 0,
/// })?;
/// let cfg = StreamingPipelineConfig {
///     vpin: VpinConfig { bucket_volume: 1_000.0, support_buckets: 10, cdf_lookback: 100 },
///     hhi: HhiConfig { lookback_events: 50 },
///     thresholds: AlertThresholds { vpin_cdf: 0.99, hhi: 0.5 },
/// };
/// let report = run_streaming_pipeline(&events, cfg)?;
/// assert_eq!(report.metrics.processed_events, 1_000);
/// // Calm flow: VPIN |120 - 130| / 250 = 0.04 in every bucket.
/// let calm = report.snapshots[699];
/// assert!((calm.vpin.unwrap() - 0.04).abs() < 1e-9);
/// assert_eq!(calm.vpin_cdf, Some(0.5)); // a steady stream ties with itself
/// // 50 events over four venues split 13/13/12/12, so HHI is 626 / 2500, not exactly 1/4.
/// assert!((calm.hhi.unwrap() - 0.2504).abs() < 1e-9);
/// // The alert fires from event 723 to 728, then the CDF decays on the VPIN plateau.
/// let alerts: Vec<usize> = (0..1_000).filter(|&i| report.snapshots[i].is_alert).collect();
/// assert_eq!(alerts, (723..=728).collect::<Vec<_>>());
/// assert_eq!(report.alert_count, 6);
/// # Ok::<(), openquant::streaming_hpc::StreamingHpcError>(())
/// ```
///
/// # Errors
///
/// As for [`StreamingEarlyWarningEngine::new`] and [`StreamingEarlyWarningEngine::on_event`];
/// the run stops at the first invalid event and returns no partial report.
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

/// Runs [`run_streaming_pipeline`] on many independent streams in parallel with
/// [`crate::hpc_parallel::run_parallel`], one stream per atom, and keeps a [`StreamSummary`]
/// of each (AFML Chapter 22 on HPC for streaming analytics).
///
/// Parallelism is across streams; each stream is processed sequentially. Only the alert count
/// and the last snapshot of each stream are kept: call [`run_streaming_pipeline`] per stream
/// for the full path. An empty `streams` slice gives an empty report.
///
/// ```
/// use openquant::hpc_parallel::{ExecutionMode, HpcParallelConfig, PartitionStrategy};
/// use openquant::streaming_hpc::{
///     generate_synthetic_flash_crash_stream, run_streaming_pipeline_parallel, AlertThresholds,
///     HhiConfig, StreamingPipelineConfig, SyntheticStreamConfig, VpinConfig,
/// };
///
/// let cfg = StreamingPipelineConfig {
///     vpin: VpinConfig { bucket_volume: 1_000.0, support_buckets: 10, cdf_lookback: 100 },
///     hhi: HhiConfig { lookback_events: 50 },
///     thresholds: AlertThresholds { vpin_cdf: 0.99, hhi: 0.5 },
/// };
/// // Eight streams, crashing at events 100, 200, ..., 800.
/// let streams = (1..=8)
///     .map(|k| {
///         generate_synthetic_flash_crash_stream(SyntheticStreamConfig {
///             events: 1_000,
///             crash_start_fraction: k as f64 / 10.0,
///             calm_venues: 4,
///             shock_venue: 0,
///         })
///     })
///     .collect::<Result<Vec<_>, _>>()?;
/// let parallel = HpcParallelConfig {
///     mode: ExecutionMode::Threaded { num_threads: 4 },
///     partition: PartitionStrategy::Linear,
///     mp_batches: 1,
///     progress_every: 1,
/// };
/// let report = run_streaming_pipeline_parallel(&streams, cfg, parallel)?;
/// // Crashes before the CDF history fills (events 100 to 300) are never ranked.
/// let alerts: Vec<usize> = report.stream_summaries.iter().map(|s| s.alert_count).collect();
/// assert_eq!(alerts, [0, 0, 0, 6, 6, 6, 6, 6]);
/// assert_eq!(report.stream_summaries[0].latest_hhi, Some(1.0));
/// # Ok::<(), openquant::streaming_hpc::StreamingHpcError>(())
/// ```
///
/// # Errors
///
/// - [`StreamingHpcError::InvalidConfig`] if `pipeline_cfg` is invalid (as for
///   [`StreamingEarlyWarningEngine::new`]); it is checked once, before any stream runs.
/// - [`StreamingHpcError::Parallel`] wrapping:
///   - [`HpcParallelError::InvalidConfig`] if `parallel_cfg` is invalid;
///   - [`HpcParallelError::CallbackFailed`] if a stream fails (for example an invalid event);
///     the message carries the underlying [`StreamingHpcError`] as text;
///   - [`HpcParallelError::WorkerPanic`] or [`HpcParallelError::ChannelClosed`] if a worker
///     thread fails.
pub fn run_streaming_pipeline_parallel(
    streams: &[Vec<StreamEvent>],
    pipeline_cfg: StreamingPipelineConfig,
    parallel_cfg: HpcParallelConfig,
) -> Result<ParallelStreamingReport, StreamingHpcError> {
    // Validate the shared config once, so a bad config is reported as such rather than as
    // every stream's callback failing.
    StreamingEarlyWarningEngine::new(pipeline_cfg)?;
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

/// Parameters of [`generate_synthetic_flash_crash_stream`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyntheticStreamConfig {
    /// Number of events; > 0.
    pub events: usize,
    /// Fraction of the stream before the crash, in `(0, 1)`; the crash starts at event
    /// `round(events * crash_start_fraction)`.
    pub crash_start_fraction: f64,
    /// Number of venues the calm flow rotates over (venues `0..calm_venues`); > 0.
    pub calm_venues: usize,
    /// Venue that carries all the flow from the crash on; may be one of the calm venues.
    pub shock_venue: usize,
}

/// Generates a deterministic two-regime stream for calibrating the early-warning engine,
/// loosely modelled on the Flash Crash of 2010 (AFML §22.6.4).
///
/// Event `i` has `timestamp_ns = i * 1_000_000` (1 ms apart). Before the crash, event `i` is on
/// venue `i % calm_venues` with 120 bought and 130 sold, and the price grows by 0.01% per event;
/// from the crash on, every event is on `shock_venue` with 80 bought and 320 sold, and the
/// price falls by 0.25% per event. The price starts from 100 and moves before the first event
/// is emitted, so event 0 is at 100.01 (or 99.75 if the crash starts at 0).
///
/// The rounding of the crash start means a small stream can have no calm events or no crash
/// events (for example 10 events with a fraction of 0.96 round to a crash at event 10).
///
/// ```
/// use openquant::streaming_hpc::{generate_synthetic_flash_crash_stream, SyntheticStreamConfig};
///
/// let events = generate_synthetic_flash_crash_stream(SyntheticStreamConfig {
///     events: 10,
///     crash_start_fraction: 0.5,
///     calm_venues: 2,
///     shock_venue: 7,
/// })?;
/// assert_eq!(events.len(), 10);
/// let e = events[3];
/// assert_eq!((e.venue_id, e.buy_volume, e.sell_volume), (1, 120.0, 130.0));
/// let e = events[5];
/// assert_eq!((e.venue_id, e.buy_volume, e.sell_volume), (7, 80.0, 320.0));
/// assert_eq!(events[5].timestamp_ns, 5_000_000);
/// assert!((events[0].price - 100.01).abs() < 1e-9);
/// let expected_last = 100.0 * 1.0001_f64.powi(5) * 0.9975_f64.powi(5);
/// assert!((events[9].price - expected_last).abs() < 1e-9);
/// # Ok::<(), openquant::streaming_hpc::StreamingHpcError>(())
/// ```
///
/// # Errors
///
/// [`StreamingHpcError::InvalidConfig`] if `events` is 0, `crash_start_fraction` is not finite
/// and strictly between 0 and 1, or `calm_venues` is 0.
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
