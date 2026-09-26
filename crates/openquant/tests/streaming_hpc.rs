use openquant::hpc_parallel::{ExecutionMode, HpcParallelConfig, PartitionStrategy};
use openquant::streaming_hpc::{
    generate_synthetic_flash_crash_stream, run_streaming_pipeline, run_streaming_pipeline_parallel,
    AlertThresholds, HhiConfig, HhiState, StreamEvent, StreamingEarlyWarningEngine,
    StreamingHpcError, StreamingPipelineConfig, SyntheticStreamConfig, VpinConfig, VpinState,
};

/// Every pipeline config in this file is built here.
fn cfg(
    bucket_volume: f64,
    support_buckets: usize,
    cdf_lookback: usize,
    lookback_events: usize,
    vpin_cdf_threshold: f64,
    hhi_threshold: f64,
) -> StreamingPipelineConfig {
    StreamingPipelineConfig {
        vpin: VpinConfig { bucket_volume, support_buckets, cdf_lookback },
        hhi: HhiConfig { lookback_events },
        thresholds: AlertThresholds { vpin_cdf: vpin_cdf_threshold, hhi: hhi_threshold },
    }
}

fn pipeline_cfg() -> StreamingPipelineConfig {
    cfg(1_000.0, 10, 100, 120, 0.99, 0.20)
}

fn event(i: usize, buy_volume: f64, sell_volume: f64, venue_id: usize) -> StreamEvent {
    StreamEvent { timestamp_ns: i as i64 * 1_000, price: 100.0, buy_volume, sell_volume, venue_id }
}

#[test]
fn bounded_memory_vpin_window_size() {
    let mut state =
        VpinState::new(VpinConfig { bucket_volume: 100.0, support_buckets: 8, cdf_lookback: 50 })
            .expect("valid");
    for _ in 0..10_000 {
        let _ = state.update(40.0, 60.0).expect("update");
    }
    assert!(state.completed_buckets() <= 8);
}

/// Two instruments on one venue (HHI = 1). A trades 45/55 in calm (VPIN 0.1) and 30/70 in its
/// shock (VPIN 0.4); B trades 25/75 in calm (VPIN 0.5) and 5/95 in its shock (VPIN 0.9). B's calm
/// VPIN is above A's shock VPIN, so no single raw-VPIN threshold alerts on both shocks without
/// alerting on B's calm. One threshold on CDF(VPIN) does (AFML §22.6.5).
#[test]
fn vpin_cdf_threshold_adapts_to_each_instruments_baseline() {
    let calm = 300;
    let shock = 50;
    for (calm_flow, shock_flow) in [((45.0, 55.0), (30.0, 70.0)), ((25.0, 75.0), (5.0, 95.0))] {
        let events: Vec<StreamEvent> = (0..calm + shock)
            .map(|i| {
                let (b, s) = if i < calm { calm_flow } else { shock_flow };
                event(i, b, s, 0)
            })
            .collect();
        let report =
            run_streaming_pipeline(&events, cfg(100.0, 5, 50, 20, 0.99, 0.5)).expect("run");
        let (pre, post) = report.snapshots.split_at(calm);
        assert!(pre.iter().all(|s| !s.is_alert), "calm flow {calm_flow:?} raised an alert");
        assert!(post.iter().any(|s| s.is_alert), "shock flow {shock_flow:?} raised no alert");
    }
    // The raw levels that make a single raw threshold impossible.
    let vpin = |b: f64, s: f64| (b - s).abs() / (b + s);
    assert!(vpin(25.0, 75.0) > vpin(30.0, 70.0));
}

/// A steady stream is not its own extreme: once the history is full, every value ties with
/// every other and the mid-distribution CDF is 0.5. Before that the CDF is undefined.
#[test]
fn vpin_cdf_is_half_for_a_steady_stream_and_none_while_warming_up() {
    let events: Vec<StreamEvent> = (0..100).map(|i| event(i, 40.0, 60.0, 0)).collect();
    let report = run_streaming_pipeline(&events, cfg(100.0, 8, 20, 10, 0.9, 0.5)).expect("run");
    // Bucket k completes at event k (one event = one bucket). VPIN exists from bucket 8 (index
    // 7); the 20th VPIN value, and so the first CDF, arrives at index 7 + 19 = 26.
    assert!(report.snapshots[25].vpin_cdf.is_none());
    assert!(report.snapshots[7].vpin.is_some());
    for s in &report.snapshots[26..] {
        assert_eq!(s.vpin_cdf, Some(0.5));
        assert!(!s.is_alert);
    }
    assert_eq!(report.alert_count, 0);
}

#[test]
fn vpin_cdf_ranks_the_current_value_in_its_history() {
    let mut state =
        VpinState::new(VpinConfig { bucket_volume: 100.0, support_buckets: 1, cdf_lookback: 4 })
            .expect("valid");
    // With one bucket per VPIN value, VPIN is each bucket's |buy - sell| / 100.
    for (buy, sell) in [(50.0, 50.0), (40.0, 60.0), (30.0, 70.0)] {
        state.update(buy, sell).expect("update");
        assert_eq!(state.current_cdf(), None);
    }
    // History {0.0, 0.2, 0.4, 0.1}: 0.1 is second of four -> (1 + 0.5) / 4.
    state.update(55.0, 45.0).expect("update");
    assert_eq!(state.current_cdf(), Some(0.375));
    // History {0.2, 0.4, 0.1, 0.8}: a new maximum -> (3 + 0.5) / 4. 0.0 has expired.
    state.update(10.0, 90.0).expect("update");
    assert_eq!(state.current_cdf(), Some(0.875));
    // History {0.4, 0.1, 0.8, 0.0}: a new minimum -> 0.5 / 4.
    state.update(50.0, 50.0).expect("update");
    assert_eq!(state.current_cdf(), Some(0.125));
}

/// Every 10 events: venue 0 prints 8 trades of 1 unit, venues 1 and 2 one trade of 46 each.
/// By volume the shares are 0.08 / 0.46 / 0.46 and HHI = 0.4296; counting events (0.8 / 0.1 / 0.1)
/// would give 0.66 and make the venue with 8% of the volume look dominant.
#[test]
fn hhi_weights_venues_by_volume_not_event_count() {
    let events: Vec<StreamEvent> = (0..200)
        .map(|i| match i % 10 {
            8 => event(i, 23.0, 23.0, 1),
            9 => event(i, 23.0, 23.0, 2),
            _ => event(i, 0.5, 0.5, 0),
        })
        .collect();
    let report = run_streaming_pipeline(&events, cfg(50.0, 2, 10, 10, 0.9, 0.5)).expect("run");
    let expected = 0.08f64.powi(2) + 2.0 * 0.46f64.powi(2);
    // Any 10 consecutive events hold exactly one period.
    for s in &report.snapshots[9..] {
        let hhi = s.hhi.expect("window full");
        assert!((hhi - expected).abs() < 1e-12, "hhi = {hhi}, expected {expected}");
    }
}

#[test]
fn hhi_state_uses_volume_shares_over_a_rolling_window() {
    let mut state = HhiState::new(HhiConfig { lookback_events: 3 }).expect("valid");
    assert_eq!(state.update(0, 1.0).expect("update"), None);
    assert_eq!(state.update(0, 1.0).expect("update"), None);
    // Window (0: 1), (0: 1), (1: 2): shares 0.5 / 0.5.
    assert_eq!(state.update(1, 2.0).expect("update"), Some(0.5));
    // Window (0: 1), (1: 2), (2: 7): shares 0.1 / 0.2 / 0.7.
    let hhi = state.update(2, 7.0).expect("update").expect("window full");
    assert!((hhi - (0.01 + 0.04 + 0.49)).abs() < 1e-12);
    assert_eq!(state.window_len(), 3);
    assert!(matches!(state.update(0, 0.0), Err(StreamingHpcError::InvalidEvent(_))));
    assert!(matches!(state.update(0, f64::NAN), Err(StreamingHpcError::InvalidEvent(_))));
}

/// The score is the alert condition as a number: >= 1 exactly when both indicators are at or
/// above their thresholds.
#[test]
fn risk_score_reaches_one_only_when_alerting() {
    let events = generate_synthetic_flash_crash_stream(SyntheticStreamConfig {
        events: 4_000,
        crash_start_fraction: 0.70,
        calm_venues: 8,
        shock_venue: 0,
    })
    .expect("synthetic stream");
    let report = run_streaming_pipeline(&events, pipeline_cfg()).expect("pipeline run");
    assert!(report.alert_count > 0);
    for s in &report.snapshots {
        if let Some(score) = s.normalized_risk_score {
            assert_eq!(score >= 1.0, s.is_alert, "score {score} but is_alert = {}", s.is_alert);
        }
    }
}

#[test]
fn rejects_unusable_vpin_cdf_thresholds() {
    let invalid = |c: StreamingPipelineConfig| {
        matches!(StreamingEarlyWarningEngine::new(c), Err(StreamingHpcError::InvalidConfig(_)))
    };
    // A CDF is a probability.
    assert!(invalid(cfg(100.0, 5, 100, 10, 1.0, 0.5)));
    assert!(invalid(cfg(100.0, 5, 100, 10, 0.0, 0.5)));
    // With 20 values the CDF is at most 1 - 0.5 / 20 = 0.975, so 0.99 could never fire.
    assert!(invalid(cfg(100.0, 5, 20, 10, 0.99, 0.5)));
    assert!(!invalid(cfg(100.0, 5, 20, 10, 0.975, 0.5)));
    assert!(invalid(cfg(100.0, 5, 1, 10, 0.4, 0.5)));
}

#[test]
fn flash_crash_segment_raises_early_warning_metrics() {
    let events = generate_synthetic_flash_crash_stream(SyntheticStreamConfig {
        events: 4_000,
        crash_start_fraction: 0.70,
        calm_venues: 8,
        shock_venue: 0,
    })
    .expect("synthetic stream");
    let report = run_streaming_pipeline(&events, pipeline_cfg()).expect("pipeline run");

    let split = (events.len() as f64 * 0.70) as usize;
    let pre = &report.snapshots[split / 2..split];
    let post = &report.snapshots[split..(split + split / 4)];

    let pre_vpin = pre.iter().filter_map(|s| s.vpin).sum::<f64>()
        / pre.iter().filter(|s| s.vpin.is_some()).count() as f64;
    let post_vpin = post.iter().filter_map(|s| s.vpin).sum::<f64>()
        / post.iter().filter(|s| s.vpin.is_some()).count() as f64;
    let pre_hhi = pre.iter().filter_map(|s| s.hhi).sum::<f64>()
        / pre.iter().filter(|s| s.hhi.is_some()).count() as f64;
    let post_hhi = post.iter().filter_map(|s| s.hhi).sum::<f64>()
        / post.iter().filter(|s| s.hhi.is_some()).count() as f64;

    assert!(post_vpin > pre_vpin);
    assert!(post_hhi > pre_hhi);
    assert!(report.alert_count > 0);
}

#[test]
fn serial_and_parallel_grouped_runs_agree_on_terminal_state() {
    let mut streams = Vec::new();
    for k in 0..12 {
        let mut stream = generate_synthetic_flash_crash_stream(SyntheticStreamConfig {
            events: 1_500,
            crash_start_fraction: 0.65,
            calm_venues: 6,
            shock_venue: k % 3,
        })
        .expect("stream");
        // Make streams non-identical while preserving deterministic order.
        for event in &mut stream {
            event.price *= 1.0 + k as f64 * 1e-6;
        }
        streams.push(stream);
    }

    let serial: Vec<_> = streams
        .iter()
        .map(|s| run_streaming_pipeline(s, pipeline_cfg()).expect("serial run"))
        .collect();

    let parallel = run_streaming_pipeline_parallel(
        &streams,
        pipeline_cfg(),
        HpcParallelConfig {
            mode: ExecutionMode::Threaded { num_threads: 4 },
            partition: PartitionStrategy::Linear,
            mp_batches: 3,
            progress_every: 2,
        },
    )
    .expect("parallel run");

    assert_eq!(parallel.stream_summaries.len(), streams.len());
    for (lhs, rhs) in serial.iter().zip(parallel.stream_summaries.iter()) {
        let last = lhs.snapshots.last().expect("non-empty stream");
        assert_eq!(lhs.metrics.processed_events, rhs.processed_events);
        assert_eq!(lhs.alert_count, rhs.alert_count);
        assert_eq!(last.vpin, rhs.latest_vpin);
        assert_eq!(last.vpin_cdf, rhs.latest_vpin_cdf);
        assert_eq!(last.hhi, rhs.latest_hhi);
    }
}

#[test]
fn supports_large_synthetic_stream_incrementally() {
    let mut events = Vec::with_capacity(50_000);
    let mut price = 100.0;
    for i in 0..50_000 {
        let venue = i % 10;
        let (buy, sell, drift) =
            if i % 5000 >= 4200 { (90.0, 280.0, -0.0012) } else { (140.0, 150.0, 0.00005) };
        price *= 1.0 + drift;
        events.push(StreamEvent {
            timestamp_ns: i as i64 * 500_000,
            price,
            buy_volume: buy,
            sell_volume: sell,
            venue_id: venue,
        });
    }
    let report = run_streaming_pipeline(&events, pipeline_cfg()).expect("large run should succeed");
    assert_eq!(report.metrics.processed_events, 50_000);
    assert!(report.metrics.events_per_sec > 0.0);
    assert!(report.snapshots.last().and_then(|s| s.vpin).is_some());
    assert!(report.snapshots.last().and_then(|s| s.hhi).is_some());
}

/// Runs `f` on a thread and fails the test if it has not finished within `secs` seconds (the
/// thread is left running; the old code looped forever here).
fn within<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(std::time::Duration::from_secs(secs)).expect("call did not return in time")
}

/// #184 item 9: `buy + sell` overflowing to +inf passed validation and looped forever, and a
/// huge finite volume cost one iteration per bucket (and stalled once `remaining - take` no
/// longer changed `remaining`).
#[test]
fn vpin_update_is_bounded_for_huge_or_overflowing_volume() {
    let vpin_cfg = VpinConfig { bucket_volume: 100.0, support_buckets: 8, cdf_lookback: 20 };

    let err = within(10, move || {
        let mut state = VpinState::new(vpin_cfg).unwrap();
        state.update(f64::MAX, f64::MAX)
    })
    .unwrap_err();
    assert_eq!(err, StreamingHpcError::InvalidEvent("buy_volume + sell_volume must be finite"));

    // 4e17 / 6e17 fills 1e16 buckets, each with toxicity 0.2.
    let (vpin, cdf) = within(10, move || {
        let mut state = VpinState::new(vpin_cfg).unwrap();
        state.update(4e17, 6e17).unwrap();
        (state.current(), state.current_cdf())
    });
    assert!((vpin.unwrap() - 0.2).abs() < 1e-12);
    assert_eq!(cdf, Some(0.5));
}

/// The bounded update must match bucket-by-bucket filling: an event split into ten equal
/// events with the same buy/sell ratio leaves the same VPIN and CDF.
#[test]
fn vpin_update_matches_the_same_volume_fed_in_pieces() {
    let vpin_cfg = VpinConfig { bucket_volume: 100.0, support_buckets: 4, cdf_lookback: 6 };
    let events = [(30.0, 5.0), (250.0, 10.0), (7.0, 93.0), (0.0, 1234.5), (61.0, 40.0), (5e3, 1e3)];
    let mut whole = VpinState::new(vpin_cfg).unwrap();
    let mut pieces = VpinState::new(vpin_cfg).unwrap();
    for (buy, sell) in events {
        whole.update(buy, sell).unwrap();
        for _ in 0..10 {
            pieces.update(buy / 10.0, sell / 10.0).unwrap();
        }
        let (a, b) = (whole.current(), pieces.current());
        assert_eq!(a.is_some(), b.is_some());
        if let (Some(a), Some(b)) = (a, b) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
        let (a, b) = (whole.current_cdf(), pieces.current_cdf());
        assert_eq!(a.is_some(), b.is_some());
        if let (Some(a), Some(b)) = (a, b) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }
    assert!(whole.current_cdf().is_some());
}
