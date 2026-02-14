use openquant::hpc_parallel::{ExecutionMode, HpcParallelConfig, PartitionStrategy};
use openquant::streaming_hpc::{
    generate_synthetic_flash_crash_stream, run_streaming_pipeline, run_streaming_pipeline_parallel,
    AlertThresholds, HhiConfig, StreamEvent, StreamingPipelineConfig, SyntheticStreamConfig,
    VpinConfig, VpinState,
};

fn pipeline_cfg() -> StreamingPipelineConfig {
    StreamingPipelineConfig {
        vpin: VpinConfig { bucket_volume: 1_000.0, support_buckets: 10 },
        hhi: HhiConfig { lookback_events: 120 },
        thresholds: AlertThresholds { vpin: 0.45, hhi: 0.30 },
    }
}

#[test]
fn bounded_memory_vpin_window_size() {
    let mut state =
        VpinState::new(VpinConfig { bucket_volume: 100.0, support_buckets: 8 }).expect("valid");
    for _ in 0..10_000 {
        let _ = state.update(40.0, 60.0).expect("update");
    }
    assert!(state.completed_buckets() <= 8);
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
