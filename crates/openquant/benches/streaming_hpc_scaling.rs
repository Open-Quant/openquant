use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use openquant::hpc_parallel::{ExecutionMode, HpcParallelConfig, PartitionStrategy};
use openquant::streaming_hpc::{
    generate_synthetic_flash_crash_stream, run_streaming_pipeline_parallel, AlertThresholds,
    HhiConfig, StreamingPipelineConfig, SyntheticStreamConfig, VpinConfig,
};
use std::thread;

fn pipeline_cfg() -> StreamingPipelineConfig {
    StreamingPipelineConfig {
        vpin: VpinConfig { bucket_volume: 1_000.0, support_buckets: 20 },
        hhi: HhiConfig { lookback_events: 200 },
        thresholds: AlertThresholds { vpin: 0.45, hhi: 0.30 },
    }
}

fn make_streams(streams: usize, events: usize) -> Vec<Vec<openquant::streaming_hpc::StreamEvent>> {
    (0..streams)
        .map(|k| {
            let mut s = generate_synthetic_flash_crash_stream(SyntheticStreamConfig {
                events,
                crash_start_fraction: 0.70,
                calm_venues: 8,
                shock_venue: k % 2,
            })
            .expect("synthetic stream");
            for event in &mut s {
                event.price *= 1.0 + k as f64 * 1e-6;
            }
            s
        })
        .collect()
}

fn bench_streaming_hpc_scaling(c: &mut Criterion) {
    let streams = make_streams(48, 4_000);
    let available = thread::available_parallelism().map_or(4, |n| n.get());
    let mut thread_options = vec![1usize, 2, 4, 8];
    thread_options.retain(|n| *n <= available.max(1));
    if !thread_options.contains(&available) && available <= 16 {
        thread_options.push(available);
    }
    thread_options.sort_unstable();
    thread_options.dedup();

    let mut group = c.benchmark_group("streaming_hpc/scaling");
    for num_threads in thread_options {
        for mp_batches in [1usize, 2, 4, 8] {
            let mode = if num_threads == 1 {
                ExecutionMode::Serial
            } else {
                ExecutionMode::Threaded { num_threads }
            };
            let bench_id = BenchmarkId::new(
                format!("threads_{num_threads}"),
                format!("mp_batches_{mp_batches}"),
            );
            group.bench_with_input(bench_id, &(mode, mp_batches), |b, (mode, mp_batches)| {
                b.iter(|| {
                    let _ = run_streaming_pipeline_parallel(
                        &streams,
                        pipeline_cfg(),
                        HpcParallelConfig {
                            mode: *mode,
                            partition: PartitionStrategy::Linear,
                            mp_batches: *mp_batches,
                            progress_every: 16,
                        },
                    )
                    .expect("parallel streaming run should succeed");
                });
            });
        }
    }
    group.finish();
}

criterion_group!(streaming_hpc_scaling, bench_streaming_hpc_scaling);
criterion_main!(streaming_hpc_scaling);
