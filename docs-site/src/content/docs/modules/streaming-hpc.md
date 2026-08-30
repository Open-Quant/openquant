---
title: "streaming_hpc"
description: "AFML Chapter 22 streaming analytics utilities for low-latency early-warning metrics with bounded-memory incremental state."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-30'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "streaming_hpc"
api_surface: "both"
risk_notes:
  - "Chapter 22 stresses turnaround-time over pure throughput: bounded rolling windows avoid unbounded latency/memory growth."
  - "For low-latency alerts, keep stream partitioning stable and calibrate `mp_batches` against scheduling overhead and cache locality."
  - "Use synthetic flash-crash replays to validate that warning thresholds react early without excessive false positives."
rust_api:
  - "StreamEvent"
  - "VpinState"
  - "HhiState"
  - "StreamingEarlyWarningEngine"
  - "run_streaming_pipeline"
  - "run_streaming_pipeline_parallel"
  - "generate_synthetic_flash_crash_stream"
  - "StreamingPipelineConfig"
  - "StreamingRunMetrics"
sidebar:
  badge: Module
---

## Subject

**Scaling, HPC and Infrastructure**

## Why This Module Exists

Streaming decisions are turnaround-time constrained; this module maintains VPIN/HHI-style indicators incrementally and supports multi-stream scaling across cores/chunk sizes.

## Mathematical Foundations

### VPIN (Rolling Volume Buckets)

$$\mathrm{VPIN}_t=\frac{1}{N}\sum_{i=t-N+1}^{t}\frac{\left|V_i^{B}-V_i^{S}\right|}{V},\qquad V_i^{B}+V_i^{S}=V$$

where $V_i^{B}$ and $V_i^{S}$ are buy- and sell-initiated volume in bucket $i$, $V$ the fixed `bucket_volume` every bucket is filled to, and $N$ = `support_buckets` the rolling window. Because buckets are equal-volume by construction, the denominator is a constant — this is the canonical Easley-Lopez de Prado form. The bar-based `get_vpin` in [`microstructural-features`](/modules/microstructural-features/) estimates the same quantity over unequal bars and so must normalise differently.

### Market Fragmentation HHI

$$\mathrm{HHI}_t=\sum_{v=1}^{K}\left(\frac{n_{v,t}}{\sum_j n_{j,t}}\right)^2$$

where $n_{v,t}$ is the event count on venue $v$ over the trailing `lookback_events` window and $K$ the number of venues. $1/K$ means flow is spread evenly; $1$ means one venue carries everything. Concentration spikes are the fragmentation half of a flash-crash signature.

### Alert Condition

$$\text{alert}_t\iff \mathrm{VPIN}_t\ge\tau_V\;\land\;\mathrm{HHI}_t\ge\tau_H,\qquad \text{risk}_t=\frac{1}{2}\left(\frac{\mathrm{VPIN}_t}{\tau_V}+\frac{\mathrm{HHI}_t}{\tau_H}\right)$$

where $\tau_V$ and $\tau_H$ are `AlertThresholds { vpin, hhi }`. Both conditions must hold — toxic flow alone, or concentrated flow alone, is common; together they are not. $\text{risk}_t$ is the threshold-normalised score reported alongside the boolean, and is undefined until both estimators have filled their windows.

## Usage Examples

### Rust

#### Incremental early-warning pipeline on streaming trades

```rust
use openquant::hpc_parallel::{ExecutionMode, HpcParallelConfig, PartitionStrategy};
use openquant::streaming_hpc::{
  run_streaming_pipeline_parallel, AlertThresholds, HhiConfig, StreamingPipelineConfig,
  SyntheticStreamConfig, VpinConfig, generate_synthetic_flash_crash_stream,
};

let streams: Vec<_> = (0..16)
  .map(|k| generate_synthetic_flash_crash_stream(SyntheticStreamConfig {
    events: 2_000,
    crash_start_fraction: 0.7,
    calm_venues: 8,
    shock_venue: k % 2,
  }))
  .collect::<Result<Vec<_>, _>>()?;

let report = run_streaming_pipeline_parallel(
  &streams,
  StreamingPipelineConfig {
    vpin: VpinConfig { bucket_volume: 1_000.0, support_buckets: 20 },
    hhi: HhiConfig { lookback_events: 200 },
    thresholds: AlertThresholds { vpin: 0.45, hhi: 0.30 },
  },
  HpcParallelConfig {
    mode: ExecutionMode::Threaded { num_threads: 8 },
    partition: PartitionStrategy::Linear,
    mp_batches: 4,
    progress_every: 8,
  },
)?;

println!("streams={} molecules={} events/s={:.0}",
  report.stream_summaries.len(),
  report.parallel_metrics.molecules_total,
  report.parallel_metrics.throughput_atoms_per_sec
);
```

## API Reference

### Python API

- `streaming_hpc.run_streaming_pipeline`
- `streaming_hpc.generate_synthetic_flash_crash_stream`

### Rust API

- `StreamEvent`
- `VpinState`
- `HhiState`
- `StreamingEarlyWarningEngine`
- `run_streaming_pipeline`
- `run_streaming_pipeline_parallel`
- `generate_synthetic_flash_crash_stream`
- `StreamingPipelineConfig`
- `StreamingRunMetrics`

## Implementation Notes

- Chapter 22 stresses turnaround-time over pure throughput: bounded rolling windows avoid unbounded latency/memory growth.
- For low-latency alerts, keep stream partitioning stable and calibrate `mp_batches` against scheduling overhead and cache locality.
- Use synthetic flash-crash replays to validate that warning thresholds react early without excessive false positives.
