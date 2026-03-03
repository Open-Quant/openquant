---
title: "hpc_parallel"
description: "AFML Chapter 20 atom/molecule execution utilities with serial/threaded modes and partition diagnostics."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "hpc_parallel"
risk_notes:
  - "Use `ExecutionMode::Serial` for deterministic debugging with identical callback semantics."
  - "If per-atom cost rises with atom index (e.g., expanding windows), nested partitioning can reduce tail stragglers versus linear chunking."
rust_api:
  - "partition_atoms"
  - "run_parallel"
  - "dispatch_async"
  - "ExecutionMode"
  - "PartitionStrategy"
  - "HpcParallelConfig"
  - "ParallelRunReport"
  - "HpcParallelMetrics"
sidebar:
  badge: Module
---

## Subject

**Scaling, HPC and Infrastructure**

## Why This Module Exists

Research pipelines bottleneck on repeated independent computations; this module exposes reproducible partitioning and dispatch controls to scale those workloads safely.

## Mathematical Foundations

### Linear Partition Boundary

$$b_i=\left\lfloor\frac{iN}{M}\right\rfloor,\;i=0,\dots,M$$

### Nested Partition Boundary

$$b_i=\left\lfloor N\sqrt{\frac{i}{M}}\right\rfloor,\;i=0,\dots,M$$

### Throughput

$$\text{throughput}=\frac{\text{atoms processed}}{\text{runtime seconds}}$$

## Usage Examples

### Rust

#### Run atom->molecule callback in threaded mode

```rust
use openquant::hpc_parallel::{run_parallel, ExecutionMode, HpcParallelConfig, PartitionStrategy};

let atoms: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
let report = run_parallel(
  &atoms,
  HpcParallelConfig {
    mode: ExecutionMode::Threaded { num_threads: 8 },
    partition: PartitionStrategy::Nested,
    mp_batches: 4,
    progress_every: 4,
  },
  |chunk| Ok::<f64, &'static str>(chunk.iter().map(|x| x.sqrt()).sum()),
)?;

println!("molecules={} atoms/s={:.0}", report.metrics.molecules_total, report.metrics.throughput_atoms_per_sec);
```

## API Reference

### Rust API

- `partition_atoms`
- `run_parallel`
- `dispatch_async`
- `ExecutionMode`
- `PartitionStrategy`
- `HpcParallelConfig`
- `ParallelRunReport`
- `HpcParallelMetrics`

## Implementation Notes

- Use `ExecutionMode::Serial` for deterministic debugging with identical callback semantics.
- If per-atom cost rises with atom index (e.g., expanding windows), nested partitioning can reduce tail stragglers versus linear chunking.
