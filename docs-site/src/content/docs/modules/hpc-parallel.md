---
title: "hpc_parallel"
description: "AFML Chapter 20 atom/molecule execution utilities with serial/threaded modes and partition diagnostics."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-31'
audience:
  - quant-dev
  - platform-engineering
module: "hpc_parallel"
api_surface: "rust-only"
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

## Concept Overview

AFML Chapter 20's atom/molecule model: a job is a list of independent atoms, atoms are grouped into molecules, and molecules are dispatched to workers. What this adds over a plain thread pool is the partitioning choice — linear for uniform-cost atoms, nested for the triangular workloads that dominate this library, where atom k touches k earlier observations — together with a metrics report and a serial mode whose callback semantics are identical to the threaded one.

## When to Use

Use it for any embarrassingly parallel research loop: per-asset feature computation, bootstrap replicas, parameter sweeps. Choose `PartitionStrategy::Nested` when per-atom cost grows with the atom index, otherwise the final molecule becomes the whole runtime; choose `Linear` when atoms cost the same. Debug with `ExecutionMode::Serial` first — the callback contract is unchanged, so a bug that reproduces there is not a concurrency bug and you have just halved the search space.

## Mathematical Foundations

### Linear Partition Boundary

$$b_i=\left\lfloor\frac{iN}{M}\right\rfloor,\;i=0,\dots,M$$

where $N$ is the number of atoms, $M$ the number of molecules (`mp_batches` x workers), and molecule $i$ covers atoms $[b_{i-1},b_i)$. Every molecule gets the same *count* of atoms, which is correct only when atoms cost the same.

### Nested Partition Boundary

$$b_i=\left\lfloor N\sqrt{\frac{i}{M}}\right\rfloor,\;i=0,\dots,M$$

where The same $N$ and $M$, for the triangular workloads that dominate this library — building an overlap or codependence matrix, where atom $k$ touches $k$ earlier observations, so its cost grows linearly with $k$. Later molecules therefore hold fewer atoms.

### Equal-Cost Condition

$$\text{cost}(i)\;\propto\;\frac{b_i^2-b_{i-1}^2}{2}=\frac{N^2}{2M}\quad\text{for every }i$$

where $b_i$ and $M$ are as above. This is why the square root is there: if atom $k$ costs $\propto k$, a molecule spanning $[b_{i-1},b_i)$ costs $\propto(b_i^2-b_{i-1}^2)/2$; substituting $b_i=N\sqrt{i/M}$ makes that $N^2/(2M)$, the same for every molecule. Linear partitioning on the same workload leaves the last molecule roughly $2M-1$ times more expensive than the first, and the run is only as fast as that straggler.

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

## Risk Notes and Caveats

- Use `ExecutionMode::Serial` for deterministic debugging with identical callback semantics.
- If per-atom cost rises with atom index (e.g., expanding windows), nested partitioning can reduce tail stragglers versus linear chunking.

## Related Modules

- [`streaming-hpc`](/modules/streaming-hpc/)
- [`combinatorial-optimization`](/modules/combinatorial-optimization/)
- [`sampling`](/modules/sampling/)
- [`backtesting-engine`](/modules/backtesting-engine/)
