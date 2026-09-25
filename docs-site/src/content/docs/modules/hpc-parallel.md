---
title: "hpc_parallel"
description: "AFML's atoms and molecules: split a job into contiguous chunks of equal work, run a callback over each in serial or on threads, and get the outputs back in order."
status: authored
last_authored: '2026-09-24'
audience:
  - quant-dev
  - platform-engineering
module: "hpc_parallel"
api_surface: "rust-only"
afml_chapter:
  - "20"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 20: §20.4 Atoms and Molecules (§20.4.1 Linear Partitions, §20.4.2 Two-Nested Loops Partitions, Snippets 20.5–20.6); §20.5 Multiprocessing Engines (Snippets 20.7–20.9)."
rust_api:
  - "partition_atoms"
  - "run_parallel"
  - "dispatch_async"
  - "AsyncParallelHandle"
  - "ExecutionMode"
  - "PartitionStrategy"
  - "HpcParallelConfig"
  - "MoleculePartition"
  - "ParallelRunReport"
  - "HpcParallelMetrics"
  - "ProgressSnapshot"
  - "HpcParallelError"
sidebar:
  badge: Module
---

Most of the expensive loops in this library are embarrassingly parallel: one label per
event, one feature per asset, one backtest per parameter set. AFML's Chapter 20 names the
pieces. An **atom** is the smallest task that cannot be split; a **molecule** is a contiguous
group of atoms handed to one worker. The whole design problem is choosing the molecules so
that every worker finishes at the same time, because a parallel run is only as fast as its
slowest molecule.

This module is the Rust version of the book's `mpPandasObj`: `partition_atoms` forms the
molecules, `run_parallel` runs a callback over each of them, in serial or on a pool of
threads, and returns the outputs in molecule order with a timing report.

## Equal atoms or equal work

`PartitionStrategy::Linear` gives every molecule the same number of atoms, AFML's `linParts`.
With $N$ atoms and $M$ molecules, molecule $i$ ends at

$$
b_i = \left\lfloor \frac{iN}{M} \right\rfloor .
$$

That is right when atoms cost the same. Many do not. Building a lower-triangular matrix —
an overlap matrix, a distance matrix, anything with a loop over $j<k$ inside a loop over $k$
— makes atom $k$ cost about $k$. A molecule spanning $[b_{i-1}, b_i)$ then costs about
$(b_i^2-b_{i-1}^2)/2$, and under linear boundaries molecule $i$ costs $2i-1$ units: the last
of $M$ molecules is $2M-1$ times as expensive as the first.

`PartitionStrategy::Nested`, AFML's `nestedParts`, picks the boundaries that make those costs
equal:

$$
b_i = \operatorname{round}\!\left(N\sqrt{i/M}\right)
\quad\Longrightarrow\quad
\frac{b_i^2-b_{i-1}^2}{2} = \frac{N^2}{2M}\ \text{ for every } i .
$$

Early molecules get many cheap atoms, late molecules a few expensive ones. The book solves a
quadratic for each boundary so that the discrete sums $\sum k$ match; this is the continuous
form of the same rule, and the two differ by an atom or so.

`HpcParallelConfig::mp_batches` is the book's `mpBatches`. The job is split into
`workers × mp_batches` molecules, so with `mp_batches > 1` each thread takes several and a
slow molecule delays only its own thread's share. It is the remedy when costs are uneven
in a way neither strategy predicts.

## From Rust

```rust
use openquant::hpc_parallel::{
    partition_atoms, run_parallel, ExecutionMode, HpcParallelConfig, HpcParallelError,
    MoleculePartition, PartitionStrategy,
};

// Row k of a lower-triangular job touches k cells, so atom k costs k.
let cost = |p: &MoleculePartition| (p.start..p.end).sum::<usize>();

let linear = partition_atoms(1_000, 4, PartitionStrategy::Linear)?;
let nested = partition_atoms(1_000, 4, PartitionStrategy::Nested)?;
assert_eq!(linear.iter().map(|p| p.end).collect::<Vec<_>>(), [250, 500, 750, 1000]);
assert_eq!(nested.iter().map(|p| p.end).collect::<Vec<_>>(), [500, 707, 866, 1000]);
// Equal atom counts, unequal work: the last linear molecule costs 7 times the first.
assert_eq!(linear.iter().map(cost).collect::<Vec<_>>(), [31_125, 93_625, 156_125, 218_625]);
// Nested molecules cost the same to within 0.2%.
assert!(nested.iter().all(|p| (cost(p) as f64 / 124_875.0 - 1.0).abs() < 2e-3));

// The job itself: each molecule returns the sums of its rows, in atom order.
let atoms: Vec<usize> = (0..1_000).collect();
let row_sums =
    |rows: &[usize]| Ok::<Vec<usize>, String>(rows.iter().map(|&k| (0..k).sum()).collect());
let config = |mode| HpcParallelConfig {
    mode,
    partition: PartitionStrategy::Nested,
    mp_batches: 4,
    progress_every: 1,
};
let serial = run_parallel(&atoms, config(ExecutionMode::Serial), row_sums)?;
let threaded =
    run_parallel(&atoms, config(ExecutionMode::Threaded { num_threads: 4 }), row_sums)?;

// Serial counts as one worker, so the two runs cut the job differently...
assert_eq!(serial.metrics.molecules_total, 4);
assert_eq!(threaded.metrics.molecules_total, 16);
// ...but outputs come back in molecule order, so the flattened results are identical.
assert_eq!(serial.outputs.concat(), threaded.outputs.concat());
assert_eq!(threaded.outputs.concat()[999], 999 * 998 / 2);
// The imbalance ratio compares atom counts, not work: 500 atoms against a mean of 250.
assert_eq!(serial.metrics.partition_imbalance_ratio, 2.0);

let failing =
    run_parallel(&atoms, config(ExecutionMode::Threaded { num_threads: 4 }), |rows| {
        if rows.contains(&0) {
            Err("row 0 is bad".to_string())
        } else {
            Ok(rows.len())
        }
    });
assert!(matches!(failing, Err(HpcParallelError::CallbackFailed { molecule_id: 0, .. })));
```

The callback receives a slice of atoms, not one atom, and returns one value per molecule;
putting the pieces together is the caller's job, here a `concat`. That matches the book,
where the callback takes a `molecule` argument and `mpPandasObj` concatenates what comes back.

`dispatch_async` runs the same thing on a background thread and returns a handle, with
`is_finished()` to poll and `wait()` to collect the report.

## What to watch for

- **`Serial` is not the same partition as `Threaded`.** It counts as one worker, so it forms
  `mp_batches` molecules where a threaded run forms `num_threads × mp_batches`. A callback
  whose result depends on where the molecule boundaries fall will give different answers in
  the two modes. The book's single-thread debugging mode has the same property. Compare
  flattened outputs, as the example does, not per-molecule ones.
- **`partition_imbalance_ratio` measures atoms, not work.** It is the largest molecule's atom
  count over the mean. A well-balanced nested partition reports 2.0 in the example, and a
  linear partition of a triangular job reports 1.0 while its last molecule does seven times
  the work of its first.
- **Nested assumes cost rises with the index.** The book's `nestedParts` takes an
  `upperTriang` flag for jobs whose cost *falls* with the index; this module has no such flag.
  For those, reverse the atoms before calling and the outputs afterwards.
- **A failing callback does not cancel the run.** In threaded mode every molecule still
  executes, and the first error to arrive is returned once all have finished; in serial mode
  the run stops at the first error. Either way no partial outputs are returned.
- **A panicking callback is an error, not a crash.** `run_parallel` catches the panic per
  molecule and returns `HpcParallelError::WorkerPanic`, as `dispatch_async`'s `wait()` does.
  The panic message is still printed by the panic hook. In threaded mode the other
  molecules still run, so a callback must not depend on state a panic could leave broken.
- **Progress is recorded, not reported.** `progress_every` controls how often a
  `ProgressSnapshot` is appended to `metrics.progress`, which you read after the run. Nothing
  is printed while it runs, unlike the book's `reportProgress`.
- **Every output is held until the end.** There is no on-the-fly reduction (AFML §20.5.5).
  If each molecule returns something large, return a summary from the callback instead.
- **These are threads, not processes.** Chapter 20's case for multiprocessing is Python's
  global interpreter lock, and so is its advice on pickling. Rust threads share the atoms
  by reference and need neither; the callback must be `Send + Sync`.
- **A nested partition of 7 atoms into 7 molecules used to panic.** Rounding the nested boundaries could push one past
  the end of the atoms, so a nested partition with nearly as many molecules as atoms either
  panicked or came back a molecule short. Every molecule now gets at least one atom.

## Related modules

- [`streaming-hpc`](/modules/streaming-hpc/) — runs many event streams through
  `run_parallel`, one stream per atom.
- [`combinatorial-optimization`](/modules/combinatorial-optimization/) — exhaustive searches,
  the other kind of job Part 5 of the book parallelises.
- [`codependence`](/modules/codependence/) — pairwise distance matrices, a triangular
  workload of exactly the kind nested partitions are for.
