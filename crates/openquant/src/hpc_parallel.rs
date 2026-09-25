//! AFML Chapter 20: multiprocessing and vectorization utilities.
//!
//! The core abstraction maps a collection of atoms into contiguous molecules and runs
//! user callbacks over those molecules in either serial or threaded mode.
//!
//! In AFML's vocabulary (§20.4 Atoms and Molecules) an **atom** is the smallest indivisible
//! task and a **molecule** is a contiguous group of atoms handed to one worker. This module
//! is the Rust counterpart of the book's `mpPandasObj` (§20.5, Snippets 20.7–20.9):
//!
//! - [`partition_atoms`] forms the molecules, either with equal atom counts
//!   ([`PartitionStrategy::Linear`], `linParts`, Snippet 20.5) or with boundaries at
//!   `round(N sqrt(i / M))` so that jobs whose atom `k` costs about `k` get equal work per
//!   molecule ([`PartitionStrategy::Nested`], `nestedParts`, Snippet 20.6; this is the
//!   continuous form of the book's quadratic and can differ from it by an atom or so).
//! - [`run_parallel`] runs a callback over every molecule, serially or on a pool of scoped
//!   threads, and returns one output per molecule **in molecule order**, with timing
//!   metrics. [`dispatch_async`] does the same on a background thread.
//!
//! Conventions: molecules are half-open atom ranges `[start, end)`, contiguous, non-empty
//! and numbered from 0 in atom order. The job is cut into `workers * mp_batches` molecules,
//! where [`ExecutionMode::Serial`] counts as one worker, so serial and threaded runs of the
//! same job use different partitions; compare flattened outputs, not per-molecule ones.
//! Combining the outputs (for example concatenating them) is the caller's job. These are
//! threads, not processes: atoms are shared by reference and the callback must be
//! `Send + Sync`.
//!
//! ```
//! use openquant::hpc_parallel::{
//!     partition_atoms, run_parallel, ExecutionMode, HpcParallelConfig, HpcParallelError,
//!     PartitionStrategy,
//! };
//!
//! # fn main() -> Result<(), HpcParallelError> {
//! // Ten atoms into three molecules.
//! let ends = |s| -> Result<Vec<usize>, HpcParallelError> {
//!     Ok(partition_atoms(10, 3, s)?.iter().map(|p| p.end).collect())
//! };
//! assert_eq!(ends(PartitionStrategy::Linear)?, [3, 6, 10]); // floor(10 i / 3)
//! assert_eq!(ends(PartitionStrategy::Nested)?, [6, 8, 10]); // round(10 sqrt(i / 3))
//!
//! // Sum the squares of 1..=100 on four threads, two molecules per thread.
//! let atoms: Vec<u64> = (1..=100).collect();
//! let cfg = HpcParallelConfig {
//!     mode: ExecutionMode::Threaded { num_threads: 4 },
//!     partition: PartitionStrategy::Linear,
//!     mp_batches: 2,
//!     progress_every: 1,
//! };
//! let report = run_parallel(&atoms, cfg, |chunk: &[u64]| {
//!     Ok::<u64, String>(chunk.iter().map(|x| x * x).sum())
//! })?;
//! assert_eq!(report.outputs.len(), 8);
//! assert_eq!(report.outputs.iter().sum::<u64>(), 338_350);
//! // Outputs are in molecule order whatever order the threads finished in.
//! assert_eq!(report.outputs[0], (1..=12u64).map(|x| x * x).sum::<u64>());
//! # Ok(())
//! # }
//! ```
#![deny(missing_docs)]

use std::fmt::{Display, Formatter};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Where [`run_parallel`] executes the callbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// On the calling thread, one molecule after another; counts as one worker when sizing
    /// the partition. Stops at the first failing molecule.
    Serial,
    /// On a pool of scoped worker threads pulling molecules from a shared queue.
    Threaded {
        /// Number of worker threads; must be `> 0`.
        num_threads: usize,
    },
}

/// How [`partition_atoms`] places molecule boundaries (AFML §20.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionStrategy {
    /// Equal-size contiguous chunks.
    ///
    /// Molecule `i` ends at `floor(i N / M)`; AFML `linParts` (§20.4.1, Snippet 20.5).
    Linear,
    /// sqrt-spaced boundaries to balance workloads that get heavier with atom index.
    ///
    /// Molecule `i` ends at `round(N sqrt(i / M))`, which equalises work when atom `k` costs
    /// about `k`; AFML `nestedParts` (§20.4.2, Snippet 20.6). There is no `upperTriang`
    /// flag: for costs that fall with the index, reverse the atoms and the outputs.
    Nested,
}

/// Configuration of [`run_parallel`] and [`dispatch_async`].
///
/// The default is threaded on [`std::thread::available_parallelism`] threads (1 if
/// unknown), linear partition, `mp_batches = 1` and a snapshot per molecule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HpcParallelConfig {
    /// Serial or threaded execution.
    pub mode: ExecutionMode,
    /// Boundary placement for the molecules.
    pub partition: PartitionStrategy,
    /// Oversubscription factor, analogous to mpBatches in AFML.
    ///
    /// The job is cut into `workers * mp_batches` molecules (capped at the atom count), so
    /// values above 1 let a slow molecule delay only its own thread's share. Must be `> 0`.
    pub mp_batches: usize,
    /// Emit progress snapshot every N completed molecules.
    ///
    /// A snapshot is also recorded when the last molecule completes. Snapshots are stored
    /// in [`HpcParallelMetrics::progress`], not printed. Must be `> 0`.
    pub progress_every: usize,
}

impl Default for HpcParallelConfig {
    fn default() -> Self {
        Self {
            mode: ExecutionMode::Threaded { num_threads: default_threads() },
            partition: PartitionStrategy::Linear,
            mp_batches: 1,
            progress_every: 1,
        }
    }
}

/// One molecule: the half-open atom range `[start, end)` handed to a single callback call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoleculePartition {
    /// Position of the molecule in atom order, from 0; also its index in
    /// [`ParallelRunReport::outputs`].
    pub molecule_id: usize,
    /// First atom index (inclusive).
    pub start: usize,
    /// One past the last atom index (exclusive).
    pub end: usize,
}

impl MoleculePartition {
    /// Number of atoms in the molecule, `end - start` (0 if `end < start`).
    pub fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Whether the molecule holds no atoms. Never true for molecules from
    /// [`partition_atoms`].
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

/// Progress of a run after some number of completed molecules.
///
/// Recorded every [`HpcParallelConfig::progress_every`] completions and at the last one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProgressSnapshot {
    /// Molecules completed so far (successfully or not).
    pub completed_molecules: usize,
    /// Molecules in the run.
    pub total_molecules: usize,
    /// Time since execution started (after partitioning).
    pub elapsed: Duration,
    /// `completed_molecules / elapsed`, or 0 if no time has elapsed.
    pub throughput_molecules_per_sec: f64,
    /// Completed atoms per second, or 0 if no time has elapsed.
    pub throughput_atoms_per_sec: f64,
}

/// Timing and partition statistics of a run.
#[derive(Debug, Clone, PartialEq)]
pub struct HpcParallelMetrics {
    /// Number of atoms processed.
    pub atoms_total: usize,
    /// Number of molecules the atoms were cut into.
    pub molecules_total: usize,
    /// Wall-clock time from the start of execution (after partitioning) to the end.
    pub runtime: Duration,
    /// `atoms_total / runtime`, or 0 if no time has elapsed.
    pub throughput_atoms_per_sec: f64,
    /// `molecules_total / runtime`, or 0 if no time has elapsed.
    pub throughput_molecules_per_sec: f64,
    /// max molecule size divided by mean molecule size.
    ///
    /// Sizes are atom counts, not work: a nested partition that balances a triangular job
    /// reports more than 1, and a linear one reports 1 while its last molecule does the
    /// most work. 0 for an empty run.
    pub partition_imbalance_ratio: f64,
    /// Snapshots in completion order (see [`HpcParallelConfig::progress_every`]).
    pub progress: Vec<ProgressSnapshot>,
}

/// The result of [`run_parallel`]: one output per molecule, in molecule order.
#[derive(Debug, Clone, PartialEq)]
pub struct ParallelRunReport<R> {
    /// Callback outputs indexed by [`MoleculePartition::molecule_id`]; concatenate or reduce
    /// them to rebuild the job's result.
    pub outputs: Vec<R>,
    /// Timing and partition statistics.
    pub metrics: HpcParallelMetrics,
}

/// Errors returned by [`run_parallel`] and [`AsyncParallelHandle::wait`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HpcParallelError {
    /// A configuration value is out of range; the message names it.
    InvalidConfig(&'static str),
    /// The callback returned an error for a molecule.
    CallbackFailed {
        /// The molecule whose callback failed.
        molecule_id: usize,
        /// The callback error, formatted with `Display`.
        message: String,
    },
    /// The callback (or a worker thread) panicked. The panic is caught and reported here by
    /// [`run_parallel`] and by [`AsyncParallelHandle::wait`]; the panic message still goes
    /// through the process's panic hook (by default, printed to stderr).
    WorkerPanic,
    /// An internal channel closed unexpectedly; the message says during which step.
    ChannelClosed(&'static str),
}

impl Display for HpcParallelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid HPC parallel config: {msg}"),
            Self::CallbackFailed { molecule_id, message } => {
                write!(f, "callback failed in molecule {molecule_id}: {message}")
            }
            Self::WorkerPanic => write!(f, "parallel worker panicked"),
            Self::ChannelClosed(ctx) => write!(f, "channel unexpectedly closed while {ctx}"),
        }
    }
}

impl std::error::Error for HpcParallelError {}

/// A handle to a run started by [`dispatch_async`].
#[derive(Debug)]
pub struct AsyncParallelHandle<R> {
    join_handle: JoinHandle<()>,
    result_rx: mpsc::Receiver<Result<ParallelRunReport<R>, HpcParallelError>>,
}

impl<R> AsyncParallelHandle<R> {
    /// Whether the background run has finished, without blocking.
    pub fn is_finished(&self) -> bool {
        self.join_handle.is_finished()
    }

    /// Blocks until the background run finishes and returns its report.
    ///
    /// # Errors
    ///
    /// - [`HpcParallelError::WorkerPanic`] if the background thread panicked (for example
    ///   because the callback panicked).
    /// - [`HpcParallelError::ChannelClosed`] if the thread ended without sending a result.
    /// - Otherwise, any error [`run_parallel`] returned for the run.
    pub fn wait(self) -> Result<ParallelRunReport<R>, HpcParallelError> {
        if self.join_handle.join().is_err() {
            return Err(HpcParallelError::WorkerPanic);
        }
        self.result_rx
            .recv()
            .map_err(|_| HpcParallelError::ChannelClosed("waiting for async coordinator result"))?
    }
}

/// Cuts `atom_count` atoms into `min(target_molecules, atom_count)` contiguous, non-empty
/// molecules (AFML §20.4, Snippets 20.5–20.6).
///
/// Boundaries are `floor(i N / M)` for [`PartitionStrategy::Linear`] and
/// `round(N sqrt(i / M))` for [`PartitionStrategy::Nested`], then clamped so every molecule
/// keeps at least one atom. Molecules are returned in atom order with `molecule_id`
/// `0, 1, ...`.
///
/// # Errors
///
/// [`HpcParallelError::InvalidConfig`] if `target_molecules == 0` and `atom_count > 0`.
/// With `atom_count == 0` the result is an empty vector whatever `target_molecules` is.
///
/// The linear boundary `i * N / M` is formed in `u128`, so it is exact for any counts.
///
/// ```
/// use openquant::hpc_parallel::{partition_atoms, PartitionStrategy};
///
/// // A lower-triangular job: atom k costs k. Nested boundaries equalise the work.
/// let nested = partition_atoms(1_000, 4, PartitionStrategy::Nested).unwrap();
/// assert_eq!(nested.iter().map(|p| p.end).collect::<Vec<_>>(), [500, 707, 866, 1000]);
/// let work: Vec<usize> = nested.iter().map(|p| (p.start..p.end).sum()).collect();
/// assert!(work.iter().all(|&w| (w as f64 / 124_875.0 - 1.0).abs() < 2e-3));
///
/// // More molecules than atoms: one atom each.
/// assert_eq!(partition_atoms(3, 8, PartitionStrategy::Linear).unwrap().len(), 3);
/// ```
pub fn partition_atoms(
    atom_count: usize,
    target_molecules: usize,
    strategy: PartitionStrategy,
) -> Result<Vec<MoleculePartition>, HpcParallelError> {
    if atom_count == 0 {
        return Ok(Vec::new());
    }
    if target_molecules == 0 {
        return Err(HpcParallelError::InvalidConfig("target_molecules must be > 0"));
    }
    let molecules = target_molecules.min(atom_count);
    let mut boundaries = Vec::with_capacity(molecules + 1);
    boundaries.push(0usize);
    for i in 1..molecules {
        let b = match strategy {
            // `i * atom_count` can exceed `usize`; the quotient is below `atom_count`.
            PartitionStrategy::Linear => {
                (i as u128 * atom_count as u128 / molecules as u128) as usize
            }
            PartitionStrategy::Nested => {
                ((atom_count as f64) * (i as f64 / molecules as f64).sqrt()).round() as usize
            }
        };
        // Leave at least one atom for each molecule still to come. Rounding the nested
        // boundaries can otherwise reach `atom_count` early.
        let last = *boundaries.last().unwrap_or(&0);
        boundaries.push(b.clamp(last + 1, atom_count - (molecules - i)));
    }
    boundaries.push(atom_count);

    let mut partitions = Vec::with_capacity(molecules);
    for i in 0..molecules {
        let start = boundaries[i];
        let end = boundaries[i + 1];
        if end > start {
            partitions.push(MoleculePartition { molecule_id: partitions.len(), start, end });
        }
    }
    Ok(partitions)
}

/// Runs `callback` over every molecule of `atoms` and returns the outputs in molecule order
/// (AFML §20.5, `mpPandasObj`, Snippets 20.7–20.9).
///
/// The atoms are cut by [`partition_atoms`] into `workers * cfg.mp_batches` molecules, with
/// one worker in [`ExecutionMode::Serial`] and `num_threads` in
/// [`ExecutionMode::Threaded`]. The callback receives each molecule's slice of atoms and
/// returns one value per molecule. Every output is held until the end; there is no
/// on-the-fly reduction. An empty `atoms` returns an empty report (after the config is
/// validated) without calling the callback.
///
/// # Errors
///
/// - [`HpcParallelError::InvalidConfig`] if `mp_batches`, `progress_every` or
///   `num_threads` is 0.
/// - [`HpcParallelError::CallbackFailed`] if the callback returns an error. In serial mode
///   the run stops at the first failing molecule. In threaded mode every molecule still
///   runs, and the error that arrives first (not necessarily the lowest `molecule_id`) is
///   returned once all have finished. No partial outputs are returned either way.
/// - [`HpcParallelError::WorkerPanic`] if the callback panics. The panic is caught per
///   molecule and treated like a callback error: serial mode stops there, threaded mode
///   finishes the other molecules first. The callback keeps being called for later
///   molecules in threaded mode, so it must not rely on state a panic could leave broken.
/// - [`HpcParallelError::ChannelClosed`] if an internal channel closes unexpectedly.
///
/// # Examples
///
/// See the [module documentation](self).
pub fn run_parallel<A, R, F, E>(
    atoms: &[A],
    cfg: HpcParallelConfig,
    callback: F,
) -> Result<ParallelRunReport<R>, HpcParallelError>
where
    A: Sync,
    R: Send,
    F: Fn(&[A]) -> Result<R, E> + Send + Sync,
    E: Display,
{
    validate_config(cfg)?;
    if atoms.is_empty() {
        return Ok(ParallelRunReport {
            outputs: Vec::new(),
            metrics: HpcParallelMetrics {
                atoms_total: 0,
                molecules_total: 0,
                runtime: Duration::ZERO,
                throughput_atoms_per_sec: 0.0,
                throughput_molecules_per_sec: 0.0,
                partition_imbalance_ratio: 0.0,
                progress: Vec::new(),
            },
        });
    }

    let worker_count = match cfg.mode {
        ExecutionMode::Serial => 1,
        ExecutionMode::Threaded { num_threads } => num_threads,
    };
    let target_molecules = worker_count.saturating_mul(cfg.mp_batches).max(1);
    let partitions = partition_atoms(atoms.len(), target_molecules, cfg.partition)?;
    let started = Instant::now();

    let (outputs, progress) = match cfg.mode {
        ExecutionMode::Serial => run_serial(atoms, &partitions, cfg.progress_every, &callback)?,
        ExecutionMode::Threaded { num_threads } => {
            run_threaded(atoms, &partitions, cfg.progress_every, num_threads, callback)?
        }
    };

    Ok(ParallelRunReport {
        outputs,
        metrics: build_metrics(atoms.len(), &partitions, started.elapsed(), progress),
    })
}

/// Starts [`run_parallel`] on a background thread and returns a handle to it.
///
/// The atoms are moved into the thread. Poll with [`AsyncParallelHandle::is_finished`] and
/// collect the report with [`AsyncParallelHandle::wait`], which reports a panicking callback
/// as [`HpcParallelError::WorkerPanic`], as [`run_parallel`] does. Configuration errors are reported by
/// `wait`, not here.
///
/// ```
/// use openquant::hpc_parallel::{dispatch_async, ExecutionMode, HpcParallelConfig, PartitionStrategy};
///
/// let cfg = HpcParallelConfig {
///     mode: ExecutionMode::Threaded { num_threads: 2 },
///     partition: PartitionStrategy::Linear,
///     mp_batches: 2,
///     progress_every: 1,
/// };
/// let handle = dispatch_async((1..=10u32).collect(), cfg, |chunk: &[u32]| {
///     Ok::<u32, String>(chunk.iter().sum())
/// });
/// let report = handle.wait().unwrap();
/// assert_eq!(report.outputs.len(), 4);
/// assert_eq!(report.outputs.iter().sum::<u32>(), 55);
/// assert_eq!(report.metrics.progress.last().unwrap().completed_molecules, 4);
/// ```
pub fn dispatch_async<A, R, F, E>(
    atoms: Vec<A>,
    cfg: HpcParallelConfig,
    callback: F,
) -> AsyncParallelHandle<R>
where
    A: Send + Sync + 'static,
    R: Send + 'static,
    F: Fn(&[A]) -> Result<R, E> + Send + Sync + 'static,
    E: Display + Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    let join_handle = thread::spawn(move || {
        let report = run_parallel(&atoms, cfg, callback);
        let _ = tx.send(report);
    });
    AsyncParallelHandle { join_handle, result_rx: rx }
}

/// Runs the callback on one molecule, turning a returned error into
/// [`HpcParallelError::CallbackFailed`] and a panic into [`HpcParallelError::WorkerPanic`].
fn call_molecule<A, R, F, E>(
    callback: &F,
    atoms: &[A],
    part: &MoleculePartition,
) -> Result<R, HpcParallelError>
where
    F: Fn(&[A]) -> Result<R, E>,
    E: Display,
{
    // The callback only sees a shared slice; any state it keeps across a panic is its own.
    match catch_unwind(AssertUnwindSafe(|| callback(&atoms[part.start..part.end]))) {
        Ok(Ok(out)) => Ok(out),
        Ok(Err(err)) => Err(HpcParallelError::CallbackFailed {
            molecule_id: part.molecule_id,
            message: err.to_string(),
        }),
        Err(_) => Err(HpcParallelError::WorkerPanic),
    }
}

fn run_serial<A, R, F, E>(
    atoms: &[A],
    partitions: &[MoleculePartition],
    progress_every: usize,
    callback: &F,
) -> Result<(Vec<R>, Vec<ProgressSnapshot>), HpcParallelError>
where
    F: Fn(&[A]) -> Result<R, E>,
    E: Display,
{
    let total = partitions.len();
    let started = Instant::now();
    let mut outputs = Vec::with_capacity(total);
    let mut progress = Vec::new();
    let mut completed_atoms = 0usize;

    for part in partitions {
        let out = call_molecule(callback, atoms, part)?;
        outputs.push(out);
        completed_atoms += part.len();
        maybe_record_progress(
            &mut progress,
            started.elapsed(),
            outputs.len(),
            total,
            completed_atoms,
            progress_every,
        );
    }
    Ok((outputs, progress))
}

fn run_threaded<A, R, F, E>(
    atoms: &[A],
    partitions: &[MoleculePartition],
    progress_every: usize,
    num_threads: usize,
    callback: F,
) -> Result<(Vec<R>, Vec<ProgressSnapshot>), HpcParallelError>
where
    A: Sync,
    R: Send,
    F: Fn(&[A]) -> Result<R, E> + Send + Sync,
    E: Display,
{
    let started = Instant::now();
    let total = partitions.len();
    let callback = Arc::new(callback);

    let (job_tx, job_rx) = mpsc::channel::<MoleculePartition>();
    let job_rx = Arc::new(Mutex::new(job_rx));
    let (result_tx, result_rx) =
        mpsc::channel::<(MoleculePartition, Result<R, HpcParallelError>)>();

    thread::scope(|scope| {
        let mut workers = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            let rx = Arc::clone(&job_rx);
            let tx = result_tx.clone();
            let cb = Arc::clone(&callback);
            workers.push(scope.spawn(move || loop {
                let next = {
                    let guard = rx.lock().expect("job receiver mutex should not be poisoned");
                    guard.recv()
                };
                let part = match next {
                    Ok(part) => part,
                    Err(_) => break,
                };
                let res = call_molecule(&*cb, atoms, &part);
                if tx.send((part, res)).is_err() {
                    break;
                }
            }));
        }
        drop(result_tx);

        for part in partitions {
            if job_tx.send(*part).is_err() {
                return Err(HpcParallelError::ChannelClosed("queueing jobs"));
            }
        }
        drop(job_tx);

        let mut ordered: Vec<Option<R>> = (0..total).map(|_| None).collect();
        let mut progress = Vec::new();
        let mut first_error: Option<HpcParallelError> = None;
        let mut completed = 0usize;
        let mut completed_atoms = 0usize;
        for _ in 0..total {
            let (part, outcome) = result_rx
                .recv()
                .map_err(|_| HpcParallelError::ChannelClosed("receiving worker results"))?;
            completed += 1;
            completed_atoms += part.len();
            match outcome {
                Ok(out) => ordered[part.molecule_id] = Some(out),
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
            maybe_record_progress(
                &mut progress,
                started.elapsed(),
                completed,
                total,
                completed_atoms,
                progress_every,
            );
        }

        for worker in workers {
            if worker.join().is_err() {
                return Err(HpcParallelError::WorkerPanic);
            }
        }
        if let Some(err) = first_error {
            return Err(err);
        }

        let mut outputs = Vec::with_capacity(total);
        for maybe in ordered {
            outputs.push(
                maybe
                    .ok_or(HpcParallelError::ChannelClosed("assembling ordered worker results"))?,
            );
        }
        Ok((outputs, progress))
    })
}

fn maybe_record_progress(
    progress: &mut Vec<ProgressSnapshot>,
    elapsed: Duration,
    completed_molecules: usize,
    total_molecules: usize,
    completed_atoms: usize,
    progress_every: usize,
) {
    let should_record = completed_molecules == total_molecules
        || completed_molecules.is_multiple_of(progress_every);
    if !should_record {
        return;
    }
    let seconds = elapsed.as_secs_f64();
    let throughput_molecules_per_sec =
        if seconds > 0.0 { completed_molecules as f64 / seconds } else { 0.0 };
    let throughput_atoms_per_sec =
        if seconds > 0.0 { completed_atoms as f64 / seconds } else { 0.0 };
    progress.push(ProgressSnapshot {
        completed_molecules,
        total_molecules,
        elapsed,
        throughput_molecules_per_sec,
        throughput_atoms_per_sec,
    });
}

fn build_metrics(
    atom_count: usize,
    partitions: &[MoleculePartition],
    runtime: Duration,
    progress: Vec<ProgressSnapshot>,
) -> HpcParallelMetrics {
    let molecules = partitions.len();
    let seconds = runtime.as_secs_f64();
    let throughput_atoms_per_sec = if seconds > 0.0 { atom_count as f64 / seconds } else { 0.0 };
    let throughput_molecules_per_sec = if seconds > 0.0 { molecules as f64 / seconds } else { 0.0 };

    let sizes: Vec<usize> = partitions.iter().map(|p| p.len()).collect();
    let mean = if sizes.is_empty() {
        0.0
    } else {
        sizes.iter().sum::<usize>() as f64 / sizes.len() as f64
    };
    let max = sizes.iter().copied().max().unwrap_or(0) as f64;
    let partition_imbalance_ratio = if mean > 0.0 { max / mean } else { 0.0 };

    HpcParallelMetrics {
        atoms_total: atom_count,
        molecules_total: molecules,
        runtime,
        throughput_atoms_per_sec,
        throughput_molecules_per_sec,
        partition_imbalance_ratio,
        progress,
    }
}

fn validate_config(cfg: HpcParallelConfig) -> Result<(), HpcParallelError> {
    if cfg.mp_batches == 0 {
        return Err(HpcParallelError::InvalidConfig("mp_batches must be > 0"));
    }
    if cfg.progress_every == 0 {
        return Err(HpcParallelError::InvalidConfig("progress_every must be > 0"));
    }
    if let ExecutionMode::Threaded { num_threads } = cfg.mode {
        if num_threads == 0 {
            return Err(HpcParallelError::InvalidConfig("num_threads must be > 0"));
        }
    }
    Ok(())
}

fn default_threads() -> usize {
    thread::available_parallelism().map_or(1, |n| n.get().max(1))
}
