//! AFML Chapter 20: multiprocessing and vectorization utilities.
//!
//! The core abstraction maps a collection of atoms into contiguous molecules and runs
//! user callbacks over those molecules in either serial or threaded mode.

use std::fmt::{Display, Formatter};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Serial,
    Threaded { num_threads: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionStrategy {
    /// Equal-size contiguous chunks.
    Linear,
    /// sqrt-spaced boundaries to balance workloads that get heavier with atom index.
    Nested,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HpcParallelConfig {
    pub mode: ExecutionMode,
    pub partition: PartitionStrategy,
    /// Oversubscription factor, analogous to mpBatches in AFML.
    pub mp_batches: usize,
    /// Emit progress snapshot every N completed molecules.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoleculePartition {
    pub molecule_id: usize,
    pub start: usize,
    pub end: usize,
}

impl MoleculePartition {
    pub fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProgressSnapshot {
    pub completed_molecules: usize,
    pub total_molecules: usize,
    pub elapsed: Duration,
    pub throughput_molecules_per_sec: f64,
    pub throughput_atoms_per_sec: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HpcParallelMetrics {
    pub atoms_total: usize,
    pub molecules_total: usize,
    pub runtime: Duration,
    pub throughput_atoms_per_sec: f64,
    pub throughput_molecules_per_sec: f64,
    /// max molecule size divided by mean molecule size.
    pub partition_imbalance_ratio: f64,
    pub progress: Vec<ProgressSnapshot>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParallelRunReport<R> {
    pub outputs: Vec<R>,
    pub metrics: HpcParallelMetrics,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HpcParallelError {
    InvalidConfig(&'static str),
    CallbackFailed { molecule_id: usize, message: String },
    WorkerPanic,
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

#[derive(Debug)]
pub struct AsyncParallelHandle<R> {
    join_handle: JoinHandle<()>,
    result_rx: mpsc::Receiver<Result<ParallelRunReport<R>, HpcParallelError>>,
}

impl<R> AsyncParallelHandle<R> {
    pub fn is_finished(&self) -> bool {
        self.join_handle.is_finished()
    }

    pub fn wait(self) -> Result<ParallelRunReport<R>, HpcParallelError> {
        if self.join_handle.join().is_err() {
            return Err(HpcParallelError::WorkerPanic);
        }
        self.result_rx
            .recv()
            .map_err(|_| HpcParallelError::ChannelClosed("waiting for async coordinator result"))?
    }
}

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
            PartitionStrategy::Linear => i * atom_count / molecules,
            PartitionStrategy::Nested => {
                ((atom_count as f64) * (i as f64 / molecules as f64).sqrt()).round() as usize
            }
        };
        let last = *boundaries.last().unwrap_or(&0);
        boundaries.push(b.clamp(last + 1, atom_count));
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
        let out = callback(&atoms[part.start..part.end]).map_err(|err| {
            HpcParallelError::CallbackFailed {
                molecule_id: part.molecule_id,
                message: err.to_string(),
            }
        })?;
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
                let res = cb(&atoms[part.start..part.end]).map_err(|err| {
                    HpcParallelError::CallbackFailed {
                        molecule_id: part.molecule_id,
                        message: err.to_string(),
                    }
                });
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
    let should_record =
        completed_molecules == total_molecules || completed_molecules % progress_every == 0;
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
