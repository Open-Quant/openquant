use openquant::hpc_parallel::{
    dispatch_async, partition_atoms, run_parallel, ExecutionMode, HpcParallelConfig,
    HpcParallelError, PartitionStrategy,
};

fn config(
    mode: ExecutionMode,
    partition: PartitionStrategy,
    mp_batches: usize,
) -> HpcParallelConfig {
    HpcParallelConfig { mode, partition, mp_batches, progress_every: 1 }
}

#[test]
fn linear_partition_covers_all_atoms_without_gaps() {
    let parts = partition_atoms(25, 4, PartitionStrategy::Linear).expect("valid partition");
    assert_eq!(parts.len(), 4);
    assert_eq!(parts.first().expect("non-empty").start, 0);
    assert_eq!(parts.last().expect("non-empty").end, 25);
    for w in parts.windows(2) {
        assert_eq!(w[0].end, w[1].start);
    }
}

#[test]
fn nested_partition_biases_molecule_sizes() {
    let parts = partition_atoms(100, 4, PartitionStrategy::Nested).expect("valid partition");
    assert_eq!(parts.len(), 4);
    assert!(parts[0].len() > parts[3].len());
}

#[test]
fn serial_and_threaded_results_match() {
    let atoms: Vec<u64> = (1..=512).collect();
    let serial = run_parallel(
        &atoms,
        config(ExecutionMode::Serial, PartitionStrategy::Linear, 8),
        |chunk| Ok::<u64, &'static str>(chunk.iter().copied().sum()),
    )
    .expect("serial run");

    let threaded = run_parallel(
        &atoms,
        config(ExecutionMode::Threaded { num_threads: 4 }, PartitionStrategy::Linear, 2),
        |chunk| Ok::<u64, &'static str>(chunk.iter().copied().sum()),
    )
    .expect("threaded run");

    assert_eq!(serial.outputs.iter().sum::<u64>(), threaded.outputs.iter().sum::<u64>());
    assert_eq!(threaded.metrics.molecules_total, 8);
    assert_eq!(
        threaded.metrics.progress.last().expect("progress snapshots").completed_molecules,
        threaded.metrics.molecules_total
    );
}

#[test]
fn mp_batches_controls_work_granularity() {
    let atoms: Vec<i32> = (0..1000).collect();
    let low = run_parallel(
        &atoms,
        config(ExecutionMode::Threaded { num_threads: 4 }, PartitionStrategy::Linear, 1),
        |chunk| Ok::<usize, &'static str>(chunk.len()),
    )
    .expect("low batches");
    let high = run_parallel(
        &atoms,
        config(ExecutionMode::Threaded { num_threads: 4 }, PartitionStrategy::Linear, 4),
        |chunk| Ok::<usize, &'static str>(chunk.len()),
    )
    .expect("high batches");

    assert_eq!(low.metrics.molecules_total, 4);
    assert_eq!(high.metrics.molecules_total, 16);
}

#[test]
fn async_dispatch_returns_report() {
    let atoms: Vec<f64> = (0..2048).map(|v| v as f64).collect();
    let handle = dispatch_async(
        atoms,
        config(ExecutionMode::Threaded { num_threads: 4 }, PartitionStrategy::Nested, 2),
        |chunk| {
            let mut acc = 0.0;
            for x in chunk {
                acc += x.sin().abs();
            }
            Ok::<f64, &'static str>(acc)
        },
    );
    let report = handle.wait().expect("async report");
    assert_eq!(report.metrics.molecules_total, 8);
    assert_eq!(report.outputs.len(), 8);
}

#[test]
fn callback_error_is_reported_with_molecule_context() {
    let atoms: Vec<i32> = (0..40).collect();
    let err = run_parallel(
        &atoms,
        config(ExecutionMode::Serial, PartitionStrategy::Linear, 4),
        |chunk| {
            if chunk.iter().any(|v| *v >= 20) {
                return Err("synthetic failure");
            }
            Ok::<i32, &'static str>(chunk.iter().sum())
        },
    )
    .expect_err("expected callback failure");

    match err {
        HpcParallelError::CallbackFailed { molecule_id, .. } => assert!(molecule_id >= 2),
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn invalid_config_rejected() {
    let atoms = vec![1, 2, 3];
    let err = run_parallel(
        &atoms,
        HpcParallelConfig {
            mode: ExecutionMode::Threaded { num_threads: 0 },
            partition: PartitionStrategy::Linear,
            mp_batches: 1,
            progress_every: 1,
        },
        |_chunk| Ok::<usize, &'static str>(1),
    )
    .expect_err("invalid thread count");
    assert!(matches!(err, HpcParallelError::InvalidConfig(_)));
}
