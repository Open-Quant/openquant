use criterion::{criterion_group, criterion_main, Criterion};
use openquant::hpc_parallel::{run_parallel, ExecutionMode, HpcParallelConfig, PartitionStrategy};
use std::thread;

fn heavy_atoms(n: usize) -> Vec<(usize, f64)> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            (i, (x / 7.0).sin() + (x / 23.0).cos())
        })
        .collect()
}

fn workload(chunk: &[(usize, f64)]) -> f64 {
    let mut acc = 0.0;
    for (idx, x) in chunk {
        // Cost increases with atom index to make partition choice matter.
        let reps = 16 + (*idx / 32);
        let mut local = *x;
        for k in 0..reps {
            local = (local + (k as f64 + 1.0)).sin().abs();
        }
        acc += local;
    }
    acc
}

fn bench_hpc_parallel_scaling(c: &mut Criterion) {
    let atoms = heavy_atoms(4_096);
    let threaded = thread::available_parallelism().map_or(4, |n| n.get().clamp(2, 8));

    c.bench_function("hpc_parallel/serial_linear", |b| {
        b.iter(|| {
            let _ = run_parallel(
                &atoms,
                HpcParallelConfig {
                    mode: ExecutionMode::Serial,
                    partition: PartitionStrategy::Linear,
                    mp_batches: 2,
                    progress_every: 64,
                },
                |chunk| Ok::<f64, &'static str>(workload(chunk)),
            )
            .expect("serial run should succeed");
        });
    });

    c.bench_function("hpc_parallel/threaded_linear", |b| {
        b.iter(|| {
            let _ = run_parallel(
                &atoms,
                HpcParallelConfig {
                    mode: ExecutionMode::Threaded { num_threads: threaded },
                    partition: PartitionStrategy::Linear,
                    mp_batches: 4,
                    progress_every: 64,
                },
                |chunk| Ok::<f64, &'static str>(workload(chunk)),
            )
            .expect("threaded linear run should succeed");
        });
    });

    c.bench_function("hpc_parallel/threaded_nested", |b| {
        b.iter(|| {
            let _ = run_parallel(
                &atoms,
                HpcParallelConfig {
                    mode: ExecutionMode::Threaded { num_threads: threaded },
                    partition: PartitionStrategy::Nested,
                    mp_batches: 4,
                    progress_every: 64,
                },
                |chunk| Ok::<f64, &'static str>(workload(chunk)),
            )
            .expect("threaded nested run should succeed");
        });
    });
}

criterion_group!(hpc_parallel_scaling, bench_hpc_parallel_scaling);
criterion_main!(hpc_parallel_scaling);
