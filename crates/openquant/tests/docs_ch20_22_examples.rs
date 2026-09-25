//! The Rust examples on the hpc_parallel, combinatorial_optimization and streaming_hpc docs
//! pages, run as tests. `check:examples` only compiles page snippets; these execute them.
//!
//! Each body is the page's snippet, reformatted by rustfmt. If you change one, change the other.

#[test]
fn hpc_parallel_page() -> Result<(), Box<dyn std::error::Error>> {
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
    Ok(())
}

#[test]
fn combinatorial_trajectory_page() -> Result<(), Box<dyn std::error::Error>> {
    use openquant::combinatorial_optimization::{
        evaluate_trading_path, solve_trading_trajectory_exact, CombinatorialOptimizationError,
        ObjectiveSense, TradeBounds, TradingTrajectoryObjective, TradingTrajectoryObjectiveConfig,
        TradingTrajectoryPath, TradingTrajectorySchema,
    };

    struct NetPnl(TradingTrajectoryObjectiveConfig);
    impl TradingTrajectoryObjective for NetPnl {
        fn sense(&self) -> ObjectiveSense {
            ObjectiveSense::Maximize
        }
        fn evaluate(
            &self,
            path: &TradingTrajectoryPath,
        ) -> Result<f64, CombinatorialOptimizationError> {
            evaluate_trading_path(path, &self.0)
        }
    }

    let schema = TradingTrajectorySchema {
        initial_inventory: 0,
        inventory_min: 0,
        inventory_max: 3,
        step_trade_bounds: vec![TradeBounds { min_trade: -3, max_trade: 3 }; 5],
        terminal_inventory: Some(0),
        max_paths: 10_000,
    };
    let best_trades = |fixed_ticket_cost| {
        let objective = NetPnl(TradingTrajectoryObjectiveConfig {
            expected_returns: vec![0.02, 0.02, -0.01, 0.02, 0.0],
            risk_aversion: 0.001,
            impact_coefficients: vec![0.002; 5],
            fixed_ticket_cost,
            terminal_inventory_target: 0,
            terminal_inventory_penalty: 0.0,
        });
        solve_trading_trajectory_exact(&schema, &objective)
    };

    // Free to trade: sell out ahead of the dip and buy back after it.
    let free = best_trades(0.0)?;
    assert_eq!(free.best_path.trades, [3, 0, -3, 3, -3]);
    assert_eq!(free.evaluated_paths, 256);
    // Dodging the dip takes two more tickets, and at 0.03 they cost more than it saves.
    let costly = best_trades(0.03)?;
    assert_eq!(costly.best_path.trades, [3, 0, 0, 0, -3]);
    assert_eq!(costly.best_path.inventory_path, [0, 3, 3, 3, 3, 0]);
    // At 0.1 the round trip does not pay for its two tickets.
    assert_eq!(best_trades(0.1)?.best_path.trades, [0, 0, 0, 0, 0]);
    Ok(())
}

#[test]
fn combinatorial_adapter_page() -> Result<(), Box<dyn std::error::Error>> {
    use openquant::combinatorial_optimization::{
        compare_exact_and_adapter, CombinatorialOptimizationError, DecisionSchema,
        IntegerObjective, IntegerVariable, ObjectiveSense, OptimizationResult, SolverAdapter,
    };

    // A smooth bowl with one isolated spike: the kind of surface a local search misses.
    struct Spike;
    impl IntegerObjective for Spike {
        fn sense(&self) -> ObjectiveSense {
            ObjectiveSense::Maximize
        }
        fn evaluate(&self, x: &[i64]) -> Result<f64, CombinatorialOptimizationError> {
            let bowl = -((x[0] - 2).pow(2) + (x[1] - 2).pow(2)) as f64;
            Ok(if x == [8, 8] { bowl + 100.0 } else { bowl })
        }
    }

    // Coordinate ascent from the lower corner: step one variable by one while it helps.
    struct Climb;
    impl SolverAdapter for Climb {
        fn solve(
            &self,
            schema: &DecisionSchema,
            objective: &dyn IntegerObjective,
        ) -> Result<OptimizationResult, CombinatorialOptimizationError> {
            let mut x: Vec<i64> = schema.variables.iter().map(|v| v.lower).collect();
            let mut best = objective.evaluate(&x)?;
            let mut evaluated = 1;
            let mut improved = true;
            while improved {
                improved = false;
                for i in 0..x.len() {
                    for d in [-1, 1] {
                        let mut y = x.clone();
                        y[i] += d;
                        if y[i] < schema.variables[i].lower || y[i] > schema.variables[i].upper {
                            continue;
                        }
                        let value = objective.evaluate(&y)?;
                        evaluated += 1;
                        if value > best {
                            (x, best, improved) = (y, value, true);
                        }
                    }
                }
            }
            Ok(OptimizationResult {
                best_decision: x,
                best_objective: best,
                evaluated_candidates: evaluated,
            })
        }
    }

    let grid = IntegerVariable { lower: 0, upper: 10, step: 1 };
    let schema = DecisionSchema { variables: vec![grid, grid], max_enumeration: 1_000 };
    let report = compare_exact_and_adapter(&schema, &Spike, &Climb)?;

    assert_eq!(report.exact.best_decision, [8, 8]);
    assert_eq!(report.exact.evaluated_candidates, 121);
    assert_eq!(report.adapter.best_decision, [2, 2]); // the top of the bowl
    assert_eq!(report.objective_gap_vs_exact, 28.0);
    Ok(())
}

#[test]
fn streaming_hpc_page() -> Result<(), Box<dyn std::error::Error>> {
    use openquant::hpc_parallel::{ExecutionMode, HpcParallelConfig, PartitionStrategy};
    use openquant::streaming_hpc::{
        generate_synthetic_flash_crash_stream, run_streaming_pipeline_parallel, AlertThresholds,
        HhiConfig, StreamEvent, StreamingEarlyWarningEngine, StreamingHpcError,
        StreamingPipelineConfig, SyntheticStreamConfig, VpinConfig,
    };

    let cfg = StreamingPipelineConfig {
        vpin: VpinConfig { bucket_volume: 1_000.0, support_buckets: 10 },
        hhi: HhiConfig { lookback_events: 50 },
        thresholds: AlertThresholds { vpin: 0.3, hhi: 0.5 },
    };
    let stream_with_crash_at = |fraction: f64| {
        generate_synthetic_flash_crash_stream(SyntheticStreamConfig {
            events: 1_000,
            crash_start_fraction: fraction,
            calm_venues: 4,
            shock_venue: 0,
        })
    };

    // Event by event: the engine holds only its two rolling windows.
    let stream = stream_with_crash_at(0.7)?;
    let mut engine = StreamingEarlyWarningEngine::new(cfg)?;
    let mut first_alert = None;
    for (i, event) in stream.iter().enumerate() {
        if engine.on_event(*event)?.is_alert && first_alert.is_none() {
            first_alert = Some(i);
        }
    }
    assert_eq!(first_alert, Some(729));

    // A bad tick is rejected before it touches the state.
    let bad = StreamEvent { price: f64::NAN, ..stream[0] };
    assert!(matches!(engine.on_event(bad), Err(StreamingHpcError::InvalidEvent(_))));

    // Eight streams, crashing at events 100, 200, ..., 800, on four threads.
    let streams =
        (1..=8).map(|k| stream_with_crash_at(k as f64 / 10.0)).collect::<Result<Vec<_>, _>>()?;
    let parallel = HpcParallelConfig {
        mode: ExecutionMode::Threaded { num_threads: 4 },
        partition: PartitionStrategy::Linear,
        mp_batches: 1,
        progress_every: 1,
    };
    let report = run_streaming_pipeline_parallel(&streams, cfg, parallel)?;
    // One summary per stream, in input order. Each alerts from 29 events after its crash to the end.
    let alerts: Vec<usize> = report.stream_summaries.iter().map(|s| s.alert_count).collect();
    assert_eq!(alerts, [871, 771, 671, 571, 471, 371, 271, 171]);
    assert_eq!(report.stream_summaries[0].latest_hhi, Some(1.0));
    Ok(())
}
