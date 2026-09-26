use chrono::{Duration, NaiveDateTime};
use openquant::backtesting_engine::{
    cpcv_path_count, run_cpcv, run_cross_validation, run_walk_forward, BacktestData, BacktestMode,
    BacktestRunConfig, BacktestSafeguards, CpcvConfig, CrossValidationConfig, WalkForwardConfig,
};

fn build_data(n: usize) -> BacktestData {
    let start = NaiveDateTime::parse_from_str("2024-01-01 09:30:00", "%Y-%m-%d %H:%M:%S")
        .expect("valid start timestamp");
    let mut returns = Vec::with_capacity(n);
    let mut spans = Vec::with_capacity(n);
    for i in 0..n {
        let t0 = start + Duration::minutes(i as i64);
        // Intentionally overlapping spans (3-minute horizon) to force purge logic.
        let t1 = t0 + Duration::minutes(3);
        spans.push((t0, t1));
        let sign = if i % 3 == 0 { -1.0 } else { 1.0 };
        returns.push(sign * (0.001 + i as f64 * 0.0002));
    }
    BacktestData { returns, label_spans: spans }
}

fn run_cfg(mode: BacktestMode) -> BacktestRunConfig {
    BacktestRunConfig {
        mode_provenance: format!("afml_ch11_ch12_{mode:?}"),
        trials_count: 19,
        safeguards: BacktestSafeguards {
            survivorship_bias_control: "Universe frozen at decision timestamp".to_string(),
            look_ahead_control: "Features lagged and label horizon aligned".to_string(),
            data_mining_control: "Validation chosen ex-ante and logged".to_string(),
            cost_assumption: "Linear + spread impact model applied".to_string(),
            multiple_testing_control: "Trials count tracked for post-selection stats".to_string(),
        },
    }
}

fn spans_overlap(lhs: (NaiveDateTime, NaiveDateTime), rhs: (NaiveDateTime, NaiveDateTime)) -> bool {
    lhs.0 <= rhs.1 && rhs.0 <= lhs.1
}

#[test]
fn cpcv_path_count_matches_phi_formula() {
    let phi_6_2 = cpcv_path_count(6, 2).expect("valid CPCV parameters");
    let phi_8_3 = cpcv_path_count(8, 3).expect("valid CPCV parameters");
    assert_eq!(phi_6_2, 5); // C(6,2)*2/6 = 5
    assert_eq!(phi_8_3, 21); // C(8,3)*3/8 = 21
}

#[test]
fn walk_forward_and_cv_produce_explicit_outputs_and_diagnostics() {
    let data = build_data(24);

    let wf = run_walk_forward(
        &data,
        &run_cfg(BacktestMode::WalkForward),
        &WalkForwardConfig { min_train_size: 10, test_size: 4, step_size: 4, pct_embargo: 0.05 },
        |split| {
            Ok(split
                .test_indices
                .iter()
                .map(|idx| data.returns[*idx] + split.train_indices.len() as f64 * 1e-5)
                .collect())
        },
    )
    .expect("walk-forward run should succeed");

    assert!(!wf.folds.is_empty());
    assert_eq!(wf.folds.len(), wf.splits.len());
    assert_eq!(wf.diagnostics.mode, BacktestMode::WalkForward);
    assert_eq!(wf.diagnostics.trials_count, 19);
    assert!(wf.diagnostics.anti_leakage.total_purged > 0);

    let cv = run_cross_validation(
        &data,
        &run_cfg(BacktestMode::CrossValidation),
        &CrossValidationConfig { n_splits: 4, pct_embargo: 0.05 },
        |split| {
            Ok(split
                .test_indices
                .iter()
                .map(|idx| data.returns[*idx] - split.split_id as f64 * 5e-6)
                .collect())
        },
    )
    .expect("cross-validation run should succeed");

    assert_eq!(cv.folds.len(), 4);
    assert_eq!(cv.diagnostics.mode, BacktestMode::CrossValidation);
    assert_eq!(cv.diagnostics.mode_provenance, "afml_ch11_ch12_CrossValidation");
}

#[test]
fn cpcv_enforces_purge_embargo_and_returns_path_distribution() {
    let data = build_data(30);
    let cfg = CpcvConfig { n_groups: 6, test_groups: 2, pct_embargo: 0.1 };

    let result = run_cpcv(
        &data,
        &run_cfg(BacktestMode::CombinatorialPurgedCrossValidation),
        &cfg,
        |split| {
            Ok(split
                .test_indices
                .iter()
                .enumerate()
                .map(|(j, idx)| {
                    // Slight split-dependent perturbation to produce non-degenerate path distribution.
                    data.returns[*idx] + split.split_id as f64 * 2e-5 + j as f64 * 1e-6
                })
                .collect())
        },
    )
    .expect("CPCV run should succeed");

    let expected_phi = cpcv_path_count(cfg.n_groups, cfg.test_groups).expect("valid phi");
    assert_eq!(result.path_count, expected_phi);
    assert_eq!(result.path_assignments.len(), expected_phi);
    assert_eq!(result.path_distribution.len(), expected_phi);
    assert_eq!(result.folds.len(), result.splits.len());
    assert_eq!(result.diagnostics.mode, BacktestMode::CombinatorialPurgedCrossValidation);
    assert_eq!(result.diagnostics.split_count, result.splits.len());

    let mut observed_sharpes =
        result.path_distribution.iter().map(|p| p.sharpe).collect::<Vec<_>>();
    observed_sharpes.sort_by(|a, b| a.partial_cmp(b).expect("sharpe values should be finite"));
    observed_sharpes.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    assert!(
        observed_sharpes.len() >= 2,
        "path distribution should not collapse to one point estimate"
    );

    for split in &result.splits {
        for train_idx in &split.train_indices {
            for test_idx in &split.test_indices {
                let overlaps =
                    spans_overlap(data.label_spans[*train_idx], data.label_spans[*test_idx]);
                assert!(
                    !overlaps,
                    "train index {train_idx} should be purged when overlapping test index {test_idx}"
                );
            }
        }
    }

    assert!(result.diagnostics.anti_leakage.total_purged > 0);
    assert!(result.diagnostics.anti_leakage.total_embargoed > 0);
}

/// #184 item 4: `n_choose_k` multiplied the `u128` numerator unchecked, so these panicked in
/// debug builds and wrapped to a wrong count in release. Now exact where the count fits, and
/// `CombinationCountOverflow` where it does not.
#[test]
fn cpcv_path_count_is_exact_or_reports_overflow() {
    use openquant::backtesting_engine::BacktestError;
    // Values from Python's math.comb(n, k) * k // n.
    assert_eq!(cpcv_path_count(62, 31).unwrap(), 232_714_176_627_630_544);
    assert_eq!(cpcv_path_count(66, 33).unwrap(), 3_609_714_217_008_132_870);
    assert_eq!(cpcv_path_count(67, 33).unwrap(), 7_007_092_303_604_022_630);
    for (n, k) in [(68, 34), (200, 100), (10_000, 5_000)] {
        let out = std::panic::catch_unwind(|| cpcv_path_count(n, k)).expect("must not panic");
        assert_eq!(out.unwrap_err(), BacktestError::CombinationCountOverflow, "({n}, {k})");
    }
}

/// #184 item 4: `start + test_size` and `start += step_size` overflowed (a panic in debug
/// builds) for sizes near `usize::MAX`. They now mean "to the end of the data".
#[test]
fn walk_forward_handles_huge_test_and_step_sizes() {
    let data = build_data(24);
    let config = WalkForwardConfig {
        min_train_size: 10,
        test_size: usize::MAX,
        step_size: usize::MAX,
        pct_embargo: 0.0,
    };
    let wf = std::panic::catch_unwind(|| {
        run_walk_forward(&data, &run_cfg(BacktestMode::WalkForward), &config, |split| {
            Ok(split.test_indices.iter().map(|i| data.returns[*i]).collect())
        })
    })
    .expect("must not panic")
    .expect("one split to the end of the data");
    assert_eq!(wf.splits.len(), 1);
    assert_eq!(wf.splits[0].test_indices, (10..24).collect::<Vec<_>>());
}
