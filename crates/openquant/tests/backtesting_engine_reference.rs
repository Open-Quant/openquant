//! Value and property tests for `backtesting_engine`. Expected values are hand-worked in the
//! comments or recomputed here by brute force from the AFML definitions (chapters 7 and 12);
//! nothing is copied from the library's output.
//!
//! The pre-existing `tests/backtesting_engine.rs` checks counts and "something was purged"; see
//! `docs/test-sensitivity-audit.md`.

use chrono::{Duration, NaiveDate, NaiveDateTime};
use openquant::backtesting_engine::{
    cpcv_path_count, run_cpcv, run_cross_validation, run_walk_forward, BacktestData, BacktestError,
    BacktestRunConfig, BacktestSafeguards, CpcvConfig, CrossValidationConfig, SplitDefinition,
    WalkForwardConfig,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

type Span = (NaiveDateTime, NaiveDateTime);

fn day(d: i64) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2024, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap() + Duration::days(d)
}

fn run_cfg() -> BacktestRunConfig {
    BacktestRunConfig {
        mode_provenance: "reference-tests".to_string(),
        trials_count: 1,
        safeguards: BacktestSafeguards {
            survivorship_bias_control: "n/a (synthetic)".to_string(),
            look_ahead_control: "n/a (synthetic)".to_string(),
            data_mining_control: "n/a (synthetic)".to_string(),
            cost_assumption: "n/a (synthetic)".to_string(),
            multiple_testing_control: "n/a (synthetic)".to_string(),
        },
    }
}

/// `n` labels that each start and end on their own day: no two spans overlap, so nothing may be
/// purged and any train/test difference is due to the split logic or the embargo alone.
fn point_labels(n: usize) -> BacktestData {
    BacktestData {
        returns: (0..n).map(|i| i as f64).collect(),
        label_spans: (0..n).map(|i| (day(i as i64), day(i as i64))).collect(),
    }
}

/// Labels with random integer-day starts (sorted) and random horizons of 0..=6 days. Integer
/// endpoints make "span A ends the instant span B starts" common, which is the boundary case a
/// closed-interval overlap test (AFML snippet 7.1 uses `<=` on both sides) must get right.
fn random_labels(n: usize, seed: u64) -> BacktestData {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut t = 0i64;
    let mut spans = Vec::with_capacity(n);
    for _ in 0..n {
        t += rng.gen_range(0..3);
        spans.push((day(t), day(t + rng.gen_range(0..7))));
    }
    BacktestData { returns: (0..n).map(|i| i as f64).collect(), label_spans: spans }
}

fn closed_overlap(a: Span, b: Span) -> bool {
    a.0 <= b.1 && b.0 <= a.1
}

/// Brute-force statement of purging: a candidate stays in train iff its label span shares no
/// instant with the span of any test label.
fn expected_purged_train(candidates: &[usize], test: &[usize], spans: &[Span]) -> Vec<usize> {
    candidates
        .iter()
        .copied()
        .filter(|i| !test.iter().any(|t| closed_overlap(spans[*i], spans[*t])))
        .collect()
}

fn echo_test_indices(split: &SplitDefinition) -> Result<Vec<f64>, BacktestError> {
    Ok(split.test_indices.iter().map(|i| *i as f64).collect())
}

/// C(n, k) by Pascal's rule, independent of the library's multiplicative formula.
fn pascal(n: usize, k: usize) -> usize {
    if k == 0 || k == n {
        1
    } else {
        pascal(n - 1, k - 1) + pascal(n - 1, k)
    }
}

/// AFML 12.4.1: CPCV with N groups and k test groups has C(N, k) splits and
/// phi = k/N * C(N, k) = C(N-1, k-1) backtest paths.
#[test]
fn cpcv_split_and_path_counts_match_combinatorics() {
    for n_groups in 2..=9usize {
        for k in 1..n_groups {
            assert_eq!(
                cpcv_path_count(n_groups, k).unwrap(),
                pascal(n_groups - 1, k - 1),
                "phi({n_groups},{k})"
            );
            let data = point_labels(2 * n_groups);
            let cfg = CpcvConfig { n_groups, test_groups: k, pct_embargo: 0.0 };
            let res = run_cpcv(&data, &run_cfg(), &cfg, echo_test_indices).unwrap();
            assert_eq!(res.splits.len(), pascal(n_groups, k), "splits({n_groups},{k})");
            assert_eq!(res.path_distribution.len(), pascal(n_groups - 1, k - 1));
        }
    }
}

/// Fold statistics, hand-worked. Out-of-sample returns (0.02, -0.01, 0.03, 0.00):
///   mean = 0.04 / 4 = 0.01
///   deviations (0.01, -0.02, 0.02, -0.01), squares sum to 1e-3
///   sample variance (ddof = 1) = 1e-3 / 3, std = sqrt(1e-3 / 3) = 0.018257418583505537
///   the engine's per-fold "sharpe" is the t-statistic mean / std * sqrt(n):
///       (0.01 * 2)^2 / (1e-3 / 3) = 1.2  ->  sharpe = sqrt(1.2) = 1.0954451150103321
#[test]
fn fold_performance_hand_worked() {
    let data = point_labels(8);
    let cfg = CrossValidationConfig { n_splits: 2, pct_embargo: 0.0 };
    let res = run_cross_validation(&data, &run_cfg(), &cfg, |_| Ok(vec![0.02, -0.01, 0.03, 0.00]))
        .unwrap();
    for fold in &res.folds {
        assert_eq!(fold.observations, 4);
        // a handful of additions and one sqrt on O(1e-2) numbers: 1e-15 is rounding
        assert!((fold.mean_return - 0.01).abs() < 1e-15, "mean {}", fold.mean_return);
        assert!(
            (fold.std_return - (1e-3f64 / 3.0).sqrt()).abs() < 1e-15,
            "std {}",
            fold.std_return
        );
        assert!((fold.sharpe - 1.2f64.sqrt()).abs() < 1e-13, "sharpe {}", fold.sharpe);
    }
}

/// Walk-forward, hand-worked: 10 samples, min_train 4, test 2, step 2, no overlap, no embargo.
///   split 0: train 0..4, test 4..6
///   split 1: train 0..6, test 6..8
///   split 2: train 0..8, test 8..10
#[test]
fn walk_forward_splits_hand_worked() {
    let data = point_labels(10);
    let cfg = WalkForwardConfig { min_train_size: 4, test_size: 2, step_size: 2, pct_embargo: 0.0 };
    let res = run_walk_forward(&data, &run_cfg(), &cfg, echo_test_indices).unwrap();
    let got: Vec<(Vec<usize>, Vec<usize>)> =
        res.splits.iter().map(|s| (s.train_indices.clone(), s.test_indices.clone())).collect();
    let want = vec![
        ((0..4).collect::<Vec<_>>(), vec![4, 5]),
        ((0..6).collect(), vec![6, 7]),
        ((0..8).collect(), vec![8, 9]),
    ];
    assert_eq!(got, want);
    assert_eq!(res.diagnostics.anti_leakage.total_purged, 0);
    // echoing the test indices back: fold 1 sees (6, 7) -> mean 6.5, std sqrt(0.5)
    assert_eq!(res.folds[1].mean_return, 6.5);
    assert!((res.folds[1].std_return - 0.5f64.sqrt()).abs() < 1e-15);
}

/// Purging must be exact, not merely "some": for every split of every mode, train equals the
/// brute-force set {candidates whose span touches no test span}, and `purged_count` is the
/// number removed. Embargo is off so purging is the only effect.
#[test]
fn purging_is_exact_for_random_label_spans() {
    for seed in 0..20u64 {
        let n = 60;
        let data = random_labels(n, seed);
        let spans = &data.label_spans;
        let all: Vec<usize> = (0..n).collect();

        let check = |splits: &[SplitDefinition], walk_forward: bool| {
            for s in splits {
                let candidates: Vec<usize> = all
                    .iter()
                    .copied()
                    .filter(|i| !s.test_indices.contains(i))
                    .filter(|i| !walk_forward || *i < s.test_indices[0])
                    .collect();
                let want = expected_purged_train(&candidates, &s.test_indices, spans);
                assert_eq!(s.train_indices, want, "seed {seed} split {}", s.split_id);
                assert_eq!(s.purged_count, candidates.len() - want.len(), "seed {seed}");
                assert_eq!(s.embargo_count, 0);
            }
        };

        let cv = CrossValidationConfig { n_splits: 5, pct_embargo: 0.0 };
        check(
            &run_cross_validation(&data, &run_cfg(), &cv, echo_test_indices).unwrap().splits,
            false,
        );

        let cpcv = CpcvConfig { n_groups: 6, test_groups: 2, pct_embargo: 0.0 };
        check(&run_cpcv(&data, &run_cfg(), &cpcv, echo_test_indices).unwrap().splits, false);

        let wf = WalkForwardConfig {
            min_train_size: 30,
            test_size: 10,
            step_size: 10,
            pct_embargo: 0.0,
        };
        check(&run_walk_forward(&data, &run_cfg(), &wf, echo_test_indices).unwrap().splits, true);
    }
}

/// Embargo after the test block, hand-worked. 20 point labels, 4 folds of 5, pct_embargo = 0.1:
/// h = 0.1 * 20 = 2 samples (exactly 2.0 in floating point, so floor and ceil agree).
/// First fold: test 0..5, the two samples after it (5, 6) are embargoed, train = 7..20.
#[test]
fn embargo_after_test_block_hand_worked() {
    let data = point_labels(20);
    let cfg = CrossValidationConfig { n_splits: 4, pct_embargo: 0.1 };
    let res = run_cross_validation(&data, &run_cfg(), &cfg, echo_test_indices).unwrap();
    let first = &res.splits[0];
    assert_eq!(first.test_indices, (0..5).collect::<Vec<_>>());
    assert_eq!(first.train_indices, (7..20).collect::<Vec<_>>());
    assert_eq!(first.embargo_count, 2);
    assert_eq!(first.purged_count, 0);
}

/// AFML 7.4.2: "we do not need to affect the training observations prior to a test set" - the
/// embargo only follows the test block. Second fold: test 5..10, embargo removes 10 and 11,
/// train = {0..5} + {12..20}.
#[test]
fn embargo_does_not_remove_samples_before_the_test_block() {
    let data = point_labels(20);
    let cfg = CrossValidationConfig { n_splits: 4, pct_embargo: 0.1 };
    let res = run_cross_validation(&data, &run_cfg(), &cfg, echo_test_indices).unwrap();
    let second = &res.splits[1];
    assert_eq!(second.test_indices, (5..10).collect::<Vec<_>>());
    let want: Vec<usize> = (0..5).chain(12..20).collect();
    assert_eq!(second.train_indices, want);
    assert_eq!(second.embargo_count, 2);
}

/// Labels that each span three days (start on day i, end on day i + 2), so a label overlaps the
/// two before and the two after it.
fn three_day_labels(n: usize) -> BacktestData {
    BacktestData {
        returns: (0..n).map(|i| i as f64).collect(),
        label_spans: (0..n).map(|i| (day(i as i64), day(i as i64 + 2))).collect(),
    }
}

/// Snippet 7.3 resumes training `h` samples after the purge, not after the test block. 20
/// three-day labels, 4 folds, h = 2; second fold tests 5..10 (latest label end: day 11).
///   purge: 3 and 4 end on days 5 and 6, inside the test window; 10 and 11 start on days 10
///          and 11, inside it too. 12 starts on day 12, after it.
///   embargo: the two samples from 12 on, i.e. 12 and 13. Nothing before the block.
/// So train = {0, 1, 2} + {14..20}, 4 purged and 2 embargoed. (Counting h from the block's edge
/// would embargo 10 and 11, which the purge already removed, and keep 12 and 13.)
#[test]
fn embargo_starts_where_the_purge_ends() {
    let data = three_day_labels(20);
    let cfg = CrossValidationConfig { n_splits: 4, pct_embargo: 0.1 };
    let res = run_cross_validation(&data, &run_cfg(), &cfg, echo_test_indices).unwrap();
    let second = &res.splits[1];
    assert_eq!(second.test_indices, (5..10).collect::<Vec<_>>());
    let want: Vec<usize> = (0..3).chain(14..20).collect();
    assert_eq!(second.train_indices, want);
    assert_eq!(second.purged_count, 4);
    assert_eq!(second.embargo_count, 2);
}

/// In walk-forward every training sample precedes the test block, so an embargo that only
/// follows test blocks removes nothing. 20 point labels, first 10 train, h = 2.
#[test]
fn walk_forward_embargo_removes_nothing() {
    let data = point_labels(20);
    let cfg =
        WalkForwardConfig { min_train_size: 10, test_size: 5, step_size: 5, pct_embargo: 0.1 };
    let res = run_walk_forward(&data, &run_cfg(), &cfg, echo_test_indices).unwrap();
    for s in &res.splits {
        assert_eq!(s.train_indices, (0..s.test_indices[0]).collect::<Vec<_>>());
        assert_eq!(s.embargo_count, 0);
    }
}

/// CPCV embargoes after each run of adjacent test groups. 12 point labels in 6 groups of 2,
/// k = 2, h = ceil(0.05 * 12) = 1. Splits are the pairs in lexicographic order.
///   split 0, groups (0, 1): test 0..4, one block; embargo 4. train = 5..12.
///   split 1, groups (0, 2): test {0, 1, 4, 5}, two blocks; embargo 2 and 6. train = {3} + 7..12.
#[test]
fn cpcv_embargo_follows_each_test_block() {
    let data = point_labels(12);
    let cfg = CpcvConfig { n_groups: 6, test_groups: 2, pct_embargo: 0.05 };
    let res = run_cpcv(&data, &run_cfg(), &cfg, echo_test_indices).unwrap();
    assert_eq!(res.splits[0].train_indices, (5..12).collect::<Vec<_>>());
    assert_eq!(res.splits[0].embargo_count, 1);
    let want: Vec<usize> = std::iter::once(3).chain(7..12).collect();
    assert_eq!(res.splits[1].test_indices, vec![0, 1, 4, 5]);
    assert_eq!(res.splits[1].train_indices, want);
    assert_eq!(res.splits[1].embargo_count, 2);
}

/// CPCV paths (AFML figure 12.1). N = 6 groups of 2 samples, k = 2: splits are the 15 pairs
/// (i, j), i < j, in lexicographic order. Group g is tested in 5 splits; path p takes group g's
/// forecasts from the p-th of those (in split order). The evaluator returns
/// `1000 * split_id + index`, so each path value reveals which split supplied each sample.
#[test]
fn cpcv_paths_follow_afml_assignment_and_cover_every_sample_once() {
    let (n_groups, per_group) = (6usize, 2usize);
    let n = n_groups * per_group;
    let data = point_labels(n);
    let cfg = CpcvConfig { n_groups, test_groups: 2, pct_embargo: 0.0 };
    let res = run_cpcv(&data, &run_cfg(), &cfg, |s: &SplitDefinition| {
        Ok(s.test_indices.iter().map(|i| (1000 * s.split_id + *i) as f64).collect())
    })
    .unwrap();

    // independent enumeration of the splits
    let mut pairs = Vec::new();
    for i in 0..n_groups {
        for j in (i + 1)..n_groups {
            pairs.push((i, j));
        }
    }
    assert_eq!(pairs.len(), 15);
    for (split_id, (i, j)) in pairs.iter().enumerate() {
        let s = &res.splits[split_id];
        let want_test: Vec<usize> =
            (0..n).filter(|idx| idx / per_group == *i || idx / per_group == *j).collect();
        let want_train: Vec<usize> = (0..n).filter(|idx| !want_test.contains(idx)).collect();
        assert_eq!(s.test_indices, want_test, "split {split_id}");
        assert_eq!(s.train_indices, want_train, "split {split_id}");
    }

    assert_eq!(res.path_distribution.len(), 5);
    let mut grand_total = 0.0;
    for (p, path) in res.path_distribution.iter().enumerate() {
        // every path is a full backtest: each of the 12 samples exactly once
        assert_eq!(path.observations, n, "path {p}");
        let mut want_sum = 0.0;
        for g in 0..n_groups {
            let splits_with_g: Vec<usize> =
                (0..pairs.len()).filter(|s| pairs[*s].0 == g || pairs[*s].1 == g).collect();
            assert_eq!(splits_with_g.len(), 5);
            for idx in (g * per_group)..((g + 1) * per_group) {
                want_sum += (1000 * splits_with_g[p] + idx) as f64;
            }
        }
        // integers below 2^53 summed and divided by 12: one rounding
        assert!(
            (path.mean_return - want_sum / n as f64).abs() < 1e-9,
            "path {p}: mean {} want {}",
            path.mean_return,
            want_sum / n as f64
        );
        grand_total += path.mean_return * n as f64;
    }

    // convention-free invariant: the paths together use every (split, test sample) forecast
    // exactly once, so their grand total equals the total over all splits.
    let all_splits_total: f64 = res
        .splits
        .iter()
        .map(|s| s.test_indices.iter().map(|i| (1000 * s.split_id + *i) as f64).sum::<f64>())
        .sum();
    assert!((grand_total - all_splits_total).abs() < 1e-6, "{grand_total} vs {all_splits_total}");
}
