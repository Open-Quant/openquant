use chrono::NaiveDateTime;
use openquant::cross_validation::{
    count_train_test_overlaps, ml_cross_val_score, ml_get_train_times, naive_kfold_splits,
    CpcvSplit, CrossValidationError, PurgedKFold, Scoring, SimpleClassifier,
};

fn make_series(
    start: &str,
    periods: usize,
    freq_minutes: i64,
) -> Vec<(NaiveDateTime, NaiveDateTime)> {
    let start_dt = NaiveDateTime::parse_from_str(start, "%Y-%m-%d %H:%M:%S").unwrap();
    (0..periods)
        .map(|i| {
            let idx = start_dt + chrono::Duration::minutes(i as i64 * freq_minutes);
            let val = idx + chrono::Duration::minutes(2);
            (idx, val)
        })
        .collect()
}

#[test]
fn test_get_train_times_cases() {
    let info_sets = make_series("2019-01-01 00:00:00", 10, 1);

    // case 1: train starts within test
    let test_times = vec![(
        NaiveDateTime::parse_from_str("2019-01-01 00:01:00", "%Y-%m-%d %H:%M:%S").unwrap(),
        NaiveDateTime::parse_from_str("2019-01-01 00:02:00", "%Y-%m-%d %H:%M:%S").unwrap(),
    )];
    let train = ml_get_train_times(&info_sets, &test_times);
    assert_eq!(train.len(), 7);

    // case 2: train ends within test
    let test_times = vec![(
        NaiveDateTime::parse_from_str("2019-01-01 00:08:00", "%Y-%m-%d %H:%M:%S").unwrap(),
        NaiveDateTime::parse_from_str("2019-01-01 00:11:00", "%Y-%m-%d %H:%M:%S").unwrap(),
    )];
    let train = ml_get_train_times(&info_sets, &test_times);
    assert_eq!(train.len(), 6);

    // case 3: train envelopes test
    let test_times = vec![(
        NaiveDateTime::parse_from_str("2019-01-01 00:06:00", "%Y-%m-%d %H:%M:%S").unwrap(),
        NaiveDateTime::parse_from_str("2019-01-01 00:08:00", "%Y-%m-%d %H:%M:%S").unwrap(),
    )];
    let train = ml_get_train_times(&info_sets, &test_times);
    assert_eq!(train.len(), 5);
}

#[test]
fn test_purged_kfold_basic() {
    let info_sets = make_series("2019-01-01 00:00:00", 20, 1);
    let pkf = PurgedKFold::new(3, info_sets.clone(), 0.0).unwrap();
    let splits = pkf.split(info_sets.len()).unwrap();
    assert_eq!(splits.len(), 3);
    for (train, test) in splits {
        assert!(!train.is_empty());
        assert!(!test.is_empty());
        // ensure disjoint
        for t in &test {
            assert!(!train.contains(t));
        }
    }
}

#[test]
fn test_purged_kfold_embargo() {
    let info_sets = make_series("2019-01-01 00:00:00", 100, 1);
    let pkf = PurgedKFold::new(3, info_sets.clone(), 0.02).unwrap();
    let splits = pkf.split(info_sets.len()).unwrap();
    assert_eq!(splits.len(), 3);
    for (train, test) in splits {
        // embargo should remove neighbors around test
        let min_test = *test.first().unwrap();
        let max_test = *test.last().unwrap();
        assert!(train.iter().all(|i| *i < min_test || *i > max_test));
    }
}

struct MajorityClassifier {
    prob: f64,
}

impl SimpleClassifier for MajorityClassifier {
    fn fit(&mut self, _x: &[Vec<f64>], y: &[f64], sample_weight: Option<&[f64]>) {
        let (mut w_sum, mut pos_sum) = (0.0, 0.0);
        for (i, yv) in y.iter().enumerate() {
            let w = sample_weight.map(|sw| sw[i]).unwrap_or(1.0);
            w_sum += w;
            if *yv == 1.0 {
                pos_sum += w;
            }
        }
        self.prob = if w_sum > 0.0 { pos_sum / w_sum } else { 0.5 };
    }

    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let _ = x;
        vec![self.prob; x.len()]
    }
}

#[test]
fn test_ml_cross_val_score_accuracy() {
    // simple dataset: feature is 0..9, label is parity
    let x: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64]).collect();
    let y: Vec<f64> = (0..30).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let info_sets = make_series("2019-01-01 00:00:00", 30, 1);
    let pkf = PurgedKFold::new(3, info_sets.clone(), 0.0).unwrap();
    let splits = pkf.split(x.len()).unwrap();
    let mut clf = MajorityClassifier { prob: 0.5 };
    let scores = ml_cross_val_score(&mut clf, &x, &y, None, &splits, Scoring::Accuracy);
    assert_eq!(scores.len(), 3);
    for s in scores {
        assert!((0.0..=1.0).contains(&s));
    }
}

#[test]
fn test_ml_cross_val_score_neg_log_loss() {
    let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
    let y: Vec<f64> = (0..20).map(|i| if i < 10 { 1.0 } else { 0.0 }).collect();
    let info_sets = make_series("2019-01-01 00:00:00", 20, 1);
    let pkf = PurgedKFold::new(4, info_sets.clone(), 0.0).unwrap();
    let splits = pkf.split(x.len()).unwrap();
    let mut clf = MajorityClassifier { prob: 0.5 };
    let scores = ml_cross_val_score(&mut clf, &x, &y, None, &splits, Scoring::NegLogLoss);
    assert_eq!(scores.len(), 4);
    for s in scores {
        assert!(s.is_finite());
    }
}

#[test]
fn test_ml_cross_val_score_f1() {
    let x: Vec<Vec<f64>> = (0..24).map(|i| vec![i as f64]).collect();
    let y: Vec<f64> = (0..24).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
    let info_sets = make_series("2019-01-01 00:00:00", 24, 1);
    let pkf = PurgedKFold::new(4, info_sets, 0.0).unwrap();
    let splits = pkf.split(x.len()).unwrap();
    let mut clf = MajorityClassifier { prob: 0.5 };
    let scores = ml_cross_val_score(&mut clf, &x, &y, None, &splits, Scoring::F1);
    assert_eq!(scores.len(), 4);
    for s in scores {
        assert!((0.0..=1.0).contains(&s));
    }
}

fn spans_intersect(a: (NaiveDateTime, NaiveDateTime), b: (NaiveDateTime, NaiveDateTime)) -> bool {
    a.0 <= b.1 && b.0 <= a.1
}

/// Daily bars, each label spanning three days: sample i covers [day i, day i + 3].
fn three_day_labels(n: usize) -> Vec<(NaiveDateTime, NaiveDateTime)> {
    let start = NaiveDateTime::parse_from_str("2019-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
    (0..n)
        .map(|i| {
            let s = start + chrono::Duration::days(i as i64);
            (s, s + chrono::Duration::days(3))
        })
        .collect()
}

#[test]
fn test_purged_kfold_purges_labels_overlapping_the_first_test_sample() {
    // Folds are [0..4], [4..8], [8..12]. For the middle fold the test labels cover
    // day 4 through day 10, so training samples 1, 2, 3 (ending on days 4, 5, 6) and
    // 8, 9, 10 (starting on days 8, 9, 10) overlap it. Only 0 and 11 are clean.
    let info_sets = three_day_labels(12);
    let splits = PurgedKFold::new(3, info_sets, 0.0).unwrap().split(12).unwrap();

    let expected_train: [Vec<usize>; 3] = [vec![7, 8, 9, 10, 11], vec![0, 11], vec![0, 1, 2, 3, 4]];
    for (fold, (train, test)) in splits.iter().enumerate() {
        assert_eq!(test, &(fold * 4..fold * 4 + 4).collect::<Vec<_>>());
        assert_eq!(train, &expected_train[fold], "fold {fold}");
    }
}

#[test]
fn test_purged_kfold_no_train_label_overlaps_any_test_label() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let origin = NaiveDateTime::parse_from_str("2019-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..200 {
        let n = rng.gen_range(6..60);
        let n_splits = rng.gen_range(2..=n.min(6));
        // Increasing start times with variable-length labels, as triple-barrier
        // events produce: an early label may outlive a later one.
        let mut minute = 0i64;
        let info_sets: Vec<_> = (0..n)
            .map(|_| {
                minute += rng.gen_range(1..10);
                let s = origin + chrono::Duration::minutes(minute);
                (s, s + chrono::Duration::minutes(rng.gen_range(0..40)))
            })
            .collect();

        let splits = PurgedKFold::new(n_splits, info_sets.clone(), 0.0).unwrap().split(n).unwrap();
        for (train, test) in &splits {
            for &tr in train {
                for &te in test {
                    assert!(
                        !spans_intersect(info_sets[tr], info_sets[te]),
                        "train {tr} {:?} overlaps test {te} {:?} (n={n}, splits={n_splits})",
                        info_sets[tr],
                        info_sets[te]
                    );
                }
            }
        }
    }
}

#[test]
fn test_purged_kfold_rejects_impossible_split_counts() {
    let info_sets = three_day_labels(5);
    for n_splits in [0, 1, 6] {
        assert!(PurgedKFold::new(n_splits, info_sets.clone(), 0.0).is_err(), "n_splits={n_splits}");
    }
    assert!(PurgedKFold::new(5, info_sets, 0.0).is_ok());
}

/// The numbers quoted on the `cross_validation` docs page (examples/docs_cross_validation.rs).
#[test]
fn test_docs_page_example_values() {
    use chrono::{Duration, NaiveDate};
    use openquant::cross_validation::PurgedKFold;

    let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 0, 0).unwrap();
    let info: Vec<_> =
        (0..40).map(|i| (open + Duration::hours(i), open + Duration::hours(i + 3))).collect();
    let train_of =
        |pct: f64| PurgedKFold::new(5, info.clone(), pct).unwrap().split(40).unwrap()[2].0.clone();

    let purged_only: Vec<usize> = (0..=12).chain(27..=39).collect();
    assert_eq!(train_of(0.0), purged_only);
    // A 3-sample embargo is counted from the fold's edges, inside the purged zone: no effect.
    assert_eq!(train_of(0.07), purged_only);
    assert_eq!(train_of(0.15), (0..=9).chain(30..=39).collect::<Vec<usize>>());
}

// ---------------------------------------------------------------------------------------
// Split diagnostics, naive k-fold and CPCV (re-ported from 59ac6fa, issue #33).
// ---------------------------------------------------------------------------------------

/// `periods` labels starting every `step_minutes`, each lasting `horizon_minutes`.
fn make_spans(
    start: &str,
    periods: usize,
    step_minutes: i64,
    horizon_minutes: i64,
) -> Vec<(NaiveDateTime, NaiveDateTime)> {
    let start_dt = NaiveDateTime::parse_from_str(start, "%Y-%m-%d %H:%M:%S").unwrap();
    (0..periods)
        .map(|i| {
            let t0 = start_dt + chrono::Duration::minutes(i as i64 * step_minutes);
            (t0, t0 + chrono::Duration::minutes(horizon_minutes))
        })
        .collect()
}

/// Increasing starts with variable-length labels, as triple-barrier events produce.
fn random_spans(rng: &mut rand::rngs::StdRng, n: usize) -> Vec<(NaiveDateTime, NaiveDateTime)> {
    use rand::Rng;
    let origin = NaiveDateTime::parse_from_str("2019-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let mut minute = 0i64;
    (0..n)
        .map(|_| {
            minute += rng.gen_range(1..10);
            let s = origin + chrono::Duration::minutes(minute);
            (s, s + chrono::Duration::minutes(rng.gen_range(0..40)))
        })
        .collect()
}

/// `PurgedKFold::split` as it was on main before the diagnostics were added (a31fbb4),
/// kept to prove the refactor changed no fold.
fn reference_split(
    info: &[(NaiveDateTime, NaiveDateTime)],
    n_splits: usize,
    pct_embargo: f64,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let n = info.len();
    let mut fold_sizes = vec![n / n_splits; n_splits];
    for fold_size in fold_sizes.iter_mut().take(n % n_splits) {
        *fold_size += 1;
    }
    let mut current = 0;
    let mut splits = Vec::new();
    for fold_size in fold_sizes {
        let (start, stop) = (current, current + fold_size);
        let mut keep = vec![true; n];
        keep[start..stop].fill(false);
        let test_start = info[start].0;
        let test_end = info[start..stop].iter().map(|(_, e)| *e).max().unwrap();
        for (i, (s, e)) in info.iter().enumerate() {
            let start_in = *s >= test_start && *s <= test_end;
            let end_in = *e >= test_start && *e <= test_end;
            let envelop = *s <= test_start && *e >= test_end;
            if start_in || end_in || envelop {
                keep[i] = false;
            }
        }
        let embargo = (pct_embargo * n as f64).ceil() as isize;
        if embargo > 0 {
            let after = (stop as isize + embargo).min(n as isize) as usize;
            let before = (start as isize - embargo).max(0) as usize;
            keep[before..after].fill(false);
        }
        let train = (0..n).filter(|i| keep[*i]).collect();
        splits.push((train, (start..stop).collect()));
        current = stop;
    }
    splits
}

fn assert_partition(n: usize, split: &openquant::cross_validation::PurgedSplit) {
    let d = &split.diagnostics;
    let mut removed: Vec<usize> =
        d.purged_indices.iter().chain(&d.embargo_indices).copied().collect();
    removed.sort_unstable();
    removed.dedup();
    let mut all: Vec<usize> =
        split.train_indices.iter().chain(&split.test_indices).chain(&removed).copied().collect();
    all.sort_unstable();
    assert_eq!(all, (0..n).collect::<Vec<_>>(), "train, test and removed must partition 0..n");
    let from_ranges: Vec<usize> = d.test_ranges.iter().flat_map(|&(a, b)| a..b).collect();
    assert_eq!(from_ranges, split.test_indices);
}

fn binomial(n: usize, k: usize) -> usize {
    (0..k).fold(1, |acc, i| acc * (n - i) / (i + 1))
}

// Original 59ac6fa `test_purged_kfold_basic`, on `split_with_diagnostics`.
#[test]
fn test_split_with_diagnostics_basic() {
    let info_sets = make_spans("2019-01-01 00:00:00", 20, 1, 2);
    let pkf = PurgedKFold::new(4, info_sets.clone(), 0.0).unwrap();
    let splits = pkf.split_with_diagnostics(info_sets.len()).unwrap();
    assert_eq!(splits.len(), 4);

    for split in splits {
        assert!(!split.train_indices.is_empty());
        assert!(!split.test_indices.is_empty());
        for idx in &split.test_indices {
            assert!(!split.train_indices.contains(idx));
        }
        assert_eq!(split.diagnostics.overlap_count_after_purge, 0);
    }
}

// Original 59ac6fa `test_purged_kfold_embargo`, on `split_with_diagnostics`.
#[test]
fn test_split_with_diagnostics_embargo() {
    let info_sets = make_spans("2019-01-01 00:00:00", 120, 1, 5);
    let pkf = PurgedKFold::new(4, info_sets.clone(), 0.02).unwrap();
    let splits = pkf.split_with_diagnostics(info_sets.len()).unwrap();
    let mut splits_with_embargo = 0;

    for split in splits {
        if !split.diagnostics.embargo_indices.is_empty() {
            splits_with_embargo += 1;
        }
        assert_eq!(
            count_train_test_overlaps(&info_sets, &split.train_indices, &split.test_indices)
                .unwrap(),
            0
        );
    }
    assert!(splits_with_embargo >= 1);
}

// Original 59ac6fa test.
#[test]
fn test_naive_kfold_leaks_but_purged_kfold_does_not() {
    let info_sets = make_spans("2019-01-01 00:00:00", 180, 1, 30);

    let naive = naive_kfold_splits(info_sets.len(), 6).unwrap();
    let naive_has_overlap = naive
        .iter()
        .any(|(train, test)| count_train_test_overlaps(&info_sets, train, test).unwrap() > 0);
    assert!(naive_has_overlap);

    let purged = PurgedKFold::new(6, info_sets.clone(), 0.02)
        .unwrap()
        .split_with_diagnostics(info_sets.len())
        .unwrap();
    for split in purged {
        assert_eq!(
            count_train_test_overlaps(&info_sets, &split.train_indices, &split.test_indices)
                .unwrap(),
            0
        );
        assert_eq!(split.diagnostics.overlap_count_after_purge, 0);
    }
}

// Original 59ac6fa `test_cpcv_paths_and_diagnostics`. Its `cpcv_paths` returned the
// C(N, k) splits; that method is `cpcv_splits` here, and `cpcv_paths` returns the
// phi[N, k] backtest paths of AFML §12.4.
#[test]
fn test_cpcv_splits_and_diagnostics() {
    let info_sets = make_spans("2019-01-01 00:00:00", 120, 1, 8);
    let pkf = PurgedKFold::new(5, info_sets.clone(), 0.01).unwrap();
    let splits: Vec<CpcvSplit> = pkf.cpcv_splits(info_sets.len(), 2).unwrap();

    // C(5,2) = 10
    assert_eq!(splits.len(), 10);

    for cpcv in splits {
        assert_eq!(cpcv.test_fold_ids.len(), 2);
        assert!(!cpcv.split.test_indices.is_empty());
        assert!(!cpcv.split.train_indices.is_empty());
        assert_eq!(
            count_train_test_overlaps(
                &info_sets,
                &cpcv.split.train_indices,
                &cpcv.split.test_indices
            )
            .unwrap(),
            0
        );
    }
    // phi[5, 2] = 2/5 * C(5, 2) = 4
    assert_eq!(pkf.cpcv_paths(2).unwrap().len(), 4);
}

#[test]
fn test_split_matches_pre_diagnostics_implementation() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(33);
    for _ in 0..300 {
        let n = rng.gen_range(4..80);
        let n_splits = rng.gen_range(2..=n.min(8));
        let pct_embargo = [0.0, 0.01, 0.05, 0.15, 0.4][rng.gen_range(0..5)];
        let info = random_spans(&mut rng, n);

        let pkf = PurgedKFold::new(n_splits, info.clone(), pct_embargo).unwrap();
        let expected = reference_split(&info, n_splits, pct_embargo);
        assert_eq!(pkf.split(n).unwrap(), expected, "n={n} splits={n_splits} emb={pct_embargo}");

        let diagnosed = pkf.split_with_diagnostics(n).unwrap();
        for (id, (split, (train, test))) in diagnosed.iter().zip(&expected).enumerate() {
            assert_eq!(&split.train_indices, train);
            assert_eq!(&split.test_indices, test);
            assert_eq!(split.diagnostics.split_id, id);
            assert_eq!(split.diagnostics.overlap_count_after_purge, 0);
            assert_partition(n, split);
            if pct_embargo == 0.0 {
                assert!(split.diagnostics.embargo_indices.is_empty());
            }
        }
    }
}

#[test]
fn test_split_with_diagnostics_docs_example() {
    // The docs page example: 40 hourly labels of 3 hours, 5 folds, third fold (16-23).
    use chrono::{Duration, NaiveDate};
    let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 0, 0).unwrap();
    let info: Vec<_> =
        (0..40).map(|i| (open + Duration::hours(i), open + Duration::hours(i + 3))).collect();

    let fold = |pct: f64| {
        PurgedKFold::new(5, info.clone(), pct).unwrap().split_with_diagnostics(40).unwrap()[2]
            .clone()
    };
    let purged: Vec<usize> = vec![13, 14, 15, 24, 25, 26];

    let no_embargo = fold(0.0);
    assert_eq!(no_embargo.diagnostics.test_ranges, vec![(16, 24)]);
    assert_eq!(no_embargo.diagnostics.purged_indices, purged);
    assert!(no_embargo.diagnostics.embargo_indices.is_empty());

    // ceil(0.07 * 40) = 3 samples from the fold's edges: all of them already purged.
    let narrow = fold(0.07);
    assert_eq!(narrow.diagnostics.embargo_indices, purged);
    assert_eq!(narrow.train_indices, no_embargo.train_indices);

    // ceil(0.15 * 40) = 6 samples: three beyond the purge on each side.
    let wide = fold(0.15);
    assert_eq!(wide.diagnostics.purged_indices, purged);
    let embargoed: Vec<usize> = (10..=15).chain(24..=29).collect();
    assert_eq!(wide.diagnostics.embargo_indices, embargoed);
    assert_eq!(wide.train_indices, (0..=9).chain(30..=39).collect::<Vec<usize>>());

    // The rest of the page's diagnostics example.
    let cv = PurgedKFold::new(5, info.clone(), 0.15).unwrap();
    assert_eq!(cv.cpcv_splits(40, 2).unwrap().len(), 10);
    assert_eq!(cv.cpcv_paths(2).unwrap().len(), 4);
    let (train, test) = &naive_kfold_splits(40, 5).unwrap()[2];
    assert_eq!(count_train_test_overlaps(&info, train, test), Ok(6));
}

#[test]
fn test_cpcv_split_and_path_counts_match_afml() {
    let info = make_spans("2019-01-01 00:00:00", 48, 1, 3);
    for n_splits in 2..=8 {
        let pkf = PurgedKFold::new(n_splits, info.clone(), 0.0).unwrap();
        for k in 1..n_splits {
            let splits = pkf.cpcv_splits(info.len(), k).unwrap();
            let paths = pkf.cpcv_paths(k).unwrap();
            // AFML §12.4: C(N, N - k) splits and phi[N, k] = k / N * C(N, N - k) paths.
            let n_combinations = binomial(n_splits, n_splits - k);
            assert_eq!(splits.len(), n_combinations, "N={n_splits} k={k}");
            assert_eq!(paths.len() * n_splits, k * n_combinations, "N={n_splits} k={k}");
            assert_eq!(paths.len(), binomial(n_splits - 1, k - 1));

            // Every (fold, split testing it) pair is used by exactly one path.
            let mut used = std::collections::BTreeSet::new();
            for (path_id, path) in paths.iter().enumerate() {
                assert_eq!(path.path_id, path_id);
                assert_eq!(path.split_for_fold.len(), n_splits);
                for (fold, &split_id) in path.split_for_fold.iter().enumerate() {
                    assert!(splits[split_id].test_fold_ids.contains(&fold));
                    assert!(used.insert((fold, split_id)));
                }
            }
            assert_eq!(used.len(), k * n_combinations);
        }
    }
}

#[test]
fn test_cpcv_six_groups_two_test_groups() {
    // AFML §12.4: N = 6, k = 2 gives C(6, 4) = 15 splits and phi[6, 2] = 5 paths.
    let info = make_spans("2019-01-01 00:00:00", 60, 1, 0);
    let pkf = PurgedKFold::new(6, info, 0.0).unwrap();
    let splits = pkf.cpcv_splits(60, 2).unwrap();
    assert_eq!(splits.len(), 15);
    assert_eq!(splits[0].test_fold_ids, vec![0, 1]);
    assert_eq!(splits[14].test_fold_ids, vec![4, 5]);
    // Folds 0 and 1 are adjacent, so their test samples form one block.
    assert_eq!(splits[0].split.diagnostics.test_ranges, vec![(0, 20)]);
    assert_eq!(splits[1].split.diagnostics.test_ranges, vec![(0, 10), (20, 30)]);

    let paths = pkf.cpcv_paths(2).unwrap();
    assert_eq!(paths.len(), 5);
    // Path 0 takes each fold's first split: (0,1) for folds 0 and 1, then (0,g).
    assert_eq!(paths[0].split_for_fold, vec![0, 0, 1, 2, 3, 4]);
    // Path 4 takes each fold's last split: (g,5) for folds 0-4 and (4,5) for fold 5.
    assert_eq!(paths[4].split_for_fold, vec![4, 8, 11, 13, 14, 14]);
}

#[test]
fn test_cpcv_with_one_test_fold_is_purged_kfold() {
    let info = make_spans("2019-01-01 00:00:00", 37, 2, 7);
    let pkf = PurgedKFold::new(5, info, 0.05).unwrap();
    let cpcv = pkf.cpcv_splits(37, 1).unwrap();
    let kfold = pkf.split_with_diagnostics(37).unwrap();
    assert_eq!(cpcv.len(), kfold.len());
    for (c, k) in cpcv.iter().zip(&kfold) {
        assert_eq!(c.test_fold_ids, vec![c.split_id]);
        assert_eq!(&c.split, k);
    }
    // phi[N, 1] = 1: a single path, each fold from its own split.
    let paths = pkf.cpcv_paths(1).unwrap();
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].split_for_fold, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_cpcv_no_train_label_overlaps_any_test_label() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(12);
    for _ in 0..150 {
        let n = rng.gen_range(8..60);
        let n_splits = rng.gen_range(3..=n.min(7));
        let k = rng.gen_range(1..n_splits);
        let pct_embargo = [0.0, 0.02, 0.1][rng.gen_range(0..3)];
        let info = random_spans(&mut rng, n);

        let pkf = PurgedKFold::new(n_splits, info.clone(), pct_embargo).unwrap();
        for cpcv in pkf.cpcv_splits(n, k).unwrap() {
            let s = &cpcv.split;
            assert_partition(n, s);
            for &tr in &s.train_indices {
                for &te in &s.test_indices {
                    assert!(!spans_intersect(info[tr], info[te]), "train {tr} overlaps test {te}");
                }
            }
            assert_eq!(s.diagnostics.overlap_count_after_purge, 0);
            assert_eq!(count_train_test_overlaps(&info, &s.train_indices, &s.test_indices), Ok(0));
        }
    }
}

#[test]
fn test_count_train_test_overlaps_counts_training_samples() {
    // Sample i covers [i, i + 2] minutes. Test = {5}: samples 3, 4, 6 and 7 overlap it.
    let info = make_spans("2019-01-01 00:00:00", 10, 1, 2);
    let train: Vec<usize> = (0..10).filter(|i| *i != 5).collect();
    assert_eq!(count_train_test_overlaps(&info, &train, &[5]), Ok(4));
    // A training sample counts once however many test samples it overlaps.
    assert_eq!(count_train_test_overlaps(&info, &[4], &[3, 5, 6]), Ok(1));
    assert_eq!(count_train_test_overlaps(&info, &[], &[5]), Ok(0));
}

#[test]
fn test_naive_kfold_splits_are_contiguous_complements() {
    let splits = naive_kfold_splits(10, 3).unwrap();
    let tests: Vec<Vec<usize>> = splits.iter().map(|(_, t)| t.clone()).collect();
    assert_eq!(tests, vec![vec![0, 1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);
    for (train, test) in &splits {
        let mut all: Vec<usize> = train.iter().chain(test).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
    }
}

#[test]
fn test_new_split_apis_reject_invalid_input() {
    let info = make_spans("2019-01-01 00:00:00", 10, 1, 2);
    let pkf = PurgedKFold::new(4, info.clone(), 0.0).unwrap();

    for k in [0, 4, 5] {
        let expected = CrossValidationError::InvalidTestSplits { n_test_splits: k, n_splits: 4 };
        assert_eq!(pkf.cpcv_splits(10, k).unwrap_err(), expected);
        assert_eq!(pkf.cpcv_paths(k).unwrap_err(), expected);
    }
    assert_eq!(pkf.cpcv_splits(9, 2).unwrap_err(), CrossValidationError::DatasetLengthMismatch);
    assert_eq!(
        pkf.split_with_diagnostics(11).unwrap_err(),
        CrossValidationError::DatasetLengthMismatch
    );

    for pct_embargo in [-0.01, 1.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                PurgedKFold::new(4, info.clone(), pct_embargo),
                Err(CrossValidationError::InvalidEmbargo { .. })
            ),
            "pct_embargo={pct_embargo}"
        );
    }
    let mut reversed = info.clone();
    reversed[3] = (reversed[3].1, reversed[3].0);
    assert_eq!(
        PurgedKFold::new(4, reversed, 0.0).err(),
        Some(CrossValidationError::InvalidInfoSet { index: 3 })
    );

    assert_eq!(
        count_train_test_overlaps(&info, &[0, 10], &[5]),
        Err(CrossValidationError::IndexOutOfRange { index: 10, len: 10 })
    );
    assert_eq!(
        count_train_test_overlaps(&info, &[0], &[12]),
        Err(CrossValidationError::IndexOutOfRange { index: 12, len: 10 })
    );

    for (n, splits) in [(0, 2), (5, 1), (5, 6)] {
        assert_eq!(
            naive_kfold_splits(n, splits).unwrap_err(),
            CrossValidationError::InvalidSplits { n_splits: splits, n_samples: n }
        );
    }
}
