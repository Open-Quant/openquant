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
        // Purging and the embargo leave no training sample inside the test fold's range.
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
    let scores = ml_cross_val_score(&mut clf, &x, &y, None, &splits, Scoring::Accuracy).unwrap();
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
    let scores = ml_cross_val_score(&mut clf, &x, &y, None, &splits, Scoring::NegLogLoss).unwrap();
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
    let scores = ml_cross_val_score(&mut clf, &x, &y, None, &splits, Scoring::F1).unwrap();
    assert_eq!(scores.len(), 4);
    for s in scores {
        assert!((0.0..=1.0).contains(&s));
    }
}

#[test]
fn test_an_empty_test_fold_scores_nan_under_every_rule() {
    // #185 item 10: F1 used to return 0.0 for an empty fold, which averages in as a real score.
    let x: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64]).collect();
    let y: Vec<f64> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let splits = vec![((0..6).collect::<Vec<usize>>(), Vec::new())];
    for scoring in [Scoring::Accuracy, Scoring::NegLogLoss, Scoring::F1] {
        let mut clf = MajorityClassifier { prob: 0.5 };
        let scores = ml_cross_val_score(&mut clf, &x, &y, None, &splits, scoring).unwrap();
        assert_eq!(scores.len(), 1);
        assert!(scores[0].is_nan(), "empty fold should score NaN, got {}", scores[0]);
    }
    // A non-empty fold with no positive predictions is still a genuine F1 of 0.
    // Trained on negatives only, it predicts 0 for every positive test row.
    let splits = vec![(vec![1, 3, 5], vec![0, 2, 4])];
    let mut clf = MajorityClassifier { prob: 0.0 };
    let f1 = ml_cross_val_score(&mut clf, &x, &y, None, &splits, Scoring::F1).unwrap();
    assert_eq!(f1, vec![0.0]);
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

    assert_eq!(train_of(0.0), (0..=12).chain(27..=39).collect::<Vec<usize>>());
    // The embargo starts after the purge (at 27) and only after the fold (issue #134).
    assert_eq!(train_of(0.07), (0..=12).chain(30..=39).collect::<Vec<usize>>());
    assert_eq!(train_of(0.15), (0..=12).chain(33..=39).collect::<Vec<usize>>());
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

/// A direct port of AFML Snippet 7.3 (`PurgedKFold.split`), for increasing label starts.
///
/// Two deliberate differences from the book, both shared by `PurgedKFold`: overlaps are
/// closed intervals, so a label ending exactly when the test fold starts is purged
/// (`t1 < t0` rather than `t1 <= t0`) and the right side resumes at the first label that
/// starts strictly after the fold's latest end (`searchsorted(side="right")`); and the
/// embargo width is ⌈pct · n⌉ rather than `int(pct · n)`.
fn snippet_7_3_split(
    info: &[(NaiveDateTime, NaiveDateTime)],
    n_splits: usize,
    pct_embargo: f64,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let n = info.len();
    let mbrg = (pct_embargo * n as f64).ceil() as usize;
    let mut fold_sizes = vec![n / n_splits; n_splits];
    for fold_size in fold_sizes.iter_mut().take(n % n_splits) {
        *fold_size += 1;
    }
    let mut current = 0;
    let mut splits = Vec::new();
    for fold_size in fold_sizes {
        let (i, j) = (current, current + fold_size);
        let t0 = info[i].0; // start of test set
        let max_t1 = info[i..j].iter().map(|(_, e)| *e).max().unwrap();
        let max_t1_idx = info.partition_point(|(s, _)| *s <= max_t1);
        // Left train: labels that end before the test set starts.
        let mut train: Vec<usize> = (0..n).filter(|&k| info[k].1 < t0).collect();
        // Right train, with the embargo counted from `max_t1_idx`.
        if max_t1_idx < n {
            train.extend((max_t1_idx + mbrg).min(n)..n);
        }
        splits.push((train, (i..j).collect()));
        current = j;
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
fn test_split_matches_snippet_7_3() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(33);
    for _ in 0..300 {
        let n = rng.gen_range(4..80);
        let n_splits = rng.gen_range(2..=n.min(8));
        let pct_embargo = [0.0, 0.01, 0.05, 0.15, 0.4][rng.gen_range(0..5)];
        let info = random_spans(&mut rng, n);

        let pkf = PurgedKFold::new(n_splits, info.clone(), pct_embargo).unwrap();
        let expected = snippet_7_3_split(&info, n_splits, pct_embargo);
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

    // ceil(0.07 * 40) = 3 samples, after the fold only, starting where the purge ends.
    let narrow = fold(0.07);
    assert_eq!(narrow.diagnostics.purged_indices, purged);
    assert_eq!(narrow.diagnostics.embargo_indices, vec![27, 28, 29]);
    assert_eq!(narrow.train_indices, (0..=12).chain(30..=39).collect::<Vec<usize>>());

    // ceil(0.15 * 40) = 6 samples.
    let wide = fold(0.15);
    assert_eq!(wide.diagnostics.purged_indices, purged);
    assert_eq!(wide.diagnostics.embargo_indices, (27..=32).collect::<Vec<usize>>());
    assert_eq!(wide.train_indices, (0..=12).chain(33..=39).collect::<Vec<usize>>());

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

// ---------------------------------------------------------------------------------------
// The embargo follows AFML Snippet 7.3 (issue #134).
// ---------------------------------------------------------------------------------------

/// Daily samples: sample i starts on day i and its label lasts `lengths[i]` days.
fn daily_labels(lengths: &[i64]) -> Vec<(NaiveDateTime, NaiveDateTime)> {
    let start = NaiveDateTime::parse_from_str("2019-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
    lengths
        .iter()
        .enumerate()
        .map(|(i, len)| {
            let s = start + chrono::Duration::days(i as i64);
            (s, s + chrono::Duration::days(*len))
        })
        .collect()
}

#[test]
fn test_embargo_follows_snippet_7_3_with_variable_length_labels() {
    // Sample i covers days [i, i + len]:
    //   i    0  1  2  3 | 4  5  6  7 |  8  9 10 11
    //   end  1  5  3  5 | 5 10  8  8 | 11 10 11 12
    // 12 samples, 3 folds, h = ceil(0.15 * 12) = 2.
    let info = daily_labels(&[1, 4, 1, 2, 1, 5, 2, 1, 3, 1, 1, 1]);
    let splits = PurgedKFold::new(3, info, 0.15).unwrap().split_with_diagnostics(12).unwrap();

    // Fold 0 (0-3) ends on day 5: 4 and 5 start by then and are purged; the embargo is 6, 7.
    // Fold 1 (4-7) ends on day 10, set by sample 5, not by the last sample 7 (day 8). Purged:
    // 1 and 3 (end on day 5) and 8, 9, 10 (start by day 10). The embargo is 11 (clipped).
    // Nothing before a fold is embargoed, so 0 and 2 train.
    // Fold 2 (8-11) is last, so there is no embargo; 5, 6, 7 end on or after day 8.
    let expected: [(&[usize], &[usize], &[usize]); 3] = [
        (&[8, 9, 10, 11], &[4, 5], &[6, 7]),
        (&[0, 2], &[1, 3, 8, 9, 10], &[11]),
        (&[0, 1, 2, 3, 4], &[5, 6, 7], &[]),
    ];
    for (fold, (split, (train, purged, embargoed))) in splits.iter().zip(expected).enumerate() {
        assert_eq!(split.train_indices, train, "fold {fold} train");
        assert_eq!(split.diagnostics.purged_indices, purged, "fold {fold} purged");
        assert_eq!(split.diagnostics.embargo_indices, embargoed, "fold {fold} embargoed");
    }
    // The two-sided, edge-counted rule before #134 gave [6..=11], [0, 11] and [0..=4].
}

#[test]
fn test_embargo_never_removes_samples_before_the_test_fold() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(134);
    for _ in 0..300 {
        let n = rng.gen_range(4..80);
        let n_splits = rng.gen_range(2..=n.min(8));
        let pct_embargo = [0.01, 0.05, 0.15, 0.4][rng.gen_range(0..4)];
        let info = random_spans(&mut rng, n);

        let with = PurgedKFold::new(n_splits, info.clone(), pct_embargo)
            .unwrap()
            .split_with_diagnostics(n)
            .unwrap();
        let without =
            PurgedKFold::new(n_splits, info, 0.0).unwrap().split_with_diagnostics(n).unwrap();
        for (a, b) in with.iter().zip(&without) {
            let start = a.test_indices[0];
            assert!(a.diagnostics.embargo_indices.iter().all(|&i| i > start), "n={n}");
            let before = |train: &[usize]| -> Vec<usize> {
                train.iter().copied().filter(|&i| i < start).collect()
            };
            assert_eq!(before(&a.train_indices), before(&b.train_indices), "n={n}");
        }
    }
}

#[test]
fn test_embargo_starts_where_the_purge_ends() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(73);
    for _ in 0..300 {
        let n = rng.gen_range(4..80);
        let n_splits = rng.gen_range(2..=n.min(8));
        let pct_embargo = [0.01, 0.05, 0.15, 0.4][rng.gen_range(0..4)];
        let h = (pct_embargo * n as f64).ceil() as usize;
        let info = random_spans(&mut rng, n);

        let pkf = PurgedKFold::new(n_splits, info.clone(), pct_embargo).unwrap();
        for split in pkf.split_with_diagnostics(n).unwrap() {
            let d = &split.diagnostics;
            let (_, stop) = d.test_ranges[0];
            let test_end = split.test_indices.iter().map(|&i| info[i].1).max().unwrap();
            // The first sample after the fold whose label starts after the fold's latest end.
            let resume = (stop..n).find(|&i| info[i].0 > test_end).unwrap_or(n);
            // Everything between the fold and `resume` is purged; the embargo is exactly the
            // next h samples, none of which the purge removed.
            assert!((stop..resume).all(|i| d.purged_indices.contains(&i)), "n={n}");
            let expected: Vec<usize> = (resume..(resume + h).min(n)).collect();
            assert_eq!(d.embargo_indices, expected, "n={n} h={h}");
            assert!(expected.iter().all(|i| !d.purged_indices.contains(i)), "n={n}");
        }
    }
}

#[test]
fn test_purged_kfold_and_backtesting_engine_train_on_the_same_samples() {
    use openquant::backtesting_engine::{
        run_cpcv, run_cross_validation, BacktestData, BacktestError, BacktestMode,
        BacktestRunConfig, BacktestSafeguards, CpcvConfig, CrossValidationConfig, SplitDefinition,
    };
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let run = |mode: BacktestMode| BacktestRunConfig {
        mode_provenance: format!("issue_134_{mode:?}"),
        trials_count: 1,
        safeguards: BacktestSafeguards {
            survivorship_bias_control: "n/a".to_string(),
            look_ahead_control: "n/a".to_string(),
            data_mining_control: "n/a".to_string(),
            cost_assumption: "n/a".to_string(),
            multiple_testing_control: "n/a".to_string(),
        },
    };
    let zeros = |s: &SplitDefinition| -> Result<Vec<f64>, BacktestError> {
        Ok(vec![0.0; s.test_indices.len()])
    };

    let mut rng = StdRng::seed_from_u64(132);
    let mut compared = (0, 0);
    for _ in 0..200 {
        let n = rng.gen_range(12..80);
        let n_splits = rng.gen_range(3..=6);
        let k = rng.gen_range(1..n_splits);
        let pct_embargo = [0.0, 0.02, 0.05, 0.1][rng.gen_range(0..4)];
        let info = random_spans(&mut rng, n);
        let data = BacktestData { returns: vec![0.001; n], label_spans: info.clone() };
        let pkf = PurgedKFold::new(n_splits, info, pct_embargo).unwrap();
        let case = format!("n={n} N={n_splits} k={k} pct={pct_embargo}");

        // The engine errors when a split has no training data; PurgedKFold returns it empty.
        let kfold = pkf.split(n).unwrap();
        let cfg = CrossValidationConfig { n_splits, pct_embargo };
        match run_cross_validation(&data, &run(BacktestMode::CrossValidation), &cfg, zeros) {
            Ok(cv) => {
                assert_eq!(cv.splits.len(), kfold.len());
                for (engine, (train, test)) in cv.splits.iter().zip(&kfold) {
                    assert_eq!(&engine.test_indices, test, "{case}");
                    assert_eq!(&engine.train_indices, train, "{case}");
                }
                compared.0 += 1;
            }
            Err(_) => assert!(kfold.iter().any(|(train, _)| train.is_empty()), "{case}"),
        }

        let cpcv = pkf.cpcv_splits(n, k).unwrap();
        let cfg = CpcvConfig { n_groups: n_splits, test_groups: k, pct_embargo };
        let mode = BacktestMode::CombinatorialPurgedCrossValidation;
        match run_cpcv(&data, &run(mode), &cfg, zeros) {
            Ok(result) => {
                assert_eq!(result.splits.len(), cpcv.len());
                for (engine, ours) in result.splits.iter().zip(&cpcv) {
                    assert_eq!(engine.test_groups, ours.test_fold_ids, "{case}");
                    assert_eq!(engine.test_indices, ours.split.test_indices, "{case}");
                    assert_eq!(engine.train_indices, ours.split.train_indices, "{case}");
                }
                compared.1 += 1;
            }
            Err(_) => assert!(cpcv.iter().any(|c| c.split.train_indices.is_empty()), "{case}"),
        }
    }
    assert!(compared.0 > 100 && compared.1 > 100, "{compared:?}");
}

/// Returns a fixed number of probabilities, whatever it is asked to score.
struct FixedCount(usize);

impl SimpleClassifier for FixedCount {
    fn fit(&mut self, _x: &[Vec<f64>], _y: &[f64], _sample_weight: Option<&[f64]>) {}
    fn predict_proba(&self, _x: &[Vec<f64>]) -> Vec<f64> {
        vec![1.0; self.0]
    }
}

/// #184 item 3: an out-of-range split index used to panic, and too few predictions were
/// silently truncated by `zip` while the score still divided by the test length (accuracy 1/3
/// for a model that got its one scored row right). Both are now typed errors.
#[test]
fn ml_cross_val_score_rejects_bad_indices_lengths_and_prediction_counts() {
    use openquant::cross_validation::CrossValidationError as Cv;
    let x: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64]).collect();
    let y = [1.0; 6];
    let good = vec![(vec![0, 1, 2], vec![3, 4, 5])];

    let bad_index = vec![(vec![0, 9], vec![3])];
    let out = std::panic::catch_unwind(|| {
        ml_cross_val_score(&mut FixedCount(1), &x, &y, None, &bad_index, Scoring::Accuracy)
    })
    .expect("must not panic");
    assert_eq!(out.unwrap_err(), Cv::SplitIndexOutOfRange { index: 9, n_rows: 6 });

    let out = ml_cross_val_score(&mut FixedCount(1), &x, &y, None, &good, Scoring::Accuracy);
    assert_eq!(out.unwrap_err(), Cv::PredictionCountMismatch { expected: 3, got: 1 });

    let out = ml_cross_val_score(&mut FixedCount(3), &x, &y[..5], None, &good, Scoring::F1);
    assert_eq!(out.unwrap_err(), Cv::LengthMismatch { name: "y", len: 5, expected: 6 });

    let sw = [1.0; 2];
    let out = ml_cross_val_score(&mut FixedCount(3), &x, &y, Some(&sw), &good, Scoring::F1);
    assert_eq!(out.unwrap_err(), Cv::LengthMismatch { name: "sample_weight", len: 2, expected: 6 });

    let ok = ml_cross_val_score(&mut FixedCount(3), &x, &y, None, &good, Scoring::Accuracy);
    assert_eq!(ok.unwrap(), vec![1.0]);
}
