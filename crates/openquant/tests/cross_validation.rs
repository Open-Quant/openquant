use chrono::NaiveDateTime;
use openquant::cross_validation::{
    ml_cross_val_score, ml_get_train_times, PurgedKFold, Scoring, SimpleClassifier,
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
        assert!(s >= 0.0 && s <= 1.0);
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
