use chrono::NaiveDateTime;
use openquant::cross_validation::{
    count_train_test_overlaps, ml_cross_val_score, ml_get_train_times, naive_kfold_splits,
    CpcvPath, PurgedKFold, Scoring, SimpleClassifier,
};

fn make_series(
    start: &str,
    periods: usize,
    step_minutes: i64,
    horizon_minutes: i64,
) -> Vec<(NaiveDateTime, NaiveDateTime)> {
    let start_dt = NaiveDateTime::parse_from_str(start, "%Y-%m-%d %H:%M:%S").unwrap();
    (0..periods)
        .map(|i| {
            let t0 = start_dt + chrono::Duration::minutes(i as i64 * step_minutes);
            let t1 = t0 + chrono::Duration::minutes(horizon_minutes);
            (t0, t1)
        })
        .collect()
}

#[test]
fn test_get_train_times_cases() {
    let info_sets = make_series("2019-01-01 00:00:00", 10, 1, 2);

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
    let info_sets = make_series("2019-01-01 00:00:00", 20, 1, 2);
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

#[test]
fn test_purged_kfold_embargo() {
    let info_sets = make_series("2019-01-01 00:00:00", 120, 1, 5);
    let pkf = PurgedKFold::new(4, info_sets.clone(), 0.02).unwrap();
    let splits = pkf.split_with_diagnostics(info_sets.len()).unwrap();
    let mut splits_with_embargo = 0;

    for split in splits {
        if !split.diagnostics.embargo_indices.is_empty() {
            splits_with_embargo += 1;
        }
        assert_eq!(
            count_train_test_overlaps(&info_sets, &split.train_indices, &split.test_indices),
            0
        );
    }
    assert!(splits_with_embargo >= 1);
}

#[test]
fn test_naive_kfold_leaks_but_purged_kfold_does_not() {
    let info_sets = make_series("2019-01-01 00:00:00", 180, 1, 30);

    let naive = naive_kfold_splits(info_sets.len(), 6).unwrap();
    let naive_has_overlap =
        naive.iter().any(|(train, test)| count_train_test_overlaps(&info_sets, train, test) > 0);
    assert!(naive_has_overlap);

    let purged = PurgedKFold::new(6, info_sets.clone(), 0.02)
        .unwrap()
        .split_with_diagnostics(info_sets.len())
        .unwrap();
    for split in purged {
        assert_eq!(
            count_train_test_overlaps(&info_sets, &split.train_indices, &split.test_indices),
            0
        );
        assert_eq!(split.diagnostics.overlap_count_after_purge, 0);
    }
}

#[test]
fn test_cpcv_paths_and_diagnostics() {
    let info_sets = make_series("2019-01-01 00:00:00", 120, 1, 8);
    let pkf = PurgedKFold::new(5, info_sets.clone(), 0.01).unwrap();
    let paths: Vec<CpcvPath> = pkf.cpcv_paths(info_sets.len(), 2).unwrap();

    // C(5,2) = 10
    assert_eq!(paths.len(), 10);

    for path in paths {
        assert_eq!(path.test_fold_ids.len(), 2);
        assert!(!path.split.test_indices.is_empty());
        assert!(!path.split.train_indices.is_empty());
        assert_eq!(
            count_train_test_overlaps(
                &info_sets,
                &path.split.train_indices,
                &path.split.test_indices
            ),
            0
        );
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
        vec![self.prob; x.len()]
    }
}

#[test]
fn test_ml_cross_val_score_accuracy() {
    let x: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64]).collect();
    let y: Vec<f64> = (0..30).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let info_sets = make_series("2019-01-01 00:00:00", 30, 1, 2);
    let pkf = PurgedKFold::new(3, info_sets, 0.0).unwrap();
    let splits = pkf.split(x.len()).unwrap();

    let mut clf = MajorityClassifier { prob: 0.5 };
    let scores = ml_cross_val_score(&mut clf, &x, &y, None, &splits, Scoring::Accuracy);
    assert_eq!(scores.len(), 3);
    for score in scores {
        assert!((0.0..=1.0).contains(&score));
    }
}

#[test]
fn test_ml_cross_val_score_neg_log_loss() {
    let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
    let y: Vec<f64> = (0..20).map(|i| if i < 10 { 1.0 } else { 0.0 }).collect();
    let info_sets = make_series("2019-01-01 00:00:00", 20, 1, 2);
    let pkf = PurgedKFold::new(4, info_sets, 0.0).unwrap();
    let splits = pkf.split(x.len()).unwrap();

    let mut clf = MajorityClassifier { prob: 0.5 };
    let scores = ml_cross_val_score(&mut clf, &x, &y, None, &splits, Scoring::NegLogLoss);
    assert_eq!(scores.len(), 4);
    for score in scores {
        assert!(score.is_finite());
    }
}
