use nalgebra::DMatrix;
use openquant::sampling::{get_ind_mat_average_uniqueness, get_ind_matrix};
use openquant::sb_bagging::{
    MaxFeatures, MaxSamples, SbBaggingError, SequentiallyBootstrappedBaggingClassifier,
    SequentiallyBootstrappedBaggingRegressor,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn synthetic_dataset() -> (DMatrix<f64>, Vec<u8>, Vec<f64>, Vec<Vec<u8>>) {
    let n = 240usize;
    let p = 8usize;
    let mut x = DMatrix::zeros(n, p);
    let mut y_clf = vec![0u8; n];
    let mut y_reg = vec![0.0; n];

    for i in 0..n {
        let t = i as f64;
        let f0 = (t / 12.0).sin();
        let f1 = (t / 19.0).cos();
        let f2 = (t / 7.0).sin() * 0.5;
        let signal = 1.4 * f0 + 0.8 * f1 - 0.3 * f2;
        y_clf[i] = if signal > 0.0 { 1 } else { 0 };
        y_reg[i] = if y_clf[i] == 1 { 2.0 } else { 1.0 };

        x[(i, 0)] = signal;
        x[(i, 1)] = f0;
        x[(i, 2)] = f1;
        x[(i, 3)] = f2;
        x[(i, 4)] = (t / 5.0).sin() * 0.2;
        x[(i, 5)] = (t / 17.0).cos() * 0.2;
        x[(i, 6)] = (i % 11) as f64 / 11.0;
        x[(i, 7)] = ((i * 7) % 13) as f64 / 13.0;
    }

    // One label per row of `x`, each spanning the next 6 bars.
    let bar_index: Vec<usize> = (0..n).collect();
    let t1: Vec<(usize, usize)> = (0..n).map(|start| (start, (start + 6).min(n - 1))).collect();
    let ind = get_ind_matrix(&t1, &bar_index).unwrap();

    (x, y_clf, y_reg, ind)
}

// One label per training row, each spanning the next 4 bars.
fn train_ind_mat(rows: usize) -> Vec<Vec<u8>> {
    let bar_index: Vec<usize> = (0..rows).collect();
    let t1: Vec<(usize, usize)> = (0..rows).map(|s| (s, (s + 4).min(rows - 1))).collect();
    get_ind_matrix(&t1, &bar_index).unwrap()
}

#[test]
fn test_sb_bagging_not_tree_base_estimator() {
    let (x, y, _, ind) = synthetic_dataset();
    let mut sb = SequentiallyBootstrappedBaggingClassifier::new(1);
    sb.supports_sample_weight = false;
    sb.n_estimators = 16;
    sb.max_features = MaxFeatures::Float(0.5);
    sb.max_samples = MaxSamples::Int(60);
    sb.fit(&x, &y, &ind, None).unwrap();

    let pred = sb.predict(&x).unwrap();
    assert_eq!(pred.len(), x.nrows());
}

#[test]
fn test_sb_bagging_non_sample_weights_with_bootstrap_features() {
    let (x, y, _, ind) = synthetic_dataset();
    let mut sb = SequentiallyBootstrappedBaggingClassifier::new(1);
    sb.supports_sample_weight = false;
    sb.n_estimators = 8;
    sb.max_features = MaxFeatures::Float(0.2);
    sb.bootstrap_features = true;
    sb.max_samples = MaxSamples::Int(30);
    sb.fit(&x, &y, &ind, None).unwrap();

    let pred = sb.predict(&x).unwrap();
    assert_eq!(pred.len(), x.nrows());
}

#[test]
fn test_sb_bagging_with_max_features() {
    let (x, y, _, ind) = synthetic_dataset();
    let weights = vec![1.0; x.nrows()];

    let mut sb = SequentiallyBootstrappedBaggingClassifier::new(1);
    sb.supports_sample_weight = true;
    sb.n_estimators = 12;
    sb.max_features = MaxFeatures::Float(0.2);
    sb.bootstrap_features = true;
    sb.max_samples = MaxSamples::Int(30);
    sb.fit(&x, &y, &ind, Some(&weights)).unwrap();

    let pred = sb.predict(&x).unwrap();
    assert_eq!(pred.len(), x.nrows());
}

#[test]
fn test_sb_bagging_float_max_samples_warm_start_true() {
    let (x, y, _, ind) = synthetic_dataset();
    let weights = vec![1.0; x.nrows()];

    let mut sb = SequentiallyBootstrappedBaggingClassifier::new(1);
    sb.warm_start = true;
    sb.n_estimators = 2;
    sb.max_features = MaxFeatures::Int(4);
    sb.bootstrap_features = true;
    sb.max_samples = MaxSamples::Float(0.3);

    sb.fit(&x, &y, &ind, Some(&weights)).unwrap();
    let first_len = sb.estimators_samples.len();

    sb.n_estimators += 0;
    sb.fit(&x, &y, &ind, Some(&weights)).unwrap();
    assert_eq!(sb.estimators_samples.len(), first_len);

    sb.n_estimators += 2;
    sb.fit(&x, &y, &ind, Some(&weights)).unwrap();
    assert!(sb.estimators_samples.len() >= first_len + 2);
}

#[test]
fn test_value_error_raise() {
    let (x, y, _, ind) = synthetic_dataset();
    let w = vec![1.0; x.nrows()];

    let mut bagging_1 = SequentiallyBootstrappedBaggingClassifier::new(1);
    bagging_1.supports_sample_weight = false;
    assert_eq!(
        bagging_1.fit(&x, &y, &ind, Some(&w)).unwrap_err(),
        SbBaggingError::SampleWeightNotSupported
    );

    let mut bagging_2 = SequentiallyBootstrappedBaggingClassifier::new(1);
    bagging_2.max_samples = MaxSamples::Int(2_000_000);
    assert_eq!(
        bagging_2.fit(&x, &y, &ind, Some(&w)).unwrap_err(),
        SbBaggingError::MaxSamplesOutOfRange
    );

    let mut bagging_4 = SequentiallyBootstrappedBaggingClassifier::new(1);
    bagging_4.max_features = MaxFeatures::Int(2_000_000);
    assert_eq!(
        bagging_4.fit(&x, &y, &ind, Some(&w)).unwrap_err(),
        SbBaggingError::MaxFeaturesOutOfRange
    );

    let mut bagging_5 = SequentiallyBootstrappedBaggingClassifier::new(1);
    bagging_5.oob_score = true;
    bagging_5.warm_start = true;
    assert_eq!(
        bagging_5.fit(&x, &y, &ind, Some(&w)).unwrap_err(),
        SbBaggingError::WarmStartWithOob
    );

    let mut bagging_6 = SequentiallyBootstrappedBaggingClassifier::new(1);
    bagging_6.warm_start = true;
    bagging_6.n_estimators = 3;
    bagging_6.fit(&x, &y, &ind, None).unwrap();
    bagging_6.n_estimators = 1;
    assert_eq!(
        bagging_6.fit(&x, &y, &ind, None).unwrap_err(),
        SbBaggingError::DecreasingEstimators
    );

    let mut bagging_7 = SequentiallyBootstrappedBaggingClassifier::new(1);
    bagging_7.n_estimators = 0;
    assert_eq!(bagging_7.fit(&x, &y, &ind, None).unwrap_err(), SbBaggingError::InvalidEstimators);
}

#[test]
fn test_sb_classifier() {
    let (x, y, _, _ind) = synthetic_dataset();
    let split = (x.nrows() as f64 * 0.6) as usize;
    let x_train = x.rows(0, split).into_owned();
    let x_test = x.rows(split, x.nrows() - split).into_owned();
    let y_train = &y[0..split];
    let y_test = &y[split..];

    let mut sb = SequentiallyBootstrappedBaggingClassifier::new(1);
    sb.n_estimators = 100;
    sb.max_features = MaxFeatures::Float(1.0);
    sb.oob_score = true;

    // indicator matrix needs the same number of labels as rows in train set
    let ind_train = train_ind_mat(split);

    sb.fit(&x_train, y_train, &ind_train, None).unwrap();
    let preds = sb.predict(&x_test).unwrap();

    let acc = preds.iter().zip(y_test.iter()).filter(|(p, t)| **p == **t).count() as f64
        / y_test.len() as f64;

    assert!(acc >= 0.55, "acc={acc}");
    assert!(sb.oob_score_value.unwrap_or(0.0).is_finite());
}

#[test]
fn test_sb_regressor() {
    let (x, _, y, _ind) = synthetic_dataset();
    let split = (x.nrows() as f64 * 0.6) as usize;
    let x_train = x.rows(0, split).into_owned();
    let x_test = x.rows(split, x.nrows() - split).into_owned();
    let y_train = &y[0..split];
    let y_test = &y[split..];

    let ind_train = train_ind_mat(split);

    let mut sb = SequentiallyBootstrappedBaggingRegressor::new(1);
    sb.n_estimators = 100;
    sb.max_features = MaxFeatures::Float(1.0);
    sb.oob_score = true;
    sb.fit(&x_train, y_train, &ind_train, None).unwrap();

    let preds = sb.predict(&x_test).unwrap();
    let mse = preds
        .iter()
        .zip(y_test.iter())
        .map(|(p, t)| {
            let d = p - t;
            d * d
        })
        .sum::<f64>()
        / y_test.len() as f64;
    let mae = preds.iter().zip(y_test.iter()).map(|(p, t)| (p - t).abs()).sum::<f64>()
        / y_test.len() as f64;

    assert!(mse < 0.4, "mse={mse}");
    assert!(mae < 0.5, "mae={mae}");
}

// Labels of 8 bars, one starting every 2 bars (the setup measured in #90).
fn overlapping_labels(n: usize) -> Vec<Vec<u8>> {
    let spans: Vec<(usize, usize)> = (0..n).map(|i| (2 * i, 2 * i + 7)).collect();
    let bars: Vec<usize> = (0..2 * n + 8).collect();
    get_ind_matrix(&spans, &bars).unwrap()
}

fn sample_uniqueness(ind_mat: &[Vec<u8>], drawn: &[usize]) -> f64 {
    let sub: Vec<Vec<u8>> =
        ind_mat.iter().map(|row| drawn.iter().map(|&c| row[c]).collect()).collect();
    get_ind_mat_average_uniqueness(&sub).unwrap()
}

fn mean_uniqueness(ind_mat: &[Vec<u8>], samples: &[Vec<usize>]) -> f64 {
    samples.iter().map(|s| sample_uniqueness(ind_mat, s)).sum::<f64>() / samples.len() as f64
}

#[test]
fn test_estimators_are_sequentially_bootstrapped() {
    let n = 60;
    let ind = overlapping_labels(n);
    let x = DMatrix::from_fn(n, 1, |r, _| r as f64);
    let y: Vec<u8> = (0..n).map(|r| u8::from(r % 3 == 0)).collect();

    let mut rng = StdRng::seed_from_u64(7);
    let uniform: Vec<Vec<usize>> =
        (0..300).map(|_| (0..n).map(|_| rng.gen_range(0..n)).collect()).collect();
    let standard = mean_uniqueness(&ind, &uniform);

    let mut clf = SequentiallyBootstrappedBaggingClassifier::new(7);
    clf.n_estimators = 300;
    clf.fit(&x, &y, &ind, None).unwrap();
    let sequential = mean_uniqueness(&ind, &clf.estimators_samples);
    // Uniform ~0.254 and sequential ~0.261, each with a standard error near 0.0005.
    assert!(
        sequential > standard + 0.003,
        "classifier: sequential {sequential:.4} vs uniform bootstrap {standard:.4}"
    );

    let mut reg = SequentiallyBootstrappedBaggingRegressor::new(7);
    reg.n_estimators = 300;
    let y_reg: Vec<f64> = (0..n).map(|r| r as f64).collect();
    reg.fit(&x, &y_reg, &ind, None).unwrap();
    let sequential = mean_uniqueness(&ind, &reg.estimators_samples);
    assert!(
        sequential > standard + 0.003,
        "regressor: sequential {sequential:.4} vs uniform bootstrap {standard:.4}"
    );
}

#[test]
fn test_random_state_reproduces_a_fit() {
    let n = 40;
    let ind = overlapping_labels(n);
    let x = DMatrix::from_fn(n, 1, |r, _| r as f64);
    let y: Vec<u8> = (0..n).map(|r| u8::from(r >= 20)).collect();
    let fit = |seed: u64| {
        let mut clf = SequentiallyBootstrappedBaggingClassifier::new(seed);
        clf.n_estimators = 5;
        clf.fit(&x, &y, &ind, None).unwrap();
        clf.estimators_samples
    };
    assert_eq!(fit(3), fit(3));
    assert_ne!(fit(3), fit(4));
}

#[test]
fn test_ind_mat_label_count_must_match_rows() {
    let n = 30;
    let x = DMatrix::from_fn(n, 1, |r, _| r as f64);
    let y: Vec<u8> = (0..n).map(|r| u8::from(r >= 15)).collect();
    let y_reg: Vec<f64> = (0..n).map(|r| r as f64).collect();
    for labels in [n - 5, n + 5] {
        let ind = overlapping_labels(labels);
        let mut clf = SequentiallyBootstrappedBaggingClassifier::new(1);
        assert_eq!(clf.fit(&x, &y, &ind, None), Err(SbBaggingError::DimensionMismatch));
        let mut reg = SequentiallyBootstrappedBaggingRegressor::new(1);
        assert_eq!(reg.fit(&x, &y_reg, &ind, None), Err(SbBaggingError::DimensionMismatch));
    }
}

// A weak, noisy relationship, so that in-sample and out-of-bag scores differ.
fn noisy_rows(n: usize) -> (DMatrix<f64>, Vec<u8>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(11);
    let x = DMatrix::from_fn(n, 1, |_, _| rng.gen_range(-1.0..1.0));
    let y: Vec<u8> = (0..n).map(|r| u8::from(x[(r, 0)] + rng.gen_range(-1.0..1.0) > 0.0)).collect();
    let y_reg: Vec<f64> = (0..n).map(|r| x[(r, 0)] + rng.gen_range(-1.0..1.0)).collect();
    (x, y, y_reg)
}

fn out_of_bag_rows(n: usize, drawn: &[usize]) -> Vec<usize> {
    (0..n).filter(|r| !drawn.contains(r)).collect()
}

// With one estimator the ensemble prediction is that estimator's, so the out-of-bag score
// is the score of `predict` over the rows the estimator did not draw.
#[test]
fn test_classifier_oob_score_uses_only_held_out_rows() {
    let n = 60;
    let ind = overlapping_labels(n);
    let (x, y, _) = noisy_rows(n);

    let mut clf = SequentiallyBootstrappedBaggingClassifier::new(5);
    clf.n_estimators = 1;
    clf.oob_score = true;
    clf.fit(&x, &y, &ind, None).unwrap();
    let preds = clf.predict(&x).unwrap();
    let accuracy = |rows: &[usize]| {
        rows.iter().filter(|&&r| preds[r] == y[r]).count() as f64 / rows.len() as f64
    };
    let oob = out_of_bag_rows(n, &clf.estimators_samples[0]);
    let all: Vec<usize> = (0..n).collect();
    assert!((accuracy(&oob) - accuracy(&all)).abs() > 1e-9, "data does not separate the two");
    assert!((clf.oob_score_value.unwrap() - accuracy(&oob)).abs() < 1e-12);
}

#[test]
fn test_regressor_oob_score_uses_only_held_out_rows() {
    let n = 60;
    let ind = overlapping_labels(n);
    let (x, _, y) = noisy_rows(n);

    let mut reg = SequentiallyBootstrappedBaggingRegressor::new(5);
    reg.n_estimators = 1;
    reg.oob_score = true;
    reg.fit(&x, &y, &ind, None).unwrap();
    let preds = reg.predict(&x).unwrap();
    let r2 = |rows: &[usize]| {
        let mean = rows.iter().map(|&r| y[r]).sum::<f64>() / rows.len() as f64;
        let ss_tot = rows.iter().map(|&r| (y[r] - mean).powi(2)).sum::<f64>();
        let ss_res = rows.iter().map(|&r| (y[r] - preds[r]).powi(2)).sum::<f64>();
        1.0 - ss_res / ss_tot
    };
    let oob = out_of_bag_rows(n, &reg.estimators_samples[0]);
    let all: Vec<usize> = (0..n).collect();
    assert!((r2(&oob) - r2(&all)).abs() > 1e-9, "data does not separate the two");
    assert!((reg.oob_score_value.unwrap() - r2(&oob)).abs() < 1e-12);
}

#[test]
fn test_sample_weight_changes_the_fit() {
    // Rows 0..30 follow y = x and rows 30..60 follow y = -x. Zero weight on one half leaves
    // the other half's relationship.
    let n = 60;
    let ind = overlapping_labels(n);
    let x = DMatrix::from_fn(n, 1, |r, _| (r % 30) as f64 - 14.5);
    let first_half: Vec<f64> = (0..n).map(|r| if r < 30 { 1.0 } else { 0.0 }).collect();
    let second_half: Vec<f64> = first_half.iter().map(|w| 1.0 - w).collect();

    let y: Vec<f64> = (0..n).map(|r| if r < 30 { x[(r, 0)] } else { -x[(r, 0)] }).collect();
    let probe = DMatrix::from_row_slice(1, 1, &[10.0]);
    let mut reg = SequentiallyBootstrappedBaggingRegressor::new(2);
    reg.n_estimators = 20;
    reg.fit(&x, &y, &ind, Some(&first_half)).unwrap();
    let weighted = reg.predict(&probe).unwrap()[0];
    assert!((weighted - 10.0).abs() < 1e-9, "weighted prediction {weighted}");
    reg.fit(&x, &y, &ind, None).unwrap();
    let unweighted = reg.predict(&probe).unwrap()[0];
    assert!(unweighted.abs() < 5.0, "unweighted prediction {unweighted}");

    // Class 1 where x >= 0 in the first half, where x < 0 in the second.
    let y_clf: Vec<u8> = (0..n).map(|r| u8::from((x[(r, 0)] >= 0.0) == (r < 30))).collect();
    let probe = DMatrix::from_row_slice(2, 1, &[-10.0, 10.0]);
    let mut clf = SequentiallyBootstrappedBaggingClassifier::new(2);
    clf.n_estimators = 21;
    clf.fit(&x, &y_clf, &ind, Some(&first_half)).unwrap();
    assert_eq!(clf.predict(&probe).unwrap(), vec![0, 1]);
    clf.fit(&x, &y_clf, &ind, Some(&second_half)).unwrap();
    assert_eq!(clf.predict(&probe).unwrap(), vec![1, 0]);
}

#[test]
fn test_sample_weight_is_validated() {
    let n = 20;
    let ind = overlapping_labels(n);
    let x = DMatrix::from_fn(n, 1, |r, _| r as f64);
    let y: Vec<u8> = (0..n).map(|r| u8::from(r >= 10)).collect();
    let mut clf = SequentiallyBootstrappedBaggingClassifier::new(1);
    assert_eq!(clf.fit(&x, &y, &ind, Some(&[1.0])), Err(SbBaggingError::DimensionMismatch));
    let mut bad = vec![1.0; n];
    bad[3] = -1.0;
    assert_eq!(clf.fit(&x, &y, &ind, Some(&bad)), Err(SbBaggingError::InvalidSampleWeight));
    bad[3] = f64::NAN;
    assert_eq!(clf.fit(&x, &y, &ind, Some(&bad)), Err(SbBaggingError::InvalidSampleWeight));
    assert_eq!(
        clf.fit(&x, &y, &ind, Some(&vec![0.0; n])),
        Err(SbBaggingError::InvalidSampleWeight)
    );
}

#[test]
fn test_predict_proba_is_the_vote_share_and_agrees_with_predict() {
    let (x, y, _, _) = synthetic_dataset();
    let split = 150;
    let x_train = x.rows(0, split).into_owned();
    let x_test = x.rows(split, x.nrows() - split).into_owned();
    let mut clf = SequentiallyBootstrappedBaggingClassifier::new(3);
    clf.n_estimators = 16;
    assert_eq!(clf.predict_proba(&x_test), Err(SbBaggingError::EmptyInput));
    clf.fit(&x_train, &y[..split], &train_ind_mat(split), None).unwrap();

    let proba = clf.predict_proba(&x_test).unwrap();
    let preds = clf.predict(&x_test).unwrap();
    assert_eq!(proba.len(), x_test.nrows());
    for (p, pred) in proba.iter().zip(&preds) {
        // A share of 16 votes.
        assert!((0.0..=1.0).contains(p));
        assert!((p * 16.0 - (p * 16.0).round()).abs() < 1e-12, "p={p}");
        assert_eq!(*pred, u8::from(*p >= 0.5));
    }

    // One estimator: the probability is that stump's 0/1 vote.
    clf.n_estimators = 1;
    clf.fit(&x_train, &y[..split], &train_ind_mat(split), None).unwrap();
    let single = clf.predict_proba(&x_test).unwrap();
    let votes: Vec<f64> = clf.predict(&x_test).unwrap().into_iter().map(f64::from).collect();
    assert_eq!(single, votes);
}

/// Runs `f`, failing the test with `what` if it panics (the pre-#184 behaviour).
fn no_panic<T>(what: &str, f: impl FnOnce() -> T) -> T {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or_else(|_| panic!("{what} panicked"))
}

#[test]
fn test_predict_with_wrong_column_count_is_an_error_not_a_panic() {
    let (x, y_clf, y_reg, ind) = synthetic_dataset();
    let narrow = x.columns(0, 1).into_owned();
    let wide = x.clone().insert_column(x.ncols(), 0.0);
    let mismatch = |found| SbBaggingError::FeatureCountMismatch { expected: x.ncols(), found };

    let mut clf = SequentiallyBootstrappedBaggingClassifier::new(4);
    clf.max_samples = MaxSamples::Float(0.3);
    assert_eq!(clf.predict(&narrow), Err(SbBaggingError::EmptyInput));
    clf.fit(&x, &y_clf, &ind, None).unwrap();
    let mut reg = SequentiallyBootstrappedBaggingRegressor::new(4);
    reg.max_samples = MaxSamples::Float(0.3);
    assert_eq!(reg.predict(&narrow), Err(SbBaggingError::EmptyInput));
    reg.fit(&x, &y_reg, &ind, None).unwrap();

    for bad in [&narrow, &wide] {
        let found = bad.ncols();
        // Before the fix, a narrower `x` indexed out of bounds and panicked, and a wider one
        // was silently accepted.
        assert_eq!(no_panic("classifier predict", || clf.predict(bad)), Err(mismatch(found)));
        assert_eq!(no_panic("predict_proba", || clf.predict_proba(bad)), Err(mismatch(found)));
        assert_eq!(no_panic("regressor predict", || reg.predict(bad)), Err(mismatch(found)));
    }
    assert_eq!(clf.predict(&x).unwrap().len(), x.nrows());
    assert_eq!(reg.predict(&x).unwrap().len(), x.nrows());

    // A warm-start fit may not add estimators trained on a different column count, and the
    // failed fit leaves the model as it was.
    clf.warm_start = true;
    clf.n_estimators += 2;
    assert_eq!(clf.fit(&narrow, &y_clf, &ind, None), Err(mismatch(1)));
    assert_eq!(clf.estimators_samples.len(), 10);
    reg.warm_start = true;
    reg.n_estimators += 2;
    assert_eq!(reg.fit(&narrow, &y_reg, &ind, None), Err(mismatch(1)));
    assert_eq!(reg.estimators_samples.len(), 10);

    // Without warm_start, a refit on a new column count replaces the model.
    clf.warm_start = false;
    clf.fit(&narrow, &y_clf, &ind, None).unwrap();
    assert_eq!(clf.predict(&narrow).unwrap().len(), x.nrows());
    assert_eq!(
        clf.predict(&x),
        Err(SbBaggingError::FeatureCountMismatch { expected: 1, found: x.ncols() })
    );
}

#[test]
fn test_warm_start_seed_is_random_state_plus_fitted_count_wrapping() {
    let (x, y_clf, y_reg, ind) = synthetic_dataset();

    // A warm-start fit that starts with `k` fitted estimators seeds its stream with
    // `random_state + k`, so its new estimators match a fresh fit with that seed. Near
    // `u64::MAX` the sum wraps; before the fix it overflowed (a panic in debug builds).
    for (random_state, fresh_seed) in [(5u64, 7u64), (u64::MAX - 1, 0), (u64::MAX, 1)] {
        let mut warm = SequentiallyBootstrappedBaggingClassifier::new(random_state);
        warm.max_samples = MaxSamples::Float(0.2);
        warm.warm_start = true;
        warm.n_estimators = 2;
        warm.fit(&x, &y_clf, &ind, None).unwrap();
        warm.n_estimators = 5;
        no_panic("warm-start fit", || warm.fit(&x, &y_clf, &ind, None)).unwrap();

        let mut fresh = SequentiallyBootstrappedBaggingClassifier::new(fresh_seed);
        fresh.max_samples = MaxSamples::Float(0.2);
        fresh.n_estimators = 3;
        fresh.fit(&x, &y_clf, &ind, None).unwrap();
        assert_eq!(warm.estimators_samples[2..], fresh.estimators_samples[..], "{random_state}");
    }

    let mut reg = SequentiallyBootstrappedBaggingRegressor::new(u64::MAX);
    reg.max_samples = MaxSamples::Float(0.2);
    reg.warm_start = true;
    reg.n_estimators = 1;
    reg.fit(&x, &y_reg, &ind, None).unwrap();
    reg.n_estimators = 3;
    no_panic("warm-start fit", || reg.fit(&x, &y_reg, &ind, None)).unwrap();
    assert_eq!(reg.estimators_samples.len(), 3);
}
