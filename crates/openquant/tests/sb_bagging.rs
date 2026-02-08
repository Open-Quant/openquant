use nalgebra::DMatrix;
use openquant::sampling::get_ind_matrix;
use openquant::sb_bagging::{
    MaxFeatures, MaxSamples, SbBaggingError, SequentiallyBootstrappedBaggingClassifier,
    SequentiallyBootstrappedBaggingRegressor,
};

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

    let bar_index: Vec<usize> = (0..n).collect();
    let mut t1 = Vec::new();
    for start in (0..n - 6).step_by(3) {
        t1.push((start, start + 6));
    }
    let ind = get_ind_matrix(&t1, &bar_index);

    (x, y_clf, y_reg, ind)
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
fn test_sb_bagging_non_sample_weights_with_verbose() {
    let (x, y, _, ind) = synthetic_dataset();
    let mut sb = SequentiallyBootstrappedBaggingClassifier::new(1);
    sb.supports_sample_weight = false;
    sb.verbose = 2;
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
    let bar_index: Vec<usize> = (0..split).collect();
    let t1: Vec<(usize, usize)> =
        (0..split.saturating_sub(4)).step_by(2).map(|s| (s, s + 4)).collect();
    let ind_train = get_ind_matrix(&t1, &bar_index);

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

    let bar_index: Vec<usize> = (0..split).collect();
    let t1: Vec<(usize, usize)> =
        (0..split.saturating_sub(4)).step_by(2).map(|s| (s, s + 4)).collect();
    let ind_train = get_ind_matrix(&t1, &bar_index);

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
