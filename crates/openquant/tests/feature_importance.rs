use openquant::cross_validation::{Scoring, SimpleClassifier};
use openquant::feature_importance::{
    feature_pca_analysis, get_orthogonal_features, mean_decrease_accuracy, mean_decrease_impurity,
    plot_feature_importance, single_feature_importance,
};

#[derive(Clone, Debug)]
struct LinearProbClassifier {
    w: Vec<f64>,
    b: f64,
}

impl LinearProbClassifier {
    fn new(n_features: usize) -> Self {
        Self { w: vec![0.0; n_features], b: 0.0 }
    }
}

impl SimpleClassifier for LinearProbClassifier {
    fn fit(&mut self, x: &[Vec<f64>], y: &[f64], _sample_weight: Option<&[f64]>) {
        let n_features = x[0].len();
        let mut pos_mean = vec![0.0; n_features];
        let mut neg_mean = vec![0.0; n_features];
        let mut n_pos = 0.0;
        let mut n_neg = 0.0;
        for (row, yy) in x.iter().zip(y.iter()) {
            if *yy > 0.5 {
                n_pos += 1.0;
                for j in 0..n_features {
                    pos_mean[j] += row[j];
                }
            } else {
                n_neg += 1.0;
                for j in 0..n_features {
                    neg_mean[j] += row[j];
                }
            }
        }
        if n_pos > 0.0 {
            for v in &mut pos_mean {
                *v /= n_pos;
            }
        }
        if n_neg > 0.0 {
            for v in &mut neg_mean {
                *v /= n_neg;
            }
        }
        self.w = (0..n_features).map(|j| pos_mean[j] - neg_mean[j]).collect();
        self.b = 0.0;
    }

    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let z = row.iter().zip(self.w.iter()).map(|(a, b)| a * b).sum::<f64>() + self.b;
                1.0 / (1.0 + (-z).exp())
            })
            .collect()
    }
}

/// `(x, y, feature_names, cv_splits)`.
type Dataset = (Vec<Vec<f64>>, Vec<f64>, Vec<String>, Vec<(Vec<usize>, Vec<usize>)>);

fn make_dataset() -> Dataset {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..120usize {
        let f0 = (i as f64 / 10.0).sin();
        let f1 = 0.7 * f0 + 0.3 * (i as f64 / 7.0).cos();
        let f2 = ((i * 37) % 17) as f64 / 17.0 - 0.5;
        x.push(vec![f0, f1, f2]);
        y.push(if f0 > 0.0 { 1.0 } else { 0.0 });
    }

    let names = vec!["f0".to_string(), "f1".to_string(), "f2".to_string()];
    let n = x.len();
    let fold = n / 4;
    let mut splits = Vec::new();
    for k in 0..4 {
        let start = k * fold;
        let end = if k == 3 { n } else { (k + 1) * fold };
        let test: Vec<usize> = (start..end).collect();
        let mut train: Vec<usize> = (0..start).collect();
        train.extend(end..n);
        splits.push((train, test));
    }
    (x, y, names, splits)
}

#[test]
fn test_orthogonal_features_and_pca_analysis() {
    let (x, _y, _names, _splits) = make_dataset();
    let pca = get_orthogonal_features(&x, 0.95).unwrap();
    assert_eq!(pca.len(), x.len());
    assert!(!pca[0].is_empty());

    let first_pc_mean = pca.iter().map(|r| r[0]).sum::<f64>() / pca.len() as f64;
    assert!(first_pc_mean.abs() < 1e-6);

    let fi = vec![0.6, 0.3, 0.1];
    let corr = feature_pca_analysis(&x, &fi, 0.95).unwrap();
    assert!(corr.pearson.is_finite());
    assert!(corr.spearman.is_finite());
    assert!(corr.kendall.is_finite());
    assert!(corr.weighted_kendall_rank.is_finite());
    assert!(corr.weighted_kendall_rank >= -1.0 && corr.weighted_kendall_rank <= 1.0);
}

#[test]
fn test_feature_importance_mdi_mda_sfi() {
    let (x, y, names, splits) = make_dataset();

    let per_tree = vec![
        vec![0.50, 0.35, 0.15],
        vec![0.60, 0.30, 0.10],
        vec![0.58, 0.32, 0.10],
        vec![0.52, 0.34, 0.14],
    ];
    let mdi = mean_decrease_impurity(&per_tree, &names).unwrap();
    let mdi_sum = mdi.values().map(|v| v.mean).sum::<f64>();
    assert!((mdi_sum - 1.0).abs() < 1e-9);
    assert!(mdi["f0"].mean > mdi["f1"].mean);
    assert!(mdi["f1"].mean > mdi["f2"].mean);

    let mut clf = LinearProbClassifier::new(names.len());
    let mda = mean_decrease_accuracy(&mut clf, &x, &y, &names, &splits, None, Scoring::Accuracy, 7)
        .unwrap();
    assert!(mda["f0"].mean > mda["f2"].mean);
    let mda_f1 =
        mean_decrease_accuracy(&mut clf, &x, &y, &names, &splits, None, Scoring::F1, 7).unwrap();
    assert!(mda_f1["f0"].mean > mda_f1["f2"].mean);

    let mut clf2 = LinearProbClassifier::new(names.len());
    let sfi =
        single_feature_importance(&mut clf2, &x, &y, &names, &splits, None, Scoring::Accuracy)
            .unwrap();
    assert!(sfi["f0"].mean >= sfi["f2"].mean);
    let sfi_f1 =
        single_feature_importance(&mut clf2, &x, &y, &names, &splits, None, Scoring::F1).unwrap();
    assert!(sfi_f1["f0"].mean >= sfi_f1["f2"].mean);
}

#[test]
fn test_plot_feature_importance_output_file() {
    let names = vec!["a".to_string(), "b".to_string()];
    let mdi = mean_decrease_impurity(&[vec![0.7, 0.3], vec![0.6, 0.4]], &names).unwrap();
    // temp_dir(), not a hard-coded /tmp: this test also runs on Windows nightly.
    let path = std::env::temp_dir().join("openquant_feature_importance_test.csv");
    let out = path.to_str().unwrap();
    let _ = std::fs::remove_file(out);
    plot_feature_importance(&mdi, 0.5, 0.4, Some(out)).unwrap();
    assert!(path.exists());
    std::fs::remove_file(out).unwrap();
}

/// One informative AR(1) feature with autocorrelation `phi` (unit variance) and one i.i.d.
/// noise feature; the label is the sign of the informative feature plus noise. Five
/// contiguous folds. Returns the negative-log-loss MDA of the informative feature.
fn mda_of_ar1_feature(phi: f64, data_seed: u64, seed: u64) -> f64 {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    let n = 2000;
    let mut rng = StdRng::seed_from_u64(data_seed);
    let innovation_scale = (1.0 - phi * phi).sqrt();
    let mut f: f64 = StandardNormal.sample(&mut rng);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let e: f64 = StandardNormal.sample(&mut rng);
        f = phi * f + innovation_scale * e;
        let noise: f64 = StandardNormal.sample(&mut rng);
        let label_noise: f64 = StandardNormal.sample(&mut rng);
        x.push(vec![f, noise]);
        y.push(if f + 0.5 * label_noise > 0.0 { 1.0 } else { 0.0 });
    }
    let names = vec!["signal".to_string(), "noise".to_string()];
    let fold = n / 5;
    let splits: Vec<_> = (0..5)
        .map(|k| {
            let test: Vec<usize> = (k * fold..(k + 1) * fold).collect();
            let train: Vec<usize> = (0..n).filter(|i| !test.contains(i)).collect();
            (train, test)
        })
        .collect();
    let mut clf = LinearProbClassifier::new(2);
    let mda =
        mean_decrease_accuracy(&mut clf, &x, &y, &names, &splits, None, Scoring::NegLogLoss, seed)
            .unwrap();
    mda["signal"].mean
}

/// Regression for #98. MDA used to "permute" by rotating the column one row, which leaves a
/// persistent feature almost unchanged, so its importance collapsed towards zero (0.12 of the
/// i.i.d. value at phi = 0.9, 0.02 at phi = 0.99). AFML Snippet 8.3 shuffles the column; a
/// shuffle breaks the feature-label link however persistent the feature is, so an equally
/// informative persistent feature must score about the same.
#[test]
fn mda_of_persistent_feature_matches_iid_case() {
    let iid = mda_of_ar1_feature(0.0, 11, 42);
    assert!(iid > 0.4, "iid MDA {iid}");
    for phi in [0.5, 0.9, 0.95] {
        let persistent = mda_of_ar1_feature(phi, 11, 42);
        assert!(
            (persistent - iid).abs() < 0.15 * iid,
            "phi {phi}: persistent MDA {persistent} vs iid MDA {iid}"
        );
    }
    // At phi = 0.99 a 400-row fold spans only a few decorrelation times, so a within-fold
    // shuffle draws from a narrower distribution than the whole sample and some understatement
    // is inherent to Snippet 8.3 itself. It must still be nowhere near the shift's collapse.
    let very_persistent = mda_of_ar1_feature(0.99, 11, 42);
    assert!(very_persistent > 0.4 * iid, "phi 0.99: MDA {very_persistent} vs iid MDA {iid}");
}

#[test]
fn mda_is_reproducible_for_a_seed() {
    let a = mda_of_ar1_feature(0.9, 11, 3);
    let b = mda_of_ar1_feature(0.9, 11, 3);
    let c = mda_of_ar1_feature(0.9, 11, 4);
    assert_eq!(a.to_bits(), b.to_bits());
    assert_ne!(a.to_bits(), c.to_bits());
}

/// Returns a fixed number of probabilities, whatever it is asked to score.
struct FixedCount(usize);

impl SimpleClassifier for FixedCount {
    fn fit(&mut self, _x: &[Vec<f64>], _y: &[f64], _sample_weight: Option<&[f64]>) {}
    fn predict_proba(&self, _x: &[Vec<f64>]) -> Vec<f64> {
        vec![0.5; self.0]
    }
}

/// #184 item 2: these inputs used to panic (index out of bounds). They are now typed errors.
#[test]
fn bad_splits_weights_predictions_and_columns_are_errors_not_panics() {
    use openquant::cross_validation::CrossValidationError as Cv;
    use openquant::feature_importance::FeatureImportanceError as Fi;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    let x: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64, 1.0]).collect();
    let y = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let names: Vec<String> = ["a", "b"].map(String::from).to_vec();
    let good = vec![(vec![0, 1, 2], vec![3, 4, 5])];
    let out_of_range = vec![(vec![0, 1, 2], vec![3, 4, 6])];
    let short_weight = [1.0; 4];

    type Case<'a> = (&'a [(Vec<usize>, Vec<usize>)], Option<&'a [f64]>, usize, Fi);
    let cases: [Case; 3] = [
        (
            &out_of_range,
            None,
            3,
            Fi::CrossValidation(Cv::SplitIndexOutOfRange { index: 6, n_rows: 6 }),
        ),
        (
            &good,
            Some(&short_weight),
            3,
            Fi::CrossValidation(Cv::LengthMismatch { name: "sample_weight", len: 4, expected: 6 }),
        ),
        (&good, None, 1, Fi::CrossValidation(Cv::PredictionCountMismatch { expected: 3, got: 1 })),
    ];
    for (splits, sw, n_pred, expected) in cases {
        for scoring in [Scoring::Accuracy, Scoring::NegLogLoss, Scoring::F1] {
            let mda = catch_unwind(AssertUnwindSafe(|| {
                let mut model = FixedCount(n_pred);
                mean_decrease_accuracy(&mut model, &x, &y, &names, splits, sw, scoring, 1)
            }))
            .expect("MDA must not panic");
            assert_eq!(mda.unwrap_err(), expected);
            let sfi = catch_unwind(AssertUnwindSafe(|| {
                let mut model = FixedCount(n_pred);
                single_feature_importance(&mut model, &x, &y, &names, splits, sw, scoring)
            }))
            .expect("SFI must not panic");
            assert_eq!(sfi.unwrap_err(), expected);
        }
    }

    // Rows with no columns: PCA used to panic slicing the eigenvector columns.
    let no_cols = vec![Vec::<f64>::new(); 3];
    let pca = catch_unwind(|| get_orthogonal_features(&no_cols, 0.95)).expect("must not panic");
    assert_eq!(pca.unwrap_err(), Fi::Empty("feature columns"));
    let corr = catch_unwind(|| feature_pca_analysis(&no_cols, &[], 0.95)).expect("must not panic");
    assert_eq!(corr.unwrap_err(), Fi::Empty("feature columns"));
}
