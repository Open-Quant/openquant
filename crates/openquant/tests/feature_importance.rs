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

fn make_dataset() -> (Vec<Vec<f64>>, Vec<f64>, Vec<String>, Vec<(Vec<usize>, Vec<usize>)>) {
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
    assert!(pca[0].len() >= 1);

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
    let mda =
        mean_decrease_accuracy(&mut clf, &x, &y, &names, &splits, None, Scoring::Accuracy).unwrap();
    assert!(mda["f0"].mean > mda["f2"].mean);

    let mut clf2 = LinearProbClassifier::new(names.len());
    let sfi =
        single_feature_importance(&mut clf2, &x, &y, &names, &splits, None, Scoring::Accuracy)
            .unwrap();
    assert!(sfi["f0"].mean >= sfi["f2"].mean);
}

#[test]
fn test_plot_feature_importance_output_file() {
    let names = vec!["a".to_string(), "b".to_string()];
    let mdi = mean_decrease_impurity(&[vec![0.7, 0.3], vec![0.6, 0.4]], &names).unwrap();
    let out = "/tmp/openquant_feature_importance_test.csv";
    let _ = std::fs::remove_file(out);
    plot_feature_importance(&mdi, 0.5, 0.4, Some(out)).unwrap();
    assert!(std::path::Path::new(out).exists());
    std::fs::remove_file(out).unwrap();
}
