use openquant::fingerprint::{
    ClassificationModelFingerprint, ClassificationPredictor, RegressionModelFingerprint,
    RegressionPredictor,
};

struct LinearRegModel {
    w: Vec<f64>,
}

impl RegressionPredictor for LinearRegModel {
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|r| r.iter().zip(self.w.iter()).map(|(a, b)| a * b).sum::<f64>())
            .collect()
    }
}

struct NonLinearRegModel;

impl RegressionPredictor for NonLinearRegModel {
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|r| 2.0 * r[0] + 0.6 * r[1] * r[1] + 1.1 * r[0] * r[2])
            .collect()
    }
}

struct NonLinearClsModel;

impl ClassificationPredictor for NonLinearClsModel {
    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|r| {
                let z = 1.5 * r[0] + 0.8 * r[1] * r[1] + 0.5 * r[0] * r[2] - 0.2 * r[3];
                1.0 / (1.0 + (-z).exp())
            })
            .collect()
    }
}

fn synthetic_x(n: usize, p: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            (0..p)
                .map(|j| ((i * (j + 3) + j * 7) % 31) as f64 / 31.0 - 0.5)
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn test_linear_effect() {
    let x = synthetic_x(120, 13);
    let mut fp = RegressionModelFingerprint::new();

    let reg_rf_like = NonLinearRegModel;
    fp.fit(&reg_rf_like, &x, 20, None).unwrap();
    let (linear_effect, _, _) = fp.get_effects().unwrap();
    assert!(linear_effect.norm[&0] > 0.20);
    assert!(linear_effect.norm[&1] > 0.01);

    let reg_linear = LinearRegModel {
        w: vec![1.2, 0.0, 0.8, 0.0, 0.0, 2.4, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
    };
    fp.fit(&reg_linear, &x, 20, None).unwrap();
    let (linear_effect_linear, _, _) = fp.get_effects().unwrap();
    let linear_effect_linear_norm = linear_effect_linear.norm.clone();
    assert!(linear_effect_linear.norm[&5] > linear_effect_linear.norm[&2]);
    assert!(linear_effect_linear.norm[&2] > linear_effect_linear.norm[&1]);

    fp.fit(&reg_linear, &x, 70, None).unwrap();
    let (linear_effect_70, _, _) = fp.get_effects().unwrap();
    for feature in [0usize, 5, 6, 12] {
        assert!(
            (linear_effect_linear_norm[&feature] - linear_effect_70.norm[&feature]).abs() < 0.05
        );
    }
}

#[test]
fn test_non_linear_effect() {
    let x = synthetic_x(100, 13);
    let mut fp = RegressionModelFingerprint::new();

    fp.fit(&NonLinearRegModel, &x, 20, None).unwrap();
    let (_, non_linear_effect, _) = fp.get_effects().unwrap();
    assert!(non_linear_effect.raw[&1] > 0.01);

    let linear = LinearRegModel {
        w: vec![1.0; 13],
    };
    fp.fit(&linear, &x, 20, None).unwrap();
    let (_, non_linear_linear_model, _) = fp.get_effects().unwrap();
    let non_linear_linear_model_raw = non_linear_linear_model.raw.clone();
    for v in non_linear_linear_model.raw.values() {
        assert!(v.abs() < 1e-8);
    }

    fp.fit(&linear, &x, 70, None).unwrap();
    let (_, non_linear_70, _) = fp.get_effects().unwrap();
    for feature in [0usize, 5, 6, 12] {
        assert!((non_linear_linear_model_raw[&feature] - non_linear_70.raw[&feature]).abs() < 0.05);
    }
}

#[test]
fn test_pairwise_effect() {
    let x = synthetic_x(100, 13);
    let pairs = vec![(0usize, 2usize), (1, 3), (5, 7)];
    let mut fp = RegressionModelFingerprint::new();

    fp.fit(&NonLinearRegModel, &x, 20, Some(&pairs)).unwrap();
    let (_, _, pairwise) = fp.get_effects().unwrap();
    let p = pairwise.unwrap();
    assert!(p.raw["(0, 2)"] > 0.01);

    let linear = LinearRegModel {
        w: vec![1.0; 13],
    };
    fp.fit(&linear, &x, 20, Some(&pairs)).unwrap();
    let (_, _, pairwise_linear) = fp.get_effects().unwrap();
    let p_lin = pairwise_linear.unwrap();
    for pair in &pairs {
        let key = format!("({}, {})", pair.0, pair.1);
        assert!(p_lin.raw[&key].abs() < 1e-9);
    }
}

#[test]
fn test_classification_fingerprint() {
    let x = synthetic_x(160, 10);
    let clf = NonLinearClsModel;
    let mut fp = ClassificationModelFingerprint::new();
    fp.fit(&clf, &x, 20, Some(&[(0, 1), (2, 3), (8, 9)])).unwrap();

    let (linear, non_linear, pairwise) = fp.get_effects().unwrap();
    for feature in [0usize, 2, 3, 8, 9] {
        assert!(linear.raw[&feature].is_finite());
        assert!(non_linear.raw[&feature].is_finite());
    }
    let p = pairwise.unwrap();
    assert!(p.raw["(2, 3)"] >= 0.0);
    assert!(p.raw["(0, 1)"] >= 0.0);
}

#[test]
fn test_plot_effects() {
    let x = synthetic_x(100, 13);
    let mut fp = RegressionModelFingerprint::new();
    fp.fit(&NonLinearRegModel, &x, 20, None).unwrap();
    let lines = fp.plot_effects().unwrap();
    assert!(!lines.is_empty());

    fp.fit(&NonLinearRegModel, &x, 20, Some(&[(1, 2), (3, 5)])).unwrap();
    let lines_pair = fp.plot_effects().unwrap();
    assert!(lines_pair.iter().any(|s| s.contains("pairwise")));
}
