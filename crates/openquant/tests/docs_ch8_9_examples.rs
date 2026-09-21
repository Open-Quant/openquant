//! The Rust examples on the feature_importance, fingerprint and hyperparameter_tuning docs
//! pages, run as tests. `check:examples` only compiles page snippets; these execute them, so the
//! numbers quoted on those pages (and drawn in ch9-scoring.svg) are pinned.
//!
//! Each body is the page's snippet, reformatted by rustfmt. If you change one, change the other.
#![allow(clippy::type_complexity)]

#[test]
fn feature_importance_page() -> Result<(), Box<dyn std::error::Error>> {
    use chrono::{Duration, NaiveDate};
    use openquant::cross_validation::{PurgedKFold, Scoring, SimpleClassifier};
    use openquant::feature_importance::{mean_decrease_accuracy, single_feature_importance};

    /// Scores by the difference of class means, squashed by a sigmoid of sharpness `k`.
    struct MeanDiff {
        k: f64,
        w: Vec<f64>,
        b: f64,
    }

    impl SimpleClassifier for MeanDiff {
        fn fit(&mut self, x: &[Vec<f64>], y: &[f64], _sample_weight: Option<&[f64]>) {
            let m = x[0].len();
            let (mut pos, mut neg, mut n_pos, mut n_neg) = (vec![0.0; m], vec![0.0; m], 0.0, 0.0);
            for (row, label) in x.iter().zip(y) {
                let (sum, count) =
                    if *label > 0.5 { (&mut pos, &mut n_pos) } else { (&mut neg, &mut n_neg) };
                *count += 1.0;
                row.iter().enumerate().for_each(|(j, v)| sum[j] += v);
            }
            self.w = (0..m).map(|j| pos[j] / n_pos - neg[j] / n_neg).collect();
            self.b =
                -(0..m).map(|j| self.w[j] * (pos[j] / n_pos + neg[j] / n_neg) / 2.0).sum::<f64>();
        }
        fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
            let z = |r: &Vec<f64>| r.iter().zip(&self.w).map(|(a, b)| a * b).sum::<f64>() + self.b;
            x.iter().map(|r| 1.0 / (1.0 + (-self.k * z(r)).exp())).collect()
        }
    }

    // Reproducible noise in [-1, 1) from a hash, so the example needs no RNG.
    fn noise(i: usize, salt: u64) -> f64 {
        let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ salt.wrapping_mul(0xD1B5_4A32_D192_ED03);
        h ^= h >> 31;
        h = h.wrapping_mul(0x7FB5_D329_728E_A185);
        h ^= h >> 27;
        (h >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    // 600 samples: one strong feature, one weak, one irrelevant.
    let n = 600;
    let x: Vec<Vec<f64>> = (0..n).map(|i| vec![noise(i, 1), noise(i, 2), noise(i, 3)]).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| f64::from(u8::from(x[i][0] + 0.5 * x[i][1] + 0.4 * noise(i, 9) > 0.0)))
        .collect();
    let names: Vec<String> = ["strong", "weak", "noise"].map(String::from).to_vec();

    let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 0, 0).unwrap();
    let info: Vec<_> =
        (0..n as i64).map(|i| (open + Duration::hours(i), open + Duration::hours(i + 3))).collect();
    let splits = PurgedKFold::new(5, info, 0.0)?.split(n)?;

    let mut model = MeanDiff { k: 4.0, w: vec![], b: 0.0 };
    let mda =
        mean_decrease_accuracy(&mut model, &x, &y, &names, &splits, None, Scoring::NegLogLoss)?;
    let sfi =
        single_feature_importance(&mut model, &x, &y, &names, &splits, None, Scoring::NegLogLoss)?;

    assert!((mda["strong"].mean - 0.751).abs() < 1e-3);
    assert!((mda["weak"].mean - 0.234).abs() < 1e-3);
    assert!(mda["noise"].mean.abs() < 0.01);
    // Alone, the irrelevant feature scores a coin flip; the strong one is far better.
    assert!((sfi["noise"].mean + 0.696).abs() < 1e-3);
    assert!((sfi["strong"].mean + 0.330).abs() < 1e-3);
    Ok(())
}

#[test]
fn fingerprint_page() -> Result<(), Box<dyn std::error::Error>> {
    use openquant::fingerprint::{RegressionModelFingerprint, RegressionPredictor};

    struct Known;
    impl RegressionPredictor for Known {
        fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
            x.iter().map(|r| 2.0 * r[0] + r[1] * r[1] + r[0] * r[2]).collect()
        }
    }

    // Reproducible noise in [-1, 1) from a hash, so the example needs no RNG.
    fn noise(i: usize, salt: u64) -> f64 {
        let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ salt.wrapping_mul(0xD1B5_4A32_D192_ED03);
        h ^= h >> 31;
        h = h.wrapping_mul(0x7FB5_D329_728E_A185);
        h ^= h >> 27;
        (h >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    let x: Vec<Vec<f64>> = (0..600).map(|i| vec![noise(i, 1), noise(i, 2), noise(i, 3)]).collect();

    let mut fingerprint = RegressionModelFingerprint::new();
    fingerprint.fit(&Known, &x, 20, Some(&[(0, 2), (0, 1)]))?;
    let (linear, non_linear, pairwise) = fingerprint.get_effects()?;
    let pairwise = pairwise.expect("pairs were requested");

    // x0 carries the linear effect: mean |2v| over [-1, 1] is about 1.
    assert!((linear.raw[&0] - 1.058).abs() < 1e-3);
    assert!(linear.norm[&0] > 0.98);
    // x1 is the only non-linear feature; its linear effect is nil because x1^2 is symmetric.
    assert!(non_linear.norm[&1] > 0.999);
    assert!(linear.raw[&1] < 0.01);
    // x2 does nothing alone and everything with x0. Keys are the pair, formatted "(k, l)".
    assert!(linear.raw[&2] < 0.02 && non_linear.raw[&2] < 1e-12);
    assert!((pairwise.raw["(0, 2)"] - 0.281).abs() < 1e-3);
    assert!(pairwise.raw["(0, 1)"] < 1e-12);
    Ok(())
}

#[test]
fn hyperparameter_tuning_page() -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::BTreeMap;

    use chrono::{Duration, NaiveDate};
    use openquant::cross_validation::SimpleClassifier;
    use openquant::hyperparameter_tuning::{
        grid_search, randomized_search, HyperParamValue, ParamSet, RandomParamDistribution,
        SearchData, SearchScoring,
    };

    /// Scores by the difference of class means, squashed by a sigmoid of sharpness `k`.
    struct MeanDiff {
        k: f64,
        w: Vec<f64>,
        b: f64,
    }

    impl SimpleClassifier for MeanDiff {
        fn fit(&mut self, x: &[Vec<f64>], y: &[f64], _sample_weight: Option<&[f64]>) {
            let m = x[0].len();
            let (mut pos, mut neg, mut n_pos, mut n_neg) = (vec![0.0; m], vec![0.0; m], 0.0, 0.0);
            for (row, label) in x.iter().zip(y) {
                let (sum, count) =
                    if *label > 0.5 { (&mut pos, &mut n_pos) } else { (&mut neg, &mut n_neg) };
                *count += 1.0;
                row.iter().enumerate().for_each(|(j, v)| sum[j] += v);
            }
            self.w = (0..m).map(|j| pos[j] / n_pos - neg[j] / n_neg).collect();
            self.b =
                -(0..m).map(|j| self.w[j] * (pos[j] / n_pos + neg[j] / n_neg) / 2.0).sum::<f64>();
        }
        fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
            let z = |r: &Vec<f64>| r.iter().zip(&self.w).map(|(a, b)| a * b).sum::<f64>() + self.b;
            x.iter().map(|r| 1.0 / (1.0 + (-self.k * z(r)).exp())).collect()
        }
    }

    // Reproducible noise in [-1, 1) from a hash, so the example needs no RNG.
    fn noise(i: usize, salt: u64) -> f64 {
        let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ salt.wrapping_mul(0xD1B5_4A32_D192_ED03);
        h ^= h >> 31;
        h = h.wrapping_mul(0x7FB5_D329_728E_A185);
        h ^= h >> 27;
        (h >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    let n = 600;
    let x: Vec<Vec<f64>> = (0..n).map(|i| vec![noise(i, 1), noise(i, 2), noise(i, 3)]).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| f64::from(u8::from(x[i][0] + 0.5 * x[i][1] + 0.4 * noise(i, 9) > 0.0)))
        .collect();
    let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 0, 0).unwrap();
    let info: Vec<_> =
        (0..n as i64).map(|i| (open + Duration::hours(i), open + Duration::hours(i + 3))).collect();

    let data = || SearchData { x: &x, y: &y, sample_weight: None, samples_info_sets: &info };
    let build = |p: &ParamSet| MeanDiff { k: p["k"].as_f64().unwrap(), w: vec![], b: 0.0 };
    let grid = BTreeMap::from([(
        "k".to_string(),
        [0.5, 2.0, 4.0, 8.0, 32.0].map(HyperParamValue::Float).to_vec(),
    )]);

    let by_loss = grid_search(build, &grid, data(), 5, 0.01, SearchScoring::NegLogLoss)?;
    assert_eq!(by_loss.best_params["k"], HyperParamValue::Float(8.0));
    assert!((by_loss.best_score + 0.2238).abs() < 1e-4);

    // Accuracy cannot tell the five apart, and a tie goes to the last one tried.
    let by_accuracy = grid_search(build, &grid, data(), 5, 0.01, SearchScoring::Accuracy)?;
    assert!(by_accuracy.trials.iter().all(|t| (t.mean_score - 0.8933).abs() < 1e-4));
    assert_eq!(by_accuracy.best_params["k"], HyperParamValue::Float(32.0));

    // A log-uniform draw covers three orders of magnitude evenly and lands near the same optimum.
    let space = BTreeMap::from([(
        "k".to_string(),
        RandomParamDistribution::LogUniform { low: 0.1, high: 100.0 },
    )]);
    let random =
        randomized_search(build, &space, 12, 7, data(), 5, 0.01, SearchScoring::NegLogLoss)?;
    assert!((random.best_params["k"].as_f64().unwrap() - 8.253).abs() < 1e-3);
    Ok(())
}
