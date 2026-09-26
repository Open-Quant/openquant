//! The program behind the examples on the `cross_validation` docs page.
//! Its output is quoted there; `tests/cross_validation.rs` pins the same values.

use chrono::{Duration, NaiveDate, NaiveDateTime};
use openquant::cross_validation::{ml_cross_val_score, PurgedKFold, Scoring, SimpleClassifier};

/// Predicts the training base rate, whatever the features.
struct BaseRate(f64);

impl SimpleClassifier for BaseRate {
    fn fit(&mut self, _x: &[Vec<f64>], y: &[f64], _sample_weight: Option<&[f64]>) {
        self.0 = y.iter().sum::<f64>() / y.len() as f64;
    }
    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        vec![self.0; x.len()]
    }
}

/// 40 hourly labels, each resolved 3 hours after it starts.
pub fn info_sets() -> Vec<(NaiveDateTime, NaiveDateTime)> {
    let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 0, 0).unwrap();
    (0..40).map(|i| (open + Duration::hours(i), open + Duration::hours(i + 3))).collect()
}

fn ranges(idx: &[usize]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut start = idx[0];
    for w in 0..idx.len() {
        if w + 1 == idx.len() || idx[w + 1] != idx[w] + 1 {
            out.push(if start == idx[w] {
                format!("{start}")
            } else {
                format!("{start}-{}", idx[w])
            });
            if w + 1 < idx.len() {
                start = idx[w + 1];
            }
        }
    }
    out.join(", ")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for pct_embargo in [0.0, 0.07, 0.15] {
        let cv = PurgedKFold::new(5, info_sets(), pct_embargo)?;
        let (train, test) = &cv.split(40)?[2];
        println!(
            "embargo {pct_embargo:.2}: test {}  train {}  ({} of 32 kept)",
            ranges(test),
            ranges(train),
            train.len()
        );
    }

    let x: Vec<Vec<f64>> = (0..40).map(|i| vec![i as f64]).collect();
    let y: Vec<f64> = (0..40).map(|i| f64::from(i % 4 == 0)).collect();
    let splits = PurgedKFold::new(5, info_sets(), 0.0)?.split(40)?;
    let scores =
        ml_cross_val_score(&mut BaseRate(0.0), &x, &y, None, &splits, Scoring::NegLogLoss)?;
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    println!("neg log loss per fold: {scores:.4?}  mean {mean:.4}");
    Ok(())
}
