use chrono::NaiveDateTime;

/// Simple classifier interface for cross-validation.
pub trait SimpleClassifier {
    fn fit(&mut self, x: &[Vec<f64>], y: &[f64], sample_weight: Option<&[f64]>);
    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64>;
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        self.predict_proba(x).into_iter().map(|p| if p >= 0.5 { 1.0 } else { 0.0 }).collect()
    }
}

#[derive(Clone, Copy)]
pub enum Scoring {
    Accuracy,
    NegLogLoss,
}

pub fn ml_cross_val_score<C: SimpleClassifier>(
    classifier: &mut C,
    x: &[Vec<f64>],
    y: &[f64],
    sample_weight: Option<&[f64]>,
    splits: &[(Vec<usize>, Vec<usize>)],
    scoring: Scoring,
) -> Vec<f64> {
    let mut scores = Vec::new();
    for (train_idx, test_idx) in splits {
        let x_train: Vec<Vec<f64>> = train_idx.iter().map(|i| x[*i].clone()).collect();
        let y_train: Vec<f64> = train_idx.iter().map(|i| y[*i]).collect();
        let sw_train: Option<Vec<f64>> =
            sample_weight.map(|sw| train_idx.iter().map(|i| sw[*i]).collect());

        classifier.fit(&x_train, &y_train, sw_train.as_deref());
        let x_test: Vec<Vec<f64>> = test_idx.iter().map(|i| x[*i].clone()).collect();
        let y_test: Vec<f64> = test_idx.iter().map(|i| y[*i]).collect();
        let probs = classifier.predict_proba(&x_test);
        let preds: Vec<f64> = probs.iter().map(|p| if *p >= 0.5 { 1.0 } else { 0.0 }).collect();

        let score = match scoring {
            Scoring::Accuracy => {
                let correct = preds
                    .iter()
                    .zip(y_test.iter())
                    .filter(|(p, y_true)| (**p - *y_true).abs() < 1e-12)
                    .count();
                correct as f64 / y_test.len() as f64
            }
            Scoring::NegLogLoss => {
                let eps = 1e-15;
                let mut loss = 0.0;
                for (p, y_true) in probs.iter().zip(y_test.iter()) {
                    let p_clip = p.max(eps).min(1.0 - eps);
                    loss += -(*y_true * p_clip.ln() + (1.0 - *y_true) * (1.0 - p_clip).ln());
                }
                -(loss / y_test.len() as f64)
            }
        };
        scores.push(score);
    }
    scores
}

/// Remove training intervals that overlap with test intervals.
pub fn ml_get_train_times(
    info_sets: &[(NaiveDateTime, NaiveDateTime)],
    test_times: &[(NaiveDateTime, NaiveDateTime)],
) -> Vec<(NaiveDateTime, NaiveDateTime)> {
    let mut out = Vec::new();
    for (start, end) in info_sets {
        let mut keep = true;
        for (test_start, test_end) in test_times {
            let start_in = *start >= *test_start && *start <= *test_end;
            let end_in = *end >= *test_start && *end <= *test_end;
            let envelop = *start <= *test_start && *end >= *test_end;
            if start_in || end_in || envelop {
                keep = false;
                break;
            }
        }
        if keep {
            out.push((*start, *end));
        }
    }
    out
}

pub struct PurgedKFold {
    n_splits: usize,
    samples_info_sets: Vec<(NaiveDateTime, NaiveDateTime)>,
    pct_embargo: f64,
}

impl PurgedKFold {
    pub fn new(
        n_splits: usize,
        samples_info_sets: Vec<(NaiveDateTime, NaiveDateTime)>,
        pct_embargo: f64,
    ) -> Result<Self, String> {
        if samples_info_sets.is_empty() {
            return Err("samples_info_sets cannot be empty".into());
        }
        Ok(Self { n_splits, samples_info_sets, pct_embargo })
    }

    pub fn split(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>, String> {
        if n_samples != self.samples_info_sets.len() {
            return Err("Dataset length must match samples_info_sets".into());
        }
        let n = n_samples;
        let mut fold_sizes = vec![n / self.n_splits; self.n_splits];
        for i in 0..(n % self.n_splits) {
            fold_sizes[i] += 1;
        }
        let mut current = 0;
        let mut splits = Vec::new();
        for fold_size in fold_sizes {
            let start = current;
            let stop = current + fold_size;
            let test_indices: Vec<usize> = (start..stop).collect();
            let mut train_mask = vec![true; n];

            // purge overlaps
            let test_start = self.samples_info_sets[test_indices[0]].1;
            let test_end = self.samples_info_sets[*test_indices.last().unwrap()].1;
            for (i, (s, e)) in self.samples_info_sets.iter().enumerate() {
                let start_in = *s >= test_start && *s <= test_end;
                let end_in = *e >= test_start && *e <= test_end;
                let envelop = *s <= test_start && *e >= test_end;
                if start_in || end_in || envelop {
                    train_mask[i] = false;
                }
            }

            // embargo
            let embargo = (self.pct_embargo * n as f64).ceil() as isize;
            if embargo > 0 {
                let after = (stop as isize + embargo).min(n as isize);
                let before = (start as isize - embargo).max(0);
                for i in start..(after as usize) {
                    if i < n {
                        train_mask[i] = false;
                    }
                }
                for i in before as usize..start {
                    train_mask[i] = false;
                }
            }

            let train_indices: Vec<usize> = train_mask
                .iter()
                .enumerate()
                .filter_map(|(i, keep)| if *keep { Some(i) } else { None })
                .collect();
            splits.push((train_indices, test_indices));
            current = stop;
        }
        Ok(splits)
    }
}
