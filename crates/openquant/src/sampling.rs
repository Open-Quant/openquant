use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;

/// Indicator matrix (rows=bar_index, cols=labels), values 0/1.
pub fn get_ind_matrix(label_endtime: &[(usize, usize)], bar_index: &[usize]) -> Vec<Vec<u8>> {
    // validate
    for (s, e) in label_endtime {
        assert!(s <= e, "label endtime out of order");
    }
    let mut ind = vec![vec![0u8; label_endtime.len()]; bar_index.len()];
    for (col, (start, end)) in label_endtime.iter().enumerate() {
        for (row_idx, bar) in bar_index.iter().enumerate() {
            if *bar >= *start && *bar <= *end {
                ind[row_idx][col] = 1;
            }
        }
    }
    ind
}

/// Average uniqueness of indicator matrix (single value).
pub fn get_ind_mat_average_uniqueness(ind_mat: &[Vec<u8>]) -> f64 {
    let cols = ind_mat.first().map(|r| r.len()).unwrap_or(0);
    if cols == 0 {
        return 0.0;
    }
    let mut uniq_sum = 0.0;
    let mut count = 0;
    for col in 0..cols {
        let mut numer = 0.0;
        let mut denom = 0.0;
        for row in ind_mat {
            if row[col] == 1 {
                let conc = row.iter().map(|v| *v as f64).sum::<f64>();
                numer += 1.0 / conc;
                denom += 1.0;
            }
        }
        if denom > 0.0 {
            uniq_sum += numer / denom;
            count += 1;
        }
    }
    if count > 0 {
        uniq_sum / count as f64
    } else {
        0.0
    }
}

/// Per-label uniqueness series.
pub fn get_ind_mat_label_uniqueness(ind_mat: &[Vec<u8>]) -> Vec<Vec<f64>> {
    let cols = ind_mat.first().map(|r| r.len()).unwrap_or(0);
    let mut out = vec![Vec::new(); cols];
    for col in 0..cols {
        let mut vals = Vec::new();
        for row in ind_mat {
            if row[col] == 1 {
                let conc = row.iter().map(|v| *v as f64).sum::<f64>();
                vals.push(1.0 / conc);
            } else {
                vals.push(0.0);
            }
        }
        out[col] = vals;
    }
    out
}

/// Core step from sequential bootstrap: average uniqueness given current concurrency.
pub fn bootstrap_loop_run(ind_mat: &[Vec<u8>], prev_concurrency: &[f64]) -> Vec<f64> {
    let cols = ind_mat.first().map(|r| r.len()).unwrap_or(0);
    let mut avg_unique = vec![0.0; cols];
    for i in 0..cols {
        let mut prev_avg = 0.0;
        let mut n = 0.0;
        for (j, row) in ind_mat.iter().enumerate() {
            let val = row[i] as f64;
            if val > 0.0 {
                let new_el = val / (val + prev_concurrency[j]);
                let avg = (prev_avg * n + new_el) / (n + 1.0);
                n += 1.0;
                prev_avg = avg;
            }
        }
        avg_unique[i] = prev_avg;
    }
    avg_unique
}

/// Sequential bootstrap (indices of samples).
pub fn seq_bootstrap(
    ind_mat: &[Vec<u8>],
    sample_length: Option<usize>,
    warmup_samples: Option<Vec<usize>>,
) -> Vec<usize> {
    let n_labels = ind_mat.first().map(|r| r.len()).unwrap_or(0);
    let target_len = sample_length.unwrap_or(n_labels);
    let mut phi: Vec<usize> = Vec::new();
    let mut warm = warmup_samples.unwrap_or_default();
    let mut prev_conc = vec![0.0; ind_mat.len()];

    while phi.len() < target_len {
        let avg_unique = bootstrap_loop_run(ind_mat, &prev_conc);
        let sum: f64 = avg_unique.iter().sum();
        let prob_iter = avg_unique.iter().map(|p| if sum > 0.0 { *p / sum } else { 1.0 });
        let dist = WeightedIndex::new(prob_iter).unwrap();
        let mut rng = thread_rng();
        let choice = warm.pop().unwrap_or_else(|| dist.sample(&mut rng));
        phi.push(choice);
        for (i, row) in ind_mat.iter().enumerate() {
            prev_conc[i] += row[choice] as f64;
        }
    }
    phi
}

/// Average uniqueness from triple barrier events (index + t1).
pub fn get_av_uniqueness_from_triple_barrier(
    samples_info: &[(usize, usize)],
    price_bars_len: usize,
) -> Vec<f64> {
    let bars: Vec<usize> = (0..price_bars_len).collect();
    let ind = get_ind_matrix(samples_info, &bars);
    let uniq = get_ind_mat_label_uniqueness(&ind);
    uniq.iter()
        .map(|u| {
            let sum: f64 = u.iter().filter(|v| **v > 0.0).sum();
            let cnt = u.iter().filter(|v| **v > 0.0).count() as f64;
            if cnt > 0.0 {
                sum / cnt
            } else {
                0.0
            }
        })
        .collect()
}

/// Number of concurrent events per bar.
pub fn num_concurrent_events(
    price_index_len: usize,
    t1: &[(usize, usize)],
    _t_events: &[usize],
) -> Vec<usize> {
    let mut counts = vec![0usize; price_index_len];
    for &(start, end) in t1 {
        if start > end {
            continue;
        }
        let end_idx = end.min(price_index_len - 1);
        for i in start..=end_idx {
            counts[i] += 1;
        }
    }
    counts
}
