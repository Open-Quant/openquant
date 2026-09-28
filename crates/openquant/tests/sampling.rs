use openquant::sampling::{
    get_av_uniqueness_from_triple_barrier, get_ind_mat_average_uniqueness,
    get_ind_mat_label_uniqueness, get_ind_matrix, num_concurrent_events, seq_bootstrap,
    seq_bootstrap_with_rng,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn setup_labels() -> (Vec<usize>, Vec<(usize, usize)>) {
    // price bars hourly range 0..=168 (per test_sampling)
    let price_bars: Vec<usize> = (0..=168).collect();
    let t_events = [1, 2, 5, 7, 10, 11, 12, 20];
    let t1: Vec<(usize, usize)> = t_events.iter().map(|t| (*t, t + 2)).collect();
    (price_bars, t1)
}

fn book_ind_mat(bar_index: &[usize], label_endtime: &[(usize, usize)]) -> Vec<Vec<u8>> {
    let mut ind = vec![vec![0u8; label_endtime.len()]; bar_index.len()];
    for (i, (start, end)) in label_endtime.iter().enumerate() {
        for (row_idx, bar) in bar_index.iter().enumerate() {
            if *bar >= *start && *bar <= *end {
                ind[row_idx][i] = 1;
            }
        }
    }
    ind
}

#[test]
fn test_num_concurrent_events() {
    let (price_bars, t1) = setup_labels();
    let t_events: Vec<usize> = vec![1, 2, 5, 7, 10, 11, 12, 20];
    let num = num_concurrent_events(price_bars.len(), &t1, &t_events);
    let start = *t_events.first().unwrap();
    let end = t1.iter().map(|(_, e)| *e).max().unwrap();
    let slice = &num[start..=end];
    // value counts: 0 ->5, 1 ->11, 2 ->5, 3 ->1 (from Python test)
    let mut counts = std::collections::HashMap::new();
    for v in slice {
        *counts.entry(v).or_insert(0) += 1;
    }
    assert_eq!(*counts.get(&0).unwrap(), 5);
    assert_eq!(*counts.get(&1).unwrap(), 11);
    assert_eq!(*counts.get(&2).unwrap(), 5);
    assert_eq!(*counts.get(&3).unwrap(), 1);
}

#[test]
fn test_get_av_uniqueness() {
    let (price_bars, t1) = setup_labels();
    let av = get_av_uniqueness_from_triple_barrier(&t1, price_bars.len()).unwrap();
    assert_eq!(av.len(), t1.len());
    assert!((av[0] - 0.66).abs() < 1e-2);
    assert!((av[2] - 0.83).abs() < 1e-2);
    assert!((av[5] - 0.44).abs() < 1e-2);
    assert!((av.last().unwrap() - 1.0).abs() < 1e-2);
}

#[test]
fn test_seq_bootstrap_and_ind_matrix() {
    let (price_bars, t1) = setup_labels();
    let trimmed: Vec<usize> = price_bars
        .iter()
        .cloned()
        .filter(|t| *t >= t1.first().unwrap().0 && *t <= t1.last().unwrap().1)
        .collect();
    let mut bar_index = Vec::new();
    bar_index.extend(t1.iter().map(|(s, _)| *s));
    bar_index.extend(t1.iter().map(|(_, e)| *e));
    bar_index.extend(trimmed.clone());
    bar_index.sort();
    bar_index.dedup();

    let ind_mat = get_ind_matrix(&t1, &bar_index).unwrap();
    let book_ind = book_ind_mat(&bar_index, &t1);
    assert_eq!(ind_mat, book_ind);
    assert_eq!(ind_mat.len(), 22);
    assert_eq!(ind_mat[0][0], 1);
    assert_eq!(ind_mat[1][1], 1);
    assert_eq!(ind_mat[4][2], 1);
    assert_eq!(ind_mat[14][6], 0);

    let boot = seq_bootstrap(&ind_mat, None, None).unwrap();
    assert_eq!(boot.len(), t1.len());
    let boot2 = seq_bootstrap(&ind_mat, Some(100), None).unwrap();
    assert_eq!(boot2.len(), 100);

    // Book example
    let mut ind = vec![vec![0u8; 3]; 6];
    ind[0] = vec![1, 0, 0];
    ind[1] = vec![1, 0, 0];
    ind[2] = vec![1, 1, 0];
    ind[3] = vec![0, 1, 0];
    ind[4] = vec![0, 0, 1];
    ind[5] = vec![0, 0, 1];
    let _ = seq_bootstrap(&ind, Some(3), Some(vec![1])).unwrap();

    // Monte Carlo uniqueness comparison (AFML Snippet 4.9, section 4.5.3): on the book's
    // three-label example the sequential bootstrap should give samples with a higher average
    // uniqueness than the standard (uniform) bootstrap. The book reports roughly 0.6 vs 0.7.
    //
    // Enumerating every draw path gives the exact expectations for three draws:
    //   standard:   E[u] = 0.64198, sd = 0.15635  (27 equally likely samples)
    //   sequential: E[u] = 0.70564, sd = 0.13858
    // so the true gap is 0.0637. With N draws per side the standard error of the difference
    // of the two means is sqrt((0.15635^2 + 0.13858^2) / N). The old test used N = 100
    // unseeded draws: SE = 0.0209, so the gap was only ~3.0 SE and `avg_seq >= avg_std`
    // failed by chance about once in 900 runs (issue #223).
    //
    // Now both samplers are seeded, so the test is deterministic, and N = 20_000 keeps the
    // tolerances sound for any seed (or a change of RNG algorithm): SE of the difference is
    // 0.00148, so requiring a gap > 0.04 leaves a margin of 16 SE, and each mean is checked
    // against its exact value to within 0.01 (> 9 SE of either mean, whose SEs are 0.00111
    // and 0.00098).
    const DRAWS: usize = 20_000;
    let mut std_rng = StdRng::seed_from_u64(223);
    let mut seq_rng = StdRng::seed_from_u64(49);
    let mut standard_sum = 0.0;
    let mut seq_sum = 0.0;
    let columns = |samp: &[usize]| -> Vec<Vec<u8>> {
        ind.iter().map(|row| samp.iter().map(|c| row[*c]).collect()).collect()
    };
    for _ in 0..DRAWS {
        let boot_samp = seq_bootstrap_with_rng(&ind, Some(3), None, &mut seq_rng).unwrap();
        let random_samp: Vec<usize> = (0..3).map(|_| std_rng.random_range(0..3usize)).collect();
        standard_sum += get_ind_mat_average_uniqueness(&columns(&random_samp)).unwrap();
        seq_sum += get_ind_mat_average_uniqueness(&columns(&boot_samp)).unwrap();
    }
    let avg_seq = seq_sum / DRAWS as f64;
    let avg_std = standard_sum / DRAWS as f64;
    assert!((avg_std - 0.64198).abs() < 0.01, "standard bootstrap mean uniqueness {avg_std}");
    assert!((avg_seq - 0.70564).abs() < 0.01, "sequential bootstrap mean uniqueness {avg_seq}");
    assert!(
        avg_seq - avg_std > 0.04,
        "sequential ({avg_seq}) should beat standard ({avg_std}) average uniqueness"
    );
}

#[test]
fn test_get_ind_mat_uniqueness() {
    let mut ind = vec![vec![0u8; 3]; 6];
    ind[0] = vec![1, 0, 0];
    ind[1] = vec![1, 0, 0];
    ind[2] = vec![1, 1, 0];
    ind[3] = vec![0, 1, 0];
    ind[4] = vec![0, 0, 1];
    ind[5] = vec![0, 0, 1];
    // AFML section 4.5.3's worked example. By hand: bar 2 is shared by labels 0 and 1, so the
    // average uniqueness of the labels is (1 + 1 + 1/2) / 3 = 5/6, (1/2 + 1) / 2 = 3/4 and 1,
    // and the mean over the seven non-zero cells is 6/7 = 0.8571.
    let uniq = get_ind_mat_label_uniqueness(&ind).unwrap();
    let avg = get_ind_mat_average_uniqueness(&ind).unwrap();
    assert!(
        (uniq[0].iter().filter(|v| **v > 0.0).sum::<f64>()
            / uniq[0].iter().filter(|v| **v > 0.0).count() as f64
            - 0.8333)
            .abs()
            <= 1e-2
    );
    assert!(
        (uniq[1].iter().filter(|v| **v > 0.0).sum::<f64>()
            / uniq[1].iter().filter(|v| **v > 0.0).count() as f64
            - 0.75)
            .abs()
            <= 1e-2
    );
    assert!(
        (uniq[2].iter().filter(|v| **v > 0.0).sum::<f64>()
            / uniq[2].iter().filter(|v| **v > 0.0).count() as f64
            - 1.0)
            .abs()
            <= 1e-2
    );
    assert!((avg - 0.8571).abs() <= 1e-2);
}

#[test]
fn test_bootstrap_loop_run() {
    let mut ind = vec![vec![0u8; 3]; 6];
    ind[0] = vec![1, 0, 0];
    ind[1] = vec![1, 0, 0];
    ind[2] = vec![1, 1, 0];
    ind[3] = vec![0, 1, 0];
    ind[4] = vec![0, 0, 1];
    ind[5] = vec![0, 0, 1];
    let mut prev_conc = vec![0.0; ind.len()];
    let first = openquant::sampling::bootstrap_loop_run(&ind, &prev_conc).unwrap();
    assert_eq!(first, vec![1.0, 1.0, 1.0]);
    for i in 0..ind.len() {
        prev_conc[i] += ind[i][1] as f64;
    }
    let second = openquant::sampling::bootstrap_loop_run(&ind, &prev_conc).unwrap();
    let sum: f64 = second.iter().sum();
    let probs: Vec<f64> = second.iter().map(|v| *v / sum).collect();
    // AFML section 4.5.3: after drawing label 1, the next draw's probabilities are
    // (5/6, 1/2, 1) / (7/3) = (5/14, 3/14, 6/14).
    let target = [5.0 / 14.0, 3.0 / 14.0, 6.0 / 14.0];
    for (p, t) in probs.iter().zip(target.iter()) {
        assert!((p - t).abs() <= 1e-6);
    }
}

#[test]
fn test_value_error_raise() {
    let (_, t1) = setup_labels();
    // create NaN-like by invalid bounds
    let mut bad = t1.clone();
    bad.push((9999, 1));
    // should panic or error
    let bar_index: Vec<usize> = (0..10).collect();
    let res = std::panic::catch_unwind(|| get_ind_matrix(&bad, &bar_index).unwrap());
    assert!(res.is_err());
}
