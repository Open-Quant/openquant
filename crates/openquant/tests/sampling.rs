use openquant::sampling::{
    get_av_uniqueness_from_triple_barrier, get_ind_mat_average_uniqueness,
    get_ind_mat_label_uniqueness, get_ind_matrix, num_concurrent_events, seq_bootstrap,
};

fn setup_labels() -> (Vec<usize>, Vec<(usize, usize)>) {
    // price bars hourly range 0..=168 (per test_sampling)
    let price_bars: Vec<usize> = (0..=168).collect();
    let t_events = vec![1, 2, 5, 7, 10, 11, 12, 20];
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
    let av = get_av_uniqueness_from_triple_barrier(&t1, price_bars.len());
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

    let ind_mat = get_ind_matrix(&t1, &bar_index);
    let book_ind = book_ind_mat(&bar_index, &t1);
    assert_eq!(ind_mat, book_ind);
    assert_eq!(ind_mat.len(), 22);
    assert_eq!(ind_mat[0][0], 1);
    assert_eq!(ind_mat[1][1], 1);
    assert_eq!(ind_mat[4][2], 1);
    assert_eq!(ind_mat[14][6], 0);

    let boot = seq_bootstrap(&ind_mat, None, None);
    assert_eq!(boot.len(), t1.len());
    let boot2 = seq_bootstrap(&ind_mat, Some(100), None);
    assert_eq!(boot2.len(), 100);

    // Book example
    let mut ind = vec![vec![0u8; 3]; 6];
    ind[0] = vec![1, 0, 0];
    ind[1] = vec![1, 0, 0];
    ind[2] = vec![1, 1, 0];
    ind[3] = vec![0, 1, 0];
    ind[4] = vec![0, 0, 1];
    ind[5] = vec![0, 0, 1];
    let _ = seq_bootstrap(&ind, Some(3), Some(vec![1]));

    // Monte Carlo uniqueness comparison
    let mut standard_unq = Vec::new();
    let mut seq_unq = Vec::new();
    for _ in 0..100 {
        let boot_samp = seq_bootstrap(&ind, Some(3), None);
        let random_samp: Vec<usize> = (0..3).map(|_| rand::random::<usize>() % 3).collect();
        standard_unq.push(get_ind_mat_average_uniqueness(
            &ind.iter()
                .map(|row| random_samp.iter().map(|c| row[*c]).collect())
                .collect::<Vec<Vec<u8>>>(),
        ));
        seq_unq.push(get_ind_mat_average_uniqueness(
            &ind.iter()
                .map(|row| boot_samp.iter().map(|c| row[*c]).collect())
                .collect::<Vec<Vec<u8>>>(),
        ));
    }
    let avg_seq = seq_unq.iter().sum::<f64>() / seq_unq.len() as f64;
    let avg_std = standard_unq.iter().sum::<f64>() / standard_unq.len() as f64;
    assert!(avg_seq >= avg_std);
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
    let uniq = get_ind_mat_label_uniqueness(&ind);
    let avg = get_ind_mat_average_uniqueness(&ind);
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
    let first = openquant::sampling::bootstrap_loop_run(&ind, &prev_conc);
    assert_eq!(first, vec![1.0, 1.0, 1.0]);
    for i in 0..ind.len() {
        prev_conc[i] += ind[i][1] as f64;
    }
    let second = openquant::sampling::bootstrap_loop_run(&ind, &prev_conc);
    let sum: f64 = second.iter().sum();
    let probs: Vec<f64> = second.iter().map(|v| *v / sum).collect();
    let target = vec![0.35714286, 0.21428571, 0.42857143];
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
    let res = std::panic::catch_unwind(|| get_ind_matrix(&bad, &bar_index));
    assert!(res.is_err());
}
