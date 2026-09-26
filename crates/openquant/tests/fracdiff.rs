use csv::ReaderBuilder;
use openquant::fracdiff::{frac_diff, frac_diff_ffd, get_weights, get_weights_ffd};

fn load_close_series() -> Vec<f64> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/shared/dollar_bar_sample.csv");
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path).unwrap();
    let mut close = Vec::new();
    for rec in rdr.records() {
        let rec = rec.unwrap();
        close.push(rec[4].parse::<f64>().unwrap());
    }
    close
}

#[test]
fn test_get_weights() {
    let weights = get_weights(0.9, 100);
    assert_eq!(weights.len(), 100);
    assert_eq!(*weights.last().unwrap(), 1.0);
}

#[test]
fn test_get_weights_ffd() {
    let weights = get_weights_ffd(0.9, 1e-3, 100);
    assert_eq!(weights.len(), 12);
    assert_eq!(*weights.last().unwrap(), 1.0);
}

#[test]
fn test_frac_diff() {
    let data = load_close_series();
    for i in 1..10 {
        let diff_amt = i as f64 / 10.0;
        let fd = frac_diff(&data, diff_amt, 0.01);
        assert_eq!(fd.len(), data.len());
        assert!(fd[0].is_nan());
    }
}

#[test]
fn test_frac_diff_ffd() {
    let data = load_close_series();
    for i in 1..10 {
        let diff_amt = i as f64 / 10.0;
        let fd = frac_diff_ffd(&data, diff_amt, 1e-5);
        assert_eq!(fd.len(), data.len());
        assert!(fd[0].is_nan());
    }
}

/// Runs `f` on a worker thread and fails the test if it does not finish within `secs`.
/// Guards the regressions below, which used to loop until memory ran out (#169).
fn within_deadline<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(std::time::Duration::from_secs(secs))
        .expect("did not return within the deadline: the lim cap was ignored")
}

#[test]
fn test_get_weights_ffd_honours_lim_of_one() {
    // #169: with lim == 1 the cap was never checked, so thresh = 0 and a non-integer d
    // expanded forever. With a positive threshold it stopped, but returned more than one
    // weight; that case is checked first so the old code fails fast.
    assert_eq!(within_deadline(5, || get_weights_ffd(0.5, 1e-2, 1)), vec![1.0]);
    assert_eq!(within_deadline(5, || get_weights_ffd(0.5, 0.0, 1)), vec![1.0]);
}

#[test]
fn test_get_weights_ffd_never_exceeds_lim() {
    for lim in 0..=20 {
        for thresh in [1e-2, 1e-12, 0.0, -1.0, f64::NAN] {
            let w = within_deadline(5, move || get_weights_ffd(0.3, thresh, lim));
            assert!(w.len() <= lim, "lim {lim}, thresh {thresh}: {} weights", w.len());
            if thresh <= 0.0 || thresh.is_nan() {
                assert_eq!(w.len(), lim, "a non-positive threshold should run to the cap");
            }
            if lim > 0 {
                assert_eq!(*w.last().unwrap(), 1.0);
            }
        }
    }
}

#[test]
fn test_frac_diff_ffd_on_a_single_value_with_zero_threshold() {
    // frac_diff_ffd passes lim = series.len(), so a one-element series hit the lim == 1 case.
    // Old code: a positive threshold gave a window wider than the series (all NaN) ...
    let out = within_deadline(5, || frac_diff_ffd(&[4.0], 0.5, 1e-3));
    assert_eq!(out, vec![4.0]);
    // ... and a zero threshold never returned.
    let out = within_deadline(5, || frac_diff_ffd(&[4.0], 0.5, 0.0));
    assert_eq!(out, vec![4.0]);
}
