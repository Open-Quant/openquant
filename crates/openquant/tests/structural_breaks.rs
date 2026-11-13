use csv::ReaderBuilder;
use openquant::structural_breaks::{
    _get_betas, _get_values_diff, get_chow_type_stat, get_chu_stinchcombe_white_statistics,
    get_sadf, SadfLags, StructuralBreakError,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde_json::Value;
use std::path::Path;

/// AFML chapter 17 recomputed in numpy by tests/fixtures/structural_breaks/generate.py.
fn reference() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/structural_breaks/reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn num(value: &Value, path: &[&str]) -> f64 {
    path.iter().fold(value, |v, key| &v[*key]).as_f64().unwrap_or_else(|| panic!("{path:?}"))
}

fn assert_rel(got: f64, want: f64, rel: f64, what: &str) {
    assert!(
        (got - want).abs() <= rel * want.abs().max(1.0),
        "{what}: got {got}, want {want} (rel tol {rel})"
    );
}

fn load_close_prices() -> Vec<f64> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/shared/dollar_bar_sample.csv");
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path).unwrap();
    let mut close = Vec::new();
    for rec in rdr.records() {
        let rec = rec.unwrap();
        close.push(rec[4].parse::<f64>().unwrap());
    }
    close
}

fn log_prices() -> Vec<f64> {
    load_close_prices().into_iter().map(|v| v.ln()).collect()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn max(values: &[f64]) -> f64 {
    values.iter().cloned().fold(f64::NAN, f64::max)
}

#[test]
fn test_chu_stinchcombe_value_diff_function() {
    let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let one_sided = _get_values_diff("one_sided", &series, 0, 1).unwrap();
    let two_sided = _get_values_diff("two_sided", &series, 0, 1).unwrap();

    assert_eq!(one_sided, -1.0);
    assert_eq!(two_sided, 1.0);

    let invalid = _get_values_diff("rubbish", &series, 0, 1);
    assert!(matches!(invalid, Err(StructuralBreakError::InvalidTestType(_))));
}

#[test]
fn test_chow_test() {
    let min_length = 10usize;
    let log_prices = log_prices();
    let stats = get_chow_type_stat(&log_prices, min_length).expect("chow stats");

    let reference = reference();
    let want = &reference["chow"];
    assert_eq!(log_prices.len() - min_length * 2, stats.len());
    assert_eq!(num(want, &["len"]) as usize, stats.len());
    assert_rel(max(&stats), num(want, &["max"]), 1e-8, "chow max");
    assert_rel(mean(&stats), num(want, &["mean"]), 1e-8, "chow mean");
    assert_rel(stats[3], num(want, &["at_3"]), 1e-8, "chow [3]");
}

#[test]
fn test_chu_stinchcombe_white_test() {
    let log_prices = log_prices();
    let one_sided =
        get_chu_stinchcombe_white_statistics(&log_prices, "one_sided").expect("one sided");
    let two_sided =
        get_chu_stinchcombe_white_statistics(&log_prices, "two_sided").expect("two sided");

    assert_eq!(log_prices.len() - 2, one_sided.critical_value.len());
    assert_eq!(log_prices.len() - 2, two_sided.critical_value.len());

    // AFML 17.3.2 values for the critical values and the statistic (#104 fixed sigma_t^2 for
    // sigma_t, #173 the divisor of sigma_t^2).
    let reference = reference();
    assert_csw_matches(&reference, &one_sided, &two_sided, "critical_value");
    assert_csw_matches(&reference, &one_sided, &two_sided, "stat");

    let invalid = get_chu_stinchcombe_white_statistics(&log_prices, "rubbish text");
    assert!(matches!(invalid, Err(StructuralBreakError::InvalidTestType(_))));
}

fn assert_csw_matches(
    reference: &Value,
    one_sided: &openquant::structural_breaks::ChuStinchcombeWhiteResult,
    two_sided: &openquant::structural_breaks::ChuStinchcombeWhiteResult,
    field: &str,
) {
    for (name, result) in [("one_sided", one_sided), ("two_sided", two_sided)] {
        let want = &reference["chu_stinchcombe_white"][name];
        let values = if field == "stat" { &result.stat } else { &result.critical_value };
        let what = format!("{name} {field}");
        assert_rel(max(values), num(want, &[field, "max"]), 1e-10, &format!("{what} max"));
        assert_rel(mean(values), num(want, &[field, "mean"]), 1e-10, &format!("{what} mean"));
        assert_rel(values[20], num(want, &[field, "at_20"]), 1e-10, &format!("{what} [20]"));
    }
}

/// #147's FINDING, fixed by #173: sigma_t^2 is the mean of the squared differences up to bar t
/// (AFML 17.3.2); it used to divide their sum by one fewer than their number.
#[test]
fn test_chu_stinchcombe_white_statistic_matches_afml() {
    let log_prices = log_prices();
    let one_sided = get_chu_stinchcombe_white_statistics(&log_prices, "one_sided").unwrap();
    let two_sided = get_chu_stinchcombe_white_statistics(&log_prices, "two_sided").unwrap();
    assert_csw_matches(&reference(), &one_sided, &two_sided, "stat");
}

#[test]
fn test_chu_stinchcombe_white_by_hand() {
    // y = 0, 1, 3. At t = 2 (0-based) the differences are 1 and 2, so sigma^2 = (1 + 4) / 2.
    // S = (3 - 0) / (sigma sqrt 2) = 3 / sqrt 5 against n = 0, and (3 - 1) / sigma = 4 / sqrt 10
    // against n = 1; the first is larger. The old divisor (1 instead of 2) gave 3 / sqrt 10.
    let out = get_chu_stinchcombe_white_statistics(&[0.0, 1.0, 3.0], "one_sided").unwrap();
    assert_eq!(out.stat.len(), 1);
    assert_rel(out.stat[0], 3.0 / 5f64.sqrt(), 1e-14, "S at t = 2");
    assert_rel(out.critical_value[0], (4.6 + 2f64.ln()).sqrt(), 1e-14, "c at t = 2");
}

#[test]
fn test_chu_stinchcombe_white_is_scale_invariant() {
    // A standardised statistic must not depend on the units of the series (#104).
    let log_prices = log_prices();
    for test_type in ["one_sided", "two_sided"] {
        let base = get_chu_stinchcombe_white_statistics(&log_prices, test_type).unwrap();
        for scale in [0.01, 100.0] {
            let scaled_prices = log_prices.iter().map(|v| v * scale).collect::<Vec<_>>();
            let scaled = get_chu_stinchcombe_white_statistics(&scaled_prices, test_type).unwrap();
            for (a, b) in base.stat.iter().zip(&scaled.stat) {
                assert!(
                    (a - b).abs() <= 1e-9 * a.abs().max(1.0),
                    "{test_type} x{scale}: {a} vs {b}"
                );
            }
            assert_eq!(base.critical_value, scaled.critical_value);
        }
    }
}

#[test]
fn test_chu_stinchcombe_white_size_on_random_walks() {
    // With no break, the one-sided statistic should rarely exceed its critical value, and how
    // rarely must not depend on the volatility of the walk (#104).
    for step_vol in [1.0, 0.01] {
        let mut rng = StdRng::seed_from_u64(104);
        let normal = Normal::new(0.0, step_vol).unwrap();
        let (mut above, mut total) = (0usize, 0usize);
        for _ in 0..20 {
            let mut y = vec![0.0];
            for _ in 0..300 {
                y.push(y.last().unwrap() + normal.sample(&mut rng));
            }
            let out = get_chu_stinchcombe_white_statistics(&y, "one_sided").unwrap();
            above += out.stat.iter().zip(&out.critical_value).filter(|(s, c)| s > c).count();
            total += out.stat.len();
        }
        let share = above as f64 / total as f64;
        assert!(share < 0.15, "step volatility {step_vol}: rejection share {share}");
    }
}

#[test]
#[ignore = "long-running hotspot; run explicitly with `cargo test -p openquant --test structural_breaks test_sadf_test -- --ignored`"]
fn test_sadf_test() {
    let log_prices = log_prices();
    let lags_int = 5usize;
    let lags_array = vec![1usize, 2, 5, 7];
    let min_length = 20usize;

    let sm_power = get_sadf(&log_prices, "sm_power", true, min_length, SadfLags::Fixed(lags_int))
        .expect("sm_power sadf");
    let linear = get_sadf(&log_prices, "linear", true, min_length, SadfLags::Fixed(lags_int))
        .expect("linear sadf");
    let linear_no_const =
        get_sadf(&log_prices, "linear", false, min_length, SadfLags::Array(lags_array.clone()))
            .expect("linear no const sadf");
    let quadratic = get_sadf(&log_prices, "quadratic", true, min_length, SadfLags::Fixed(lags_int))
        .expect("quadratic sadf");
    let sm_poly_1 = get_sadf(&log_prices, "sm_poly_1", true, min_length, SadfLags::Fixed(lags_int))
        .expect("sm_poly_1 sadf");
    let sm_poly_2 = get_sadf(&log_prices, "sm_poly_2", true, min_length, SadfLags::Fixed(lags_int))
        .expect("sm_poly_2 sadf");
    let sm_exp = get_sadf(&log_prices, "sm_exp", true, min_length, SadfLags::Fixed(lags_int))
        .expect("sm_exp sadf");

    let expected_len = log_prices.len() - min_length - lags_int - 1;
    assert_eq!(expected_len, sm_power.len());
    assert_eq!(expected_len, linear.len());
    assert_eq!(expected_len, quadratic.len());
    assert_eq!(expected_len, sm_poly_1.len());
    assert_eq!(expected_len, sm_poly_2.len());
    assert_eq!(expected_len, sm_exp.len());

    // AFML 17.4.2-17.4.3 values over the whole series (also checked value by value on a 60-bar
    // prefix, fast, in sadf_*_match_afml_on_prefix).
    let reference = reference();
    assert_sadf_matches(
        &reference,
        &[
            ("linear", &linear),
            ("linear_no_const", &linear_no_const),
            ("quadratic", &quadratic),
            ("sm_power", &sm_power),
            ("sm_poly_1", &sm_poly_1),
            ("sm_poly_2", &sm_poly_2),
            ("sm_exp", &sm_exp),
        ],
    );

    let ones = vec![1.0; log_prices.len()];
    let trivial =
        get_sadf(&ones, "sm_power", true, min_length, SadfLags::Fixed(lags_int)).expect("ones");
    assert!(trivial.iter().all(|v| v.is_infinite() && v.is_sign_negative()));

    let invalid =
        get_sadf(&log_prices, "rubbish_string", true, min_length, SadfLags::Fixed(lags_int));
    assert!(matches!(invalid, Err(StructuralBreakError::InvalidModel(_))));

    let singular_matrix = vec![vec![1.0, 0.0, 0.0], vec![-1.0, 3.0, 3.0], vec![1.0, 2.0, 2.0]];
    let (b_mean, b_var) = _get_betas(&singular_matrix, &singular_matrix).expect("betas");
    assert!(b_mean.iter().all(|v| v.is_nan()));
    assert!(b_var.iter().all(|row| row.iter().all(|v| v.is_nan())));
}

fn assert_sadf_matches(reference: &Value, models: &[(&str, &Vec<f64>)]) {
    for (name, values) in models {
        let want = &reference["sadf"]["models"][*name];
        assert_eq!(num(want, &["len"]) as usize, values.len(), "{name} len");
        // 1e-7: the normal-equations inverse (snippet 17.4, as in the library) is off from a
        // QR solve by up to ~2e-9 relative in these statistics.
        assert_rel(mean(values), num(want, &["mean"]), 1e-7, &format!("{name} mean"));
        assert_rel(max(values), num(want, &["max"]), 1e-7, &format!("{name} max"));
        assert_rel(values[29], num(want, &["at_29"]), 1e-7, &format!("{name} [29]"));
    }
}

/// SADF at a row only looks at earlier rows, so on the first 60 bars every value equals the
/// full-series value (the generator checks this); here each one is compared.
fn assert_sadf_prefix_matches(models: &[(&str, bool, SadfLags)]) {
    let reference = reference();
    let prefix = &reference["sadf_prefix"];
    let n_bars = num(prefix, &["n_bars"]) as usize;
    let log_prices = log_prices();
    for (name, add_const, lags) in models {
        let model = if name.starts_with("linear") { "linear" } else { name };
        let got = get_sadf(&log_prices[..n_bars], model, *add_const, 20, lags.clone()).unwrap();
        let want = prefix["models"][*name]["values"].as_array().unwrap();
        assert_eq!(want.len(), got.len(), "{name} len");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_rel(*g, w.as_f64().unwrap(), 1e-7, &format!("{name} prefix [{i}]"));
        }
    }
}

#[test]
fn sadf_linear_models_match_afml_on_prefix() {
    assert_sadf_prefix_matches(&[
        ("linear", true, SadfLags::Fixed(5)),
        ("linear_no_const", false, SadfLags::Array(vec![1, 2, 5, 7])),
    ]);
}

/// #166: 'quadratic' lacked the linear t of Snippet 17.2's 'ctt'; the sm_* models took the
/// sup of the signed beta/se where AFML 17.4.3 takes |beta|/se; sm_power took log(0) on the
/// first row. The full-series means are checked in test_sadf_test.
#[test]
fn sadf_quadratic_and_martingale_models_match_afml() {
    assert_sadf_prefix_matches(&[
        ("quadratic", true, SadfLags::Fixed(5)),
        ("sm_power", true, SadfLags::Fixed(5)),
        ("sm_poly_1", true, SadfLags::Fixed(5)),
        ("sm_poly_2", true, SadfLags::Fixed(5)),
        ("sm_exp", true, SadfLags::Fixed(5)),
    ]);
}

#[test]
fn sadf_quadratic_is_unchanged_by_a_quadratic_trend_in_levels() {
    // Adding c t^2 to y adds a quadratic in t to y_{t-1} and a linear one to dy_t. Both lie in
    // the span of the const, t and t^2 columns, so beta on y_{t-1} (and the statistic) cannot
    // move. Without the linear t column (before #166) it does.
    let y = log_prices()[..60].to_vec();
    let bent = y.iter().enumerate().map(|(t, v)| v + 1e-4 * (t * t) as f64).collect::<Vec<_>>();
    let base = get_sadf(&y, "quadratic", true, 20, SadfLags::Fixed(2)).unwrap();
    let moved = get_sadf(&bent, "quadratic", true, 20, SadfLags::Fixed(2)).unwrap();
    for (a, b) in base.iter().zip(&moved) {
        assert_rel(*b, *a, 1e-6, "quadratic under a quadratic trend");
    }
}

#[test]
fn sadf_martingale_statistics_ignore_the_sign_of_the_trend() {
    // AFML 17.4.3 takes |beta| / se: a series and its reciprocal (log y negated) give the same
    // statistic for the log models, and y and -y for the level model.
    let prices = load_close_prices()[..60].to_vec();
    let inverse = prices.iter().map(|p| 1.0 / p).collect::<Vec<_>>();
    let negated = prices.iter().map(|p| -p).collect::<Vec<_>>();
    let lags = SadfLags::Fixed(1);
    for (model, other) in [
        ("sm_poly_1", &negated),
        ("sm_poly_2", &inverse),
        ("sm_exp", &inverse),
        ("sm_power", &inverse),
    ] {
        let a = get_sadf(&prices, model, true, 20, lags.clone()).unwrap();
        let b = get_sadf(other, model, true, 20, lags.clone()).unwrap();
        assert!(a.iter().all(|v| v.is_finite() && *v >= 0.0), "{model}: {a:?}");
        for (x, y) in a.iter().zip(&b) {
            assert_rel(*y, *x, 1e-8, model);
        }
    }
}

#[test]
fn sadf_sm_power_uses_the_first_row() {
    // Time counts from 1, so the first row has log t = 0 and the window starting there is
    // used. Before #166 it was log 0 = -inf, and the regression starting at row 0 was dropped:
    // with min_length equal to the number of rows minus one, only that window and one other
    // exist.
    let prices = load_close_prices()[..30].to_vec();
    let rows = prices.len() - 2; // one lag
    let out = get_sadf(&prices, "sm_power", true, rows - 1, SadfLags::Fixed(1)).unwrap();
    assert_eq!(out.len(), 1);
    // The value is the larger |t| of the windows starting at rows 0 and 1, computed here.
    let y = prices[2..].iter().map(|p| vec![p.ln()]).collect::<Vec<_>>();
    let x = (0..rows).map(|i| vec![((i + 1) as f64).ln(), 1.0]).collect::<Vec<_>>();
    let t_stat = |start: usize| {
        let (b, v) = _get_betas(&x[start..], &y[start..]).unwrap();
        (b[0] / v[0][0].sqrt()).abs()
    };
    let (from_row_0, from_row_1) = (t_stat(0), t_stat(1));
    assert!(from_row_0.is_finite() && from_row_1.is_finite());
    assert_rel(out[0], from_row_0.max(from_row_1), 1e-12, "sm_power sup");
}

/// `get_sadf`'s search as it was before the windows shared running `XᵀX` / `Xᵀy` sums: a fresh
/// `_get_betas` regression per window, on the rows built by `sadf_rows`.
fn sadf_one_regression_per_window(
    x: &[Vec<f64>],
    y: &[f64],
    min_length: usize,
    absolute: bool,
) -> Vec<f64> {
    let mut out = Vec::new();
    for pos in min_length..y.len() {
        let (x_t, y_t) = (&x[..=pos], &y[..=pos]);
        let mut bsadf = f64::NEG_INFINITY;
        for start in 0..=(y_t.len() - min_length) {
            let y_subset = y_t[start..].iter().map(|v| vec![*v]).collect::<Vec<_>>();
            let (b_mean, b_var) = _get_betas(&x_t[start..], &y_subset).unwrap();
            if b_mean[0].is_nan() {
                continue;
            }
            let ratio = b_mean[0] / b_var[0][0].sqrt();
            let stat = if absolute { ratio.abs() } else { ratio };
            if stat > bsadf {
                bsadf = stat;
            }
        }
        out.push(bsadf);
    }
    out
}

/// The regression rows `get_sadf` builds (AFML Snippet 17.2), for the reference above.
fn sadf_rows(
    series: &[f64],
    model: &str,
    add_const: bool,
    lags: &[usize],
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let diff: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();
    let start = lags.iter().max().unwrap_or(&0) + 1;
    let mut x = Vec::new();
    let mut y = Vec::new();
    for (i, idx) in (start..series.len()).enumerate() {
        let t = i as f64;
        let lagged = lags.iter().map(|lag| diff[idx - lag - 1]);
        let (row, target) = match model {
            "linear" | "quadratic" => {
                let mut row = vec![series[idx - 1]];
                row.extend(lagged);
                if add_const {
                    row.push(1.0);
                }
                row.push(t);
                if model == "quadratic" {
                    row.push(t * t);
                }
                (row, diff[idx - 1])
            }
            "sm_poly_1" => (vec![t * t, 1.0, t], series[idx]),
            "sm_poly_2" => (vec![t * t, 1.0, t], series[idx].ln()),
            "sm_exp" => (vec![t, 1.0], series[idx].ln()),
            "sm_power" => (vec![((i + 1) as f64).ln(), 1.0], series[idx].ln()),
            _ => unreachable!(),
        };
        x.push(row);
        y.push(target);
    }
    (x, y)
}

/// `β₀ / se(β₀)` by Householder QR of `X`, which never forms `XᵀX`: a more accurate
/// reference than either normal-equations implementation.
fn qr_stat(x: &[Vec<f64>], y: &[f64]) -> Option<f64> {
    use nalgebra::{DMatrix, DVector};
    let (n, k) = (x.len(), x[0].len());
    if n <= k {
        return None;
    }
    let xm = DMatrix::from_fn(n, k, |r, c| x[r][c]);
    let yv = DVector::from_column_slice(y);
    let qr = xm.clone().qr();
    let r = qr.r();
    let beta = r.solve_upper_triangular(&(qr.q().transpose() * &yv))?;
    let e = &yv - &xm * &beta;
    let s2 = e.dot(&e) / (n - k) as f64;
    let r_inv = r.try_inverse()?;
    let var_b0 = (&r_inv * r_inv.transpose())[(0, 0)] * s2;
    Some(beta[0] / var_b0.sqrt())
}

/// The SADF search with [`qr_stat`] for each window.
fn sadf_by_qr(x: &[Vec<f64>], y: &[f64], min_length: usize, absolute: bool) -> Vec<f64> {
    (min_length..y.len())
        .map(|end| {
            (0..=end + 1 - min_length)
                .filter_map(|start| qr_stat(&x[start..=end], &y[start..=end]))
                .map(|s| if absolute { s.abs() } else { s })
                .fold(f64::NEG_INFINITY, |best, s| if s > best { s } else { best })
        })
        .collect()
}

#[test]
fn sadf_matches_one_regression_per_window() {
    // Accumulating XᵀX and Xᵀy one row at a time sums them in a different order from
    // nalgebra's product, so the statistics are not bit-for-bit the old ones. On well-posed
    // windows they agree to ~1e-10 relative or better. The quadratic model with no lags is
    // badly conditioned on short windows (y₋₁ barely moves, so it is almost collinear with the
    // constant, next to t and t²): there any normal-equations solve is only good to ~1e-4,
    // and old and new differ by about that much. Both are then about equally far from a QR
    // solve, so the change is rounding, not a loss of accuracy. Random walks in log price,
    // all six models.
    let mut rng = StdRng::seed_from_u64(1708);
    let normal = Normal::new(0.0, 0.01).unwrap();
    for model in ["linear", "quadratic", "sm_poly_1", "sm_poly_2", "sm_exp", "sm_power"] {
        let tol = if model == "quadratic" { 1e-3 } else { 1e-8 };
        let (mut new_vs_old, mut new_vs_qr, mut old_vs_qr) = (0.0f64, 0.0f64, 0.0f64);
        for case in 0..12 {
            let mut log_p = 4.6_f64;
            let prices: Vec<f64> = (0..30 + 4 * case)
                .map(|_| {
                    log_p += normal.sample(&mut rng);
                    log_p.exp()
                })
                .collect();
            let lags: Vec<usize> = match case % 3 {
                0 => vec![],
                1 => vec![1, 2],
                _ => vec![1, 3],
            };
            let add_const = case % 2 == 0;
            let min_length = 8 + case % 5;
            let absolute = model.starts_with("sm_");
            let got =
                get_sadf(&prices, model, add_const, min_length, SadfLags::Array(lags.clone()))
                    .unwrap();
            let (x, y) = sadf_rows(&prices, model, add_const, &lags);
            let old = sadf_one_regression_per_window(&x, &y, min_length, absolute);
            let qr = sadf_by_qr(&x, &y, min_length, absolute);
            assert_eq!(got.len(), old.len(), "{model} case {case}");
            for (i, ((g, o), q)) in got.iter().zip(&old).zip(&qr).enumerate() {
                if o.is_infinite() || g.is_infinite() {
                    assert_eq!(g, o, "{model} case {case} [{i}]");
                    continue;
                }
                let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1.0);
                assert!(rel(*g, *o) <= tol, "{model} case {case} [{i}]: got {g}, old {o}");
                new_vs_old = new_vs_old.max(rel(*g, *o));
                if q.is_finite() {
                    new_vs_qr = new_vs_qr.max(rel(*g, *q));
                    old_vs_qr = old_vs_qr.max(rel(*o, *q));
                }
            }
        }
        eprintln!(
            "{model}: new vs old {new_vs_old:.1e}, new vs QR {new_vs_qr:.1e}, old vs QR {old_vs_qr:.1e}"
        );
        assert!(
            new_vs_qr <= 4.0 * old_vs_qr + 1e-12,
            "{model}: new is further from QR ({new_vs_qr:e}) than old ({old_vs_qr:e})"
        );
    }
}

/// A deliberately direct prefix recomputation protects the summation order when
/// the production implementation reuses the previous prefix.
#[test]
fn csw_matches_direct_prefix_recomputation() {
    for prices in [
        vec![0.0; 16],
        (0..128).map(|i| (i as f64 / 7.0).sin() + i as f64 / 90.0).collect(),
        vec![0.0, 1e150, -1e150, 1e150, 0.0],
        vec![0.0, 1e-150, -1e-150, 1e-150, 0.0],
    ] {
        for kind in ["one_sided", "two_sided"] {
            let actual = get_chu_stinchcombe_white_statistics(&prices, kind).unwrap();
            for t in 2..prices.len() {
                let mut sum = 0.0;
                for j in 1..=t {
                    let change = prices[j] - prices[j - 1];
                    sum += change * change;
                }
                let variance = sum / t as f64;
                let mut best = f64::NEG_INFINITY;
                let mut critical = f64::NAN;
                for s in 0..t {
                    let delta = prices[t] - prices[s];
                    let delta = if kind == "two_sided" { delta.abs() } else { delta };
                    let distance = (t - s) as f64;
                    let value = (1.0 / (variance.sqrt() * distance.sqrt())) * delta;
                    if value > best {
                        best = value;
                        critical = (4.6 + distance.ln()).sqrt();
                    }
                }
                assert_eq!(actual.stat[t - 2].to_bits(), best.to_bits());
                assert_eq!(actual.critical_value[t - 2].to_bits(), critical.to_bits());
            }
        }
    }
}
