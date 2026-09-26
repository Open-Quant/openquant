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
