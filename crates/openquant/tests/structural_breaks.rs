use csv::ReaderBuilder;
use openquant::structural_breaks::{
    _get_betas, _get_values_diff, get_chow_type_stat, get_chu_stinchcombe_white_statistics,
    get_sadf, SadfLags, StructuralBreakError,
};
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
        .join("../../tests/fixtures/structural_breaks/dollar_bar_sample.csv");
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

    // The critical values (AFML 17.3.2) do not depend on how the statistic is scaled; the
    // statistic itself is checked in test_chu_stinchcombe_white_statistic_matches_afml.
    let reference = reference();
    assert_csw_matches(&reference, &one_sided, &two_sided, "critical_value");

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
        assert_rel(max(values), num(want, &[field, "max"]), 1e-8, &format!("{what} max"));
        assert_rel(mean(values), num(want, &[field, "mean"]), 1e-8, &format!("{what} mean"));
        assert_rel(values[20], num(want, &[field, "at_20"]), 1e-8, &format!("{what} [20]"));
    }
}

#[test]
#[ignore = "FINDING: get_chu_stinchcombe_white_statistics divides y_t - y_n by sigma_t^2 * sqrt(t - n); AFML 17.3.2 divides by sigma_t * sqrt(t - n), and sigma_t^2 averages t-1 squared differences where the library divides by t-2 (one-sided max 3729.0 vs 5.39)"]
fn test_chu_stinchcombe_white_statistic_matches_afml() {
    let log_prices = log_prices();
    let one_sided = get_chu_stinchcombe_white_statistics(&log_prices, "one_sided").unwrap();
    let two_sided = get_chu_stinchcombe_white_statistics(&log_prices, "two_sided").unwrap();
    assert_csw_matches(&reference(), &one_sided, &two_sided, "stat");
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

    // AFML 17.4.2 values. The quadratic and sub/super-martingale models depart from AFML and
    // are checked in sadf_quadratic_and_martingale_models_match_afml.
    let reference = reference();
    assert_sadf_matches(&reference, &[("linear", &linear), ("linear_no_const", &linear_no_const)]);

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
        assert_rel(values[29], num(want, &["at_29"]), 1e-7, &format!("{name} [29]"));
    }
}

#[test]
#[ignore = "FINDING: get_sadf 'quadratic' regresses on const + t^2 without the linear t of AFML's 'ctt'; the sm_* models take the sup of the signed beta/se where AFML 17.4.3 takes |beta|/se; sm_power uses log(0) for the first row and drops that window. Long-running."]
fn sadf_quadratic_and_martingale_models_match_afml() {
    let log_prices = log_prices();
    let lags = SadfLags::Fixed(5);
    let run = |model: &str| get_sadf(&log_prices, model, true, 20, lags.clone()).unwrap();
    let (quadratic, sm_power, sm_poly_1, sm_poly_2, sm_exp) =
        (run("quadratic"), run("sm_power"), run("sm_poly_1"), run("sm_poly_2"), run("sm_exp"));
    assert_sadf_matches(
        &reference(),
        &[
            ("quadratic", &quadratic),
            ("sm_power", &sm_power),
            ("sm_poly_1", &sm_poly_1),
            ("sm_poly_2", &sm_poly_2),
            ("sm_exp", &sm_exp),
        ],
    );
}
