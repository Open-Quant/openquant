use csv::ReaderBuilder;
use openquant::structural_breaks::{
    _get_betas, _get_values_diff, get_chow_type_stat, get_chu_stinchcombe_white_statistics,
    get_sadf, SadfLags, StructuralBreakError,
};
use std::path::Path;

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

    assert_eq!(log_prices.len() - min_length * 2, stats.len());
    assert!((max(&stats) - 0.179).abs() < 0.001);
    assert!((mean(&stats) + 0.653).abs() < 0.001);
    assert!((stats[3] + 0.6649).abs() < 0.001);
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

    assert!((max(&one_sided.critical_value) - 3.265).abs() < 0.001);
    assert!((mean(&one_sided.critical_value) - 2.7809).abs() < 0.001);
    assert!((one_sided.critical_value[20] - 2.4466).abs() < 0.001);

    assert!((max(&one_sided.stat) - 3729.001).abs() < 0.001);
    assert!((mean(&one_sided.stat) - 836.509).abs() < 0.001);
    assert!((one_sided.stat[20] - 380.137).abs() < 0.001);

    assert!((max(&two_sided.critical_value) - 3.235).abs() < 0.001);
    assert!((mean(&two_sided.critical_value) - 2.769).abs() < 0.001);
    assert!((two_sided.critical_value[20] - 2.715).abs() < 0.001);

    assert!((max(&two_sided.stat) - 5518.519).abs() < 0.001);
    assert!((mean(&two_sided.stat) - 1264.582).abs() < 0.001);
    assert!((two_sided.stat[20] - 921.2979).abs() < 0.001);

    let invalid = get_chu_stinchcombe_white_statistics(&log_prices, "rubbish text");
    assert!(matches!(invalid, Err(StructuralBreakError::InvalidTestType(_))));
}

#[test]
fn test_sadf_test() {
    let log_prices = log_prices();
    let lags_int = 5usize;
    let lags_array = vec![1usize, 2, 5, 7];
    let min_length = 20usize;

    let sm_power =
        get_sadf(&log_prices, "sm_power", true, min_length, SadfLags::Fixed(lags_int))
            .expect("sm_power sadf");
    let linear =
        get_sadf(&log_prices, "linear", true, min_length, SadfLags::Fixed(lags_int))
            .expect("linear sadf");
    let linear_no_const = get_sadf(
        &log_prices,
        "linear",
        false,
        min_length,
        SadfLags::Array(lags_array.clone()),
    )
    .expect("linear no const sadf");
    let quadratic =
        get_sadf(&log_prices, "quadratic", true, min_length, SadfLags::Fixed(lags_int))
            .expect("quadratic sadf");
    let sm_poly_1 =
        get_sadf(&log_prices, "sm_poly_1", true, min_length, SadfLags::Fixed(lags_int))
            .expect("sm_poly_1 sadf");
    let sm_poly_2 =
        get_sadf(&log_prices, "sm_poly_2", true, min_length, SadfLags::Fixed(lags_int))
            .expect("sm_poly_2 sadf");
    let sm_exp =
        get_sadf(&log_prices, "sm_exp", true, min_length, SadfLags::Fixed(lags_int))
            .expect("sm_exp sadf");

    let expected_len = log_prices.len() - min_length - lags_int - 1;
    assert_eq!(expected_len, sm_power.len());
    assert_eq!(expected_len, linear.len());
    assert_eq!(expected_len, quadratic.len());
    assert_eq!(expected_len, sm_poly_1.len());
    assert_eq!(expected_len, sm_poly_2.len());
    assert_eq!(expected_len, sm_exp.len());

    assert!((mean(&sm_power) - 17.814).abs() < 0.001);
    assert!((sm_power[29] + 4.281).abs() < 0.001);
    assert!((mean(&linear) + 0.669).abs() < 0.001);
    assert!((linear[29] + 0.717).abs() < 0.001);
    assert!((mean(&linear_no_const) - 1.899).abs() < 0.001);
    assert!((linear_no_const[29] - 1.252).abs() < 0.001);
    assert!((mean(&quadratic) + 0.651).abs() < 0.001);
    assert!((quadratic[29] + 1.065).abs() < 0.001);
    assert!((mean(&sm_poly_1) - 21.02).abs() < 0.001);
    assert!((sm_poly_1[29] - 0.8268).abs() < 0.001);
    assert!((mean(&sm_poly_2) - 21.01).abs() < 0.001);
    assert!((sm_poly_2[29] - 0.822).abs() < 0.001);
    assert!((mean(&sm_exp) - 17.632).abs() < 0.001);
    assert!((sm_exp[29] + 5.821).abs() < 0.001);

    let ones = vec![1.0; log_prices.len()];
    let trivial =
        get_sadf(&ones, "sm_power", true, min_length, SadfLags::Fixed(lags_int)).expect("ones");
    assert!(trivial.iter().all(|v| v.is_infinite() && v.is_sign_negative()));

    let invalid = get_sadf(&log_prices, "rubbish_string", true, min_length, SadfLags::Fixed(lags_int));
    assert!(matches!(invalid, Err(StructuralBreakError::InvalidModel(_))));

    let singular_matrix = vec![
        vec![1.0, 0.0, 0.0],
        vec![-1.0, 3.0, 3.0],
        vec![1.0, 2.0, 2.0],
    ];
    let (b_mean, b_var) = _get_betas(&singular_matrix, &singular_matrix).expect("betas");
    assert!(b_mean.iter().all(|v| v.is_nan()));
    assert!(b_var.iter().all(|row| row.iter().all(|v| v.is_nan())));
}
