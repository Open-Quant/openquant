use csv::ReaderBuilder;
use openquant::microstructural_features::{
    encode_tick_rule_array, get_bar_based_amihud_lambda, get_bar_based_hasbrouck_lambda,
    get_bar_based_kyle_lambda, get_bekker_parkinson_vol, get_bvc_buy_volume,
    get_corwin_schultz_estimator, get_konto_entropy, get_lempel_ziv_entropy, get_plug_in_entropy,
    get_roll_impact, get_roll_measure, get_shannon_entropy, get_trades_based_hasbrouck_lambda,
    get_vpin, quantile_mapping, MicrostructuralFeaturesGenerator,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use serde::Deserialize;
use std::path::Path;

/// Summary of one feature series, as written by
/// `tests/fixtures/microstructural_features/generate.py` (AFML ch. 19 in pandas).
#[derive(Deserialize)]
struct Summary {
    max: f64,
    mean: f64,
    position: usize,
    at_position: f64,
    first_finite: usize,
}

#[derive(Deserialize)]
struct Reference {
    roll_measure: Summary,
    roll_impact: Summary,
    corwin_schultz: Summary,
    becker_parkinson: Summary,
    kyle_lambda: Summary,
    amihud_lambda: Summary,
    hasbrouck_lambda: Summary,
    vpin_1: Summary,
    vpin_20: Summary,
}

fn load_reference() -> Reference {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/microstructural_features/reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn finite_max(v: &[f64]) -> f64 {
    v.iter().cloned().filter(|x| x.is_finite()).fold(f64::NAN, f64::max)
}

fn finite_mean(v: &[f64]) -> f64 {
    let kept: Vec<f64> = v.iter().cloned().filter(|x| x.is_finite()).collect();
    kept.iter().sum::<f64>() / kept.len() as f64
}

fn assert_rel(name: &str, actual: f64, expected: f64, rel: f64) {
    assert!(
        (actual - expected).abs() <= rel * expected.abs(),
        "{name}: got {actual:e}, expected {expected:e} (rel tol {rel:e})"
    );
}

/// Same arithmetic as the reference, so only rounding separates the two.
const REL_TOL: f64 = 1e-9;

fn assert_matches(name: &str, series: &[f64], expected: &Summary) {
    assert!(
        series[..expected.first_finite].iter().all(|x| x.is_nan()),
        "{name}: expected NaN before position {}",
        expected.first_finite
    );
    assert!(series[expected.first_finite].is_finite(), "{name}: first finite value misplaced");
    assert_rel(&format!("{name} max"), finite_max(series), expected.max, REL_TOL);
    assert_rel(&format!("{name} mean"), finite_mean(series), expected.mean, REL_TOL);
    assert_rel(
        &format!("{name}[{}]", expected.position),
        series[expected.position],
        expected.at_position,
        REL_TOL,
    );
}

/// `(close, high, low, cum_dollar, cum_volume)` columns.
type DollarBarColumns = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

fn load_dollar_bars() -> DollarBarColumns {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/shared/dollar_bar_sample.csv");
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path).unwrap();
    let mut close = Vec::new();
    let mut high = Vec::new();
    let mut low = Vec::new();
    let mut cum_dollar = Vec::new();
    let mut cum_vol = Vec::new();
    for rec in rdr.records() {
        let rec = rec.unwrap();
        close.push(rec[4].parse::<f64>().unwrap());
        high.push(rec[2].parse::<f64>().unwrap());
        low.push(rec[3].parse::<f64>().unwrap());
        cum_dollar.push(rec[6].parse::<f64>().unwrap());
        cum_vol.push(rec[5].parse::<f64>().unwrap());
    }
    (close, high, low, cum_dollar, cum_vol)
}

#[test]
fn test_second_generation_intra_bar() {
    let (close, _high, _low, cum_dollar, cum_vol) = load_dollar_bars();
    let reference = load_reference();
    let kyle = get_bar_based_kyle_lambda(&close, &cum_vol, 20).unwrap();
    let amihud = get_bar_based_amihud_lambda(&close, &cum_dollar, 20).unwrap();
    let hasbrouck = get_bar_based_hasbrouck_lambda(&close, &cum_dollar, 20).unwrap();

    assert_matches("kyle", &kyle, &reference.kyle_lambda);
    assert_matches("amihud", &amihud, &reference.amihud_lambda);
    assert_matches("hasbrouck", &hasbrouck, &reference.hasbrouck_lambda);
}

#[test]
fn test_third_generation_features() {
    let (close, _high, _low, _cum_dollar, cum_vol) = load_dollar_bars();
    let bvc = get_bvc_buy_volume(&close, &cum_vol, 20).unwrap();
    let vpin1 = get_vpin(&cum_vol, &bvc, 1).unwrap();
    let vpin20 = get_vpin(&cum_vol, &bvc, 20).unwrap();
    let reference = load_reference();

    assert_matches("vpin_1", &vpin1, &reference.vpin_1);
    assert_matches("vpin_20", &vpin20, &reference.vpin_20);
}

#[test]
fn test_tick_rule_encoding() {
    assert!(encode_tick_rule_array(&[-1, 1, 0, 20000000]).is_err());
    let enc = encode_tick_rule_array(&[-1, 1, 0, 0]).unwrap();
    assert_eq!(enc, "bacc");
}

#[test]
fn test_entropy_calculations() {
    let message = "11100001";
    let message_array = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let shannon = get_shannon_entropy(message);
    let plug_in = get_plug_in_entropy(message, 1).unwrap();
    let qmap = quantile_mapping(&message_array, 2).unwrap();
    let plug_in_arr = get_plug_in_entropy(&encode_array_f64(&message_array, &qmap), 1).unwrap();
    let lempel = get_lempel_ziv_entropy(message);
    let konto = get_konto_entropy(message, 0);
    // Worked by hand from AFML ch. 18 for "11100001":
    // Shannon: four 1s and four 0s, so -2 * (1/2) log2(1/2) = 1.
    assert!((shannon - 1.0).abs() < 1e-12);
    // Lempel-Ziv (snippet 18.2): library 1, 11, 0, 00, 01 -> 5 words / 8 chars.
    assert!((lempel - 5.0 / 8.0).abs() < 1e-12);
    // Plug-in, word length 1 (snippet 18.1): the snippet's pmf reads msg[i-1] for
    // i in 1..len, i.e. "1110000" (the last character is not counted):
    // p(1) = 3/7, p(0) = 4/7.
    let p1: f64 = 3.0 / 7.0;
    let p0: f64 = 4.0 / 7.0;
    assert!((plug_in - -(p1 * p1.log2() + p0 * p0.log2())).abs() < 1e-12);
    // Kontoyiannis, expanding window (snippets 18.3-18.4): points i = 1..4 with
    // match length + 1 of 2, 2, 1, 4, so h = mean(log2(i+1) / L_i).
    let konto_expected =
        (2f64.log2() / 2.0 + 3f64.log2() / 2.0 + 4f64.log2() / 1.0 + 5f64.log2() / 4.0) / 4.0;
    assert!((konto - konto_expected).abs() < 1e-12);
    assert!((plug_in - plug_in_arr).abs() < 1e-9);
}

fn encode_array_f64(arr: &[f64], enc: &[(f64, char)]) -> String {
    openquant::microstructural_features::encode_array(arr, enc).unwrap()
}

fn load_tick_data_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/microstructural_features/tick_data.csv")
}

fn build_tick_num_from_volume_bars(threshold: f64) -> Vec<usize> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(load_tick_data_path()).unwrap();
    let mut cum_vol = 0.0;
    let mut tick_num = Vec::new();
    let mut idx = 0usize;
    for rec in rdr.records() {
        let rec = rec.unwrap();
        let vol = rec[2].parse::<f64>().unwrap();
        cum_vol += vol;
        idx += 1;
        if cum_vol >= threshold {
            tick_num.push(idx);
            cum_vol = 0.0;
        }
    }
    tick_num
}

#[test]
fn test_feature_generator_function() {
    // Build tick_num via simple volume bar threshold like Python get_volume_bars(threshold=20)
    let tick_num = build_tick_num_from_volume_bars(20.0);
    // build encodings from tick data
    let mut tick_rdr =
        ReaderBuilder::new().has_headers(true).from_path(load_tick_data_path()).unwrap();
    let mut volume_vals = Vec::new();
    let mut price_vals = Vec::new();
    for rec in tick_rdr.records() {
        let rec = rec.unwrap();
        volume_vals.push(rec[2].parse::<f64>().unwrap());
        price_vals.push(rec[1].parse::<f64>().unwrap());
    }
    // log returns
    let mut log_ret = Vec::new();
    for i in 1..price_vals.len() {
        log_ret.push((price_vals[i] / price_vals[i - 1]).ln());
    }
    let volume_enc = quantile_mapping(&volume_vals, 10).unwrap();
    let pct_enc = quantile_mapping(&log_ret, 10).unwrap();
    let mut r#gen = MicrostructuralFeaturesGenerator::new_from_csv(
        load_tick_data_path().to_str().unwrap(),
        &tick_num,
        Some(volume_enc.clone()),
        Some(pct_enc.clone()),
    )
    .unwrap();
    let feats = r#gen.get_features_from_csv(load_tick_data_path().to_str().unwrap()).unwrap();
    assert!(!feats.is_empty());
    // basic shape and a few value checks vs Python expectations
    // columns order: date_time(ts), avg_tick_size, tick_rule_sum, vwap, kyle, amihud, hasbrouck, entropies...
    let first = &feats[0];
    assert!(first.len() >= 7);
    // avg_tick_size should be positive
    assert!(first[1].is_finite() && first[1] > 0.0);
}

#[test]
fn test_csv_format_validation() {
    // ensure valid csv passes
    let r#gen = MicrostructuralFeaturesGenerator::new_from_csv(
        load_tick_data_path().to_str().unwrap(),
        &[1, 2, 3],
        None,
        None,
    );
    assert!(r#gen.is_ok());
}

#[test]
fn test_first_generation_features() {
    let (close, high, low, cum_dollar, _cum_vol) = load_dollar_bars();
    let roll = get_roll_measure(&close, 20);
    let roll_imp = get_roll_impact(&close, &cum_dollar, 20).unwrap();
    let cs = get_corwin_schultz_estimator(&high, &low, 20).unwrap();
    let bekker = get_bekker_parkinson_vol(&high, &low, 20).unwrap();

    assert_eq!(roll.len(), close.len());
    assert_eq!(roll_imp.len(), close.len());
    assert_eq!(cs.len(), close.len());
    assert_eq!(bekker.len(), close.len());

    let reference = load_reference();
    assert_matches("roll", &roll, &reference.roll_measure);
    assert_matches("roll_impact", &roll_imp, &reference.roll_impact);
    assert_matches("corwin_schultz", &cs, &reference.corwin_schultz);
    assert_matches("bekker_parkinson", &bekker, &reference.becker_parkinson);
}

#[test]
fn test_feature_generator_emits_one_row_per_tick_threshold() {
    // 30 ticks; trade size is 1 for ticks 1-10, 2 for 11-20, 3 for 21-30. With bars
    // closing at ticks 10, 20 and 30 the average tick size per bar is exactly 1, 2, 3.
    let path = std::env::temp_dir()
        .join(format!("openquant_ticks_{}_one_row_per_threshold.csv", std::process::id()));
    let mut csv = String::from("Date and Time,Price,Volume\n");
    for tick in 0..30 {
        let price = 100.0 + (tick % 5) as f64 * 0.25;
        csv.push_str(&format!("2011/07/31 23:31:{:02}.000,{price},{}\n", 10 + tick, tick / 10 + 1));
    }
    std::fs::write(&path, csv).unwrap();

    let mut r#gen = MicrostructuralFeaturesGenerator::new_from_csv(
        path.to_str().unwrap(),
        &[10, 20, 30],
        None,
        None,
    )
    .unwrap();
    let feats = r#gen.get_features_from_csv(path.to_str().unwrap()).unwrap();
    std::fs::remove_file(&path).ok();

    let avg_tick_sizes: Vec<f64> = feats.iter().map(|row| row[1]).collect();
    assert_eq!(avg_tick_sizes, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_trades_based_hasbrouck_lambda_recovers_lambda_under_balanced_flow() {
    // r_t = lambda * b_t * sqrt(p_t V_t) + noise, with buys and sells equally likely (#105).
    let lambda = 1e-5;
    let mut rng = StdRng::seed_from_u64(105);
    let noise = Normal::new(0.0, 1e-4).unwrap();
    let n = 5_000;
    let mut log_ret = Vec::with_capacity(n);
    let mut dollar_volume = Vec::with_capacity(n);
    let mut sides = Vec::with_capacity(n);
    for _ in 0..n {
        let side = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        let dv: f64 = rng.random_range(1e4..1e6);
        log_ret.push(lambda * side * dv.sqrt() + noise.sample(&mut rng));
        dollar_volume.push(dv);
        sides.push(side);
    }
    let est = get_trades_based_hasbrouck_lambda(&log_ret, &dollar_volume, &sides).unwrap();
    assert!((est - lambda).abs() < 0.02 * lambda, "estimate {est:e}, true {lambda:e}");
}

/// #185 item 16: Snippet 18.4 clamps the window before using it for the points, the look-back
/// and the `log2(window + 1)` numerator.
#[test]
fn konto_entropy_clamps_the_window_everywhere() {
    // "aaaa" with window 5 is window 2: point 2 matches "aa" two back, L = 3, log2(3) / 3.
    // The old code kept window 5 for the numerator and gave log2(6) / 3.
    let h = get_konto_entropy("aaaa", 5);
    assert!((h - 3f64.log2() / 3.0).abs() < 1e-12, "{h}");
    // Any window at or above len/2 is the same estimate as len/2 itself.
    let msg = "abbabaabbaababba";
    for w in [8, 9, 50] {
        assert_eq!(get_konto_entropy(msg, w), get_konto_entropy(msg, 8));
    }
}

#[test]
fn bvc_buy_volume_rejects_a_window_below_two() {
    let close = [10.0, 11.0, 10.0, 11.0];
    for window in [0, 1] {
        assert!(get_bvc_buy_volume(&close, &[100.0; 4], window).is_err(), "window {window}");
    }
    assert!(get_bvc_buy_volume(&close, &[100.0; 4], 2).is_ok());
}

#[test]
fn sigma_mapping_rejects_bad_steps_and_matches_quantile_mapping_on_edge_cases() {
    use openquant::microstructural_features::{sigma_mapping, MicrostructuralError};
    for step in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        assert_eq!(sigma_mapping(&[0.0, 1.0], step), Err(MicrostructuralError::NonPositiveStep));
    }
    // Empty and NaN input are errors, as in quantile_mapping.
    assert_eq!(sigma_mapping(&[], 0.5), Err(MicrostructuralError::EmptyArray));
    assert_eq!(quantile_mapping(&[], 2), Err(MicrostructuralError::EmptyArray));
    assert_eq!(sigma_mapping(&[0.0, f64::NAN], 0.5), Err(MicrostructuralError::NanInArray));
    assert_eq!(quantile_mapping(&[0.0, f64::NAN], 2), Err(MicrostructuralError::NanInArray));
    // A constant array has a one-letter codebook (it was empty), so it still encodes.
    assert_eq!(sigma_mapping(&[3.0, 3.0], 0.5), Ok(vec![(3.0, '\u{0}')]));
    // Unchanged for ordinary input.
    assert_eq!(sigma_mapping(&[0.0, 0.3, 1.0], 0.5), Ok(vec![(0.0, '\u{0}'), (0.5, '\u{1}')]));
}

#[test]
fn encode_array_keeps_one_letter_per_value_or_errors() {
    use openquant::microstructural_features::{encode_array, MicrostructuralError};
    let codebook = quantile_mapping(&[1.0, 2.0, 3.0, 4.0, 5.0], 2).unwrap();
    let values = [1.0, f64::INFINITY, f64::NEG_INFINITY, 4.9];
    assert_eq!(encode_array(&values, &codebook).unwrap().chars().count(), values.len());
    // These used to return shorter strings with no warning.
    assert_eq!(encode_array(&[1.0, f64::NAN], &codebook), Err(MicrostructuralError::NanInArray));
    assert!(matches!(encode_array(&[1.0], &[]), Err(MicrostructuralError::InvalidCodebook(_))));
    assert!(matches!(
        encode_array(&[1.0], &[(f64::NAN, 'a')]),
        Err(MicrostructuralError::InvalidCodebook(_))
    ));
}

#[test]
fn encode_array_orders_overflowing_finite_distances() {
    use openquant::microstructural_features::encode_array;
    // Both distances overflow, but -1e308 is strictly nearer to the positive query.
    assert_eq!(encode_array(&[1e308], &[(-1.5e308, 'a'), (-1e308, 'b')]).unwrap(), "b");
    assert_eq!(encode_array(&[-1e308], &[(1.5e308, 'a'), (1e308, 'b')]).unwrap(), "b");
    // Reversing the codebook, repeated entries and a finite-distance contender.
    assert_eq!(encode_array(&[1e308], &[(-1e308, 'a'), (-1.5e308, 'b')]).unwrap(), "a");
    assert_eq!(encode_array(&[1e308], &[(-1e308, 'a'), (-1e308, 'b')]).unwrap(), "a");
    assert_eq!(encode_array(&[1e308], &[(-1.5e308, 'a'), (0.0, 'b')]).unwrap(), "b");
}

#[test]
fn liquidity_lambdas_keep_extreme_finite_log_returns() {
    // Independent reference: powers of two give log returns exactly ±2046 ln(2).
    let tiny = 2.0_f64.powi(-1023);
    let large = 2.0_f64.powi(1023);
    let close = [tiny, large, tiny];
    let expected = 2046.0 * std::f64::consts::LN_2;
    for values in [
        get_bar_based_amihud_lambda(&close, &[1.0; 3], 1).unwrap(),
        get_bar_based_hasbrouck_lambda(&close, &[1.0; 3], 1).unwrap(),
    ] {
        assert!(values[0].is_nan());
        assert_rel("extreme positive log return", values[1], expected, 1e-14);
        assert_rel("extreme negative log return", values[2], expected, 1e-14);
    }
}

#[test]
fn liquidity_log_recovery_preserves_other_input_behavior() {
    for close in [
        [100.0, 110.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 1.0],
        [1.0, f64::INFINITY],
        [f64::INFINITY, f64::INFINITY],
        [1.0, f64::NAN],
    ] {
        let expected = if close[0] == 0.0 { f64::NAN } else { (close[1] / close[0]).ln().abs() };
        for values in [
            get_bar_based_amihud_lambda(&close, &[1.0; 2], 1).unwrap(),
            get_bar_based_hasbrouck_lambda(&close, &[1.0; 2], 1).unwrap(),
        ] {
            if expected.is_nan() {
                assert!(values[1].is_nan());
            } else {
                assert_eq!(values[1], expected);
            }
        }
    }
}

#[test]
fn feature_generator_encodes_extreme_finite_log_returns() {
    let path = std::env::temp_dir()
        .join(format!("openquant_ticks_{}_extreme_log_returns.csv", std::process::id()));
    let tiny = 2.0_f64.powi(-1023);
    let large = 2.0_f64.powi(1023);
    let csv = format!(
        "Date and Time,Price,Volume\n2011/07/31 23:31:10.000,{tiny},1\n2011/07/31 23:31:11.000,{large},1\n"
    );
    std::fs::write(&path, csv).unwrap();
    let mut generator = MicrostructuralFeaturesGenerator::new_from_csv(
        path.to_str().unwrap(),
        &[2],
        None,
        Some(vec![(0.0, 'a'), (2046.0 * std::f64::consts::LN_2, 'b')]),
    )
    .unwrap();
    let features = generator.get_features_from_csv(path.to_str().unwrap());
    std::fs::remove_file(path).unwrap();
    let features = features.unwrap();
    // Seven liquidity columns, four tick-rule entropy columns, then return entropy.
    // The return message is 'ab', whose Shannon entropy is exactly one bit.
    assert_eq!(features.len(), 1);
    assert_eq!(features[0][11], 1.0);
}

#[test]
fn liquidity_lambdas_do_not_log_a_rounded_subnormal_ratio() {
    // The true quotient is 1.5 * the smallest subnormal. Binary64 division
    // rounds it to 2 * the smallest subnormal, changing its log by ln(4/3).
    let close = [2.0_f64.powi(52), 1.5 * f64::MIN_POSITIVE];
    let expected = 1074.0 * std::f64::consts::LN_2 - 1.5_f64.ln();
    for values in [
        get_bar_based_amihud_lambda(&close, &[1.0; 2], 1).unwrap(),
        get_bar_based_hasbrouck_lambda(&close, &[1.0; 2], 1).unwrap(),
    ] {
        assert_rel("subnormal positive ratio", values[1], expected, 1e-14);
    }
}

#[test]
fn bvc_positive_deviation_is_invariant_to_price_units() {
    // Independently evaluated 100 * Phi(+/-1/sqrt(2)), using Python math.erfc.
    let expected = [23.975006109347675, 76.02499389065233];
    for exponent in [0, -38, -40, -44, -60] {
        // Binary rescaling preserves the differences exactly, without variance underflow.
        let scale = 2.0_f64.powi(exponent);
        let close = [10.0, 11.0, 10.0, 11.0].map(|p| p * scale);
        let actual = get_bvc_buy_volume(&close, &[100.0; 4], 2).unwrap();
        assert!(actual[..2].iter().all(|x| x.is_nan()));
        for (&value, &reference) in actual[2..].iter().zip(&expected) {
            assert!((value - reference).abs() < 1e-8, "scale 2^{exponent}: {value} != {reference}");
        }
    }
}

#[test]
fn bvc_preserves_zero_deviation_and_ordinary_outputs_bitwise() {
    let scale = 2.0_f64.powi(-44);
    // Pre-fix outputs: preserve Phi(dp / 1e-12) for constant nonzero changes,
    // not an unconditional sign split or half volume. Ordinary deviations stay untouched.
    let cases = [
        (vec![10.0; 4], 2, vec![0x4049000000000000_u64; 2]),
        (
            vec![10.0 * scale, 11.0 * scale, 12.0 * scale, 13.0 * scale],
            2,
            vec![0x404a221ccad359cd; 2],
        ),
        (
            vec![13.0 * scale, 12.0 * scale, 11.0 * scale, 10.0 * scale],
            2,
            vec![0x4047dde3352ca633; 2],
        ),
        (
            vec![10.0, 12.0, 11.0, 14.0, 14.0, 12.0, 15.0],
            3,
            vec![0x40572177b8e9e3ae, 0x4049000000000000, 0x403556be02a9f9b6, 0x405615a9cd2e4d0e],
        ),
    ];
    for (close, window, expected_bits) in cases {
        let actual = get_bvc_buy_volume(&close, &vec![100.0; close.len()], window).unwrap();
        assert!(actual[..window].iter().all(|x| x.is_nan()));
        assert_eq!(actual[window..].iter().map(|x| x.to_bits()).collect::<Vec<_>>(), expected_bits);
    }
}

#[test]
fn bvc_preserves_missing_nonfinite_and_window_behavior() {
    for window in [0, 1] {
        assert!(get_bvc_buy_volume(&[1.0, 2.0], &[100.0; 2], window).is_err());
    }
    assert!(get_bvc_buy_volume(&[1.0, 2.0], &[100.0], 2).is_err());
    assert!(get_bvc_buy_volume(&[], &[], 2).unwrap().is_empty());
    let short = get_bvc_buy_volume(&[1.0, 2.0], &[100.0; 2], 2).unwrap();
    assert_eq!(short.len(), 2);
    assert!(short.iter().all(|x| x.is_nan()));
    for missing in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let close = [1.0, missing, 2.0, 3.0, 4.0, 3.0];
        let actual = get_bvc_buy_volume(&close, &[100.0; 6], 2).unwrap();
        assert!(actual[..4].iter().all(|x| x.is_nan()));
        assert_eq!(actual[4], 100.0); // A complete clean window recovers.
        assert!((actual[5] - 23.975006109347675).abs() < 1e-8);
    }
    // Finite price changes, but sigma=+inf from squared-deviation overflow: z=0.
    let actual = get_bvc_buy_volume(&[1e200, 2e200, 1e200, 2e200], &[100.0; 4], 2).unwrap();
    assert_eq!(&actual[2..], &[50.0, 50.0]);
}

#[test]
fn entropy_replays_are_bitwise_reproducible() {
    let message: String = (0..64)
        .flat_map(|i| std::iter::repeat_n(char::from_u32(0x100 + i).unwrap(), i as usize + 1))
        .collect();
    let shannon = get_shannon_entropy(&message).to_bits();
    let plugin = get_plug_in_entropy(&message, 2).unwrap().to_bits();
    for _ in 0..64 {
        assert_eq!(get_shannon_entropy(&message).to_bits(), shannon);
        assert_eq!(get_plug_in_entropy(&message, 2).unwrap().to_bits(), plugin);
    }
}
