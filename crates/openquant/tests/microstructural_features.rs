use csv::ReaderBuilder;
use openquant::microstructural_features::{
    encode_tick_rule_array, get_bar_based_amihud_lambda, get_bar_based_hasbrouck_lambda,
    get_bar_based_kyle_lambda, get_bekker_parkinson_vol, get_bvc_buy_volume,
    get_corwin_schultz_estimator, get_konto_entropy, get_lempel_ziv_entropy, get_plug_in_entropy,
    get_roll_impact, get_roll_measure, get_shannon_entropy, get_trades_based_hasbrouck_lambda,
    get_vpin, quantile_mapping, MicrostructuralFeaturesGenerator,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
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
        .join("../../tests/fixtures/microstructural_features/dollar_bar_sample.csv");
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
    let mut gen = MicrostructuralFeaturesGenerator::new_from_csv(
        load_tick_data_path().to_str().unwrap(),
        &tick_num,
        Some(volume_enc.clone()),
        Some(pct_enc.clone()),
    )
    .unwrap();
    let feats = gen.get_features_from_csv(load_tick_data_path().to_str().unwrap()).unwrap();
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
    let gen = MicrostructuralFeaturesGenerator::new_from_csv(
        load_tick_data_path().to_str().unwrap(),
        &[1, 2, 3],
        None,
        None,
    );
    assert!(gen.is_ok());
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

    let mut gen = MicrostructuralFeaturesGenerator::new_from_csv(
        path.to_str().unwrap(),
        &[10, 20, 30],
        None,
        None,
    )
    .unwrap();
    let feats = gen.get_features_from_csv(path.to_str().unwrap()).unwrap();
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
        let side = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
        let dv: f64 = rng.gen_range(1e4..1e6);
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
    assert_eq!(
        sigma_mapping(&[0.0, 0.3, 1.0], 0.5),
        Ok(vec![(0.0, '\u{0}'), (0.5, '\u{1}')])
    );
}

#[test]
fn encode_array_keeps_one_letter_per_value_or_errors() {
    use openquant::microstructural_features::{encode_array, MicrostructuralError};
    let codebook = quantile_mapping(&[1.0, 2.0, 3.0, 4.0, 5.0], 2).unwrap();
    let values = [1.0, f64::INFINITY, f64::NEG_INFINITY, 4.9];
    assert_eq!(encode_array(&values, &codebook).unwrap().chars().count(), values.len());
    // These used to return shorter strings with no warning.
    assert_eq!(encode_array(&[1.0, f64::NAN], &codebook), Err(MicrostructuralError::NanInArray));
    assert!(matches!(
        encode_array(&[1.0], &[]),
        Err(MicrostructuralError::InvalidCodebook(_))
    ));
    assert!(matches!(
        encode_array(&[1.0], &[(f64::NAN, 'a')]),
        Err(MicrostructuralError::InvalidCodebook(_))
    ));
}
