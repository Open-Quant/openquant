use csv::ReaderBuilder;
use openquant::microstructural_features::{
    get_bekker_parkinson_vol, get_corwin_schultz_estimator, get_roll_impact, get_roll_measure,
    get_bar_based_kyle_lambda, get_bar_based_amihud_lambda, get_bar_based_hasbrouck_lambda,
    encode_tick_rule_array, quantile_mapping, get_shannon_entropy, get_lempel_ziv_entropy,
    get_plug_in_entropy, get_konto_entropy, get_bvc_buy_volume, get_vpin, MicrostructuralFeaturesGenerator,
};
use std::path::Path;

fn load_dollar_bars() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
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
    let (close, _high, _low, cum_dollar, volume) = load_dollar_bars();
    let volume: Vec<f64> = {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/microstructural_features/dollar_bar_sample.csv");
        let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path).unwrap();
        let mut v = Vec::new();
        for rec in rdr.records() {
            let rec = rec.unwrap();
            v.push(rec[5].parse::<f64>().unwrap());
        }
        v
    };
    let kyle = get_bar_based_kyle_lambda(&close, &volume, 20);
    let amihud = get_bar_based_amihud_lambda(&close, &cum_dollar, 20);
    let hasbrouck = get_bar_based_hasbrouck_lambda(&close, &cum_dollar, 20);
    let max = |v: &[f64]| v.iter().cloned().fold(f64::NAN, f64::max);
    let mean = |v: &[f64]| {
        let mut sum = 0.0;
        let mut count: f64 = 0.0;
        for x in v.iter() {
            if x.is_finite() {
                sum += *x;
                count += 1.0;
            }
        }
        if count > 0.0 { sum / count } else { f64::NAN }
    };
    assert!((max(&kyle) - 0.000163423).abs() < 1e-6);
    assert!((mean(&kyle) - 7.02e-5).abs() < 1e-5);
    assert!((kyle[25] - 7.76e-5).abs() < 1e-5);

    assert!((max(&amihud) - 4.057838e-11).abs() < 1e-13);
    assert!((mean(&amihud) - 1.7213e-11).abs() < 1e-12);
    assert!((amihud[25] - 1.8439e-11).abs() < 1e-12);

    assert!((max(&hasbrouck) - 3.39527e-7).abs() < 1e-9);
    assert!((mean(&hasbrouck) - 1.44037e-7).abs() < 1e-8);
    assert!((hasbrouck[25] - 1.5433e-7).abs() < 1e-8);
}

#[test]
fn test_third_generation_features() {
    let (close, _high, _low, _cum_dollar, cum_vol) = load_dollar_bars();
    let bvc = get_bvc_buy_volume(&close, &cum_vol, 20);
    let vpin1 = get_vpin(&cum_vol, &bvc, 1);
    let vpin20 = get_vpin(&cum_vol, &bvc, 20);
    let max = |v: &[f64]| v.iter().cloned().fold(f64::NAN, f64::max);
    let mean = |v: &[f64]| {
        let mut sum = 0.0;
        let mut count: f64 = 0.0;
        for x in v.iter() {
            if x.is_finite() {
                sum += *x;
                count += 1.0;
            }
        }
        if count > 0.0 { sum / count } else { f64::NAN }
    };
    assert!((max(&vpin1) - 0.999).abs() < 1e-3);
    assert!((mean(&vpin1) - 0.501).abs() < 1e-3);
    assert!((vpin1[25] - 0.554).abs() < 1e-3);

    assert!((max(&vpin20) - 0.6811).abs() < 1e-3);
    assert!((mean(&vpin20) - 0.500).abs() < 1e-3);
    assert!((vpin20[45] - 0.4638).abs() < 1e-3);
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
    let plug_in = get_plug_in_entropy(message, 1);
    let qmap = quantile_mapping(&message_array, 2).unwrap();
    let plug_in_arr = get_plug_in_entropy(&encode_array_f64(&message_array, &qmap), 1);
    let lempel = get_lempel_ziv_entropy(message);
    let konto = get_konto_entropy(message, 0);
    assert!((shannon - 1.0).abs() < 1e-3);
    assert!((lempel - 0.625).abs() < 1e-3);
    assert!((plug_in - 0.985).abs() < 1e-3);
    assert!((konto - 0.9682).abs() < 1e-3);
    assert!((plug_in - plug_in_arr).abs() < 1e-9);
}

fn encode_array_f64(arr: &[f64], enc: &[(f64, char)]) -> String {
    openquant::microstructural_features::encode_array(arr, enc)
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
    let mut tick_rdr = ReaderBuilder::new().has_headers(true).from_path(load_tick_data_path()).unwrap();
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
    ).unwrap();
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
        &[1,2,3],
        None,
        None
    );
    assert!(gen.is_ok());
}

#[test]
fn test_first_generation_features() {
    let (close, high, low, cum_dollar, _cum_vol) = load_dollar_bars();
    let roll = get_roll_measure(&close, 20);
    let roll_imp = get_roll_impact(&close, &cum_dollar, 20);
    let cs = get_corwin_schultz_estimator(&high, &low, 20);
    let bekker = get_bekker_parkinson_vol(&high, &low, 20);

    assert_eq!(roll.len(), close.len());
    assert_eq!(roll_imp.len(), close.len());
    assert_eq!(cs.len(), close.len());
    assert_eq!(bekker.len(), close.len());

    let max = |v: &[f64]| v.iter().cloned().fold(f64::NAN, f64::max);
    let mean = |v: &[f64]| {
        let mut sum = 0.0;
        let mut count: f64 = 0.0;
        for x in v.iter() {
            if x.is_finite() {
                sum += *x;
                count += 1.0;
            }
        }
        if count > 0.0 { sum / count } else { f64::NAN }
    };
    assert!((max(&roll) - 7.1584).abs() < 1e-3);
    assert!((mean(&roll) - 2.341).abs() < 1e-3);
    assert!((roll[25] - 1.176).abs() < 1e-3);

    assert!((max(&roll_imp) - 1.022e-7).abs() < 1e-8);
    assert!((mean(&roll_imp) - 3.3445e-8).abs() < 1e-8);
    assert!((roll_imp[25] - 1.6807e-8).abs() < 1e-6);

    assert!((max(&cs) - 0.01652).abs() < 1e-4);
    assert!((mean(&cs) - 0.00151602).abs() < 1e-4);
    assert!((cs[25] - 0.00139617).abs() < 1e-4);

    assert!((max(&bekker) - 0.018773).abs() < 1e-4);
    assert!((mean(&bekker) - 0.001456).abs() < 1e-4);
    assert!((bekker[25] - 0.000517).abs() < 1e-4);
}
