use csv::ReaderBuilder;
use openquant::util::fast_ewma::ewma;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TickRow {
    #[serde(rename = "Price")]
    price: f64,
}

fn load_prices() -> Vec<f64> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/microstructural_features/tick_data.csv");
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path).unwrap();
    let mut prices = Vec::new();
    for rec in rdr.deserialize::<TickRow>() {
        prices.push(rec.unwrap().price);
    }
    prices
}

#[test]
fn test_ewma() {
    let price_arr = load_prices();
    let ewma_res = ewma(&price_arr, 20).unwrap();

    assert_eq!(ewma_res.len(), price_arr.len());
    assert_eq!(ewma_res[0], price_arr[0]);
    // By hand: the first two prices are 1205 and 1005, alpha = 2 / 21, so the adjusted EWMA is
    // (1005 + (19/21) 1205) / (1 + 19/21) = (21 * 1005 + 19 * 1205) / 40 = 44000 / 40 = 1100.
    assert_eq!(&price_arr[..2], &[1205.0, 1005.0]);
    assert!((ewma_res[1] - 1100.0).abs() < 1e-9);
}

/// `ewma` as it was before it stopped adding weight terms too small to change the weight.
fn ewma_every_term(arr_in: &[f64], window: usize) -> Vec<f64> {
    let alpha = 2.0 / (window as f64 + 1.0);
    let mut weight = 1.0;
    let mut ewma_old = arr_in[0];
    let mut out = vec![0.0; arr_in.len()];
    out[0] = ewma_old;
    for i in 1..arr_in.len() {
        weight += (1.0 - alpha).powi(i as i32);
        ewma_old = ewma_old * (1.0 - alpha) + arr_in[i];
        out[i] = ewma_old / weight;
    }
    out
}

#[test]
fn ewma_matches_summing_every_weight_term() {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let prices = load_prices();
    let long: Vec<f64> = (0..300_000).map(|i| 100.0 + (i as f64 / 37.0).sin()).collect();
    // Short windows stop summing weight terms early; the longest never stop on these lengths.
    for window in [1, 2, 3, 5, 20, 64, 100, 250, 1000, 5000, 100_000, 10_000_000] {
        for series in [&prices, &long] {
            let got = ewma(series, window).unwrap();
            let want = ewma_every_term(series, window);
            assert_eq!(bits(&got), bits(&want), "window {window}, {} values", series.len());
        }
    }
}
