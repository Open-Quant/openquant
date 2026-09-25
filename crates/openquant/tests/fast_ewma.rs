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
