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
    let ewma_res = ewma(&price_arr, 20);

    assert_eq!(ewma_res.len(), price_arr.len());
    assert_eq!(ewma_res[0], price_arr[0]);
    assert!((ewma_res[1] - 1100.0).abs() < 1e-5);
}
