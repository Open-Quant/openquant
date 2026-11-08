use chrono::{Duration, NaiveDate};
use openquant::sample_weights::get_weights_by_return;

#[test]
fn finite_extreme_prices_preserve_symmetric_return_attribution() {
    let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    for prices in [[1e-300, 1e300, 1e-300], [1e300, 1e-300, 1e300]] {
        let close: Vec<_> = prices
            .iter()
            .enumerate()
            .map(|(i, p)| (start + Duration::days(i as i64), *p))
            .collect();
        let events: Vec<_> = close[1..].iter().map(|(t, _)| (*t, *t, 1.0)).collect();
        let weights = get_weights_by_return(&events, &close).unwrap();
        assert_eq!(weights.iter().map(|(_, w)| *w).collect::<Vec<_>>(), vec![1.0, 1.0]);
    }
}

#[test]
fn subnormal_price_ratio_avoids_rounding_the_ratio_before_its_log() {
    let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    let prices: [f64; 3] = [1.5, f64::from_bits(1), 2.0];
    let close: Vec<_> =
        prices.iter().enumerate().map(|(i, p)| (start + Duration::days(i as i64), *p)).collect();
    let events: Vec<_> = close[1..].iter().map(|(t, _)| (*t, *t, 1.0)).collect();
    let raw = [(prices[1].ln() - prices[0].ln()).abs(), (prices[2].ln() - prices[1].ln()).abs()];
    let weights = get_weights_by_return(&events, &close).unwrap();
    for ((_, actual), expected) in
        weights.iter().zip(raw.map(|w| 2.0 * w / raw.iter().sum::<f64>()))
    {
        assert!((actual - expected).abs() < 1e-14);
    }
}
