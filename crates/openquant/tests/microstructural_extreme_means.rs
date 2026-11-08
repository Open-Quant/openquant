use openquant::microstructural_features::{get_avg_tick_size, get_vpin};

#[test]
fn mean_trade_size_stays_within_finite_sample_range() {
    for size in [f64::MAX, -f64::MAX] {
        assert_eq!(get_avg_tick_size(&[size, size, size]), size);
    }
    let actual = get_avg_tick_size(&[f64::MAX, f64::MAX, -f64::MAX]);
    assert!((actual / (f64::MAX / 3.0) - 1.0).abs() < 1e-15);
    assert_eq!(get_avg_tick_size(&[f64::from_bits(1); 3]), f64::from_bits(1));
    assert_eq!(get_avg_tick_size(&[1.0, 2.0, 6.0]), 3.0);
}

#[test]
fn mean_trade_size_preserves_nonfinite_and_empty_input_behavior() {
    assert!(get_avg_tick_size(&[]).is_nan());
    assert!(get_avg_tick_size(&[f64::NAN]).is_nan());
    assert!(get_avg_tick_size(&[f64::INFINITY, f64::NEG_INFINITY]).is_nan());
    assert_eq!(get_avg_tick_size(&[f64::INFINITY, 1.0]), f64::INFINITY);
}

#[test]
fn vpin_all_buy_and_balanced_volume_bars_remain_bounded() {
    let volume = [f64::MAX; 4];
    let all_buy = get_vpin(&volume, &volume, 3).unwrap();
    assert!(all_buy[..2].iter().all(|v| v.is_nan()));
    assert_eq!(&all_buy[2..], &[1.0, 1.0]);
    let balanced = get_vpin(&volume, &[f64::MAX / 2.0; 4], 3).unwrap();
    assert_eq!(&balanced[2..], &[0.0, 0.0]);
    let alternating = get_vpin(&volume, &[0.0, f64::MAX, 0.0, f64::MAX], 3).unwrap();
    assert_eq!(&alternating[2..], &[1.0, 1.0]);
}
