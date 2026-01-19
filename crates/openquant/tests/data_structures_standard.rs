use chrono::{DateTime, Duration, NaiveDateTime, Utc};
use openquant::data_structures::{standard_bars, time_bars, StandardBarType, Trade};

fn ts(seconds: i64) -> NaiveDateTime {
    DateTime::<Utc>::from_timestamp(seconds, 0).expect("timestamp").naive_utc()
}

fn sample_trades() -> Vec<Trade> {
    vec![
        Trade { timestamp: ts(0), price: 100.0, volume: 1.0 },
        Trade { timestamp: ts(30), price: 101.0, volume: 2.0 },
        Trade { timestamp: ts(60), price: 102.0, volume: 3.0 },
        Trade { timestamp: ts(150), price: 103.0, volume: 4.0 },
        Trade { timestamp: ts(210), price: 104.0, volume: 5.0 },
    ]
}

#[test]
fn tick_volume_dollar_bars() {
    let trades = sample_trades();

    let tick_bars = standard_bars(&trades, 2.0, StandardBarType::Tick);
    assert_eq!(tick_bars.len(), 2);
    assert_eq!(tick_bars[0].open, 100.0);
    assert_eq!(tick_bars[0].close, 101.0);
    assert_eq!(tick_bars[0].high, 101.0);
    assert_eq!(tick_bars[0].low, 100.0);
    assert_eq!(tick_bars[0].volume, 3.0);
    assert_eq!(tick_bars[0].dollar_value, 302.0);
    assert_eq!(tick_bars[0].tick_count, 2);

    assert_eq!(tick_bars[1].open, 102.0);
    assert_eq!(tick_bars[1].close, 103.0);
    assert_eq!(tick_bars[1].high, 103.0);
    assert_eq!(tick_bars[1].low, 102.0);
    assert_eq!(tick_bars[1].volume, 7.0);
    assert_eq!(tick_bars[1].dollar_value, 718.0);
    assert_eq!(tick_bars[1].tick_count, 2);

    let volume_bars = standard_bars(&trades, 5.0, StandardBarType::Volume);
    assert_eq!(volume_bars.len(), 2);
    assert_eq!(volume_bars[0].close, 102.0);
    assert_eq!(volume_bars[0].volume, 6.0);
    assert_eq!(volume_bars[0].dollar_value, 608.0);
    assert_eq!(volume_bars[0].tick_count, 3);
    assert_eq!(volume_bars[1].close, 104.0);
    assert_eq!(volume_bars[1].volume, 9.0);
    assert_eq!(volume_bars[1].dollar_value, 932.0);
    assert_eq!(volume_bars[1].tick_count, 2);

    let dollar_bars = standard_bars(&trades, 500.0, StandardBarType::Dollar);
    assert_eq!(dollar_bars.len(), 2);
    assert_eq!(dollar_bars[0].close, 102.0);
    assert_eq!(dollar_bars[0].dollar_value, 608.0);
    assert_eq!(dollar_bars[0].tick_count, 3);
    assert_eq!(dollar_bars[1].close, 104.0);
    assert_eq!(dollar_bars[1].dollar_value, 932.0);
    assert_eq!(dollar_bars[1].tick_count, 2);
}

#[test]
fn time_bars_split_on_interval() {
    let trades = sample_trades();
    let bars = time_bars(&trades, Duration::seconds(120));
    assert_eq!(bars.len(), 2);

    // First bar closes on the trade at 150s (crosses the 120s threshold from start at 0s).
    assert_eq!(bars[0].open, 100.0);
    assert_eq!(bars[0].close, 103.0);
    assert_eq!(bars[0].volume, 10.0);
    assert_eq!(bars[0].tick_count, 4);

    // Second bar captures the remaining trade at 210s.
    assert_eq!(bars[1].open, 104.0);
    assert_eq!(bars[1].close, 104.0);
    assert_eq!(bars[1].volume, 5.0);
    assert_eq!(bars[1].tick_count, 1);
}
