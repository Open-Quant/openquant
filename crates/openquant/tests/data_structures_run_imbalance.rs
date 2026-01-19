use chrono::{DateTime, NaiveDateTime, Utc};
use openquant::data_structures::{imbalance_bars, run_bars, ImbalanceBarType, Trade};

fn ts(seconds: i64) -> NaiveDateTime {
    DateTime::<Utc>::from_timestamp(seconds, 0).expect("timestamp").naive_utc()
}

fn trades_for_runs() -> Vec<Trade> {
    vec![
        Trade { timestamp: ts(0), price: 100.0, volume: 1.0 },
        Trade { timestamp: ts(1), price: 101.0, volume: 1.0 },
        Trade { timestamp: ts(2), price: 102.0, volume: 1.0 },
        Trade { timestamp: ts(3), price: 103.0, volume: 1.0 },
        Trade { timestamp: ts(4), price: 102.0, volume: 1.0 },
        Trade { timestamp: ts(5), price: 101.0, volume: 1.0 },
        Trade { timestamp: ts(6), price: 100.0, volume: 1.0 },
        Trade { timestamp: ts(7), price: 99.0, volume: 1.0 },
        Trade { timestamp: ts(8), price: 98.0, volume: 1.0 },
    ]
}

fn trades_for_imbalance() -> Vec<Trade> {
    vec![
        Trade { timestamp: ts(0), price: 100.0, volume: 1.0 },
        Trade { timestamp: ts(1), price: 101.0, volume: 1.0 },
        Trade { timestamp: ts(2), price: 102.0, volume: 1.0 },
        Trade { timestamp: ts(3), price: 101.0, volume: 1.0 },
        Trade { timestamp: ts(4), price: 100.0, volume: 1.0 },
        Trade { timestamp: ts(5), price: 101.0, volume: 1.0 },
        Trade { timestamp: ts(6), price: 102.0, volume: 1.0 },
        Trade { timestamp: ts(7), price: 103.0, volume: 1.0 },
    ]
}

#[test]
fn run_bars_close_on_consecutive_moves() {
    let trades = trades_for_runs();
    let bars = run_bars(&trades, 3);

    assert_eq!(bars.len(), 2);

    let first = &bars[0];
    assert_eq!(first.open, 100.0);
    assert_eq!(first.close, 103.0);
    assert_eq!(first.high, 103.0);
    assert_eq!(first.low, 100.0);
    assert_eq!(first.volume, 4.0);
    assert_eq!(first.tick_count, 4);

    let second = &bars[1];
    assert_eq!(second.open, 102.0);
    assert_eq!(second.close, 99.0);
    assert_eq!(second.high, 102.0);
    assert_eq!(second.low, 99.0);
    assert_eq!(second.volume, 4.0);
    assert_eq!(second.tick_count, 4);
}

#[test]
fn tick_imbalance_bars_accumulate_signed_ticks() {
    let trades = trades_for_imbalance();
    let bars = imbalance_bars(&trades, 2.0, ImbalanceBarType::Tick);

    assert_eq!(bars.len(), 2);

    let first = &bars[0];
    assert_eq!(first.open, 100.0);
    assert_eq!(first.close, 102.0);
    assert_eq!(first.tick_count, 3);
    assert_eq!(first.volume, 3.0);

    let second = &bars[1];
    assert_eq!(second.open, 101.0);
    assert_eq!(second.close, 103.0);
    assert_eq!(second.tick_count, 5);
    assert_eq!(second.volume, 5.0);
}
