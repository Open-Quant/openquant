use chrono::{DateTime, NaiveDateTime, Utc};
use openquant::data_processing::{align_calendar_rows, clean_ohlcv_rows, OhlcvRow};

fn ts(seconds: i64) -> NaiveDateTime {
    DateTime::<Utc>::from_timestamp(seconds, 0).expect("timestamp").naive_utc()
}

fn sample_rows() -> Vec<OhlcvRow> {
    vec![
        OhlcvRow {
            timestamp: ts(0),
            symbol: "AAPL".to_string(),
            open: 100.0,
            high: 101.0,
            low: 99.0,
            close: 100.5,
            volume: 10.0,
            adj_close: 100.5,
        },
        OhlcvRow {
            timestamp: ts(60),
            symbol: "AAPL".to_string(),
            open: 100.5,
            high: 101.2,
            low: 100.3,
            close: 101.0,
            volume: 11.0,
            adj_close: 101.0,
        },
        OhlcvRow {
            timestamp: ts(60),
            symbol: "AAPL".to_string(),
            open: 100.6,
            high: 101.3,
            low: 100.4,
            close: 101.1,
            volume: 12.0,
            adj_close: 101.1,
        },
    ]
}

#[test]
fn clean_and_align_rows() {
    let rows = sample_rows();
    let (clean, report) = clean_ohlcv_rows(&rows, true);
    assert_eq!(clean.len(), 2);
    assert_eq!(report.rows_removed_by_deduplication, 1);
    assert_eq!(report.duplicate_key_count, 0);

    let aligned = align_calendar_rows(&clean, 60).expect("align");
    assert_eq!(aligned.len(), 2);
    assert!(!aligned[0].is_missing_bar);
    assert!(!aligned[1].is_missing_bar);
}
