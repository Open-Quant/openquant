use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use openquant::data_processing::{
    align_calendar_columns, align_calendar_rows, clean_ohlcv_rows, quality_report,
    CalendarAlignmentReport, OhlcvRow,
};

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

    let (aligned, alignment) = align_calendar_rows(&clean, 60).expect("align");
    assert_eq!(alignment, CalendarAlignmentReport::default());
    assert_eq!(aligned.len(), 2);
    assert!(!aligned[0].is_missing_bar);
    assert!(!aligned[1].is_missing_bar);
}

fn bar(symbol: &str, timestamp: NaiveDateTime, close: f64) -> OhlcvRow {
    OhlcvRow {
        timestamp,
        symbol: symbol.to_string(),
        open: close,
        high: close,
        low: close,
        close,
        volume: 1.0,
        adj_close: close,
    }
}

fn day(d: u32, hour: u32) -> NaiveDateTime {
    // September 2024: the 2nd is a Monday.
    NaiveDate::from_ymd_opt(2024, 9, d).unwrap().and_hms_opt(hour, 0, 0).unwrap()
}

/// #168: bars not on a symbol's grid were dropped without a trace.
#[test]
fn align_calendar_reports_off_grid_bars() {
    // Daily bars at midnight, except the 4th, stamped at 16:00. On a daily grid from the
    // 2nd at midnight that bar has no slot, so the 4th shows as missing and the bar is
    // reported, not silently lost.
    let rows = vec![
        bar("AAA", day(2, 0), 1.0),
        bar("AAA", day(3, 0), 2.0),
        bar("AAA", day(4, 16), 3.0),
        bar("AAA", day(5, 0), 4.0),
        bar("BBB", day(2, 0), 5.0),
        bar("BBB", day(2, 0), 5.5),
        bar("BBB", day(3, 0), 6.0),
    ];
    let (aligned, report) = align_calendar_rows(&rows, 86_400).expect("align");
    assert_eq!(aligned.len(), 6);
    assert_eq!(
        aligned.iter().filter(|r| r.symbol == "AAA").map(|r| r.is_missing_bar).collect::<Vec<_>>(),
        vec![false, false, true, false]
    );
    assert!(aligned.iter().all(|r| r.close != Some(3.0)));
    assert_eq!(report.off_grid_bars, vec![("AAA".to_string(), day(4, 16))]);
    assert_eq!(report.rows_removed_by_deduplication, 1);

    // The column form reports the same.
    let (cols, col_report) = align_calendar_columns(
        &openquant::data_processing::OhlcvColumns {
            timestamps_us: rows.iter().map(|r| r.timestamp.and_utc().timestamp_micros()).collect(),
            symbols: rows.iter().map(|r| r.symbol.clone()).collect(),
            open: rows.iter().map(|r| r.open).collect(),
            high: rows.iter().map(|r| r.high).collect(),
            low: rows.iter().map(|r| r.low).collect(),
            close: rows.iter().map(|r| r.close).collect(),
            volume: rows.iter().map(|r| r.volume).collect(),
            adj_close: rows.iter().map(|r| r.adj_close).collect(),
        },
        86_400,
    )
    .expect("align columns");
    assert_eq!(cols.timestamps_us.len(), 6);
    assert_eq!(col_report, report);
}

/// #168: any spacing over one day was a gap, so every weekend was one on daily data, and
/// no spacing on intraday data ever was.
#[test]
fn gap_count_follows_the_bar_frequency() {
    // Mon 2 .. Fri 6, then Mon 9: a weekend, not a gap.
    let weekdays: Vec<OhlcvRow> =
        [2, 3, 4, 5, 6, 9].iter().map(|&d| bar("AAA", day(d, 0), 1.0)).collect();
    let report = quality_report(&weekdays, 0);
    assert_eq!(report.inferred_interval_us, Some(86_400_000_000));
    assert_eq!(report.gap_interval_count, 0);

    // Drop Wednesday the 4th: that is a gap. Fri 6 -> Tue 10 skips Monday: another.
    let missing: Vec<OhlcvRow> =
        [2, 3, 5, 6, 10].iter().map(|&d| bar("AAA", day(d, 0), 1.0)).collect();
    assert_eq!(quality_report(&missing, 0).gap_interval_count, 2);

    // Daily bars stamped at a US close in UTC, shifting by an hour at a DST change, are
    // still daily: 21:00 -> 20:00 the next day is 23 hours.
    let dst = vec![
        bar("AAA", day(2, 21), 1.0),
        bar("AAA", day(3, 21), 1.0),
        bar("AAA", day(4, 20), 1.0),
        bar("AAA", day(5, 20), 1.0),
        bar("AAA", day(6, 20), 1.0),
        bar("AAA", day(9, 20), 1.0),
    ];
    let report = quality_report(&dst, 0);
    assert_eq!(report.inferred_interval_us, Some(86_400_000_000));
    assert_eq!(report.gap_interval_count, 0);

    // Hourly bars with one hour missing: a gap the one-day rule could not see.
    let hourly: Vec<OhlcvRow> =
        [9, 10, 11, 13, 14, 15].iter().map(|&h| bar("AAA", day(2, h), 1.0)).collect();
    let report = quality_report(&hourly, 0);
    assert_eq!(report.inferred_interval_us, Some(3_600_000_000));
    assert_eq!(report.gap_interval_count, 1);

    // Spacing is pooled over symbols; a symbol with one bar has none.
    let mut pooled = weekdays.clone();
    pooled.push(bar("BBB", day(2, 0), 1.0));
    pooled.push(bar("BBB", day(4, 0), 1.0));
    let report = quality_report(&pooled, 0);
    assert_eq!(report.inferred_interval_us, Some(86_400_000_000));
    assert_eq!(report.gap_interval_count, 1); // BBB skips Tuesday
    let single = quality_report(&[bar("AAA", day(2, 0), 1.0)], 0);
    assert_eq!((single.inferred_interval_us, single.gap_interval_count), (None, 0));
}

/// #220: polars 0.55 neither keeps the left order of a join nor sorts stably unless asked.
/// Keep-first/keep-last must follow the input order of duplicates, and aligned output must
/// be sorted by `(symbol, timestamp)`, with enough rows that an unordered join would show.
#[test]
fn dedupe_keeps_input_order_and_alignment_output_is_sorted() {
    let symbols = ["ZZZ", "MMM", "AAA", "QQQ"];
    let mut rows = Vec::new();
    // Interleaved symbols, descending time, every key three times with different closes.
    for t in (0..200).rev() {
        for (s, sym) in symbols.iter().enumerate() {
            if (t + s) % 7 == 3 {
                continue; // leave holes so alignment has missing bars
            }
            for copy in 0..3 {
                rows.push(bar(sym, ts(60 * t as i64), (1000 * s + 10 * t + copy) as f64));
            }
        }
    }

    for keep_last in [false, true] {
        let (clean, report) = clean_ohlcv_rows(&rows, keep_last);
        assert_eq!(report.rows_removed_by_deduplication, rows.len() * 2 / 3);
        let copy = if keep_last { 2.0 } else { 0.0 };
        for row in &clean {
            let s = symbols.iter().position(|sym| *sym == row.symbol).unwrap() as f64;
            let t = (row.timestamp.and_utc().timestamp() / 60) as f64;
            assert_eq!(row.close, 1000.0 * s + 10.0 * t + copy, "keep_last={keep_last}");
        }
        let keys: Vec<_> = clean.iter().map(|r| (r.symbol.clone(), r.timestamp)).collect();
        assert!(keys.windows(2).all(|w| w[0] < w[1]), "clean output not sorted");
    }

    let (aligned, report) = align_calendar_rows(&rows, 60).expect("align");
    assert!(report.off_grid_bars.is_empty());
    let keys: Vec<_> = aligned.iter().map(|r| (r.symbol.clone(), r.timestamp)).collect();
    assert!(keys.windows(2).all(|w| w[0] < w[1]), "aligned output not sorted");
    for row in &aligned {
        let s = symbols.iter().position(|sym| *sym == row.symbol).unwrap();
        let t = (row.timestamp.and_utc().timestamp() / 60) as usize;
        assert_eq!(row.is_missing_bar, (t + s) % 7 == 3);
        if !row.is_missing_bar {
            assert_eq!(row.close, Some((1000 * s + 10 * t + 2) as f64));
        } else {
            assert_eq!(row.close, None);
        }
    }
}
