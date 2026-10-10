//! Executes the Rust example on the data_processing module page.
//! Keep the page snippet and this body in sync (apart from rustfmt).

#[test]
fn data_processing_page() -> Result<(), Box<dyn std::error::Error>> {
    use chrono::NaiveDate;
    use openquant::data_processing::{
        align_calendar_rows, clean_ohlcv_rows, quality_report, DataProcessingError, OhlcvRow,
    };

    let bar = |day, close| OhlcvRow {
        timestamp: NaiveDate::from_ymd_opt(2024, 1, day).unwrap().and_hms_opt(0, 0, 0).unwrap(),
        symbol: "AAA".to_string(),
        open: close,
        high: close,
        low: close,
        close,
        volume: 1.0,
        adj_close: close,
    };
    let rows = vec![bar(2, 10.0), bar(2, 10.5), bar(3, 11.0), bar(5, 12.0)];
    let raw_report = quality_report(&rows, 0);
    assert_eq!(raw_report.row_count, 4);
    assert_eq!(raw_report.duplicate_key_count, 1);

    let (clean, report) = clean_ohlcv_rows(&rows, true);
    assert_eq!(clean.iter().map(|r| r.close).collect::<Vec<_>>(), [10.5, 11.0, 12.0]);
    assert_eq!(clean_ohlcv_rows(&rows, false).0[0].close, 10.0);
    assert_eq!(report.duplicate_key_count, 0);
    assert_eq!(report.rows_removed_by_deduplication, 1);
    assert_eq!(report.inferred_interval_us, Some(86_400_000_000));
    assert_eq!(report.gap_interval_count, 1);
    // Report-only calls carry forward the supplied removal count.
    assert_eq!(quality_report(&clean, 1), report);

    let (aligned, alignment) = align_calendar_rows(&rows, 86_400)?;
    let missing: Vec<bool> = aligned.iter().map(|r| r.is_missing_bar).collect();
    assert_eq!(missing, [false, false, true, false]);
    assert_eq!(aligned[2].timestamp, bar(4, 0.0).timestamp);
    assert_eq!(aligned[2].close, None);
    assert_eq!(alignment.rows_removed_by_deduplication, 1);
    assert!(alignment.off_grid_bars.is_empty());

    // A two-day grid starts on Jan 2: Jan 3 and Jan 5 do not land on it.
    let (coarse, omitted) = align_calendar_rows(&rows, 172_800)?;
    assert_eq!(coarse.len(), 2); // Jan 2, Jan 4; no rounding or aggregation
    assert_eq!(coarse[0].close, Some(10.5));
    assert_eq!(coarse[1].close, None);
    assert_eq!(
        omitted.off_grid_bars,
        vec![
            ("AAA".to_string(), bar(3, 0.0).timestamp),
            ("AAA".to_string(), bar(5, 0.0).timestamp),
        ]
    );
    assert!(matches!(align_calendar_rows(&rows, 0), Err(DataProcessingError::NonPositiveInterval)));
    println!(
        "raw={} clean={} removed={} gaps={}",
        raw_report.row_count,
        report.row_count,
        report.rows_removed_by_deduplication,
        report.gap_interval_count
    );
    println!("daily missing={missing:?}; two-day off-grid={}", omitted.off_grid_bars.len());
    Ok(())
}
