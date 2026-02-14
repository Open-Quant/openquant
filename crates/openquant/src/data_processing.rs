use chrono::{DateTime, NaiveDateTime, Utc};
use polars::prelude::*;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq)]
pub struct OhlcvRow {
    pub timestamp: NaiveDateTime,
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub adj_close: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OhlcvColumns {
    pub timestamps_us: Vec<i64>,
    pub symbols: Vec<String>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
    pub adj_close: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlignedOhlcvRow {
    pub timestamp: NaiveDateTime,
    pub symbol: String,
    pub open: Option<f64>,
    pub high: Option<f64>,
    pub low: Option<f64>,
    pub close: Option<f64>,
    pub volume: Option<f64>,
    pub adj_close: Option<f64>,
    pub is_missing_bar: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlignedOhlcvColumns {
    pub timestamps_us: Vec<i64>,
    pub symbols: Vec<String>,
    pub open: Vec<Option<f64>>,
    pub high: Vec<Option<f64>>,
    pub low: Vec<Option<f64>>,
    pub close: Vec<Option<f64>>,
    pub volume: Vec<Option<f64>>,
    pub adj_close: Vec<Option<f64>>,
    pub is_missing_bar: Vec<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DataQualityReport {
    pub row_count: usize,
    pub symbol_count: usize,
    pub duplicate_key_count: usize,
    pub gap_interval_count: usize,
    pub ts_min: Option<NaiveDateTime>,
    pub ts_max: Option<NaiveDateTime>,
    pub rows_removed_by_deduplication: usize,
}

fn require_ohlcv_columns(df: &DataFrame) -> Result<(), String> {
    for name in ["symbol", "ts_us", "open", "high", "low", "close", "volume", "adj_close"] {
        df.column(name).map_err(|e| format!("missing required column '{name}': {e}"))?;
    }
    Ok(())
}

fn sort_ohlcv_df(df: &DataFrame) -> Result<DataFrame, String> {
    df.sort(
        ["symbol", "ts_us"],
        SortMultipleOptions::new().with_order_descending_multi([false, false]),
    )
    .map_err(|e| format!("polars sort failed: {e}"))
}

fn micros_to_naive(ts_us: i64) -> Option<NaiveDateTime> {
    DateTime::<Utc>::from_timestamp_micros(ts_us).map(|dt| dt.naive_utc())
}

fn quality_report_from_sorted_df(
    sorted: &DataFrame,
    rows_removed_by_deduplication: usize,
) -> Result<DataQualityReport, String> {
    require_ohlcv_columns(sorted)?;

    let symbols = sorted
        .column("symbol")
        .map_err(|e| format!("symbol column error: {e}"))?
        .str()
        .map_err(|e| format!("symbol dtype error: {e}"))?;
    let ts = sorted
        .column("ts_us")
        .map_err(|e| format!("ts_us column error: {e}"))?
        .i64()
        .map_err(|e| format!("ts_us dtype error: {e}"))?;

    let mut symbol_set: HashSet<&str> = HashSet::new();
    let mut duplicate_key_count = 0usize;
    let mut gap_interval_count = 0usize;
    let day_us = 24 * 3600 * 1_000_000i64;

    let mut prev_symbol: Option<&str> = None;
    let mut prev_ts: Option<i64> = None;

    for i in 0..sorted.height() {
        let s = symbols.get(i).ok_or_else(|| format!("null symbol at row {i}"))?;
        let t = ts.get(i).ok_or_else(|| format!("null ts_us at row {i}"))?;
        symbol_set.insert(s);

        if let (Some(ps), Some(pt)) = (prev_symbol, prev_ts) {
            if ps == s && pt == t {
                duplicate_key_count += 1;
            } else if ps == s && t - pt > day_us {
                gap_interval_count += 1;
            }
        }

        prev_symbol = Some(s);
        prev_ts = Some(t);
    }

    let ts_min = ts.min().and_then(micros_to_naive);
    let ts_max = ts.max().and_then(micros_to_naive);

    Ok(DataQualityReport {
        row_count: sorted.height(),
        symbol_count: symbol_set.len(),
        duplicate_key_count,
        gap_interval_count,
        ts_min,
        ts_max,
        rows_removed_by_deduplication,
    })
}

pub fn quality_report_df(
    df: &DataFrame,
    rows_removed_by_deduplication: usize,
) -> Result<DataQualityReport, String> {
    require_ohlcv_columns(df)?;
    if df.height() == 0 {
        return Ok(DataQualityReport {
            row_count: 0,
            symbol_count: 0,
            duplicate_key_count: 0,
            gap_interval_count: 0,
            ts_min: None,
            ts_max: None,
            rows_removed_by_deduplication,
        });
    }
    let sorted = sort_ohlcv_df(df)?;
    quality_report_from_sorted_df(&sorted, rows_removed_by_deduplication)
}

pub fn clean_ohlcv_df(
    df: &DataFrame,
    keep_last: bool,
) -> Result<(DataFrame, DataQualityReport), String> {
    require_ohlcv_columns(df)?;

    if df.height() == 0 {
        let empty = sort_ohlcv_df(df)?;
        let report = DataQualityReport {
            row_count: 0,
            symbol_count: 0,
            duplicate_key_count: 0,
            gap_interval_count: 0,
            ts_min: None,
            ts_max: None,
            rows_removed_by_deduplication: 0,
        };
        return Ok((empty, report));
    }

    let sorted = sort_ohlcv_df(df)?;
    let before = sorted.height();

    let cleaned = sorted
        .unique_stable(
            Some(&["symbol".to_string(), "ts_us".to_string()]),
            if keep_last { UniqueKeepStrategy::Last } else { UniqueKeepStrategy::First },
            None,
        )
        .map_err(|e| format!("polars unique failed: {e}"))?;

    let removed = before.saturating_sub(cleaned.height());
    let mut report = quality_report_from_sorted_df(&cleaned, removed)?;
    report.duplicate_key_count = 0;

    Ok((cleaned, report))
}

pub fn align_calendar_df(df: &DataFrame, interval_seconds: i64) -> Result<DataFrame, String> {
    if interval_seconds <= 0 {
        return Err("interval_seconds must be > 0".to_string());
    }

    let (cleaned, _) = clean_ohlcv_df(df, true)?;
    if cleaned.height() == 0 {
        let mut out = cleaned.clone();
        out.with_column(Series::new("is_missing_bar".into(), Vec::<bool>::new()))
            .map_err(|e| format!("failed to add is_missing_bar: {e}"))?;
        return Ok(out);
    }

    let symbols = cleaned
        .column("symbol")
        .map_err(|e| format!("symbol column error: {e}"))?
        .str()
        .map_err(|e| format!("symbol dtype error: {e}"))?;
    let ts = cleaned
        .column("ts_us")
        .map_err(|e| format!("ts_us column error: {e}"))?
        .i64()
        .map_err(|e| format!("ts_us dtype error: {e}"))?;

    let step_us = interval_seconds * 1_000_000;

    let mut cal_symbols: Vec<String> = Vec::new();
    let mut cal_ts: Vec<i64> = Vec::new();

    let mut i = 0usize;
    while i < cleaned.height() {
        let symbol = symbols.get(i).ok_or_else(|| format!("null symbol at row {i}"))?;
        let start = ts.get(i).ok_or_else(|| format!("null ts_us at row {i}"))?;

        let mut j = i + 1;
        while j < cleaned.height() && symbols.get(j) == Some(symbol) {
            j += 1;
        }

        let end = ts.get(j - 1).ok_or_else(|| format!("null ts_us at row {}", j - 1))?;

        let mut cur = start;
        while cur <= end {
            cal_symbols.push(symbol.to_string());
            cal_ts.push(cur);
            cur += step_us;
        }

        i = j;
    }

    let calendar = df!("symbol" => cal_symbols, "ts_us" => cal_ts)
        .map_err(|e| format!("calendar df build failed: {e}"))?;

    let mut out = calendar
        .left_join(&cleaned, ["symbol", "ts_us"], ["symbol", "ts_us"])
        .map_err(|e| format!("calendar join failed: {e}"))?;

    let mut missing = out.column("open").map_err(to_string_err)?.is_null().into_series();
    missing.rename("is_missing_bar".into());
    out.with_column(missing).map_err(|e| format!("failed to add is_missing_bar: {e}"))?;

    Ok(out)
}

fn validate_lengths(columns: &OhlcvColumns) -> Result<(), String> {
    let n = columns.timestamps_us.len();
    let lengths = [
        columns.symbols.len(),
        columns.open.len(),
        columns.high.len(),
        columns.low.len(),
        columns.close.len(),
        columns.volume.len(),
        columns.adj_close.len(),
    ];
    if lengths.iter().any(|&len| len != n) {
        return Err(format!(
            "ohlcv vector length mismatch: ts={n}, symbol={}, open={}, high={}, low={}, close={}, volume={}, adj_close={}",
            columns.symbols.len(),
            columns.open.len(),
            columns.high.len(),
            columns.low.len(),
            columns.close.len(),
            columns.volume.len(),
            columns.adj_close.len()
        ));
    }
    Ok(())
}

fn to_polars_df(columns: &OhlcvColumns) -> Result<DataFrame, String> {
    validate_lengths(columns)?;
    df!(
        "symbol" => columns.symbols.clone(),
        "ts_us" => columns.timestamps_us.clone(),
        "open" => columns.open.clone(),
        "high" => columns.high.clone(),
        "low" => columns.low.clone(),
        "close" => columns.close.clone(),
        "volume" => columns.volume.clone(),
        "adj_close" => columns.adj_close.clone(),
    )
    .map_err(|e| format!("polars df build failed: {e}"))
}

fn df_to_ohlcv_columns(df: &DataFrame) -> Result<OhlcvColumns, String> {
    let timestamps_us = df
        .column("ts_us")
        .map_err(|e| format!("missing ts_us: {e}"))?
        .i64()
        .map_err(|e| format!("ts_us type error: {e}"))?
        .into_no_null_iter()
        .collect::<Vec<_>>();
    let symbols = df
        .column("symbol")
        .map_err(|e| format!("missing symbol: {e}"))?
        .str()
        .map_err(|e| format!("symbol type error: {e}"))?
        .into_no_null_iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let open = df
        .column("open")
        .map_err(|e| format!("missing open: {e}"))?
        .f64()
        .map_err(|e| format!("open type error: {e}"))?
        .into_no_null_iter()
        .collect::<Vec<_>>();
    let high = df
        .column("high")
        .map_err(|e| format!("missing high: {e}"))?
        .f64()
        .map_err(|e| format!("high type error: {e}"))?
        .into_no_null_iter()
        .collect::<Vec<_>>();
    let low = df
        .column("low")
        .map_err(|e| format!("missing low: {e}"))?
        .f64()
        .map_err(|e| format!("low type error: {e}"))?
        .into_no_null_iter()
        .collect::<Vec<_>>();
    let close = df
        .column("close")
        .map_err(|e| format!("missing close: {e}"))?
        .f64()
        .map_err(|e| format!("close type error: {e}"))?
        .into_no_null_iter()
        .collect::<Vec<_>>();
    let volume = df
        .column("volume")
        .map_err(|e| format!("missing volume: {e}"))?
        .f64()
        .map_err(|e| format!("volume type error: {e}"))?
        .into_no_null_iter()
        .collect::<Vec<_>>();
    let adj_close = df
        .column("adj_close")
        .map_err(|e| format!("missing adj_close: {e}"))?
        .f64()
        .map_err(|e| format!("adj_close type error: {e}"))?
        .into_no_null_iter()
        .collect::<Vec<_>>();

    Ok(OhlcvColumns { timestamps_us, symbols, open, high, low, close, volume, adj_close })
}

fn to_string_err<T: core::fmt::Display>(e: T) -> String {
    e.to_string()
}

pub fn quality_report_columns(
    columns: &OhlcvColumns,
    rows_removed_by_deduplication: usize,
) -> Result<DataQualityReport, String> {
    let df = to_polars_df(columns)?;
    quality_report_df(&df, rows_removed_by_deduplication)
}

pub fn clean_ohlcv_columns(
    columns: &OhlcvColumns,
    keep_last: bool,
) -> Result<(OhlcvColumns, DataQualityReport), String> {
    let df = to_polars_df(columns)?;
    let (clean_df, report) = clean_ohlcv_df(&df, keep_last)?;
    let clean_cols = df_to_ohlcv_columns(&clean_df)?;
    Ok((clean_cols, report))
}

pub fn align_calendar_columns(
    columns: &OhlcvColumns,
    interval_seconds: i64,
) -> Result<AlignedOhlcvColumns, String> {
    let df = to_polars_df(columns)?;
    let out = align_calendar_df(&df, interval_seconds)?;

    let timestamps_us = out
        .column("ts_us")
        .map_err(|e| format!("missing ts_us: {e}"))?
        .i64()
        .map_err(|e| format!("ts_us type error: {e}"))?
        .into_no_null_iter()
        .collect::<Vec<_>>();
    let symbols = out
        .column("symbol")
        .map_err(|e| format!("missing symbol: {e}"))?
        .str()
        .map_err(|e| format!("symbol type error: {e}"))?
        .into_no_null_iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();

    let open = out
        .column("open")
        .map_err(|e| format!("missing open: {e}"))?
        .f64()
        .map_err(|e| format!("open type error: {e}"))?
        .into_iter()
        .collect::<Vec<_>>();
    let high = out
        .column("high")
        .map_err(|e| format!("missing high: {e}"))?
        .f64()
        .map_err(|e| format!("high type error: {e}"))?
        .into_iter()
        .collect::<Vec<_>>();
    let low = out
        .column("low")
        .map_err(|e| format!("missing low: {e}"))?
        .f64()
        .map_err(|e| format!("low type error: {e}"))?
        .into_iter()
        .collect::<Vec<_>>();
    let close = out
        .column("close")
        .map_err(|e| format!("missing close: {e}"))?
        .f64()
        .map_err(|e| format!("close type error: {e}"))?
        .into_iter()
        .collect::<Vec<_>>();
    let volume = out
        .column("volume")
        .map_err(|e| format!("missing volume: {e}"))?
        .f64()
        .map_err(|e| format!("volume type error: {e}"))?
        .into_iter()
        .collect::<Vec<_>>();
    let adj_close = out
        .column("adj_close")
        .map_err(|e| format!("missing adj_close: {e}"))?
        .f64()
        .map_err(|e| format!("adj_close type error: {e}"))?
        .into_iter()
        .collect::<Vec<_>>();
    let is_missing_bar = out
        .column("is_missing_bar")
        .map_err(|e| format!("missing is_missing_bar: {e}"))?
        .bool()
        .map_err(|e| format!("is_missing_bar type error: {e}"))?
        .into_no_null_iter()
        .collect::<Vec<_>>();

    Ok(AlignedOhlcvColumns {
        timestamps_us,
        symbols,
        open,
        high,
        low,
        close,
        volume,
        adj_close,
        is_missing_bar,
    })
}

fn rows_to_columns(rows: &[OhlcvRow]) -> OhlcvColumns {
    let mut out = OhlcvColumns {
        timestamps_us: Vec::with_capacity(rows.len()),
        symbols: Vec::with_capacity(rows.len()),
        open: Vec::with_capacity(rows.len()),
        high: Vec::with_capacity(rows.len()),
        low: Vec::with_capacity(rows.len()),
        close: Vec::with_capacity(rows.len()),
        volume: Vec::with_capacity(rows.len()),
        adj_close: Vec::with_capacity(rows.len()),
    };
    for row in rows {
        out.timestamps_us.push(
            DateTime::<Utc>::from_naive_utc_and_offset(row.timestamp, Utc).timestamp_micros(),
        );
        out.symbols.push(row.symbol.clone());
        out.open.push(row.open);
        out.high.push(row.high);
        out.low.push(row.low);
        out.close.push(row.close);
        out.volume.push(row.volume);
        out.adj_close.push(row.adj_close);
    }
    out
}

fn columns_to_rows(columns: &OhlcvColumns) -> Vec<OhlcvRow> {
    let mut rows = Vec::with_capacity(columns.timestamps_us.len());
    for i in 0..columns.timestamps_us.len() {
        let dt = DateTime::<Utc>::from_timestamp_micros(columns.timestamps_us[i])
            .expect("valid datetime")
            .naive_utc();
        rows.push(OhlcvRow {
            timestamp: dt,
            symbol: columns.symbols[i].clone(),
            open: columns.open[i],
            high: columns.high[i],
            low: columns.low[i],
            close: columns.close[i],
            volume: columns.volume[i],
            adj_close: columns.adj_close[i],
        });
    }
    rows
}

fn aligned_columns_to_rows(columns: &AlignedOhlcvColumns) -> Vec<AlignedOhlcvRow> {
    let mut rows = Vec::with_capacity(columns.timestamps_us.len());
    for i in 0..columns.timestamps_us.len() {
        let dt = DateTime::<Utc>::from_timestamp_micros(columns.timestamps_us[i])
            .expect("valid datetime")
            .naive_utc();
        rows.push(AlignedOhlcvRow {
            timestamp: dt,
            symbol: columns.symbols[i].clone(),
            open: columns.open[i],
            high: columns.high[i],
            low: columns.low[i],
            close: columns.close[i],
            volume: columns.volume[i],
            adj_close: columns.adj_close[i],
            is_missing_bar: columns.is_missing_bar[i],
        });
    }
    rows
}

pub fn clean_ohlcv_rows(rows: &[OhlcvRow], keep_last: bool) -> (Vec<OhlcvRow>, DataQualityReport) {
    let cols = rows_to_columns(rows);
    let (clean_cols, report) = clean_ohlcv_columns(&cols, keep_last).expect("validated rows");
    (columns_to_rows(&clean_cols), report)
}

pub fn quality_report(
    rows: &[OhlcvRow],
    rows_removed_by_deduplication: usize,
) -> DataQualityReport {
    let cols = rows_to_columns(rows);
    quality_report_columns(&cols, rows_removed_by_deduplication).expect("validated rows")
}

pub fn align_calendar_rows(
    rows: &[OhlcvRow],
    interval_seconds: i64,
) -> Result<Vec<AlignedOhlcvRow>, String> {
    let cols = rows_to_columns(rows);
    let aligned_cols = align_calendar_columns(&cols, interval_seconds)?;
    Ok(aligned_columns_to_rows(&aligned_cols))
}
