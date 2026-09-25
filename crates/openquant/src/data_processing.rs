//! OHLCV data hygiene: deduplication, calendar alignment and a data-quality report.
//!
//! This is the preparation step before bars, labels or features (it has no AFML chapter;
//! AFML chapter 2 assumes clean input). Each operation comes in three forms that share one
//! implementation:
//!
//! - `*_df` functions take a Polars [`DataFrame`] with columns `symbol` (string), `ts_us`
//!   (`i64` microseconds since the Unix epoch, UTC) and `open`, `high`, `low`, `close`,
//!   `volume`, `adj_close` (`f64`);
//! - `*_columns` functions take the same data as [`OhlcvColumns`];
//! - row functions take [`OhlcvRow`]s with a [`NaiveDateTime`] interpreted as UTC.
//!
//! Output is sorted by `(symbol, timestamp)`. A duplicate is a repeated `(symbol, timestamp)`
//! key. A gap is two consecutive bars of one symbol more than **one day** apart, whatever
//! the bar interval, so weekends count as gaps on daily data.
//!
//! ```
//! use chrono::NaiveDate;
//! use openquant::data_processing::{align_calendar_rows, clean_ohlcv_rows, OhlcvRow};
//!
//! # fn main() -> Result<(), openquant::data_processing::DataProcessingError> {
//! let bar = |day, close| OhlcvRow {
//!     timestamp: NaiveDate::from_ymd_opt(2024, 1, day).unwrap().and_hms_opt(0, 0, 0).unwrap(),
//!     symbol: "AAA".to_string(),
//!     open: close,
//!     high: close,
//!     low: close,
//!     close,
//!     volume: 1.0,
//!     adj_close: close,
//! };
//! // Jan 2 is duplicated and Jan 4 is missing.
//! let rows = vec![bar(2, 10.0), bar(2, 10.5), bar(3, 11.0), bar(5, 12.0)];
//!
//! let (clean, report) = clean_ohlcv_rows(&rows, true);
//! assert_eq!(clean.len(), 3);
//! assert_eq!(clean[0].close, 10.5); // keep_last keeps the later duplicate
//! assert_eq!(report.rows_removed_by_deduplication, 1);
//! assert_eq!(report.gap_interval_count, 1); // Jan 3 -> Jan 5
//!
//! let aligned = align_calendar_rows(&rows, 86_400)?;
//! let missing: Vec<bool> = aligned.iter().map(|r| r.is_missing_bar).collect();
//! assert_eq!(missing, vec![false, false, true, false]);
//! assert_eq!(aligned[2].close, None);
//! # Ok(())
//! # }
//! ```
#![deny(missing_docs)]

use chrono::{DateTime, NaiveDateTime, Utc};
use polars::prelude::*;
use std::collections::HashSet;

/// Errors returned by the data-processing functions.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DataProcessingError {
    /// A Polars operation failed. `reason` is Polars' message; its error type is neither
    /// `Clone` nor `PartialEq`, so it is not kept as a source.
    #[error("{context}: {reason}")]
    Frame {
        /// What was being done, e.g. `"missing required column 'close'"`.
        context: String,
        /// Polars' error message.
        reason: String,
    },
    /// A `symbol` or `ts_us` value is null.
    #[error("null {column} at row {row}")]
    NullValue {
        /// The column holding the null.
        column: &'static str,
        /// Row position (in sorted order).
        row: usize,
    },
    /// `interval_seconds` is zero or negative.
    #[error("interval_seconds must be > 0")]
    NonPositiveInterval,
    /// The OHLCV column vectors differ in length; the payload lists every length.
    #[error("{0}")]
    LengthMismatch(String),
}

impl DataProcessingError {
    fn frame(context: impl Into<String>, err: impl core::fmt::Display) -> Self {
        Self::Frame { context: context.into(), reason: err.to_string() }
    }
}

/// One OHLCV bar.
#[derive(Debug, Clone, PartialEq)]
pub struct OhlcvRow {
    /// Bar timestamp, interpreted as UTC.
    pub timestamp: NaiveDateTime,
    /// Instrument identifier.
    pub symbol: String,
    /// Opening price.
    pub open: f64,
    /// High price.
    pub high: f64,
    /// Low price.
    pub low: f64,
    /// Closing price.
    pub close: f64,
    /// Traded volume.
    pub volume: f64,
    /// Close adjusted for splits and dividends.
    pub adj_close: f64,
}

/// OHLCV bars in columnar form; every vector must have the same length.
#[derive(Debug, Clone, PartialEq)]
pub struct OhlcvColumns {
    /// Timestamps in microseconds since the Unix epoch (UTC).
    pub timestamps_us: Vec<i64>,
    /// Instrument identifiers.
    pub symbols: Vec<String>,
    /// Opening prices.
    pub open: Vec<f64>,
    /// High prices.
    pub high: Vec<f64>,
    /// Low prices.
    pub low: Vec<f64>,
    /// Closing prices.
    pub close: Vec<f64>,
    /// Traded volumes.
    pub volume: Vec<f64>,
    /// Adjusted closes.
    pub adj_close: Vec<f64>,
}

/// One bar on a regular calendar grid; price fields are `None` where no bar existed.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignedOhlcvRow {
    /// Grid timestamp (UTC).
    pub timestamp: NaiveDateTime,
    /// Instrument identifier.
    pub symbol: String,
    /// Opening price, if a bar existed.
    pub open: Option<f64>,
    /// High price, if a bar existed.
    pub high: Option<f64>,
    /// Low price, if a bar existed.
    pub low: Option<f64>,
    /// Closing price, if a bar existed.
    pub close: Option<f64>,
    /// Traded volume, if a bar existed.
    pub volume: Option<f64>,
    /// Adjusted close, if a bar existed.
    pub adj_close: Option<f64>,
    /// `true` when the grid point had no bar in the input.
    pub is_missing_bar: bool,
}

/// Calendar-aligned bars in columnar form; see [`AlignedOhlcvRow`] for the fields.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignedOhlcvColumns {
    /// Grid timestamps in microseconds since the Unix epoch (UTC).
    pub timestamps_us: Vec<i64>,
    /// Instrument identifiers.
    pub symbols: Vec<String>,
    /// Opening prices (`None` for missing bars).
    pub open: Vec<Option<f64>>,
    /// High prices (`None` for missing bars).
    pub high: Vec<Option<f64>>,
    /// Low prices (`None` for missing bars).
    pub low: Vec<Option<f64>>,
    /// Closing prices (`None` for missing bars).
    pub close: Vec<Option<f64>>,
    /// Traded volumes (`None` for missing bars).
    pub volume: Vec<Option<f64>>,
    /// Adjusted closes (`None` for missing bars).
    pub adj_close: Vec<Option<f64>>,
    /// `true` where the grid point had no bar in the input.
    pub is_missing_bar: Vec<bool>,
}

/// Summary diagnostics of an OHLCV data set.
#[derive(Debug, Clone, PartialEq)]
pub struct DataQualityReport {
    /// Number of rows.
    pub row_count: usize,
    /// Number of distinct symbols.
    pub symbol_count: usize,
    /// Rows whose `(symbol, timestamp)` repeats the previous row's (0 after cleaning).
    pub duplicate_key_count: usize,
    /// Consecutive bars of one symbol more than one day apart.
    pub gap_interval_count: usize,
    /// Earliest timestamp (UTC), `None` when empty.
    pub ts_min: Option<NaiveDateTime>,
    /// Latest timestamp (UTC), `None` when empty.
    pub ts_max: Option<NaiveDateTime>,
    /// Rows dropped by deduplication (passed through by the report functions, computed by
    /// the clean functions).
    pub rows_removed_by_deduplication: usize,
}

fn require_ohlcv_columns(df: &DataFrame) -> Result<(), DataProcessingError> {
    for name in ["symbol", "ts_us", "open", "high", "low", "close", "volume", "adj_close"] {
        df.column(name).map_err(|e| {
            DataProcessingError::frame(format!("missing required column '{name}'"), e)
        })?;
    }
    Ok(())
}

fn sort_ohlcv_df(df: &DataFrame) -> Result<DataFrame, DataProcessingError> {
    df.sort(
        ["symbol", "ts_us"],
        SortMultipleOptions::new().with_order_descending_multi([false, false]),
    )
    .map_err(|e| DataProcessingError::frame("polars sort failed", e))
}

fn micros_to_naive(ts_us: i64) -> Option<NaiveDateTime> {
    DateTime::<Utc>::from_timestamp_micros(ts_us).map(|dt| dt.naive_utc())
}

fn quality_report_from_sorted_df(
    sorted: &DataFrame,
    rows_removed_by_deduplication: usize,
) -> Result<DataQualityReport, DataProcessingError> {
    require_ohlcv_columns(sorted)?;

    let symbols = sorted
        .column("symbol")
        .map_err(|e| DataProcessingError::frame("symbol column error", e))?
        .str()
        .map_err(|e| DataProcessingError::frame("symbol dtype error", e))?;
    let ts = sorted
        .column("ts_us")
        .map_err(|e| DataProcessingError::frame("ts_us column error", e))?
        .i64()
        .map_err(|e| DataProcessingError::frame("ts_us dtype error", e))?;

    let mut symbol_set: HashSet<&str> = HashSet::new();
    let mut duplicate_key_count = 0usize;
    let mut gap_interval_count = 0usize;
    let day_us = 24 * 3600 * 1_000_000i64;

    let mut prev_symbol: Option<&str> = None;
    let mut prev_ts: Option<i64> = None;

    for i in 0..sorted.height() {
        let s =
            symbols.get(i).ok_or(DataProcessingError::NullValue { column: "symbol", row: i })?;
        let t = ts.get(i).ok_or(DataProcessingError::NullValue { column: "ts_us", row: i })?;
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

/// Data-quality report for an OHLCV [`DataFrame`], without modifying it.
///
/// `rows_removed_by_deduplication` is copied into the report, so a caller that deduplicated
/// earlier can carry the count forward.
///
/// # Errors
///
/// - [`DataProcessingError::Frame`] if a required column is missing, has the wrong dtype, or
///   a Polars operation fails.
/// - [`DataProcessingError::NullValue`] if a `symbol` or `ts_us` value is null.
pub fn quality_report_df(
    df: &DataFrame,
    rows_removed_by_deduplication: usize,
) -> Result<DataQualityReport, DataProcessingError> {
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

/// Sorts an OHLCV [`DataFrame`] by `(symbol, ts_us)` and drops duplicate keys, keeping the
/// last occurrence when `keep_last` is true and the first otherwise.
///
/// Returns the cleaned frame and its quality report, with `rows_removed_by_deduplication`
/// set to the number of rows dropped. Prices are not validated or filled.
///
/// # Errors
///
/// - [`DataProcessingError::Frame`] if a required column is missing, has the wrong dtype, or
///   a Polars operation fails.
/// - [`DataProcessingError::NullValue`] if a `symbol` or `ts_us` value is null.
pub fn clean_ohlcv_df(
    df: &DataFrame,
    keep_last: bool,
) -> Result<(DataFrame, DataQualityReport), DataProcessingError> {
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
        .map_err(|e| DataProcessingError::frame("polars unique failed", e))?;

    let removed = before.saturating_sub(cleaned.height());
    let mut report = quality_report_from_sorted_df(&cleaned, removed)?;
    report.duplicate_key_count = 0;

    Ok((cleaned, report))
}

/// Cleans an OHLCV [`DataFrame`] (keeping the last duplicate) and reindexes each symbol onto
/// a regular grid from its first to its last timestamp in steps of `interval_seconds`.
///
/// Grid points without a bar get null prices and `is_missing_bar = true`. The grid starts at
/// each symbol's first timestamp, so **bars whose timestamp is not on that grid are dropped**;
/// the interval should divide the data's spacing. A short interval over a long span produces
/// many rows.
///
/// # Errors
///
/// - [`DataProcessingError::NonPositiveInterval`] if `interval_seconds <= 0`.
/// - [`DataProcessingError::Frame`] if a required column is missing, has the wrong dtype, or
///   a Polars operation fails.
/// - [`DataProcessingError::NullValue`] if a `symbol` or `ts_us` value is null.
pub fn align_calendar_df(
    df: &DataFrame,
    interval_seconds: i64,
) -> Result<DataFrame, DataProcessingError> {
    if interval_seconds <= 0 {
        return Err(DataProcessingError::NonPositiveInterval);
    }

    let (cleaned, _) = clean_ohlcv_df(df, true)?;
    if cleaned.height() == 0 {
        let mut out = cleaned.clone();
        out.with_column(Series::new("is_missing_bar".into(), Vec::<bool>::new()))
            .map_err(|e| DataProcessingError::frame("failed to add is_missing_bar", e))?;
        return Ok(out);
    }

    let symbols = cleaned
        .column("symbol")
        .map_err(|e| DataProcessingError::frame("symbol column error", e))?
        .str()
        .map_err(|e| DataProcessingError::frame("symbol dtype error", e))?;
    let ts = cleaned
        .column("ts_us")
        .map_err(|e| DataProcessingError::frame("ts_us column error", e))?
        .i64()
        .map_err(|e| DataProcessingError::frame("ts_us dtype error", e))?;

    let step_us = interval_seconds * 1_000_000;

    let mut cal_symbols: Vec<String> = Vec::new();
    let mut cal_ts: Vec<i64> = Vec::new();

    let mut i = 0usize;
    while i < cleaned.height() {
        let symbol =
            symbols.get(i).ok_or(DataProcessingError::NullValue { column: "symbol", row: i })?;
        let start = ts.get(i).ok_or(DataProcessingError::NullValue { column: "ts_us", row: i })?;

        let mut j = i + 1;
        while j < cleaned.height() && symbols.get(j) == Some(symbol) {
            j += 1;
        }

        let end =
            ts.get(j - 1).ok_or(DataProcessingError::NullValue { column: "ts_us", row: j - 1 })?;

        let mut cur = start;
        while cur <= end {
            cal_symbols.push(symbol.to_string());
            cal_ts.push(cur);
            cur += step_us;
        }

        i = j;
    }

    let calendar = df!("symbol" => cal_symbols, "ts_us" => cal_ts)
        .map_err(|e| DataProcessingError::frame("calendar df build failed", e))?;

    let mut out = calendar
        .left_join(&cleaned, ["symbol", "ts_us"], ["symbol", "ts_us"])
        .map_err(|e| DataProcessingError::frame("calendar join failed", e))?;

    let mut missing = out
        .column("open")
        .map_err(|e| DataProcessingError::frame("column lookup", e))?
        .is_null()
        .into_series();
    missing.rename("is_missing_bar".into());
    out.with_column(missing)
        .map_err(|e| DataProcessingError::frame("failed to add is_missing_bar", e))?;

    Ok(out)
}

fn validate_lengths(columns: &OhlcvColumns) -> Result<(), DataProcessingError> {
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
        return Err(DataProcessingError::LengthMismatch(format!(
            "ohlcv vector length mismatch: ts={n}, symbol={}, open={}, high={}, low={}, close={}, volume={}, adj_close={}",
            columns.symbols.len(),
            columns.open.len(),
            columns.high.len(),
            columns.low.len(),
            columns.close.len(),
            columns.volume.len(),
            columns.adj_close.len()
        )));
    }
    Ok(())
}

fn to_polars_df(columns: &OhlcvColumns) -> Result<DataFrame, DataProcessingError> {
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
    .map_err(|e| DataProcessingError::frame("polars df build failed", e))
}

fn df_to_ohlcv_columns(df: &DataFrame) -> Result<OhlcvColumns, DataProcessingError> {
    let timestamps_us = df
        .column("ts_us")
        .map_err(|e| DataProcessingError::frame("missing ts_us", e))?
        .i64()
        .map_err(|e| DataProcessingError::frame("ts_us type error", e))?
        .into_no_null_iter()
        .collect::<Vec<_>>();
    let symbols = df
        .column("symbol")
        .map_err(|e| DataProcessingError::frame("missing symbol", e))?
        .str()
        .map_err(|e| DataProcessingError::frame("symbol type error", e))?
        .into_no_null_iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let open = df
        .column("open")
        .map_err(|e| DataProcessingError::frame("missing open", e))?
        .f64()
        .map_err(|e| DataProcessingError::frame("open type error", e))?
        .into_no_null_iter()
        .collect::<Vec<_>>();
    let high = df
        .column("high")
        .map_err(|e| DataProcessingError::frame("missing high", e))?
        .f64()
        .map_err(|e| DataProcessingError::frame("high type error", e))?
        .into_no_null_iter()
        .collect::<Vec<_>>();
    let low = df
        .column("low")
        .map_err(|e| DataProcessingError::frame("missing low", e))?
        .f64()
        .map_err(|e| DataProcessingError::frame("low type error", e))?
        .into_no_null_iter()
        .collect::<Vec<_>>();
    let close = df
        .column("close")
        .map_err(|e| DataProcessingError::frame("missing close", e))?
        .f64()
        .map_err(|e| DataProcessingError::frame("close type error", e))?
        .into_no_null_iter()
        .collect::<Vec<_>>();
    let volume = df
        .column("volume")
        .map_err(|e| DataProcessingError::frame("missing volume", e))?
        .f64()
        .map_err(|e| DataProcessingError::frame("volume type error", e))?
        .into_no_null_iter()
        .collect::<Vec<_>>();
    let adj_close = df
        .column("adj_close")
        .map_err(|e| DataProcessingError::frame("missing adj_close", e))?
        .f64()
        .map_err(|e| DataProcessingError::frame("adj_close type error", e))?
        .into_no_null_iter()
        .collect::<Vec<_>>();

    Ok(OhlcvColumns { timestamps_us, symbols, open, high, low, close, volume, adj_close })
}

/// Data-quality report for [`OhlcvColumns`]; see [`quality_report_df`].
///
/// # Errors
///
/// - [`DataProcessingError::LengthMismatch`] if the column vectors differ in length.
/// - [`DataProcessingError::Frame`] if a Polars operation fails.
pub fn quality_report_columns(
    columns: &OhlcvColumns,
    rows_removed_by_deduplication: usize,
) -> Result<DataQualityReport, DataProcessingError> {
    let df = to_polars_df(columns)?;
    quality_report_df(&df, rows_removed_by_deduplication)
}

/// Sorts and deduplicates [`OhlcvColumns`]; see [`clean_ohlcv_df`].
///
/// # Errors
///
/// - [`DataProcessingError::LengthMismatch`] if the column vectors differ in length.
/// - [`DataProcessingError::Frame`] if a Polars operation fails.
pub fn clean_ohlcv_columns(
    columns: &OhlcvColumns,
    keep_last: bool,
) -> Result<(OhlcvColumns, DataQualityReport), DataProcessingError> {
    let df = to_polars_df(columns)?;
    let (clean_df, report) = clean_ohlcv_df(&df, keep_last)?;
    let clean_cols = df_to_ohlcv_columns(&clean_df)?;
    Ok((clean_cols, report))
}

/// Aligns [`OhlcvColumns`] to a regular grid; see [`align_calendar_df`].
///
/// # Errors
///
/// - [`DataProcessingError::NonPositiveInterval`] if `interval_seconds <= 0`.
/// - [`DataProcessingError::LengthMismatch`] if the column vectors differ in length.
/// - [`DataProcessingError::Frame`] if a Polars operation fails.
pub fn align_calendar_columns(
    columns: &OhlcvColumns,
    interval_seconds: i64,
) -> Result<AlignedOhlcvColumns, DataProcessingError> {
    let df = to_polars_df(columns)?;
    let out = align_calendar_df(&df, interval_seconds)?;

    let timestamps_us = out
        .column("ts_us")
        .map_err(|e| DataProcessingError::frame("missing ts_us", e))?
        .i64()
        .map_err(|e| DataProcessingError::frame("ts_us type error", e))?
        .into_no_null_iter()
        .collect::<Vec<_>>();
    let symbols = out
        .column("symbol")
        .map_err(|e| DataProcessingError::frame("missing symbol", e))?
        .str()
        .map_err(|e| DataProcessingError::frame("symbol type error", e))?
        .into_no_null_iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();

    let open = out
        .column("open")
        .map_err(|e| DataProcessingError::frame("missing open", e))?
        .f64()
        .map_err(|e| DataProcessingError::frame("open type error", e))?
        .into_iter()
        .collect::<Vec<_>>();
    let high = out
        .column("high")
        .map_err(|e| DataProcessingError::frame("missing high", e))?
        .f64()
        .map_err(|e| DataProcessingError::frame("high type error", e))?
        .into_iter()
        .collect::<Vec<_>>();
    let low = out
        .column("low")
        .map_err(|e| DataProcessingError::frame("missing low", e))?
        .f64()
        .map_err(|e| DataProcessingError::frame("low type error", e))?
        .into_iter()
        .collect::<Vec<_>>();
    let close = out
        .column("close")
        .map_err(|e| DataProcessingError::frame("missing close", e))?
        .f64()
        .map_err(|e| DataProcessingError::frame("close type error", e))?
        .into_iter()
        .collect::<Vec<_>>();
    let volume = out
        .column("volume")
        .map_err(|e| DataProcessingError::frame("missing volume", e))?
        .f64()
        .map_err(|e| DataProcessingError::frame("volume type error", e))?
        .into_iter()
        .collect::<Vec<_>>();
    let adj_close = out
        .column("adj_close")
        .map_err(|e| DataProcessingError::frame("missing adj_close", e))?
        .f64()
        .map_err(|e| DataProcessingError::frame("adj_close type error", e))?
        .into_iter()
        .collect::<Vec<_>>();
    let is_missing_bar = out
        .column("is_missing_bar")
        .map_err(|e| DataProcessingError::frame("missing is_missing_bar", e))?
        .bool()
        .map_err(|e| DataProcessingError::frame("is_missing_bar type error", e))?
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

/// Sorts and deduplicates [`OhlcvRow`]s; see [`clean_ohlcv_df`].
///
/// Rows always form equal-length columns, so this does not return a `Result`.
///
/// # Panics
///
/// Panics if the underlying Polars operations fail, which well-formed rows do not trigger.
pub fn clean_ohlcv_rows(rows: &[OhlcvRow], keep_last: bool) -> (Vec<OhlcvRow>, DataQualityReport) {
    let cols = rows_to_columns(rows);
    let (clean_cols, report) = clean_ohlcv_columns(&cols, keep_last).expect("validated rows");
    (columns_to_rows(&clean_cols), report)
}

/// Data-quality report for [`OhlcvRow`]s; see [`quality_report_df`].
///
/// # Panics
///
/// Panics if the underlying Polars operations fail, which well-formed rows do not trigger.
pub fn quality_report(
    rows: &[OhlcvRow],
    rows_removed_by_deduplication: usize,
) -> DataQualityReport {
    let cols = rows_to_columns(rows);
    quality_report_columns(&cols, rows_removed_by_deduplication).expect("validated rows")
}

/// Aligns [`OhlcvRow`]s to a regular grid of `interval_seconds`; see [`align_calendar_df`].
///
/// # Errors
///
/// - [`DataProcessingError::NonPositiveInterval`] if `interval_seconds <= 0`.
/// - [`DataProcessingError::Frame`] if a Polars operation fails.
pub fn align_calendar_rows(
    rows: &[OhlcvRow],
    interval_seconds: i64,
) -> Result<Vec<AlignedOhlcvRow>, DataProcessingError> {
    let cols = rows_to_columns(rows);
    let aligned_cols = align_calendar_columns(&cols, interval_seconds)?;
    Ok(aligned_columns_to_rows(&aligned_cols))
}
