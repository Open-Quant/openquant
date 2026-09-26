use openquant::data_processing::{
    align_calendar_columns, clean_ohlcv_columns, quality_report_columns, CalendarAlignmentReport,
};
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_polars::PyDataFrame;

use crate::helpers::{build_ohlcv_columns, format_naive_datetime, report_to_pydict, to_py_err};

/// `{"rows_removed_by_deduplication", "off_grid_bar_count", "off_grid_bars"}`, the last a
/// list of `(symbol, ts_us)`.
fn alignment_report_to_pydict(
    py: Python<'_>,
    report: CalendarAlignmentReport,
) -> PyResult<PyObject> {
    let off_grid: Vec<(String, i64)> = report
        .off_grid_bars
        .into_iter()
        .map(|(symbol, ts)| (symbol, ts.and_utc().timestamp_micros()))
        .collect();
    let d = PyDict::new(py);
    d.set_item("rows_removed_by_deduplication", report.rows_removed_by_deduplication)?;
    d.set_item("off_grid_bar_count", off_grid.len())?;
    d.set_item("off_grid_bars", off_grid)?;
    Ok(d.into_pyobject(py).unwrap().into_any().unbind())
}

/// `(timestamps_us, symbols, open, high, low, close, volume, adj_close, quality_report)`.
type CleanOhlcvColumns =
    (Vec<i64>, Vec<String>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, PyObject);

/// `(timestamps_us, symbols, open, high, low, close, volume, adj_close, is_missing_bar)`.
type AlignedOhlcvColumns = (
    Vec<i64>,
    Vec<String>,
    Vec<Option<f64>>,
    Vec<Option<f64>>,
    Vec<Option<f64>>,
    Vec<Option<f64>>,
    Vec<Option<f64>>,
    Vec<Option<f64>>,
    Vec<bool>,
);

/// Sort OHLCV columns by `(symbol, timestamp)` and drop duplicate keys.
///
/// A duplicate is a repeated `(symbol, timestamp)` key; one row per key is kept. Output is
/// sorted by symbol, then timestamp. Prices are neither validated nor filled. This is data
/// preparation ahead of bars, labels or features (no AFML chapter). A gap in the report is
/// two consecutive bars of one symbol more than one day apart, whatever the bar interval,
/// so weekends count as gaps on daily data.
///
/// Parameters
/// ----------
/// timestamps_us : list[int]
///     Bar timestamps in microseconds since the Unix epoch (UTC).
/// symbols : list[str]
///     Instrument identifier per row.
/// open : list[float]
///     Opening prices.
/// high : list[float]
///     High prices.
/// low : list[float]
///     Low prices.
/// close : list[float]
///     Closing prices.
/// volume : list[float]
///     Traded volumes.
/// adj_close : list[float]
///     Closes adjusted for splits and dividends.
/// dedupe_keep_last : bool
///     Keep the last occurrence of a duplicated key if True, the first if False.
///
/// Returns
/// -------
/// tuple[list[int], list[str], list[float], list[float], list[float], list[float],
///       list[float], list[float], dict[str, Any]]
///     `(timestamps_us, symbols, open, high, low, close, volume, adj_close, report)` for
///     the cleaned rows. `report` has keys `row_count`, `symbol_count`,
///     `duplicate_key_count` (always 0 after cleaning), `gap_interval_count`, `ts_min` and
///     `ts_max` (UTC `"%Y-%m-%d %H:%M:%S"` strings, None when empty) and
///     `rows_removed_by_deduplication`.
///
/// Raises
/// ------
/// ValueError
///     If the column lists differ in length, or a Polars operation fails.
#[pyfunction(name = "clean_ohlcv")]
// Python keyword signature.
#[allow(clippy::too_many_arguments)]
fn data_clean_ohlcv(
    py: Python<'_>,
    timestamps_us: Vec<i64>,
    symbols: Vec<String>,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
    adj_close: Vec<f64>,
    dedupe_keep_last: bool,
) -> PyResult<CleanOhlcvColumns> {
    let cols =
        build_ohlcv_columns(timestamps_us, symbols, open, high, low, close, volume, adj_close)?;
    let (clean, report) = clean_ohlcv_columns(&cols, dedupe_keep_last).map_err(to_py_err)?;

    let out_report = PyDict::new(py);
    out_report.set_item("row_count", report.row_count)?;
    out_report.set_item("symbol_count", report.symbol_count)?;
    out_report.set_item("duplicate_key_count", report.duplicate_key_count)?;
    out_report.set_item("gap_interval_count", report.gap_interval_count)?;
    out_report.set_item("inferred_interval_us", report.inferred_interval_us)?;
    out_report.set_item("ts_min", report.ts_min.map(|v| format_naive_datetime(&v)))?;
    out_report.set_item("ts_max", report.ts_max.map(|v| format_naive_datetime(&v)))?;
    out_report.set_item("rows_removed_by_deduplication", report.rows_removed_by_deduplication)?;
    Ok((
        clean.timestamps_us,
        clean.symbols,
        clean.open,
        clean.high,
        clean.low,
        clean.close,
        clean.volume,
        clean.adj_close,
        out_report.into_pyobject(py).unwrap().into_any().unbind(),
    ))
}

/// Data-quality report for OHLCV columns, without modifying them.
///
/// Rows are sorted by `(symbol, timestamp)` before counting. A duplicate is a repeated
/// `(symbol, timestamp)` key. A gap is two consecutive bars of one symbol more than one
/// day apart, whatever the bar interval, so weekends count as gaps on daily data.
///
/// Parameters
/// ----------
/// timestamps_us : list[int]
///     Bar timestamps in microseconds since the Unix epoch (UTC).
/// symbols : list[str]
///     Instrument identifier per row.
/// open : list[float]
///     Opening prices.
/// high : list[float]
///     High prices.
/// low : list[float]
///     Low prices.
/// close : list[float]
///     Closing prices.
/// volume : list[float]
///     Traded volumes.
/// adj_close : list[float]
///     Closes adjusted for splits and dividends.
///
/// Returns
/// -------
/// dict[str, Any]
///     Keys `row_count`, `symbol_count`, `duplicate_key_count`, `gap_interval_count`,
///     `ts_min` and `ts_max` (UTC `"%Y-%m-%d %H:%M:%S"` strings, None when empty) and
///     `rows_removed_by_deduplication` (always 0 here).
///
/// Raises
/// ------
/// ValueError
///     If the column lists differ in length, or a Polars operation fails.
#[pyfunction(name = "quality_report")]
// Python keyword signature.
#[allow(clippy::too_many_arguments)]
fn data_quality_report(
    py: Python<'_>,
    timestamps_us: Vec<i64>,
    symbols: Vec<String>,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
    adj_close: Vec<f64>,
) -> PyResult<PyObject> {
    let cols =
        build_ohlcv_columns(timestamps_us, symbols, open, high, low, close, volume, adj_close)?;
    let report = quality_report_columns(&cols, 0).map_err(to_py_err)?;
    let out_report = PyDict::new(py);
    out_report.set_item("row_count", report.row_count)?;
    out_report.set_item("symbol_count", report.symbol_count)?;
    out_report.set_item("duplicate_key_count", report.duplicate_key_count)?;
    out_report.set_item("gap_interval_count", report.gap_interval_count)?;
    out_report.set_item("inferred_interval_us", report.inferred_interval_us)?;
    out_report.set_item("ts_min", report.ts_min.map(|v| format_naive_datetime(&v)))?;
    out_report.set_item("ts_max", report.ts_max.map(|v| format_naive_datetime(&v)))?;
    out_report.set_item("rows_removed_by_deduplication", 0)?;
    Ok(out_report.into_pyobject(py).unwrap().into_any().unbind())
}

/// Clean OHLCV columns and reindex each symbol onto a regular time grid.
///
/// Rows are first deduplicated (keeping the last duplicate), then each symbol is reindexed
/// from its first to its last timestamp in steps of `interval_seconds`. Grid points without
/// a bar get None prices and `is_missing_bar = True`. The grid starts at each symbol's
/// first timestamp, so bars whose timestamp is not on that grid are dropped; the interval
/// should divide the data's spacing. A short interval over a long span produces many rows.
///
/// Parameters
/// ----------
/// timestamps_us : list[int]
///     Bar timestamps in microseconds since the Unix epoch (UTC).
/// symbols : list[str]
///     Instrument identifier per row.
/// open : list[float]
///     Opening prices.
/// high : list[float]
///     High prices.
/// low : list[float]
///     Low prices.
/// close : list[float]
///     Closing prices.
/// volume : list[float]
///     Traded volumes.
/// adj_close : list[float]
///     Closes adjusted for splits and dividends.
/// interval_seconds : int
///     Grid spacing in seconds; must be positive.
///
/// Returns
/// -------
/// tuple[list[int], list[str], list[float | None], list[float | None], list[float | None],
///       list[float | None], list[float | None], list[float | None], list[bool]]
///     `(timestamps_us, symbols, open, high, low, close, volume, adj_close, is_missing_bar)`
///     on the grid, sorted by symbol then timestamp. Price fields are None where the grid
///     point had no bar.
///
/// Raises
/// ------
/// ValueError
///     If the column lists differ in length, `interval_seconds <= 0`, or a Polars
///     operation fails.
#[pyfunction(name = "align_calendar")]
#[pyo3(signature = (
    timestamps_us,
    symbols,
    open,
    high,
    low,
    close,
    volume,
    adj_close,
    interval_seconds,
    return_report=false
))]
// Python keyword signature.
#[allow(clippy::too_many_arguments)]
fn data_align_calendar(
    py: Python<'_>,
    timestamps_us: Vec<i64>,
    symbols: Vec<String>,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
    adj_close: Vec<f64>,
    interval_seconds: i64,
    return_report: bool,
) -> PyResult<PyObject> {
    let cols =
        build_ohlcv_columns(timestamps_us, symbols, open, high, low, close, volume, adj_close)?;
    let (out, report) = align_calendar_columns(&cols, interval_seconds).map_err(to_py_err)?;
    let aligned: AlignedOhlcvColumns = (
        out.timestamps_us,
        out.symbols,
        out.open,
        out.high,
        out.low,
        out.close,
        out.volume,
        out.adj_close,
        out.is_missing_bar,
    );
    if return_report {
        let report = alignment_report_to_pydict(py, report)?;
        Ok((aligned, report).into_pyobject(py)?.into_any().unbind())
    } else {
        Ok(aligned.into_pyobject(py)?.into_any().unbind())
    }
}

/// Sort an OHLCV DataFrame by `(symbol, ts_us)` and drop duplicate keys.
///
/// DataFrame form of `clean_ohlcv`. The frame needs columns `symbol` (str), `ts_us`
/// (int64 microseconds since the Unix epoch, UTC) and `open`, `high`, `low`, `close`,
/// `volume`, `adj_close` (float64). A duplicate is a repeated `(symbol, ts_us)` key. Prices
/// are neither validated nor filled. A gap in the report is two consecutive bars of one
/// symbol more than one day apart.
///
/// Parameters
/// ----------
/// pydf : polars.DataFrame
///     OHLCV frame with the columns listed above.
/// dedupe_keep_last : bool
///     Keep the last occurrence of a duplicated key if True, the first if False.
///
/// Returns
/// -------
/// tuple[polars.DataFrame, dict[str, Any]]
///     `(cleaned_frame, report)`. `report` has keys `row_count`, `symbol_count`,
///     `duplicate_key_count` (always 0 after cleaning), `gap_interval_count`, `ts_min` and
///     `ts_max` (UTC `"%Y-%m-%d %H:%M:%S"` strings, None when empty) and
///     `rows_removed_by_deduplication`.
///
/// Raises
/// ------
/// ValueError
///     If a required column is missing or has the wrong dtype, a `symbol` or `ts_us` value
///     is null, or a Polars operation fails.
#[pyfunction(name = "clean_ohlcv_df")]
fn data_clean_ohlcv_df(
    py: Python<'_>,
    pydf: PyDataFrame,
    dedupe_keep_last: bool,
) -> PyResult<(PyDataFrame, PyObject)> {
    let df: DataFrame = pydf.into();
    let (out_df, report) =
        openquant::data_processing::clean_ohlcv_df(&df, dedupe_keep_last).map_err(to_py_err)?;
    let out_report = report_to_pydict(py, report)?;
    Ok((PyDataFrame(out_df), out_report))
}

/// Data-quality report for an OHLCV DataFrame, without modifying it.
///
/// DataFrame form of `quality_report`. The frame needs columns `symbol` (str), `ts_us`
/// (int64 microseconds since the Unix epoch, UTC) and `open`, `high`, `low`, `close`,
/// `volume`, `adj_close` (float64). A gap is two consecutive bars of one symbol more than
/// one day apart, whatever the bar interval.
///
/// Parameters
/// ----------
/// pydf : polars.DataFrame
///     OHLCV frame with the columns listed above.
///
/// Returns
/// -------
/// dict[str, Any]
///     Keys `row_count`, `symbol_count`, `duplicate_key_count`, `gap_interval_count`,
///     `ts_min` and `ts_max` (UTC `"%Y-%m-%d %H:%M:%S"` strings, None when empty) and
///     `rows_removed_by_deduplication` (always 0 here).
///
/// Raises
/// ------
/// ValueError
///     If a required column is missing or has the wrong dtype, a `symbol` or `ts_us` value
///     is null, or a Polars operation fails.
#[pyfunction(name = "quality_report_df")]
fn data_quality_report_df(py: Python<'_>, pydf: PyDataFrame) -> PyResult<PyObject> {
    let df: DataFrame = pydf.into();
    let report = openquant::data_processing::quality_report_df(&df, 0).map_err(to_py_err)?;
    report_to_pydict(py, report)
}

/// Clean an OHLCV DataFrame and reindex each symbol onto a regular time grid.
///
/// DataFrame form of `align_calendar`. The frame needs columns `symbol` (str), `ts_us`
/// (int64 microseconds since the Unix epoch, UTC) and `open`, `high`, `low`, `close`,
/// `volume`, `adj_close` (float64). Duplicates are dropped (keeping the last), then each
/// symbol is reindexed from its first to its last `ts_us` in steps of `interval_seconds`.
/// Grid points without a bar get null prices. Bars not on the grid (which starts at each
/// symbol's first timestamp) are dropped, so the interval should divide the data's spacing.
///
/// Parameters
/// ----------
/// pydf : polars.DataFrame
///     OHLCV frame with the columns listed above.
/// interval_seconds : int
///     Grid spacing in seconds; must be positive.
///
/// Returns
/// -------
/// polars.DataFrame
///     The aligned frame with the input columns plus a boolean `is_missing_bar` column,
///     sorted by `symbol` then `ts_us`.
///
/// Raises
/// ------
/// ValueError
///     If `interval_seconds <= 0`, a required column is missing or has the wrong dtype, a
///     `symbol` or `ts_us` value is null, or a Polars operation fails.
#[pyfunction(name = "align_calendar_df")]
#[pyo3(signature = (pydf, interval_seconds, return_report=false))]
fn data_align_calendar_df(
    py: Python<'_>,
    pydf: PyDataFrame,
    interval_seconds: i64,
    return_report: bool,
) -> PyResult<PyObject> {
    let df: DataFrame = pydf.into();
    let (out_df, report) =
        openquant::data_processing::align_calendar_df(&df, interval_seconds).map_err(to_py_err)?;
    let frame = PyDataFrame(out_df);
    if return_report {
        let report = alignment_report_to_pydict(py, report)?;
        Ok((frame, report).into_pyobject(py)?.into_any().unbind())
    } else {
        Ok(frame.into_pyobject(py)?.into_any().unbind())
    }
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "data")?;
    m.add_function(wrap_pyfunction!(data_clean_ohlcv, &m)?)?;
    m.add_function(wrap_pyfunction!(data_quality_report, &m)?)?;
    m.add_function(wrap_pyfunction!(data_align_calendar, &m)?)?;
    m.add_function(wrap_pyfunction!(data_clean_ohlcv_df, &m)?)?;
    m.add_function(wrap_pyfunction!(data_quality_report_df, &m)?)?;
    m.add_function(wrap_pyfunction!(data_align_calendar_df, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("data", m)?;
    Ok(())
}
