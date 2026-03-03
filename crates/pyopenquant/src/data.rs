use openquant::data_processing::{align_calendar_columns, clean_ohlcv_columns, quality_report_columns};
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_polars::PyDataFrame;

use crate::helpers::{build_ohlcv_columns, report_to_pydict, to_py_err};

#[pyfunction(name = "clean_ohlcv")]
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
) -> PyResult<(
    Vec<i64>,
    Vec<String>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    PyObject,
)> {
    let cols =
        build_ohlcv_columns(timestamps_us, symbols, open, high, low, close, volume, adj_close)?;
    let (clean, report) = clean_ohlcv_columns(&cols, dedupe_keep_last).map_err(to_py_err)?;

    let out_report = PyDict::new(py);
    out_report.set_item("row_count", report.row_count)?;
    out_report.set_item("symbol_count", report.symbol_count)?;
    out_report.set_item("duplicate_key_count", report.duplicate_key_count)?;
    out_report.set_item("gap_interval_count", report.gap_interval_count)?;
    out_report
        .set_item("ts_min", report.ts_min.map(|v| v.format("%Y-%m-%d %H:%M:%S").to_string()))?;
    out_report
        .set_item("ts_max", report.ts_max.map(|v| v.format("%Y-%m-%d %H:%M:%S").to_string()))?;
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

#[pyfunction(name = "quality_report")]
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
    out_report
        .set_item("ts_min", report.ts_min.map(|v| v.format("%Y-%m-%d %H:%M:%S").to_string()))?;
    out_report
        .set_item("ts_max", report.ts_max.map(|v| v.format("%Y-%m-%d %H:%M:%S").to_string()))?;
    out_report.set_item("rows_removed_by_deduplication", 0)?;
    Ok(out_report.into_pyobject(py).unwrap().into_any().unbind())
}

#[pyfunction(name = "align_calendar")]
fn data_align_calendar(
    timestamps_us: Vec<i64>,
    symbols: Vec<String>,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
    adj_close: Vec<f64>,
    interval_seconds: i64,
) -> PyResult<(
    Vec<i64>,
    Vec<String>,
    Vec<Option<f64>>,
    Vec<Option<f64>>,
    Vec<Option<f64>>,
    Vec<Option<f64>>,
    Vec<Option<f64>>,
    Vec<Option<f64>>,
    Vec<bool>,
)> {
    let cols =
        build_ohlcv_columns(timestamps_us, symbols, open, high, low, close, volume, adj_close)?;
    let out = align_calendar_columns(&cols, interval_seconds).map_err(to_py_err)?;
    Ok((
        out.timestamps_us,
        out.symbols,
        out.open,
        out.high,
        out.low,
        out.close,
        out.volume,
        out.adj_close,
        out.is_missing_bar,
    ))
}

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

#[pyfunction(name = "quality_report_df")]
fn data_quality_report_df(py: Python<'_>, pydf: PyDataFrame) -> PyResult<PyObject> {
    let df: DataFrame = pydf.into();
    let report = openquant::data_processing::quality_report_df(&df, 0).map_err(to_py_err)?;
    report_to_pydict(py, report)
}

#[pyfunction(name = "align_calendar_df")]
fn data_align_calendar_df(pydf: PyDataFrame, interval_seconds: i64) -> PyResult<PyDataFrame> {
    let df: DataFrame = pydf.into();
    let out_df =
        openquant::data_processing::align_calendar_df(&df, interval_seconds).map_err(to_py_err)?;
    Ok(PyDataFrame(out_df))
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
