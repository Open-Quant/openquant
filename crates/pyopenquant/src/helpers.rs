use nalgebra::DMatrix;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub fn to_py_err<T: core::fmt::Debug>(err: T) -> PyErr {
    PyValueError::new_err(format!("{err:?}"))
}

pub fn matrix_from_rows(rows: Vec<Vec<f64>>) -> PyResult<DMatrix<f64>> {
    let nrows = rows.len();
    if nrows == 0 {
        return Err(PyValueError::new_err("prices matrix must have at least one row"));
    }
    let ncols = rows[0].len();
    if ncols == 0 {
        return Err(PyValueError::new_err("prices matrix must have at least one column"));
    }
    if rows.iter().any(|r| r.len() != ncols) {
        return Err(PyValueError::new_err("prices matrix must be rectangular"));
    }
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Ok(DMatrix::from_vec(nrows, ncols, flat))
}

pub fn parse_naive_datetimes(values: Vec<String>) -> PyResult<Vec<chrono::NaiveDateTime>> {
    values
        .into_iter()
        .map(|v| {
            chrono::NaiveDateTime::parse_from_str(&v, "%Y-%m-%d %H:%M:%S").map_err(|e| {
                PyValueError::new_err(format!(
                    "invalid datetime '{v}' (expected '%Y-%m-%d %H:%M:%S'): {e}"
                ))
            })
        })
        .collect()
}

pub fn format_naive_datetimes(values: Vec<chrono::NaiveDateTime>) -> Vec<String> {
    values.into_iter().map(|v| v.format("%Y-%m-%d %H:%M:%S").to_string()).collect()
}

pub fn pair_timestamps_values(
    timestamps: Vec<String>,
    values: Vec<f64>,
    left_name: &str,
    right_name: &str,
) -> PyResult<Vec<(chrono::NaiveDateTime, f64)>> {
    let ts = parse_naive_datetimes(timestamps)?;
    if ts.len() != values.len() {
        return Err(PyValueError::new_err(format!(
            "{left_name}/{right_name} length mismatch: {} vs {}",
            ts.len(),
            values.len()
        )));
    }
    Ok(ts.into_iter().zip(values).collect())
}

pub fn parse_vertical_barriers(
    values: Option<Vec<(String, String)>>,
) -> PyResult<Option<Vec<(chrono::NaiveDateTime, chrono::NaiveDateTime)>>> {
    let Some(values) = values else {
        return Ok(None);
    };

    let mut out = Vec::with_capacity(values.len());
    for (start, end) in values {
        let start_ts =
            chrono::NaiveDateTime::parse_from_str(&start, "%Y-%m-%d %H:%M:%S").map_err(|e| {
                PyValueError::new_err(format!("invalid start barrier datetime '{start}': {e}"))
            })?;
        let end_ts =
            chrono::NaiveDateTime::parse_from_str(&end, "%Y-%m-%d %H:%M:%S").map_err(|e| {
                PyValueError::new_err(format!("invalid end barrier datetime '{end}': {e}"))
            })?;
        out.push((start_ts, end_ts));
    }
    Ok(Some(out))
}

pub fn parse_one_naive_datetime(value: &str) -> PyResult<chrono::NaiveDateTime> {
    chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S")
        .or_else(|_| {
            chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d")
                .map(|d| d.and_hms_opt(0, 0, 0).expect("valid fixed midnight"))
        })
        .map_err(|e| {
            PyValueError::new_err(format!(
                "invalid datetime '{value}' (expected '%Y-%m-%d %H:%M:%S' or '%Y-%m-%d'): {e}"
            ))
        })
}

pub fn build_trades(
    timestamps: Vec<String>,
    prices: Vec<f64>,
    volumes: Vec<f64>,
) -> PyResult<Vec<openquant::data_structures::Trade>> {
    if timestamps.len() != prices.len() || prices.len() != volumes.len() {
        return Err(PyValueError::new_err(format!(
            "timestamps/prices/volumes length mismatch: {} / {} / {}",
            timestamps.len(),
            prices.len(),
            volumes.len()
        )));
    }
    let mut trades = Vec::with_capacity(prices.len());
    for i in 0..prices.len() {
        trades.push(openquant::data_structures::Trade {
            timestamp: parse_one_naive_datetime(&timestamps[i])?,
            price: prices[i],
            volume: volumes[i],
        });
    }
    Ok(trades)
}

pub fn bars_to_rows(
    bars: Vec<openquant::data_structures::StandardBar>,
) -> Vec<(String, String, f64, f64, f64, f64, f64, f64, usize)> {
    bars.into_iter()
        .map(|b| {
            (
                b.start_timestamp.format("%Y-%m-%d %H:%M:%S").to_string(),
                b.timestamp.format("%Y-%m-%d %H:%M:%S").to_string(),
                b.open,
                b.high,
                b.low,
                b.close,
                b.volume,
                b.dollar_value,
                b.tick_count,
            )
        })
        .collect()
}

pub fn build_ohlcv_columns(
    timestamps_us: Vec<i64>,
    symbols: Vec<String>,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
    adj_close: Vec<f64>,
) -> PyResult<openquant::data_processing::OhlcvColumns> {
    let n = timestamps_us.len();
    let lengths = [
        symbols.len(),
        open.len(),
        high.len(),
        low.len(),
        close.len(),
        volume.len(),
        adj_close.len(),
    ];
    if lengths.iter().any(|&len| len != n) {
        return Err(PyValueError::new_err(format!(
            "ohlcv vector length mismatch: ts={n}, symbol={}, open={}, high={}, low={}, close={}, volume={}, adj_close={}",
            symbols.len(),
            open.len(),
            high.len(),
            low.len(),
            close.len(),
            volume.len(),
            adj_close.len(),
        )));
    }
    Ok(openquant::data_processing::OhlcvColumns { timestamps_us, symbols, open, high, low, close, volume, adj_close })
}

pub fn report_to_pydict(
    py: Python<'_>,
    report: openquant::data_processing::DataQualityReport,
) -> PyResult<PyObject> {
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
    Ok(out_report.into_pyobject(py).unwrap().into_any().unbind())
}

pub fn build_labeling_events(
    close_timestamps: Vec<String>,
    close_prices: Vec<f64>,
    t_events: Vec<String>,
    target_timestamps: Vec<String>,
    target_values: Vec<f64>,
    pt: f64,
    sl: f64,
    min_ret: f64,
    vertical_barrier_times: Option<Vec<(String, String)>>,
    side_prediction: Option<Vec<(String, f64)>>,
) -> PyResult<(
    Vec<(chrono::NaiveDateTime, f64)>,
    Vec<(chrono::NaiveDateTime, openquant::labeling::Event)>,
)> {
    let close =
        pair_timestamps_values(close_timestamps, close_prices, "close_timestamps", "close_prices")?;
    let t_events = parse_naive_datetimes(t_events)?;
    let target = pair_timestamps_values(
        target_timestamps,
        target_values,
        "target_timestamps",
        "target_values",
    )?;
    let vbars = parse_vertical_barriers(vertical_barrier_times)?;

    let side_storage: Option<Vec<(chrono::NaiveDateTime, f64)>> =
        if let Some(side) = side_prediction {
            let (timestamps, values): (Vec<String>, Vec<f64>) = side.into_iter().unzip();
            Some(pair_timestamps_values(timestamps, values, "side timestamps", "side values")?)
        } else {
            None
        };

    let events = openquant::labeling::triple_barrier_events(
        &close,
        &t_events,
        &target,
        openquant::labeling::TripleBarrierConfig { pt, sl, min_ret, vertical_barrier_times: vbars.as_deref() },
        side_storage.as_deref(),
    );
    Ok((close, events))
}
