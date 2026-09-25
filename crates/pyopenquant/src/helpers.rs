use nalgebra::DMatrix;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Python-facing bar row:
/// `(start_timestamp, timestamp, open, high, low, close, volume, dollar_value, tick_count)`.
pub type BarRow = (String, String, f64, f64, f64, f64, f64, f64, usize);

/// Parsed labeling inputs: `(close series, events keyed by timestamp)`.
pub type LabelingInputs =
    (Vec<(chrono::NaiveDateTime, f64)>, Vec<(chrono::NaiveDateTime, openquant::labeling::Event)>);

/// A core error, raised as `ValueError` carrying the core's own message.
pub fn to_py_err<T: core::fmt::Display>(err: T) -> PyErr {
    PyValueError::new_err(err.to_string())
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
    // `rows` is row-major (one inner Vec per row). `DMatrix::from_vec` is column-major and
    // would interleave rows and columns for any non-square input.
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Ok(DMatrix::from_row_slice(nrows, ncols, &flat))
}

/// Wire format for timestamps crossing the Python boundary, e.g. `2024-01-02 09:30:01.760917`.
///
/// On input `%.f` makes the fraction optional and accepts any number of digits up to
/// nanoseconds.
pub const DATETIME_FORMAT: &str = "%Y-%m-%d %H:%M:%S%.f";

/// Parse one wire-format timestamp, with or without a fractional second.
pub fn parse_datetime_str(value: &str) -> chrono::ParseResult<chrono::NaiveDateTime> {
    chrono::NaiveDateTime::parse_from_str(value, DATETIME_FORMAT)
}

/// Format one timestamp in the wire format, keeping any fractional second.
///
/// Written as Python's `str(datetime)` and pandas' `str(Timestamp)` write it: no fraction on a
/// whole second (so whole-second output is byte-identical to what it always was), six digits
/// for a whole microsecond, nine otherwise. A string built by `str()` on the Python side
/// therefore comes back unchanged and can be used as a join key.
pub fn format_naive_datetime(value: &chrono::NaiveDateTime) -> String {
    let fmt = match chrono::Timelike::nanosecond(value) {
        0 => "%Y-%m-%d %H:%M:%S",
        n if n % 1_000 == 0 => "%Y-%m-%d %H:%M:%S%.6f",
        _ => "%Y-%m-%d %H:%M:%S%.9f",
    };
    value.format(fmt).to_string()
}

/// Parse one timestamp, naming it in the error as `what` (e.g. "datetime").
pub fn parse_naive_datetime(value: &str, what: &str) -> PyResult<chrono::NaiveDateTime> {
    parse_datetime_str(value).map_err(|e| {
        PyValueError::new_err(format!(
            "invalid {what} '{value}' (expected '%Y-%m-%d %H:%M:%S' with an optional \
             fractional second): {e}"
        ))
    })
}

pub fn parse_naive_datetimes(values: Vec<String>) -> PyResult<Vec<chrono::NaiveDateTime>> {
    values.iter().map(|v| parse_naive_datetime(v, "datetime")).collect()
}

pub fn format_naive_datetimes(values: Vec<chrono::NaiveDateTime>) -> Vec<String> {
    values.iter().map(format_naive_datetime).collect()
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
        let start_ts = parse_naive_datetime(&start, "start barrier datetime")?;
        let end_ts = parse_naive_datetime(&end, "end barrier datetime")?;
        out.push((start_ts, end_ts));
    }
    Ok(Some(out))
}

pub fn parse_one_naive_datetime(value: &str) -> PyResult<chrono::NaiveDateTime> {
    parse_datetime_str(value)
        .or_else(|_| {
            chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d")
                .map(|d| d.and_hms_opt(0, 0, 0).expect("valid fixed midnight"))
        })
        .map_err(|e| {
            PyValueError::new_err(format!(
                "invalid datetime '{value}' (expected '%Y-%m-%d %H:%M:%S' with an optional \
                 fractional second, or '%Y-%m-%d'): {e}"
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

pub fn bars_to_rows(bars: Vec<openquant::data_structures::StandardBar>) -> Vec<BarRow> {
    bars.into_iter()
        .map(|b| {
            (
                format_naive_datetime(&b.start_timestamp),
                format_naive_datetime(&b.timestamp),
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

// Takes the raw OHLCV columns it validates into `OhlcvColumns`; a params struct would duplicate that type.
#[allow(clippy::too_many_arguments)]
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
    Ok(openquant::data_processing::OhlcvColumns {
        timestamps_us,
        symbols,
        open,
        high,
        low,
        close,
        volume,
        adj_close,
    })
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
    out_report.set_item("ts_min", report.ts_min.map(|v| format_naive_datetime(&v)))?;
    out_report.set_item("ts_max", report.ts_max.map(|v| format_naive_datetime(&v)))?;
    out_report.set_item("rows_removed_by_deduplication", report.rows_removed_by_deduplication)?;
    Ok(out_report.into_pyobject(py).unwrap().into_any().unbind())
}

/// Raw Python-side inputs shared by the triple-barrier labeling bindings.
pub struct LabelingEventArgs {
    pub close_timestamps: Vec<String>,
    pub close_prices: Vec<f64>,
    pub t_events: Vec<String>,
    pub target_timestamps: Vec<String>,
    pub target_values: Vec<f64>,
    pub pt: f64,
    pub sl: f64,
    pub min_ret: f64,
    pub vertical_barrier_times: Option<Vec<(String, String)>>,
    pub side_prediction: Option<Vec<(String, f64)>>,
}

pub fn build_labeling_events(args: LabelingEventArgs) -> PyResult<LabelingInputs> {
    let LabelingEventArgs {
        close_timestamps,
        close_prices,
        t_events,
        target_timestamps,
        target_values,
        pt,
        sl,
        min_ret,
        vertical_barrier_times,
        side_prediction,
    } = args;
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
        openquant::labeling::TripleBarrierConfig {
            pt,
            sl,
            min_ret,
            vertical_barrier_times: vbars.as_deref(),
        },
        side_storage.as_deref(),
    );
    Ok((close, events))
}

#[cfg(test)]
mod tests {
    use super::{format_naive_datetime, parse_datetime_str};

    #[test]
    fn whole_second_timestamps_round_trip_byte_for_byte() {
        for s in ["2024-01-02 09:30:01", "1999-12-31 23:59:59", "2024-02-29 00:00:00"] {
            let ts = parse_datetime_str(s).unwrap();
            assert_eq!(format_naive_datetime(&ts), s);
        }
    }

    #[test]
    fn sub_second_timestamps_keep_their_fraction() {
        for s in [
            "2024-01-02 09:30:01.500000",
            "2024-01-02 09:30:01.000001",
            "2024-01-02 09:30:01.760917",
            "2024-01-02 09:30:01.000000001",
            "2024-01-02 09:30:01.123456789",
        ] {
            let ts = parse_datetime_str(s).unwrap();
            assert_eq!(format_naive_datetime(&ts), s);
        }
        let a = parse_datetime_str("2024-01-02 09:30:01.000001").unwrap();
        let b = parse_datetime_str("2024-01-02 09:30:01.000002").unwrap();
        assert_eq!((b - a).num_microseconds(), Some(1));
    }

    #[test]
    fn fraction_is_written_as_python_str_writes_it() {
        // None on a whole second, six digits to the microsecond, nine below it.
        for (input, output) in [
            ("2024-01-02 09:30:01.5", "2024-01-02 09:30:01.500000"),
            ("2024-01-02 09:30:01.120", "2024-01-02 09:30:01.120000"),
            ("2024-01-02 09:30:01.000000", "2024-01-02 09:30:01"),
            ("2024-01-02 09:30:01.1234567", "2024-01-02 09:30:01.123456700"),
        ] {
            let ts = parse_datetime_str(input).unwrap();
            assert_eq!(format_naive_datetime(&ts), output);
        }
    }

    #[test]
    fn malformed_timestamps_are_still_rejected() {
        for s in ["2024-01-02", "2024-01-02 09:30", "2024-01-02T09:30:01", "x"] {
            assert!(parse_datetime_str(s).is_err(), "{s} should not parse");
        }
    }
}
