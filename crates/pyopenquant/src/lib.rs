use nalgebra::DMatrix;
use openquant::data_processing::{align_calendar_rows, clean_ohlcv_rows, quality_report, OhlcvRow};
use openquant::data_structures::{standard_bars, time_bars, StandardBarType, Trade};
use openquant::filters::Threshold;
use openquant::pipeline::{
    run_mid_frequency_pipeline, ResearchPipelineConfig, ResearchPipelineInput,
};
use openquant::portfolio_optimization::{
    allocate_inverse_variance, allocate_max_sharpe, allocate_min_vol,
};
use openquant::risk_metrics::RiskMetrics;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyModule;

fn to_py_err<T: core::fmt::Debug>(err: T) -> PyErr {
    PyValueError::new_err(format!("{err:?}"))
}

fn matrix_from_rows(rows: Vec<Vec<f64>>) -> PyResult<DMatrix<f64>> {
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

fn parse_naive_datetimes(values: Vec<String>) -> PyResult<Vec<chrono::NaiveDateTime>> {
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

fn format_naive_datetimes(values: Vec<chrono::NaiveDateTime>) -> Vec<String> {
    values.into_iter().map(|v| v.format("%Y-%m-%d %H:%M:%S").to_string()).collect()
}

fn parse_one_naive_datetime(value: &str) -> PyResult<chrono::NaiveDateTime> {
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

fn build_trades(
    timestamps: Vec<String>,
    prices: Vec<f64>,
    volumes: Vec<f64>,
) -> PyResult<Vec<Trade>> {
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
        trades.push(Trade {
            timestamp: parse_one_naive_datetime(&timestamps[i])?,
            price: prices[i],
            volume: volumes[i],
        });
    }
    Ok(trades)
}

fn bars_to_rows(bars: Vec<openquant::data_structures::StandardBar>) -> Vec<(String, String, f64, f64, f64, f64, f64, f64, usize)> {
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

fn build_ohlcv_rows(
    timestamps: Vec<String>,
    symbols: Vec<String>,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
    adj_close: Vec<f64>,
) -> PyResult<Vec<OhlcvRow>> {
    let n = timestamps.len();
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
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        rows.push(OhlcvRow {
            timestamp: parse_one_naive_datetime(&timestamps[i])?,
            symbol: symbols[i].clone(),
            open: open[i],
            high: high[i],
            low: low[i],
            close: close[i],
            volume: volume[i],
            adj_close: adj_close[i],
        });
    }
    Ok(rows)
}

#[pyfunction(name = "calculate_value_at_risk")]
fn risk_calculate_value_at_risk(returns: Vec<f64>, confidence_level: f64) -> PyResult<f64> {
    RiskMetrics::default().calculate_value_at_risk(&returns, confidence_level).map_err(to_py_err)
}

#[pyfunction(name = "calculate_expected_shortfall")]
fn risk_calculate_expected_shortfall(returns: Vec<f64>, confidence_level: f64) -> PyResult<f64> {
    RiskMetrics::default()
        .calculate_expected_shortfall(&returns, confidence_level)
        .map_err(to_py_err)
}

#[pyfunction(name = "calculate_conditional_drawdown_risk")]
fn risk_calculate_conditional_drawdown_risk(
    returns: Vec<f64>,
    confidence_level: f64,
) -> PyResult<f64> {
    RiskMetrics::default()
        .calculate_conditional_drawdown_risk(&returns, confidence_level)
        .map_err(to_py_err)
}

#[pyfunction(name = "cusum_filter_indices")]
fn filters_cusum_filter_indices(close: Vec<f64>, threshold: f64) -> Vec<usize> {
    openquant::filters::cusum_filter_indices(&close, Threshold::Scalar(threshold))
}

#[pyfunction(name = "cusum_filter_timestamps")]
fn filters_cusum_filter_timestamps(
    close: Vec<f64>,
    timestamps: Vec<String>,
    threshold: f64,
) -> PyResult<Vec<String>> {
    let ts = parse_naive_datetimes(timestamps)?;
    if close.len() != ts.len() {
        return Err(PyValueError::new_err(format!(
            "close/timestamps length mismatch: {} vs {}",
            close.len(),
            ts.len()
        )));
    }
    let out =
        openquant::filters::cusum_filter_timestamps(&close, &ts, Threshold::Scalar(threshold));
    Ok(format_naive_datetimes(out))
}

#[pyfunction(name = "z_score_filter_indices")]
fn filters_z_score_filter_indices(
    close: Vec<f64>,
    mean_window: usize,
    std_window: usize,
    threshold: f64,
) -> Vec<usize> {
    openquant::filters::z_score_filter_indices(&close, mean_window, std_window, threshold)
}

#[pyfunction(name = "z_score_filter_timestamps")]
fn filters_z_score_filter_timestamps(
    close: Vec<f64>,
    timestamps: Vec<String>,
    mean_window: usize,
    std_window: usize,
    threshold: f64,
) -> PyResult<Vec<String>> {
    let ts = parse_naive_datetimes(timestamps)?;
    if close.len() != ts.len() {
        return Err(PyValueError::new_err(format!(
            "close/timestamps length mismatch: {} vs {}",
            close.len(),
            ts.len()
        )));
    }
    let out = openquant::filters::z_score_filter_timestamps(
        &close,
        &ts,
        mean_window,
        std_window,
        threshold,
    );
    Ok(format_naive_datetimes(out))
}

#[pyfunction(name = "get_ind_matrix")]
fn sampling_get_ind_matrix(
    label_endtime: Vec<(usize, usize)>,
    bar_index: Vec<usize>,
) -> Vec<Vec<u8>> {
    openquant::sampling::get_ind_matrix(&label_endtime, &bar_index)
}

#[pyfunction(name = "get_ind_mat_average_uniqueness")]
fn sampling_get_ind_mat_average_uniqueness(ind_mat: Vec<Vec<u8>>) -> f64 {
    openquant::sampling::get_ind_mat_average_uniqueness(&ind_mat)
}

#[pyfunction(name = "seq_bootstrap")]
fn sampling_seq_bootstrap(
    ind_mat: Vec<Vec<u8>>,
    sample_length: Option<usize>,
    warmup_samples: Option<Vec<usize>>,
) -> Vec<usize> {
    openquant::sampling::seq_bootstrap(&ind_mat, sample_length, warmup_samples)
}

#[pyfunction(name = "build_time_bars")]
fn bars_build_time_bars(
    timestamps: Vec<String>,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    interval_seconds: i64,
) -> PyResult<Vec<(String, String, f64, f64, f64, f64, f64, f64, usize)>> {
    if interval_seconds <= 0 {
        return Err(PyValueError::new_err("interval_seconds must be > 0"));
    }
    let trades = build_trades(timestamps, prices, volumes)?;
    let bars = time_bars(&trades, chrono::Duration::seconds(interval_seconds));
    Ok(bars_to_rows(bars))
}

#[pyfunction(name = "build_tick_bars")]
fn bars_build_tick_bars(
    timestamps: Vec<String>,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    ticks_per_bar: usize,
) -> PyResult<Vec<(String, String, f64, f64, f64, f64, f64, f64, usize)>> {
    if ticks_per_bar == 0 {
        return Err(PyValueError::new_err("ticks_per_bar must be > 0"));
    }
    let trades = build_trades(timestamps, prices, volumes)?;
    let bars = standard_bars(&trades, ticks_per_bar as f64, StandardBarType::Tick);
    Ok(bars_to_rows(bars))
}

#[pyfunction(name = "build_volume_bars")]
fn bars_build_volume_bars(
    timestamps: Vec<String>,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    volume_per_bar: f64,
) -> PyResult<Vec<(String, String, f64, f64, f64, f64, f64, f64, usize)>> {
    if !volume_per_bar.is_finite() || volume_per_bar <= 0.0 {
        return Err(PyValueError::new_err("volume_per_bar must be > 0"));
    }
    let trades = build_trades(timestamps, prices, volumes)?;
    let bars = standard_bars(&trades, volume_per_bar, StandardBarType::Volume);
    Ok(bars_to_rows(bars))
}

#[pyfunction(name = "build_dollar_bars")]
fn bars_build_dollar_bars(
    timestamps: Vec<String>,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    dollar_value_per_bar: f64,
) -> PyResult<Vec<(String, String, f64, f64, f64, f64, f64, f64, usize)>> {
    if !dollar_value_per_bar.is_finite() || dollar_value_per_bar <= 0.0 {
        return Err(PyValueError::new_err("dollar_value_per_bar must be > 0"));
    }
    let trades = build_trades(timestamps, prices, volumes)?;
    let bars = standard_bars(&trades, dollar_value_per_bar, StandardBarType::Dollar);
    Ok(bars_to_rows(bars))
}

#[pyfunction(name = "clean_ohlcv")]
fn data_clean_ohlcv(
    py: Python<'_>,
    timestamps: Vec<String>,
    symbols: Vec<String>,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
    adj_close: Vec<f64>,
    dedupe_keep_last: bool,
) -> PyResult<(Vec<(String, String, f64, f64, f64, f64, f64, f64)>, PyObject)> {
    let rows = build_ohlcv_rows(timestamps, symbols, open, high, low, close, volume, adj_close)?;
    let (clean, report) = clean_ohlcv_rows(&rows, dedupe_keep_last);
    let out_rows = clean
        .into_iter()
        .map(|r| {
            (
                r.timestamp.format("%Y-%m-%d %H:%M:%S").to_string(),
                r.symbol,
                r.open,
                r.high,
                r.low,
                r.close,
                r.volume,
                r.adj_close,
            )
        })
        .collect::<Vec<_>>();

    let out_report = PyDict::new_bound(py);
    out_report.set_item("row_count", report.row_count)?;
    out_report.set_item("symbol_count", report.symbol_count)?;
    out_report.set_item("duplicate_key_count", report.duplicate_key_count)?;
    out_report.set_item("gap_interval_count", report.gap_interval_count)?;
    out_report.set_item(
        "ts_min",
        report.ts_min.map(|v| v.format("%Y-%m-%d %H:%M:%S").to_string()),
    )?;
    out_report.set_item(
        "ts_max",
        report.ts_max.map(|v| v.format("%Y-%m-%d %H:%M:%S").to_string()),
    )?;
    out_report.set_item("rows_removed_by_deduplication", report.rows_removed_by_deduplication)?;
    Ok((out_rows, out_report.into_py(py)))
}

#[pyfunction(name = "quality_report")]
fn data_quality_report(
    py: Python<'_>,
    timestamps: Vec<String>,
    symbols: Vec<String>,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
    adj_close: Vec<f64>,
) -> PyResult<PyObject> {
    let mut rows = build_ohlcv_rows(timestamps, symbols, open, high, low, close, volume, adj_close)?;
    rows.sort_by(|a, b| a.symbol.cmp(&b.symbol).then_with(|| a.timestamp.cmp(&b.timestamp)));
    let report = quality_report(&rows, 0);
    let out_report = PyDict::new_bound(py);
    out_report.set_item("row_count", report.row_count)?;
    out_report.set_item("symbol_count", report.symbol_count)?;
    out_report.set_item("duplicate_key_count", report.duplicate_key_count)?;
    out_report.set_item("gap_interval_count", report.gap_interval_count)?;
    out_report.set_item(
        "ts_min",
        report.ts_min.map(|v| v.format("%Y-%m-%d %H:%M:%S").to_string()),
    )?;
    out_report.set_item(
        "ts_max",
        report.ts_max.map(|v| v.format("%Y-%m-%d %H:%M:%S").to_string()),
    )?;
    out_report.set_item("rows_removed_by_deduplication", 0)?;
    Ok(out_report.into_py(py))
}

#[pyfunction(name = "align_calendar")]
fn data_align_calendar(
    timestamps: Vec<String>,
    symbols: Vec<String>,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
    adj_close: Vec<f64>,
    interval_seconds: i64,
) -> PyResult<Vec<(String, String, Option<f64>, Option<f64>, Option<f64>, Option<f64>, Option<f64>, Option<f64>, bool)>> {
    let rows = build_ohlcv_rows(timestamps, symbols, open, high, low, close, volume, adj_close)?;
    let out = align_calendar_rows(&rows, interval_seconds).map_err(to_py_err)?;
    Ok(out
        .into_iter()
        .map(|r| {
            (
                r.timestamp.format("%Y-%m-%d %H:%M:%S").to_string(),
                r.symbol,
                r.open,
                r.high,
                r.low,
                r.close,
                r.volume,
                r.adj_close,
                r.is_missing_bar,
            )
        })
        .collect())
}

#[pyfunction(name = "get_signal")]
fn bet_sizing_get_signal(prob: Vec<f64>, num_classes: usize, pred: Option<Vec<f64>>) -> Vec<f64> {
    openquant::bet_sizing::get_signal(&prob, num_classes, pred.as_deref())
}

#[pyfunction(name = "discrete_signal")]
fn bet_sizing_discrete_signal(signal0: Vec<f64>, step_size: f64) -> Vec<f64> {
    openquant::bet_sizing::discrete_signal(&signal0, step_size)
}

#[pyfunction(name = "bet_size")]
fn bet_sizing_bet_size(w_param: f64, price_div: f64, func: String) -> PyResult<f64> {
    if func != "sigmoid" && func != "power" {
        return Err(PyValueError::new_err(format!(
            "invalid func '{func}'; expected 'sigmoid' or 'power'"
        )));
    }
    std::panic::catch_unwind(|| openquant::bet_sizing::bet_size(w_param, price_div, &func))
        .map_err(|_| PyRuntimeError::new_err("bet_size panicked for the supplied arguments"))
}

#[pyfunction(name = "allocate_inverse_variance")]
fn portfolio_allocate_inverse_variance(
    prices: Vec<Vec<f64>>,
) -> PyResult<(Vec<f64>, f64, f64, f64)> {
    let m = matrix_from_rows(prices)?;
    let out = allocate_inverse_variance(&m).map_err(to_py_err)?;
    Ok((out.weights, out.portfolio_risk, out.portfolio_return, out.portfolio_sharpe))
}

#[pyfunction(name = "allocate_min_vol")]
fn portfolio_allocate_min_vol(prices: Vec<Vec<f64>>) -> PyResult<(Vec<f64>, f64, f64, f64)> {
    let m = matrix_from_rows(prices)?;
    let out = allocate_min_vol(&m, None, None).map_err(to_py_err)?;
    Ok((out.weights, out.portfolio_risk, out.portfolio_return, out.portfolio_sharpe))
}

#[pyfunction(name = "allocate_max_sharpe")]
fn portfolio_allocate_max_sharpe(
    prices: Vec<Vec<f64>>,
    risk_free_rate: Option<f64>,
) -> PyResult<(Vec<f64>, f64, f64, f64)> {
    let m = matrix_from_rows(prices)?;
    let out =
        allocate_max_sharpe(&m, risk_free_rate.unwrap_or(0.0), None, None).map_err(to_py_err)?;
    Ok((out.weights, out.portfolio_risk, out.portfolio_return, out.portfolio_sharpe))
}

#[pyfunction(name = "run_mid_frequency_pipeline")]
#[pyo3(signature = (
    timestamps,
    close,
    model_probabilities,
    asset_prices,
    model_sides=None,
    asset_names=None,
    cusum_threshold=0.001,
    num_classes=2,
    step_size=0.1,
    risk_free_rate=0.0,
    confidence_level=0.05
))]
fn pipeline_run_mid_frequency_pipeline(
    py: Python<'_>,
    timestamps: Vec<String>,
    close: Vec<f64>,
    model_probabilities: Vec<f64>,
    asset_prices: Vec<Vec<f64>>,
    model_sides: Option<Vec<f64>>,
    asset_names: Option<Vec<String>>,
    cusum_threshold: f64,
    num_classes: usize,
    step_size: f64,
    risk_free_rate: f64,
    confidence_level: f64,
) -> PyResult<PyObject> {
    let timestamps = parse_naive_datetimes(timestamps)?;
    let asset_prices = matrix_from_rows(asset_prices)?;

    let asset_names = asset_names.unwrap_or_else(|| {
        (0..asset_prices.ncols()).map(|i| format!("asset_{i}")).collect::<Vec<_>>()
    });

    let input = ResearchPipelineInput {
        timestamps: &timestamps,
        close: &close,
        model_probabilities: &model_probabilities,
        model_sides: model_sides.as_deref(),
        asset_prices: &asset_prices,
        asset_names: &asset_names,
    };
    let config = ResearchPipelineConfig {
        cusum_threshold,
        num_classes,
        step_size,
        risk_free_rate,
        confidence_level,
    };
    let out = run_mid_frequency_pipeline(input, &config).map_err(to_py_err)?;

    let root = PyDict::new_bound(py);

    let events = PyDict::new_bound(py);
    events.set_item("indices", out.events.indices)?;
    events.set_item("timestamps", format_naive_datetimes(out.events.timestamps))?;
    events.set_item("probabilities", out.events.probabilities)?;
    events.set_item("sides", out.events.sides)?;
    root.set_item("events", events)?;

    let signals = PyDict::new_bound(py);
    signals.set_item("timestamps", format_naive_datetimes(timestamps.clone()))?;
    signals.set_item("values", out.signals.timeline_signal)?;
    signals.set_item("event_signal", out.signals.event_signal)?;
    root.set_item("signals", signals)?;

    let portfolio = PyDict::new_bound(py);
    portfolio.set_item("asset_names", out.portfolio.asset_names)?;
    portfolio.set_item("weights", out.portfolio.weights)?;
    portfolio.set_item("portfolio_risk", out.portfolio.portfolio_risk)?;
    portfolio.set_item("portfolio_return", out.portfolio.portfolio_return)?;
    portfolio.set_item("portfolio_sharpe", out.portfolio.portfolio_sharpe)?;
    root.set_item("portfolio", portfolio)?;

    let risk = PyDict::new_bound(py);
    risk.set_item("value_at_risk", out.risk.value_at_risk)?;
    risk.set_item("expected_shortfall", out.risk.expected_shortfall)?;
    risk.set_item("conditional_drawdown_risk", out.risk.conditional_drawdown_risk)?;
    risk.set_item("realized_sharpe", out.risk.realized_sharpe)?;
    root.set_item("risk", risk)?;

    let backtest = PyDict::new_bound(py);
    backtest.set_item("timestamps", format_naive_datetimes(out.backtest.timestamps))?;
    backtest.set_item("strategy_returns", out.backtest.strategy_returns)?;
    backtest.set_item("equity_curve", out.backtest.equity_curve)?;
    backtest.set_item("drawdowns", out.backtest.drawdowns)?;
    backtest.set_item("time_under_water_years", out.backtest.time_under_water_years)?;
    root.set_item("backtest", backtest)?;

    let leakage_checks = PyDict::new_bound(py);
    leakage_checks.set_item("inputs_aligned", out.leakage_checks.inputs_aligned)?;
    leakage_checks.set_item("event_indices_sorted", out.leakage_checks.event_indices_sorted)?;
    leakage_checks.set_item("has_forward_look_bias", out.leakage_checks.has_forward_look_bias)?;
    root.set_item("leakage_checks", leakage_checks)?;

    Ok(root.into_py(py))
}

#[pymodule]
fn _core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let risk = PyModule::new_bound(py, "risk")?;
    risk.add_function(wrap_pyfunction!(risk_calculate_value_at_risk, &risk)?)?;
    risk.add_function(wrap_pyfunction!(risk_calculate_expected_shortfall, &risk)?)?;
    risk.add_function(wrap_pyfunction!(risk_calculate_conditional_drawdown_risk, &risk)?)?;
    m.add_submodule(&risk)?;
    m.add("risk", risk)?;

    let filters = PyModule::new_bound(py, "filters")?;
    filters.add_function(wrap_pyfunction!(filters_cusum_filter_indices, &filters)?)?;
    filters.add_function(wrap_pyfunction!(filters_cusum_filter_timestamps, &filters)?)?;
    filters.add_function(wrap_pyfunction!(filters_z_score_filter_indices, &filters)?)?;
    filters.add_function(wrap_pyfunction!(filters_z_score_filter_timestamps, &filters)?)?;
    m.add_submodule(&filters)?;
    m.add("filters", filters)?;

    let sampling = PyModule::new_bound(py, "sampling")?;
    sampling.add_function(wrap_pyfunction!(sampling_get_ind_matrix, &sampling)?)?;
    sampling.add_function(wrap_pyfunction!(sampling_get_ind_mat_average_uniqueness, &sampling)?)?;
    sampling.add_function(wrap_pyfunction!(sampling_seq_bootstrap, &sampling)?)?;
    m.add_submodule(&sampling)?;
    m.add("sampling", sampling)?;

    let bars = PyModule::new_bound(py, "bars")?;
    bars.add_function(wrap_pyfunction!(bars_build_time_bars, &bars)?)?;
    bars.add_function(wrap_pyfunction!(bars_build_tick_bars, &bars)?)?;
    bars.add_function(wrap_pyfunction!(bars_build_volume_bars, &bars)?)?;
    bars.add_function(wrap_pyfunction!(bars_build_dollar_bars, &bars)?)?;
    m.add_submodule(&bars)?;
    m.add("bars", bars)?;

    let data = PyModule::new_bound(py, "data")?;
    data.add_function(wrap_pyfunction!(data_clean_ohlcv, &data)?)?;
    data.add_function(wrap_pyfunction!(data_quality_report, &data)?)?;
    data.add_function(wrap_pyfunction!(data_align_calendar, &data)?)?;
    m.add_submodule(&data)?;
    m.add("data", data)?;

    let bet_sizing = PyModule::new_bound(py, "bet_sizing")?;
    bet_sizing.add_function(wrap_pyfunction!(bet_sizing_get_signal, &bet_sizing)?)?;
    bet_sizing.add_function(wrap_pyfunction!(bet_sizing_discrete_signal, &bet_sizing)?)?;
    bet_sizing.add_function(wrap_pyfunction!(bet_sizing_bet_size, &bet_sizing)?)?;
    m.add_submodule(&bet_sizing)?;
    m.add("bet_sizing", bet_sizing)?;

    let portfolio = PyModule::new_bound(py, "portfolio")?;
    portfolio.add_function(wrap_pyfunction!(portfolio_allocate_inverse_variance, &portfolio)?)?;
    portfolio.add_function(wrap_pyfunction!(portfolio_allocate_min_vol, &portfolio)?)?;
    portfolio.add_function(wrap_pyfunction!(portfolio_allocate_max_sharpe, &portfolio)?)?;
    m.add_submodule(&portfolio)?;
    m.add("portfolio", portfolio)?;

    let pipeline = PyModule::new_bound(py, "pipeline")?;
    pipeline.add_function(wrap_pyfunction!(pipeline_run_mid_frequency_pipeline, &pipeline)?)?;
    m.add_submodule(&pipeline)?;
    m.add("pipeline", pipeline)?;

    Ok(())
}
