//! `openquant._core.backtesting_engine`: CPCV backtest paths from out-of-sample returns the
//! caller has already computed.
//!
//! `run_cpcv` in Rust calls an evaluator per split. Here that evaluator is a Rust closure that
//! looks up the caller's precomputed returns for the split, so no Python callable crosses the
//! boundary: the caller fits and predicts on `openquant.cross_validation.cpcv_splits`, then
//! hands the out-of-sample returns in.

use std::collections::HashMap;

use openquant::backtesting_engine::{
    cpcv_path_count, run_cpcv, BacktestData, BacktestError, BacktestRunConfig, BacktestSafeguards,
    CpcvConfig, FoldPerformance,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::cross_validation::label_spans;
use crate::helpers::to_py_err;

const SAFEGUARD_KEYS: [&str; 5] = [
    "survivorship_bias_control",
    "look_ahead_control",
    "data_mining_control",
    "cost_assumption",
    "multiple_testing_control",
];

fn safeguards_from(mut values: HashMap<String, String>) -> PyResult<BacktestSafeguards> {
    let mut take = |key: &str| {
        values.remove(key).ok_or_else(|| {
            PyValueError::new_err(format!(
                "safeguards is missing '{key}' (required: {})",
                SAFEGUARD_KEYS.join(", ")
            ))
        })
    };
    let out = BacktestSafeguards {
        survivorship_bias_control: take("survivorship_bias_control")?,
        look_ahead_control: take("look_ahead_control")?,
        data_mining_control: take("data_mining_control")?,
        cost_assumption: take("cost_assumption")?,
        multiple_testing_control: take("multiple_testing_control")?,
    };
    if let Some(extra) = values.keys().min() {
        return Err(PyValueError::new_err(format!(
            "safeguards has unknown key '{extra}' (allowed: {})",
            SAFEGUARD_KEYS.join(", ")
        )));
    }
    Ok(out)
}

fn performance_dict<'py>(py: Python<'py>, f: &FoldPerformance) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("split_id", f.split_id)?;
    d.set_item("sharpe", f.sharpe)?;
    d.set_item("mean_return", f.mean_return)?;
    d.set_item("std_return", f.std_return)?;
    d.set_item("observations", f.observations)?;
    Ok(d)
}

#[pyfunction(name = "cpcv_path_count")]
fn bt_cpcv_path_count(n_groups: usize, test_groups: usize) -> PyResult<usize> {
    cpcv_path_count(n_groups, test_groups).map_err(to_py_err)
}

// Mirrors the Rust run configuration field by field; a params struct would only restate it.
#[allow(clippy::too_many_arguments)]
#[pyfunction(name = "run_cpcv")]
#[pyo3(signature = (
    t0,
    t1,
    split_returns,
    *,
    n_groups,
    test_groups,
    pct_embargo,
    mode_provenance,
    trials_count,
    safeguards
))]
fn bt_run_cpcv<'py>(
    py: Python<'py>,
    t0: Vec<i64>,
    t1: Vec<i64>,
    split_returns: Vec<Vec<f64>>,
    n_groups: usize,
    test_groups: usize,
    pct_embargo: f64,
    mode_provenance: String,
    trials_count: usize,
    safeguards: HashMap<String, String>,
) -> PyResult<Bound<'py, PyDict>> {
    let spans = label_spans(t0, t1)?;
    // The engine reads the sample count from `returns`; per-sample returns are not used, the
    // split returns below are.
    let data = BacktestData { returns: vec![0.0; spans.len()], label_spans: spans };
    let run = BacktestRunConfig {
        mode_provenance,
        trials_count,
        safeguards: safeguards_from(safeguards)?,
    };
    let config = CpcvConfig { n_groups, test_groups, pct_embargo };

    let n_supplied = split_returns.len();
    let result = run_cpcv(&data, &run, &config, |split| {
        let returns = split_returns.get(split.split_id).ok_or_else(|| {
            BacktestError::Evaluator(format!(
                "split_returns has {n_supplied} entries but CPCV has more splits (split {} \
                 is missing); pass one array per split of cpcv_splits, in order",
                split.split_id
            ))
        })?;
        if returns.len() != split.test_indices.len() {
            return Err(BacktestError::Evaluator(format!(
                "split_returns[{}] has {} values but split {} tests {} samples",
                split.split_id,
                returns.len(),
                split.split_id,
                split.test_indices.len()
            )));
        }
        Ok(returns.clone())
    })
    .map_err(to_py_err)?;
    if n_supplied != result.splits.len() {
        return Err(PyValueError::new_err(format!(
            "split_returns has {n_supplied} entries but CPCV has {} splits",
            result.splits.len()
        )));
    }

    let out = PyDict::new(py);
    let folds =
        result.folds.iter().map(|f| performance_dict(py, f)).collect::<PyResult<Vec<_>>>()?;
    out.set_item("folds", folds)?;

    let mut splits = Vec::with_capacity(result.splits.len());
    for s in result.splits {
        let d = PyDict::new(py);
        d.set_item("split_id", s.split_id)?;
        d.set_item("train_indices", s.train_indices)?;
        d.set_item("test_indices", s.test_indices)?;
        d.set_item("test_groups", s.test_groups)?;
        d.set_item("purged_count", s.purged_count)?;
        d.set_item("embargo_count", s.embargo_count)?;
        splits.push(d);
    }
    out.set_item("splits", splits)?;
    out.set_item("path_count", result.path_count)?;
    out.set_item(
        "path_assignments",
        result.path_assignments.into_iter().map(|p| p.split_for_group).collect::<Vec<_>>(),
    )?;

    let mut paths = Vec::with_capacity(result.path_distribution.len());
    for p in &result.path_distribution {
        let d = PyDict::new(py);
        d.set_item("path_id", p.path_id)?;
        d.set_item("sharpe", p.sharpe)?;
        d.set_item("mean_return", p.mean_return)?;
        d.set_item("std_return", p.std_return)?;
        d.set_item("observations", p.observations)?;
        paths.push(d);
    }
    out.set_item("path_distribution", paths)?;

    let diag = result.diagnostics;
    let dd = PyDict::new(py);
    dd.set_item("mode", "combinatorial_purged_cross_validation")?;
    dd.set_item("mode_provenance", diag.mode_provenance)?;
    dd.set_item("trials_count", diag.trials_count)?;
    dd.set_item("split_count", diag.split_count)?;
    dd.set_item("pct_embargo", diag.pct_embargo)?;
    let sg = PyDict::new(py);
    sg.set_item("survivorship_bias_control", diag.safeguards.survivorship_bias_control)?;
    sg.set_item("look_ahead_control", diag.safeguards.look_ahead_control)?;
    sg.set_item("data_mining_control", diag.safeguards.data_mining_control)?;
    sg.set_item("cost_assumption", diag.safeguards.cost_assumption)?;
    sg.set_item("multiple_testing_control", diag.safeguards.multiple_testing_control)?;
    dd.set_item("safeguards", sg)?;
    dd.set_item("uses_label_span_purging", diag.anti_leakage.uses_label_span_purging)?;
    dd.set_item("uses_embargo", diag.anti_leakage.uses_embargo)?;
    dd.set_item("total_purged", diag.anti_leakage.total_purged)?;
    dd.set_item("total_embargoed", diag.anti_leakage.total_embargoed)?;
    out.set_item("diagnostics", dd)?;
    Ok(out)
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "backtesting_engine")?;
    m.add_function(wrap_pyfunction!(bt_cpcv_path_count, &m)?)?;
    m.add_function(wrap_pyfunction!(bt_run_cpcv, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("backtesting_engine", m)?;
    Ok(())
}
