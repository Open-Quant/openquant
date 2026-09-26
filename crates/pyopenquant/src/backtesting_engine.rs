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

/// Number of CPCV backtest paths, phi[N, k] = k / N * C(N, k) (AFML section 12.4.1).
///
/// Each of the N groups is tested in exactly phi of the C(N, k) splits, so the splits can be
/// stitched into phi full-length out-of-sample paths. For example N = 6, k = 2 gives 15 splits
/// and 5 paths.
///
/// Parameters
/// ----------
/// n_groups : int
///     Number of contiguous groups N; at least 2.
/// test_groups : int
///     Number of groups tested per split k; `1 <= test_groups < n_groups`.
///
/// Returns
/// -------
/// int
///     The path count phi[N, k].
///
/// Raises
/// ------
/// ValueError
///     If the core rejects the input (e.g. `n_groups < 2`, `test_groups == 0`,
///     `test_groups >= n_groups`, or C(N, k) too large for the platform's integers).
#[pyfunction(name = "cpcv_path_count")]
fn bt_cpcv_path_count(n_groups: usize, test_groups: usize) -> PyResult<usize> {
    cpcv_path_count(n_groups, test_groups).map_err(to_py_err)
}

// Mirrors the Rust run configuration field by field; a params struct would only restate it.
#[allow(clippy::too_many_arguments)]
/// Run a combinatorial purged cross-validation backtest on precomputed returns (AFML 12.4).
///
/// The samples are cut into `n_groups` contiguous groups (the first `n % n_groups` one sample
/// larger) and one split is built for each of the C(N, k) ways to test `test_groups` of them,
/// in lexicographic order; this is the numbering of `openquant.cross_validation.cpcv_splits`
/// with `n_splits=n_groups, n_test_splits=test_groups`. Each split trains on the other groups
/// after purging (closed-interval overlap with any test span) and an embargo of
/// `ceil(pct_embargo * n)` samples after each run of adjacent test groups. No model is fitted
/// here: fit and predict on those splits yourself, then pass one array of out-of-sample returns
/// per split. Path `j` takes each group's returns from the `j`-th split that tests it, giving
/// phi[N, k] paths that each cover every sample once. The paths reuse the same predictions,
/// so their spread is not a confidence interval.
///
/// `sharpe` in the results is a t-statistic, `mean / std * sqrt(n)` with the sample standard
/// deviation (ddof 1); it is not annualised, grows with `n`, and is 0 when the deviation is 0.
///
/// Parameters
/// ----------
/// t0 : list[int]
///     Start of each sample's label span, as int64 nanoseconds since the epoch (plain integers
///     also work). One per sample, in time order.
/// t1 : list[int]
///     End of each sample's label span, in the same units as `t0`. Must be `>= t0`.
/// split_returns : list[list[float]]
///     One array per split, in `split_id` order, each holding one finite out-of-sample return
///     per entry of that split's `test_indices`, in the same order.
/// n_groups : int
///     Number of contiguous groups N; `2 <= n_groups <= len(t0)`. Keyword-only.
/// test_groups : int
///     Number of groups tested per split k; `1 <= test_groups < n_groups`. Keyword-only.
/// pct_embargo : float
///     Embargo as a fraction of the whole sample count, in `[0, 1)`, rounded up. Keyword-only.
/// mode_provenance : str
///     Non-blank free-text note on where this configuration came from. Keyword-only.
/// trials_count : int
///     Number of configurations tried to reach this one (what a deflated Sharpe ratio needs);
///     must be positive. Keyword-only.
/// safeguards : dict[str, str]
///     How the run controls the pitfalls of AFML section 11.4, as non-blank free text, with
///     exactly the keys `survivorship_bias_control`, `look_ahead_control`,
///     `data_mining_control`, `cost_assumption` and `multiple_testing_control`. Keyword-only.
///
/// Returns
/// -------
/// dict[str, Any]
///     A dict with keys:
///
///     - `folds` (list[dict]): one per split, with `split_id` (int), `sharpe` (float),
///       `mean_return` (float), `std_return` (float) and `observations` (int).
///     - `splits` (list[dict]): one per split, with `split_id` (int), `train_indices`
///       (list[int]), `test_indices` (list[int]), `test_groups` (list[int], ascending),
///       `purged_count` (int) and `embargo_count` (int, samples embargoed that the purge had
///       not already removed).
///     - `path_count` (int): phi[N, k].
///     - `path_assignments` (list[list[int]]): per path, element `g` is the `split_id` that
///       supplies group `g`'s returns.
///     - `path_distribution` (list[dict]): one per path, with `path_id` (int), `sharpe`,
///       `mean_return`, `std_return` (floats) and `observations` (int).
///     - `diagnostics` (dict): `mode` (always "combinatorial_purged_cross_validation"),
///       `mode_provenance` (str), `trials_count` (int), `split_count` (int), `pct_embargo`
///       (float), `safeguards` (dict[str, str], as passed), `uses_label_span_purging` (bool,
///       always True), `uses_embargo` (bool, `pct_embargo > 0`), `total_purged` (int) and
///       `total_embargoed` (int).
///
/// Raises
/// ------
/// ValueError
///     If `t0` and `t1` differ in length; if `safeguards` is missing a required key or has an
///     unknown one; if `split_returns` does not have exactly one entry per split or an entry's
///     length differs from its split's test set; or if the core rejects the input (e.g. no
///     samples, a span that ends before it starts, a blank `mode_provenance` or safeguard,
///     `trials_count == 0`, `n_groups < 2`, `test_groups` not in `[1, n_groups)`,
///     `n_groups` above the sample count, `pct_embargo` outside `[0, 1)`, purging and embargo
///     emptying a training set, or an empty or non-finite `split_returns` entry).
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
