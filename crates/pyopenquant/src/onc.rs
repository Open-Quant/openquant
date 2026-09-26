use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::helpers::{matrix_from_rows, to_py_err};

/// Optimal Number of Clusters (ONC) partition of a correlation matrix.
///
/// Not from AFML: López de Prado, Machine Learning for Asset Managers (2020), section 4.4,
/// Snippets 4.1-4.2. Correlations become distances `sqrt((1 - rho) / 2)` (inputs clamped
/// to `[-1, 1]`), each item is represented by its row of that matrix, and k-means is run
/// for every `k` from 2 to `max(N - 1, 2)`, `repeat` times each; the partition with the
/// highest silhouette t-statistic `mean(S) / std(S)` wins. Clusters scoring below average
/// are re-clustered and kept only if that improves their mean t-statistic. Negative
/// correlation means far apart. Results are deterministic (fixed seeds). The search starts
/// at `k = 2`, so a matrix with no structure is still partitioned; a low mean silhouette
/// is the sign the clusters are not real. Cost grows at least as `N^3`.
///
/// Parameters
/// ----------
/// corr_mat : list[list[float]]
///     `N x N` correlation matrix (`N >= 2`), one inner list per row. Symmetry and a unit
///     diagonal are not checked.
/// repeat : int
///     Number of k-means initialisations per candidate `k`; must be positive.
///
/// Returns
/// -------
/// dict[str, Any]
///     `ordered_correlation` (list[list[float]], the input permuted so each cluster's
///     members are contiguous), `clusters` (dict[int, list[int]], cluster label to member
///     indices in the input's row order) and `silhouette_scores` (list[float], one per
///     item in the input's row order; 0 for a singleton cluster's member).
///
/// Raises
/// ------
/// ValueError
///     If `corr_mat` is empty or ragged, or the core rejects the input (e.g. `repeat` of
///     0, a non-square matrix or fewer than two rows, or NaN entries that leave no
///     candidate partition).
#[pyfunction(name = "get_onc_clusters")]
fn onc_get_onc_clusters(
    py: Python<'_>,
    corr_mat: Vec<Vec<f64>>,
    repeat: usize,
) -> PyResult<PyObject> {
    let m = matrix_from_rows(corr_mat)?;
    let result = openquant::onc::get_onc_clusters(&m, repeat).map_err(to_py_err)?;

    let d = PyDict::new(py);

    // Convert ordered correlation matrix to Vec<Vec<f64>>
    let nrows = result.ordered_correlation.nrows();
    let ncols = result.ordered_correlation.ncols();
    let ordered: Vec<Vec<f64>> = (0..nrows)
        .map(|r| (0..ncols).map(|c| result.ordered_correlation[(r, c)]).collect())
        .collect();
    d.set_item("ordered_correlation", ordered)?;

    // Convert BTreeMap<usize, Vec<usize>> to PyDict
    let clusters = PyDict::new(py);
    for (k, v) in &result.clusters {
        clusters.set_item(*k, v.clone())?;
    }
    d.set_item("clusters", clusters)?;

    d.set_item("silhouette_scores", result.silhouette_scores)?;

    Ok(d.into_pyobject(py).unwrap().into_any().unbind())
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "onc")?;
    m.add_function(wrap_pyfunction!(onc_get_onc_clusters, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("onc", m)?;
    Ok(())
}
