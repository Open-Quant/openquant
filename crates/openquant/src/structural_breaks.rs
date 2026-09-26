//! Structural break tests (AFML chapter 17).
//!
//! A structural break is a change of regime: a mean-reverting series starts trending, a quiet
//! market turns explosive. AFML treats break statistics as *features*, one number per bar
//! saying how strongly the recent past looks like a different regime. This module implements
//! three of the chapter's tests:
//!
//! - [`get_sadf`]: the supremum augmented Dickey-Fuller test for explosiveness (§17.4.2,
//!   Snippets 17.1-17.4) and the sub/super-martingale tests for trends of a given shape
//!   (§17.4.3).
//! - [`get_chow_type_stat`]: the Chow-type Dickey-Fuller test for a single switch from a
//!   random walk to an explosive process (§17.4.1).
//! - [`get_chu_stinchcombe_white_statistics`]: the Chu-Stinchcombe-White CUSUM test on levels
//!   (§17.3.2).
//!
//! Conventions:
//!
//! - Inputs are **log prices**, oldest first, except that the `sm_poly_2`, `sm_exp` and
//!   `sm_power` models of [`get_sadf`] take a logarithm themselves and so want positive
//!   prices.
//! - Outputs are shorter than the input: each function documents which bar its first value
//!   belongs to.
//! - No critical values are supplied for SADF; they depend on the sample length and
//!   `min_length` and come from simulation (Phillips, Shi and Yu, 2015).
//!
//! ```
//! use openquant::structural_breaks::{get_sadf, SadfLags};
//!
//! // A log price whose increments compound: explosive by construction.
//! let mut y = vec![4.0_f64];
//! for t in 1..120 {
//!     let wobble = if t % 2 == 0 { 0.002 } else { -0.002 };
//!     y.push(y[t - 1] + 0.0005 * 1.05_f64.powi(t as i32) + wobble);
//! }
//!
//! let sadf = get_sadf(&y, "linear", true, 20, SadfLags::Fixed(1))?;
//! // One lag uses two leading bars; the first statistic then needs min_length more.
//! assert_eq!(sadf.len(), y.len() - 2 - 20);
//! assert!(*sadf.last().unwrap() > 3.0);
//! # Ok::<(), openquant::structural_breaks::StructuralBreakError>(())
//! ```

use nalgebra::DMatrix;

/// Error returned by the structural break tests.
#[derive(Debug, thiserror::Error)]
pub enum StructuralBreakError {
    /// The CUSUM `test_type` is neither `"one_sided"` nor `"two_sided"`.
    #[error("unknown test type: {0}")]
    InvalidTestType(String),
    /// The SADF `model` is not one of those listed on [`get_sadf`].
    #[error("unknown model: {0}")]
    InvalidModel(String),
    /// Reserved for a requested option that is not implemented; no function currently
    /// returns it.
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
    /// The series is too short for the requested lags and window, or a regression was given
    /// no rows (see each function's `# Errors`).
    #[error("the series is too short for the requested lags and window")]
    InputTooShort,
    /// A bar index passed to [`_get_values_diff`] is past the end of the series.
    #[error("index out of bounds")]
    IndexOutOfBounds,
}

/// Result alias for this module's functions.
pub type StructuralBreakResult<T> = Result<T, StructuralBreakError>;

/// Output of [`get_chu_stinchcombe_white_statistics`]: one entry per bar from bar 2 on.
#[derive(Debug, Clone)]
pub struct ChuStinchcombeWhiteResult {
    /// Critical value `sqrt(4.6 + ln(t - n))` at the reference bar `n` that maximises the
    /// statistic for bar `t` (AFML §17.3.2, `b_alpha = 4.6`).
    pub critical_value: Vec<f64>,
    /// The CUSUM statistic `S_t = max_n S_{n,t}` for bar `t`, maximised over reference bars
    /// `n < t`.
    pub stat: Vec<f64>,
}

/// The lagged differences included in the ADF regressions of [`get_sadf`].
#[derive(Debug, Clone)]
pub enum SadfLags {
    /// Lags `1..=L`: the usual augmented Dickey-Fuller specification with `L` lagged
    /// differences.
    Fixed(usize),
    /// An explicit list of lag numbers, e.g. `vec![1, 5]` for the first and fifth lagged
    /// difference only.
    Array(Vec<usize>),
}

/// Chow-type Dickey-Fuller statistics for a switch from a random walk to an explosive process
/// (AFML §17.4.1).
///
/// For each candidate break the function fits `Δy_t = δ y_{t-1} D_t + ε_t` without an
/// intercept, with the dummy `D_t` zero for the first `k` differences and one afterwards, and
/// reports the `t`-statistic of `δ`. Candidates run from `k = min_length` to
/// `k = n - min_length - 1`, so the output has `n - 2 * min_length` values and value `i`
/// belongs to a break at bar `min_length + i`. The largest value marks the estimated break
/// date. A singular regression gives `NaN`.
///
/// `log_prices` are log prices, oldest first. A series of at most `2 * min_length` prices
/// returns an empty vector rather than an error. The test assumes the series *stays* explosive after
/// the break; a later collapse weakens it (§17.4.2 prefers [`get_sadf`] for that reason).
///
/// # Errors
///
/// [`StructuralBreakError::InputTooShort`] only in the degenerate case `min_length == 0` with
/// a single price.
///
/// ```
/// use openquant::structural_breaks::get_chow_type_stat;
///
/// let y: Vec<f64> = (0..60).map(|t| 4.0 + 0.001 * 1.08_f64.powi(t)).collect();
/// let chow = get_chow_type_stat(&y, 20)?;
/// assert_eq!(chow.len(), y.len() - 2 * 20);
///
/// // Too short a series is not an error: it returns nothing.
/// assert!(get_chow_type_stat(&y[..30], 20)?.is_empty());
/// # Ok::<(), openquant::structural_breaks::StructuralBreakError>(())
/// ```
pub fn get_chow_type_stat(
    _log_prices: &[f64],
    _min_length: usize,
) -> StructuralBreakResult<Vec<f64>> {
    let series_len = _log_prices.len();
    if series_len < _min_length * 2 {
        return Ok(Vec::new());
    }

    let mut diffs = Vec::with_capacity(series_len.saturating_sub(1));
    let mut lags = Vec::with_capacity(series_len.saturating_sub(1));
    for i in 1..series_len {
        diffs.push(_log_prices[i] - _log_prices[i - 1]);
        lags.push(_log_prices[i - 1]);
    }

    let mut stats = Vec::with_capacity(series_len - _min_length * 2);
    for index in _min_length..(series_len - _min_length) {
        let mut x = lags.clone();
        for lag_value in x.iter_mut().take(index) {
            *lag_value = 0.0;
        }
        let x_matrix = x.into_iter().map(|v| vec![v]).collect::<Vec<_>>();
        let y_matrix = diffs.iter().map(|v| vec![*v]).collect::<Vec<_>>();
        let (coefs, coef_vars) = _get_betas(&x_matrix, &y_matrix)?;
        let b_estimate = coefs[0];
        let b_var = coef_vars[0][0];
        stats.push(b_estimate / b_var.sqrt());
    }

    Ok(stats)
}

/// Chu-Stinchcombe-White CUSUM test on levels (AFML §17.3.2).
///
/// For each bar `t` and every earlier reference bar `n`, the statistic is
/// `S_{n,t} = (y_t - y_n) / (σ_t sqrt(t - n))`, where `σ_t²` is the mean of the squared
/// first differences up to bar `t`. `S_t` is the maximum over `n`, returned with the critical
/// value `sqrt(4.6 + ln(t - n))` at the maximising `n`. The output vectors have `n - 2`
/// entries, for bars 2 onwards.
///
/// `log_prices` are log prices, oldest first. `test_type` is `"one_sided"` (`y_t - y_n`,
/// sensitive to rises) or `"two_sided"` (`|y_t - y_n|`). The statistic follows the book and
/// is unchanged by rescaling the series; mlfinlab divides by `σ_t²` rather than `σ_t`.
///
/// # Errors
///
/// - [`StructuralBreakError::InputTooShort`] if `log_prices` has fewer than 3 values.
/// - [`StructuralBreakError::InvalidTestType`] for any other `test_type`.
///
/// ```
/// use openquant::structural_breaks::get_chu_stinchcombe_white_statistics;
///
/// let y = [0.0, 0.01, -0.01, 0.0, 0.05, 0.10];
/// let res = get_chu_stinchcombe_white_statistics(&y, "one_sided")?;
/// assert_eq!(res.stat.len(), y.len() - 2);
/// assert_eq!(res.critical_value.len(), y.len() - 2);
/// // The late jump shows up as the largest statistic.
/// assert!(res.stat[3] > res.stat[0]);
/// # Ok::<(), openquant::structural_breaks::StructuralBreakError>(())
/// ```
pub fn get_chu_stinchcombe_white_statistics(
    _log_prices: &[f64],
    _test_type: &str,
) -> StructuralBreakResult<ChuStinchcombeWhiteResult> {
    let series_len = _log_prices.len();
    if series_len < 3 {
        return Err(StructuralBreakError::InputTooShort);
    }

    let mut critical_value = Vec::with_capacity(series_len - 2);
    let mut stat = Vec::with_capacity(series_len - 2);

    for index in 2..series_len {
        let mut squared_diff_sum = 0.0;
        for i in 1..=index {
            let diff = _log_prices[i] - _log_prices[i - 1];
            squared_diff_sum += diff * diff;
        }
        // AFML 17.3.2: sigma_t^2 = (t - 1)^-1 * sum_{i=2..t} (dy_i)^2 with 1-based t, i.e. the
        // mean of the squared differences up to bar t. With 0-based `index` there are
        // `index` of them (#173; the divisor used to be index - 1).
        let sigma_sq_t = squared_diff_sum / index as f64;

        let mut max_s_n_value = f64::NEG_INFINITY;
        let mut max_s_n_critical_value: Option<f64> = None;

        for ind in 0..index {
            let values_diff = _get_values_diff(_test_type, _log_prices, index, ind)?;
            let distance = (index - ind) as f64;
            let s_n_t = (1.0 / (sigma_sq_t.sqrt() * distance.sqrt())) * values_diff;

            if s_n_t > max_s_n_value {
                max_s_n_value = s_n_t;
                max_s_n_critical_value = Some((4.6 + distance.ln()).sqrt());
            }
        }

        stat.push(max_s_n_value);
        critical_value.push(max_s_n_critical_value.unwrap_or(f64::NAN));
    }

    Ok(ChuStinchcombeWhiteResult { critical_value, stat })
}

/// Supremum ADF and sub/super-martingale statistics (AFML §17.4.2–17.4.3, Snippets 17.1–17.4).
///
/// Returns one value per regression row from position `min_length` on: the supremum, over
/// every window of at least `min_length` rows ending at that row, of the statistic below.
/// Rows start at bar `max_lag + 1` for every model, and `t` is the 0-based row position over the
/// whole sample.
///
/// | `model` | Regression | Statistic |
/// | --- | --- | --- |
/// | `"linear"` | Δy on y₋₁, lagged Δy, const (if `add_const`), t | β(y₋₁) / se |
/// | `"quadratic"` | Δy on y₋₁, lagged Δy, const (if `add_const`), t, t² (Snippet 17.2 `ctt`) | β(y₋₁) / se |
/// | `"sm_poly_1"` | y on 1, t, t² | \|β(t²)\| / se |
/// | `"sm_poly_2"` | log y on 1, t, t² | \|β(t²)\| / se |
/// | `"sm_exp"` | log y on 1, t | \|β(t)\| / se |
/// | `"sm_power"` | log y on 1, log(t + 1) (time counted from 1) | \|β\| / se |
///
/// The `sm_*` models ignore `add_const`, take the absolute value because a trend of either
/// sign is of interest (§17.4.3), and need a positive series when they take logs. Windows
/// whose regression is singular are skipped; a row with no usable window is `-inf`.
///
/// `lags` sets the lagged differences in the ADF models; for the `sm_*` models it only sets
/// where the output starts. The output has `rows - min_length` values, where
/// `rows = series.len() - max_lag - 1`, and is empty when `rows <= min_length`. Cost is
/// `O(n²)` regressions, so compute it on sampled bars. The `(t - t0)^φ` window penalty of
/// §17.4.3 is not implemented (`φ = 0`).
///
/// # Errors
///
/// - [`StructuralBreakError::InvalidModel`] for a `model` not in the table.
/// - [`StructuralBreakError::InputTooShort`] if `series` has fewer than 2 values, is no
///   longer than `max_lag + 1`, or `min_length == 0` (a window must have at least one row).
///
/// ```
/// use openquant::structural_breaks::{get_sadf, SadfLags, StructuralBreakError};
///
/// let prices: Vec<f64> = (0..60).map(|t| 100.0 * 1.01_f64.powi(t) + (t % 3) as f64).collect();
/// let smt = get_sadf(&prices, "sm_exp", false, 20, SadfLags::Fixed(1))?;
/// assert_eq!(smt.len(), prices.len() - 2 - 20);
/// // The sub/super-martingale statistic is an absolute t-ratio.
/// assert!(smt.iter().all(|v| *v >= 0.0));
///
/// assert!(matches!(
///     get_sadf(&prices, "cubic", true, 20, SadfLags::Fixed(1)),
///     Err(StructuralBreakError::InvalidModel(_))
/// ));
/// # Ok::<(), StructuralBreakError>(())
/// ```
pub fn get_sadf(
    _series: &[f64],
    _model: &str,
    _add_const: bool,
    _min_length: usize,
    _lags: SadfLags,
) -> StructuralBreakResult<Vec<f64>> {
    let (x, y, indices) = get_y_x(_series, _model, _lags, _add_const)?;
    // AFML 17.4.3: the sub/super-martingale statistic is |beta| / se, since a trend of either
    // sign is of interest. The ADF-based models keep the signed t-statistic (explosiveness
    // is beta > 0).
    let absolute = _model.starts_with("sm_");
    if y.len() <= _min_length {
        return Ok(Vec::new());
    }

    let mut sadf_values = Vec::with_capacity(y.len().saturating_sub(_min_length));
    for (pos, _) in indices.iter().enumerate().skip(_min_length) {
        let x_subset = x[..=pos].to_vec();
        let y_subset = y[..=pos].to_vec();
        let value = get_sadf_at_t(&x_subset, &y_subset, _min_length, absolute)?;
        sadf_values.push(value);
    }

    Ok(sadf_values)
}

/// The price change used by the CUSUM test: `series[index] - series[ind]` for
/// `"one_sided"`, its absolute value for `"two_sided"`.
///
/// A helper of [`get_chu_stinchcombe_white_statistics`], public (with mlfinlab's
/// underscore-prefixed name) so that tests can check it directly.
///
/// # Errors
///
/// - [`StructuralBreakError::IndexOutOfBounds`] if `index` or `ind` is past the end of
///   `series`.
/// - [`StructuralBreakError::InvalidTestType`] for any other `test_type`.
///
/// ```
/// use openquant::structural_breaks::_get_values_diff;
///
/// assert_eq!(_get_values_diff("one_sided", &[1.0, 3.0], 0, 1)?, -2.0);
/// assert_eq!(_get_values_diff("two_sided", &[1.0, 3.0], 0, 1)?, 2.0);
/// # Ok::<(), openquant::structural_breaks::StructuralBreakError>(())
/// ```
pub fn _get_values_diff(
    test_type: &str,
    series: &[f64],
    index: usize,
    ind: usize,
) -> StructuralBreakResult<f64> {
    let left = series.get(index).ok_or(StructuralBreakError::IndexOutOfBounds)?;
    let right = series.get(ind).ok_or(StructuralBreakError::IndexOutOfBounds)?;
    match test_type {
        "one_sided" => Ok(left - right),
        "two_sided" => Ok((left - right).abs()),
        _ => Err(StructuralBreakError::InvalidTestType(test_type.to_string())),
    }
}

/// Ordinary least squares of `y` on `x` without an added intercept (AFML Snippet 17.4,
/// `getBetas`).
///
/// `x` holds one row per observation and `y` one single-element row per observation.
/// Returns `(coefficients, covariance)`: the coefficient vector `(XᵀX)⁻¹ Xᵀy` and its
/// covariance matrix `(XᵀX)⁻¹ s²`, with `s² = eᵀe / (rows - cols)`. If `XᵀX` is singular both
/// are all `NaN`, which is how the SADF search recognises a window to skip.
///
/// A helper of [`get_sadf`] and [`get_chow_type_stat`], public (with mlfinlab's
/// underscore-prefixed name) so that tests can check it directly.
///
/// # Errors
///
/// [`StructuralBreakError::InputTooShort`] if `x` or `y` has no rows or an empty row, or its
/// rows differ in length.
///
/// ```
/// use openquant::structural_breaks::_get_betas;
///
/// // y = 2 x exactly: slope 2, zero residual variance.
/// let x = vec![vec![1.0], vec![2.0], vec![3.0]];
/// let y = vec![vec![2.0], vec![4.0], vec![6.0]];
/// let (beta, cov) = _get_betas(&x, &y)?;
/// assert!((beta[0] - 2.0).abs() < 1e-12);
/// assert!(cov[0][0].abs() < 1e-12);
/// # Ok::<(), openquant::structural_breaks::StructuralBreakError>(())
/// ```
pub fn _get_betas(
    _x: &[Vec<f64>],
    _y: &[Vec<f64>],
) -> StructuralBreakResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let x_matrix = to_matrix(_x)?;
    let y_matrix = to_matrix(_y)?;

    let rows = x_matrix.nrows();
    let cols = x_matrix.ncols();
    let y_cols = y_matrix.ncols();

    let xy = x_matrix.transpose() * &y_matrix;
    let xx = x_matrix.transpose() * &x_matrix;

    let Some(xx_inv) = xx.try_inverse() else {
        let b_mean = vec![f64::NAN; cols];
        let b_var = vec![vec![f64::NAN; cols]; cols];
        return Ok((b_mean, b_var));
    };

    let b_mean = &xx_inv * xy;
    let err = y_matrix - x_matrix * &b_mean;
    let err_t_err = err.transpose() * err;
    let denom = rows as f64 - cols as f64;
    let scale = err_t_err / denom;

    let b_var_matrix = if y_cols == 1 {
        let scalar = scale[(0, 0)];
        xx_inv * scalar
    } else if scale.nrows() == cols && scale.ncols() == cols {
        xx_inv.component_mul(&scale)
    } else {
        let scalar = scale[(0, 0)];
        xx_inv * scalar
    };

    let mut b_mean_vec = Vec::with_capacity(cols);
    for i in 0..cols {
        b_mean_vec.push(b_mean[(i, 0)]);
    }

    Ok((b_mean_vec, matrix_to_vec(b_var_matrix)))
}

/// Regression inputs for SADF: `(x rows, y, lag list)`.
type SadfRegressionInputs = (Vec<Vec<f64>>, Vec<f64>, Vec<usize>);

fn get_y_x(
    series: &[f64],
    model: &str,
    lags: SadfLags,
    add_const: bool,
) -> StructuralBreakResult<SadfRegressionInputs> {
    let series_len = series.len();
    if series_len < 2 {
        return Err(StructuralBreakError::InputTooShort);
    }

    let mut series_diff = Vec::with_capacity(series_len - 1);
    for i in 1..series_len {
        series_diff.push(series[i] - series[i - 1]);
    }

    let lag_values = match lags {
        SadfLags::Fixed(value) => (1..=value).collect::<Vec<_>>(),
        SadfLags::Array(values) => values.into_iter().collect(),
    };
    let max_lag = *lag_values.iter().max().unwrap_or(&0);
    let start_index = max_lag + 1;
    if series_len <= start_index {
        return Err(StructuralBreakError::InputTooShort);
    }

    let mut indices = Vec::new();
    let mut x_rows = Vec::new();
    let mut y_values = Vec::new();

    for idx in start_index..=series_len - 1 {
        let mut row = Vec::with_capacity(lag_values.len());
        for lag in &lag_values {
            let pos = idx - lag - 1;
            row.push(series_diff[pos]);
        }
        x_rows.push(row);
        indices.push(idx);
        y_values.push(series_diff[idx - 1]);
    }

    let mut x = x_rows;
    let mut y = y_values;

    match model {
        "linear" | "quadratic" => {
            let mut updated = Vec::with_capacity(x.len());
            for (i, row) in x.into_iter().enumerate() {
                let mut new_row = Vec::with_capacity(row.len() + 3);
                new_row.push(series[indices[i] - 1]);
                new_row.extend(row);
                if add_const {
                    new_row.push(1.0);
                }
                // AFML snippet 17.2 (getYX): 'ct' appends the trend; 'ctt' appends the trend
                // and then its square, so "quadratic" carries both t and t^2.
                let trend = i as f64;
                new_row.push(trend);
                if model == "quadratic" {
                    new_row.push(trend * trend);
                }
                updated.push(new_row);
            }
            x = updated;
        }
        "sm_poly_1" => {
            y = indices.iter().map(|&idx| series[idx]).collect();
            let mut updated = Vec::with_capacity(y.len());
            for i in 0..y.len() {
                let trend = i as f64;
                let row = vec![trend * trend, 1.0, trend];
                updated.push(row);
            }
            x = updated;
        }
        "sm_poly_2" => {
            y = indices.iter().map(|&idx| series[idx].ln()).collect();
            let mut updated = Vec::with_capacity(y.len());
            for i in 0..y.len() {
                let trend = i as f64;
                let row = vec![trend * trend, 1.0, trend];
                updated.push(row);
            }
            x = updated;
        }
        "sm_exp" => {
            y = indices.iter().map(|&idx| series[idx].ln()).collect();
            let mut updated = Vec::with_capacity(y.len());
            for (i, _) in y.iter().enumerate() {
                let trend = i as f64;
                let row = vec![trend, 1.0];
                updated.push(row);
            }
            x = updated;
        }
        "sm_power" => {
            y = indices.iter().map(|&idx| series[idx].ln()).collect();
            let mut updated = Vec::with_capacity(y.len());
            for (i, _) in y.iter().enumerate() {
                // AFML 17.4.3 counts time from t = 1 for the power trend, so the first row
                // has log t = 0 rather than log 0 = -inf.
                let trend = ((i + 1) as f64).ln();
                let row = vec![trend, 1.0];
                updated.push(row);
            }
            x = updated;
        }
        _ => {
            return Err(StructuralBreakError::InvalidModel(model.to_string()));
        }
    }

    Ok((x, y, indices))
}

fn get_sadf_at_t(
    x: &[Vec<f64>],
    y: &[f64],
    min_length: usize,
    absolute: bool,
) -> StructuralBreakResult<f64> {
    let y_len = y.len();
    if y_len < min_length {
        return Ok(f64::NEG_INFINITY);
    }

    let mut bsadf = f64::NEG_INFINITY;
    let start_points = 0..=(y_len - min_length);
    for start in start_points {
        let y_subset = y[start..].iter().map(|v| vec![*v]).collect::<Vec<_>>();
        let x_subset = x[start..].to_vec();

        let (b_mean, b_var) = _get_betas(&x_subset, &y_subset)?;
        if b_mean.first().map(|v| v.is_nan()).unwrap_or(true) {
            continue;
        }

        let b_estimate = b_mean[0];
        let b_std = b_var[0][0].sqrt();
        let ratio = b_estimate / b_std;
        let all_adf = if absolute { ratio.abs() } else { ratio };
        if all_adf > bsadf {
            bsadf = all_adf;
        }
    }

    Ok(bsadf)
}

fn to_matrix(data: &[Vec<f64>]) -> StructuralBreakResult<DMatrix<f64>> {
    let rows = data.len();
    if rows == 0 {
        return Err(StructuralBreakError::InputTooShort);
    }
    let cols = data[0].len();
    if cols == 0 {
        return Err(StructuralBreakError::InputTooShort);
    }
    if data.iter().any(|row| row.len() != cols) {
        return Err(StructuralBreakError::InputTooShort);
    }
    let flat = data.iter().flat_map(|row| row.iter().cloned()).collect::<Vec<_>>();
    Ok(DMatrix::from_row_slice(rows, cols, &flat))
}

fn matrix_to_vec(matrix: DMatrix<f64>) -> Vec<Vec<f64>> {
    let mut rows = Vec::with_capacity(matrix.nrows());
    for i in 0..matrix.nrows() {
        let mut row = Vec::with_capacity(matrix.ncols());
        for j in 0..matrix.ncols() {
            row.push(matrix[(i, j)]);
        }
        rows.push(row);
    }
    rows
}
