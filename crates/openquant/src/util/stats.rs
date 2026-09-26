//! Descriptive statistics shared across the crate: [`mean`], [`variance`], [`std_dev`],
//! the sample [`covariance`] of a returns matrix, and [`quantile`] / [`quantile_sorted`].
//!
//! Every estimator in the crate that needs one of these calls it from here, so the conventions
//! are stated once:
//!
//! - **Degrees of freedom.** [`variance`] and [`std_dev`] divide the sum of squared deviations
//!   by `n - ddof`, as numpy's `ddof` argument does: `ddof = 0` is the population estimator
//!   (numpy's default), `ddof = 1` the sample estimator (pandas' default). [`covariance`] is
//!   always the sample estimator (`ddof = 1`, pandas' `DataFrame.cov`).
//! - **Too few values.** Where numpy and pandas return `NaN` (no values, or `n <= ddof`), these
//!   functions return `None`, and each caller states what it substitutes (usually `0.0` or
//!   `NaN`).
//! - **`NaN` values** are not skipped: one `NaN` input makes the result `NaN` (numpy's
//!   behaviour, not pandas' `skipna`). Filter them out first where that matters.
//! - **Quantiles** interpolate between order statistics by one of the [`QuantileMethod`]s, with
//!   `q` clamped into `[0, 1]`.
//!
//! The arithmetic is two-pass (the mean first, then the squared deviations from it), summed in
//! input order, so results do not depend on which module calls them.
//!
//! ```
//! use openquant::util::stats::{mean, quantile, std_dev, QuantileMethod};
//!
//! let x = [1.0, 2.0, 3.0, 4.0];
//! assert_eq!(mean(&x), Some(2.5));
//! assert_eq!(std_dev(&x, 0), Some(1.25_f64.sqrt()));
//! assert_eq!(quantile(&x, 0.5, QuantileMethod::Linear), Some(2.5));
//! ```

use nalgebra::DMatrix;

/// Arithmetic mean, `sum(values) / n`, or `None` when `values` is empty.
///
/// ```
/// use openquant::util::stats::mean;
///
/// assert_eq!(mean(&[1.0, 2.0, 6.0]), Some(3.0));
/// assert_eq!(mean(&[]), None);
/// ```
pub fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

/// Variance with `ddof` delta degrees of freedom: `sum((x - mean)^2) / (n - ddof)`.
///
/// `ddof = 0` is the population variance and `ddof = 1` the sample variance. Returns `None`
/// when `n <= ddof` or `values` is empty (numpy and pandas would give `NaN`).
///
/// ```
/// use openquant::util::stats::variance;
///
/// let x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// assert_eq!(variance(&x, 0), Some(4.0));
/// assert_eq!(variance(&x, 1), Some(32.0 / 7.0));
/// assert_eq!(variance(&[1.0], 1), None);
/// ```
pub fn variance(values: &[f64], ddof: usize) -> Option<f64> {
    let n = values.len();
    if n == 0 || n <= ddof {
        return None;
    }
    let mu = values.iter().sum::<f64>() / n as f64;
    let squares = values
        .iter()
        .map(|v| {
            let d = *v - mu;
            d * d
        })
        .sum::<f64>();
    Some(squares / (n - ddof) as f64)
}

/// Standard deviation with `ddof` delta degrees of freedom: the square root of
/// [`variance`]`(values, ddof)`, with the same `None` cases.
///
/// ```
/// use openquant::util::stats::std_dev;
///
/// assert_eq!(std_dev(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], 0), Some(2.0));
/// assert_eq!(std_dev(&[1.0, 3.0], 1), Some(2.0_f64.sqrt()));
/// assert_eq!(std_dev(&[], 0), None);
/// ```
pub fn std_dev(values: &[f64], ddof: usize) -> Option<f64> {
    variance(values, ddof).map(f64::sqrt)
}

/// Sample covariance matrix (`ddof = 1`) of `returns`, which has one row per observation and
/// one column per asset: entry `(i, j)` is
/// `sum_r (x_ri - mean_i) (x_rj - mean_j) / (rows - 1)`.
///
/// The result is exactly symmetric (each pair is computed once). Returns `None` with fewer than
/// two rows.
///
/// ```
/// use nalgebra::DMatrix;
/// use openquant::util::stats::covariance;
///
/// // Two assets, three observations; the second column is twice the first.
/// let returns = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0]);
/// let cov = covariance(&returns).unwrap();
/// assert_eq!(cov, DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]));
/// assert!(covariance(&DMatrix::<f64>::zeros(1, 2)).is_none());
/// ```
pub fn covariance(returns: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let rows = returns.nrows();
    let cols = returns.ncols();
    if rows < 2 {
        return None;
    }
    let means: Vec<f64> = (0..cols).map(|c| returns.column(c).sum() / rows as f64).collect();
    let mut cov = DMatrix::<f64>::zeros(cols, cols);
    for i in 0..cols {
        for j in i..cols {
            let mut s = 0.0;
            for r in 0..rows {
                s += (returns[(r, i)] - means[i]) * (returns[(r, j)] - means[j]);
            }
            s /= (rows - 1) as f64;
            cov[(i, j)] = s;
            cov[(j, i)] = s;
        }
    }
    Some(cov)
}

/// How [`quantile_sorted`] picks the `q`-quantile of `n` sorted values `x_0 <= … <= x_{n-1}`.
///
/// Each method starts from the fractional index `h = q (n - 1)` (numpy's default plotting
/// position):
///
/// | method | result | numpy `method=` |
/// | --- | --- | --- |
/// | `Linear` | `x_⌊h⌋ + (h - ⌊h⌋)(x_⌈h⌉ - x_⌊h⌋)` | `"linear"` (the default) |
/// | `Higher` | `x_⌈h⌉` | `"higher"` |
/// | `Nearest` | `x_round(h)`, halves rounded away from zero | none (numpy's `"nearest"` rounds halves to even) |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantileMethod {
    /// Linear interpolation between the two order statistics around `h`.
    Linear,
    /// The order statistic at or above `h`; always an observed value.
    Higher,
    /// The order statistic nearest `h`, ties rounded away from zero (Rust's [`f64::round`]);
    /// always an observed value.
    Nearest,
}

/// The `q`-quantile of `sorted`, which must already be in ascending order, by `method`.
///
/// `q` is clamped into `[0, 1]`. Returns `None` when `sorted` is empty. The order is not
/// checked; callers choose how `NaN` values sort (for example [`f64::total_cmp`]).
///
/// ```
/// use openquant::util::stats::{quantile_sorted, QuantileMethod};
///
/// let x = [10.0, 20.0, 30.0, 40.0];
/// // h = 0.5 * 3 = 1.5
/// assert_eq!(quantile_sorted(&x, 0.5, QuantileMethod::Linear), Some(25.0));
/// assert_eq!(quantile_sorted(&x, 0.5, QuantileMethod::Higher), Some(30.0));
/// assert_eq!(quantile_sorted(&x, 0.5, QuantileMethod::Nearest), Some(30.0));
/// assert_eq!(quantile_sorted(&x, 2.0, QuantileMethod::Linear), Some(40.0));
/// assert_eq!(quantile_sorted(&[], 0.5, QuantileMethod::Linear), None);
/// ```
pub fn quantile_sorted(sorted: &[f64], q: f64, method: QuantileMethod) -> Option<f64> {
    let n = sorted.len();
    if n == 0 {
        return None;
    }
    let q = q.clamp(0.0, 1.0);
    let h = q * (n - 1) as f64;
    // A `NaN` `q` makes `h` `NaN`, which the casts below turn into index 0.
    let value = match method {
        QuantileMethod::Linear => {
            let lo = h.floor() as usize;
            let hi = h.ceil() as usize;
            if lo == hi {
                sorted[lo]
            } else {
                sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
            }
        }
        QuantileMethod::Higher => sorted[(h.ceil() as usize).min(n - 1)],
        QuantileMethod::Nearest => sorted[(h.round() as usize).min(n - 1)],
    };
    Some(value)
}

/// The `q`-quantile of `values` by `method`: [`quantile_sorted`] on a copy sorted with
/// [`f64::total_cmp`] (so `NaN`s sort to the ends by sign).
///
/// ```
/// use openquant::util::stats::{quantile, QuantileMethod};
///
/// assert_eq!(quantile(&[3.0, 1.0, 2.0], 0.25, QuantileMethod::Linear), Some(1.5));
/// assert_eq!(quantile(&[3.0, 1.0, 2.0], 0.25, QuantileMethod::Higher), Some(2.0));
/// assert_eq!(quantile(&[3.0, 1.0, 2.0], 0.25, QuantileMethod::Nearest), Some(2.0));
/// ```
pub fn quantile(values: &[f64], q: f64, method: QuantileMethod) -> Option<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    quantile_sorted(&sorted, q, method)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variance_needs_more_values_than_ddof() {
        assert_eq!(variance(&[], 0), None);
        assert_eq!(variance(&[5.0], 0), Some(0.0));
        assert_eq!(variance(&[5.0], 1), None);
        assert_eq!(variance(&[1.0, 2.0], 2), None);
    }

    #[test]
    fn nan_propagates() {
        assert!(std_dev(&[1.0, f64::NAN, 3.0], 1).unwrap().is_nan());
        assert!(mean(&[1.0, f64::NAN]).unwrap().is_nan());
    }

    #[test]
    fn quantile_edges() {
        let x = [1.0, 2.0, 3.0];
        for method in [QuantileMethod::Linear, QuantileMethod::Higher, QuantileMethod::Nearest] {
            assert_eq!(quantile_sorted(&x, 0.0, method), Some(1.0));
            assert_eq!(quantile_sorted(&x, 1.0, method), Some(3.0));
            assert_eq!(quantile_sorted(&x, -1.0, method), Some(1.0));
            assert_eq!(quantile_sorted(&x, f64::NAN, method), Some(1.0));
            assert_eq!(quantile_sorted(&[7.0], 0.3, method), Some(7.0));
        }
        // h = 0.25 * 2 = 0.5: halves round away from zero.
        assert_eq!(quantile_sorted(&x, 0.25, QuantileMethod::Nearest), Some(2.0));
        assert_eq!(quantile_sorted(&x, 0.2, QuantileMethod::Higher), Some(2.0));
    }

    #[test]
    fn covariance_is_symmetric_sample_estimate() {
        let r = DMatrix::from_row_slice(4, 2, &[0.1, -0.2, 0.3, 0.1, -0.1, 0.0, 0.2, 0.4]);
        let cov = covariance(&r).unwrap();
        assert_eq!(cov[(0, 1)], cov[(1, 0)]);
        let c0: Vec<f64> = r.column(0).iter().copied().collect();
        assert!((cov[(0, 0)] - variance(&c0, 1).unwrap()).abs() < 1e-15);
    }
}
