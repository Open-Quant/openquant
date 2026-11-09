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
//! Finite constant slices have their exact mean and zero variance/deviation. Otherwise the
//! arithmetic is two-pass (the mean first, then the squared deviations from it), summed in input
//! order. This is not an overflow-safe or compensated estimator for nonconstant inputs.
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
/// A finite constant slice returns its constant value exactly, avoiding summation roundoff.
///
/// ```
/// use openquant::util::stats::mean;
///
/// assert_eq!(mean(&[1.0, 2.0, 6.0]), Some(3.0));
/// assert_eq!(mean(&[]), None);
/// ```
pub fn mean(values: &[f64]) -> Option<f64> {
    let &first = values.first()?;
    // Equal values can acquire a rounded mean after summation. Compare bits so a mixture of
    // positive and negative zero keeps the original summation's sign convention.
    if first.is_finite() && values.iter().all(|v| v.to_bits() == first.to_bits()) {
        return Some(first);
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
    let mu = mean(values)?;
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
    // [`mean`] keeps a finite constant column's mean exact, so its variance is exactly zero;
    // `sum / rows` can round (three `0.1`s give `0.10000000000000002`). The storage is
    // column-major, so column `c` is one contiguous slice.
    let data = returns.as_slice();
    let means: Vec<f64> =
        (0..cols).map(|c| mean(&data[c * rows..(c + 1) * rows]).unwrap_or(f64::NAN)).collect();
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
    fn finite_constants_have_exact_mean() {
        for value in [0.01, -0.01, f64::MAX, -f64::MAX, f64::MIN_POSITIVE, f64::from_bits(1)] {
            for n in [1, 2, 10, 1000] {
                let values = vec![value; n];
                assert_eq!(mean(&values), Some(value), "value={value}, n={n}");
            }
        }
    }

    #[test]
    fn finite_constants_have_zero_variance() {
        for value in [0.01, -0.01, f64::MAX, -f64::MAX, f64::MIN_POSITIVE, f64::from_bits(1)] {
            for n in [1, 2, 10, 1000] {
                let values = vec![value; n];
                for ddof in [0, 1, n - 1, n, usize::MAX] {
                    let expected = (n > ddof).then_some(0.0);
                    assert_eq!(
                        variance(&values, ddof),
                        expected,
                        "value={value}, n={n}, ddof={ddof}"
                    );
                    assert_eq!(std_dev(&values, ddof), expected);
                }
            }
        }
    }

    #[test]
    fn signed_zero_statistics_keep_their_sign_conventions() {
        for values in [vec![0.0; 10], vec![-0.0; 10], vec![-0.0, 0.0], vec![0.0, -0.0]] {
            let expected_mean = values.iter().sum::<f64>() / values.len() as f64;
            assert_eq!(mean(&values).unwrap().to_bits(), expected_mean.to_bits());
            for ddof in [0, 1] {
                assert_eq!(variance(&values, ddof).unwrap().to_bits(), 0.0_f64.to_bits());
                assert_eq!(std_dev(&values, ddof).unwrap().to_bits(), 0.0_f64.to_bits());
            }
        }
    }

    #[test]
    fn nonfinite_statistics_are_not_treated_as_finite_constants() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for values in [vec![value; 10], vec![0.01, value], vec![value, 0.01]] {
                let expected_mean = values.iter().sum::<f64>() / values.len() as f64;
                let actual_mean = mean(&values).unwrap();
                if expected_mean.is_nan() {
                    assert!(actual_mean.is_nan());
                } else {
                    assert_eq!(actual_mean, expected_mean);
                }
                assert!(variance(&values, 0).unwrap().is_nan());
                assert!(std_dev(&values, 1).unwrap().is_nan());
                assert_eq!(variance(&values, values.len()), None);
            }
        }
        assert!(mean(&[f64::INFINITY, f64::NEG_INFINITY]).unwrap().is_nan());
        assert!(variance(&[f64::INFINITY, f64::NEG_INFINITY], 1).unwrap().is_nan());
    }

    #[test]
    fn nonconstant_statistics_keep_the_existing_arithmetic() {
        let mut nearly_constant = vec![0.01; 10];
        nearly_constant[9] = f64::from_bits(0.01_f64.to_bits() + 1);
        assert!(variance(&nearly_constant, 1).unwrap() > 0.0);
        for values in [
            nearly_constant,
            vec![0.03, 0.02, 0.01, -0.01, 0.02, 0.01, 0.0, -0.01, 0.01],
            vec![f64::MAX, f64::MAX / 2.0],
            vec![1e-300, 2e-300],
        ] {
            let mu = values.iter().sum::<f64>() / values.len() as f64;
            assert_eq!(mean(&values).unwrap().to_bits(), mu.to_bits());
            for ddof in [0, 1] {
                let expected = values.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>()
                    / (values.len() - ddof) as f64;
                let actual = variance(&values, ddof).unwrap();
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
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

    #[test]
    fn covariance_of_a_constant_column_is_exactly_zero() {
        // 0.1 + 0.1 + 0.1 rounds, so `sum / rows` is not 0.1 and left a variance near 1e-34.
        let r = DMatrix::from_row_slice(3, 2, &[0.1, 0.01, 0.1, -0.02, 0.1, 0.03]);
        let cov = covariance(&r).unwrap();
        assert_eq!(cov[(0, 0)], 0.0);
        assert_eq!(cov[(0, 1)], 0.0);
        assert_eq!(cov[(1, 0)], 0.0);
        assert!(cov[(1, 1)] > 0.0);
    }
}
