//! Codependence measures between two series: correlation-based distances, distance
//! correlation, mutual information and variation of information.
//!
//! These follow López de Prado, *Machine Learning for Asset Managers* (2020), chapter 3, and
//! are the distance layer under the hierarchical methods of AFML chapter 16 (see
//! [`crate::hrp`], [`crate::hcaa`] and [`crate::onc`]).
//!
//! - [`angular_distance`], [`absolute_angular_distance`] and [`squared_angular_distance`]
//!   turn a Pearson correlation `rho` into a metric on `[0, 1]`. The first treats
//!   `rho = -1` as maximally distant (long-only books); the other two as identical
//!   (long-short books).
//! - [`distance_correlation`] (Székely et al., 2007) is zero only under independence.
//! - [`get_mutual_info`] and [`variation_of_information_score`] (Meilă, 2007) are estimated
//!   from equal-width histograms, with the bin count from
//!   [`get_optimal_number_of_bins`] (Hacine-Gharbi et al., 2012) when none is given.
//!
//! All functions take two equal-length slices of paired observations and treat the pairs as
//! exchangeable: pass returns or other stationary series, not trending price levels.
//! Entropies use natural logarithms.
//!
//! ```
//! use openquant::codependence::{
//!     absolute_angular_distance, angular_distance, distance_correlation,
//! };
//!
//! # fn main() -> Result<(), openquant::codependence::CodependenceError> {
//! let x: Vec<f64> = (0..=200).map(|i| f64::from(i) / 100.0 - 1.0).collect();
//! let mirrored: Vec<f64> = x.iter().map(|v| -v).collect();
//! let squared: Vec<f64> = x.iter().map(|v| v * v).collect();
//!
//! // rho = -1: maximal angular distance, zero absolute angular distance.
//! assert!((angular_distance(&x, &mirrored)? - 1.0).abs() < 1e-12);
//! assert!(absolute_angular_distance(&x, &mirrored)?.abs() < 1e-7);
//!
//! // y = x^2 on a symmetric range is uncorrelated with x, and clearly dependent on it.
//! assert!((angular_distance(&x, &squared)? - 0.5f64.sqrt()).abs() < 1e-3);
//! assert!(distance_correlation(&x, &squared)? > 0.4);
//! # Ok(())
//! # }
//! ```
#![deny(missing_docs)]

/// Errors returned by the codependence measures.
#[derive(Debug, thiserror::Error)]
pub enum CodependenceError {
    /// The two series have different lengths.
    #[error("the two series have different lengths")]
    InputLengthMismatch,
    /// Too few observations for the measure.
    #[error("the series are too short")]
    InputTooShort,
    /// The bin count is zero, or could not be computed (a `NaN` correlation).
    #[error("the number of bins must be positive")]
    InvalidBins,
    /// A series is constant (zero variance or zero entropy) where the measure divides by it.
    #[error("a series has zero variance")]
    ZeroVariance,
    /// A series is constant, so its distance variance is zero.
    #[error("a series has zero distance variance")]
    ZeroDistanceVariance,
}

/// Result type of this module.
pub type CodependenceResult<T> = Result<T, CodependenceError>;

fn corrcoef(x: &[f64], y: &[f64]) -> CodependenceResult<f64> {
    if x.len() != y.len() {
        return Err(CodependenceError::InputLengthMismatch);
    }
    if x.len() < 2 {
        return Err(CodependenceError::InputTooShort);
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x == 0.0 || var_y == 0.0 {
        return Err(CodependenceError::ZeroVariance);
    }

    Ok(cov / (var_x * var_y).sqrt())
}

fn histogram(values: &[f64], n_bins: usize) -> CodependenceResult<Vec<usize>> {
    if n_bins == 0 {
        return Err(CodependenceError::InvalidBins);
    }

    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    for value in values {
        if *value < min_value {
            min_value = *value;
        }
        if *value > max_value {
            max_value = *value;
        }
    }

    let mut counts = vec![0usize; n_bins];
    if (max_value - min_value).abs() < f64::EPSILON {
        counts[n_bins - 1] = values.len();
        return Ok(counts);
    }

    let bin_width = (max_value - min_value) / n_bins as f64;

    for value in values {
        let mut idx = ((value - min_value) / bin_width).floor() as isize;
        if idx < 0 {
            idx = 0;
        }
        if idx as usize >= n_bins {
            idx = (n_bins as isize) - 1;
        }
        counts[idx as usize] += 1;
    }

    Ok(counts)
}

fn histogram2d(x: &[f64], y: &[f64], n_bins: usize) -> CodependenceResult<Vec<Vec<usize>>> {
    if x.len() != y.len() {
        return Err(CodependenceError::InputLengthMismatch);
    }
    if n_bins == 0 {
        return Err(CodependenceError::InvalidBins);
    }

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        if xi < min_x {
            min_x = xi;
        }
        if xi > max_x {
            max_x = xi;
        }
        if yi < min_y {
            min_y = yi;
        }
        if yi > max_y {
            max_y = yi;
        }
    }

    let mut counts = vec![vec![0usize; n_bins]; n_bins];
    if (max_x - min_x).abs() < f64::EPSILON || (max_y - min_y).abs() < f64::EPSILON {
        for _ in 0..x.len() {
            counts[n_bins - 1][n_bins - 1] += 1;
        }
        return Ok(counts);
    }

    let bin_width_x = (max_x - min_x) / n_bins as f64;
    let bin_width_y = (max_y - min_y) / n_bins as f64;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let mut ix = ((xi - min_x) / bin_width_x).floor() as isize;
        let mut iy = ((yi - min_y) / bin_width_y).floor() as isize;
        if ix < 0 {
            ix = 0;
        }
        if iy < 0 {
            iy = 0;
        }
        if ix as usize >= n_bins {
            ix = (n_bins as isize) - 1;
        }
        if iy as usize >= n_bins {
            iy = (n_bins as isize) - 1;
        }
        counts[ix as usize][iy as usize] += 1;
    }

    Ok(counts)
}

fn entropy(counts: &[usize]) -> CodependenceResult<f64> {
    let total: usize = counts.iter().sum();
    if total == 0 {
        return Err(CodependenceError::InputTooShort);
    }
    let total_f = total as f64;

    let mut value = 0.0;
    for count in counts {
        if *count == 0 {
            continue;
        }
        let p = *count as f64 / total_f;
        value -= p * p.ln();
    }

    Ok(value)
}

/// Angular distance `sqrt((1 - rho) / 2)` from the Pearson correlation `rho` of `x` and `y`.
///
/// In `[0, 1]`: 0 for `rho = 1` and 1 for `rho = -1`. Suited to long-only portfolios, where a
/// negatively correlated asset is a diversifier.
///
/// # Errors
///
/// - [`CodependenceError::InputLengthMismatch`] if `x` and `y` differ in length.
/// - [`CodependenceError::InputTooShort`] if they have fewer than two observations.
/// - [`CodependenceError::ZeroVariance`] if either series is constant.
pub fn angular_distance(x: &[f64], y: &[f64]) -> CodependenceResult<f64> {
    let corr_coef = corrcoef(x, y)?;
    Ok((0.5 * (1.0 - corr_coef)).sqrt())
}

/// Absolute angular distance `sqrt((1 - |rho|) / 2)` from the Pearson correlation of `x` and
/// `y`.
///
/// In `[0, sqrt(1/2)]`, and 0 for `rho = +-1`: perfectly anti-correlated series are treated
/// as identical, as suits long-short portfolios.
///
/// # Errors
///
/// - [`CodependenceError::InputLengthMismatch`] if `x` and `y` differ in length.
/// - [`CodependenceError::InputTooShort`] if they have fewer than two observations.
/// - [`CodependenceError::ZeroVariance`] if either series is constant.
pub fn absolute_angular_distance(x: &[f64], y: &[f64]) -> CodependenceResult<f64> {
    let corr_coef = corrcoef(x, y)?;
    Ok((0.5 * (1.0 - corr_coef.abs())).sqrt())
}

/// Squared angular distance `sqrt((1 - rho^2) / 2)` from the Pearson correlation of `x` and
/// `y`.
///
/// Like [`absolute_angular_distance`], 0 for `rho = +-1`; it spreads out high correlations
/// and compresses low ones.
///
/// # Errors
///
/// - [`CodependenceError::InputLengthMismatch`] if `x` and `y` differ in length.
/// - [`CodependenceError::InputTooShort`] if they have fewer than two observations.
/// - [`CodependenceError::ZeroVariance`] if either series is constant.
pub fn squared_angular_distance(x: &[f64], y: &[f64]) -> CodependenceResult<f64> {
    let corr_coef = corrcoef(x, y)?;
    Ok((0.5 * (1.0 - corr_coef.powi(2))).sqrt())
}

/// Distance correlation of `x` and `y` (Székely et al., 2007).
///
/// Double-centres the matrices of pairwise absolute differences within each series and
/// returns `dCov(x, y) / sqrt(dVar(x) dVar(y))`, in `[0, 1]`, which is zero only when the
/// series are independent. Builds two `n x n` matrices: memory is `O(n^2)` (about 16 MB at
/// 1,000 observations).
///
/// # Errors
///
/// - [`CodependenceError::InputLengthMismatch`] if `x` and `y` differ in length.
/// - [`CodependenceError::InputTooShort`] if they have fewer than two observations.
/// - [`CodependenceError::ZeroDistanceVariance`] if either series is constant.
pub fn distance_correlation(x: &[f64], y: &[f64]) -> CodependenceResult<f64> {
    if x.len() != y.len() {
        return Err(CodependenceError::InputLengthMismatch);
    }
    let n = x.len();
    if n < 2 {
        return Err(CodependenceError::InputTooShort);
    }

    let mut a = vec![0.0; n * n];
    let mut b = vec![0.0; n * n];

    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = (x[i] - x[j]).abs();
            b[i * n + j] = (y[i] - y[j]).abs();
        }
    }

    let mut row_mean_a = vec![0.0; n];
    let mut col_mean_a = vec![0.0; n];
    let mut row_mean_b = vec![0.0; n];
    let mut col_mean_b = vec![0.0; n];

    for i in 0..n {
        let mut sum_a = 0.0;
        let mut sum_b = 0.0;
        for j in 0..n {
            sum_a += a[i * n + j];
            sum_b += b[i * n + j];
        }
        row_mean_a[i] = sum_a / n as f64;
        row_mean_b[i] = sum_b / n as f64;
    }

    for j in 0..n {
        let mut sum_a = 0.0;
        let mut sum_b = 0.0;
        for i in 0..n {
            sum_a += a[i * n + j];
            sum_b += b[i * n + j];
        }
        col_mean_a[j] = sum_a / n as f64;
        col_mean_b[j] = sum_b / n as f64;
    }

    let mean_a = a.iter().sum::<f64>() / (n * n) as f64;
    let mean_b = b.iter().sum::<f64>() / (n * n) as f64;

    let mut d_cov_xx = 0.0;
    let mut d_cov_xy = 0.0;
    let mut d_cov_yy = 0.0;

    for i in 0..n {
        for j in 0..n {
            let a_centered = a[i * n + j] - row_mean_a[i] - col_mean_a[j] + mean_a;
            let b_centered = b[i * n + j] - row_mean_b[i] - col_mean_b[j] + mean_b;
            d_cov_xx += a_centered * a_centered;
            d_cov_xy += a_centered * b_centered;
            d_cov_yy += b_centered * b_centered;
        }
    }

    let denom = (n * n) as f64;
    d_cov_xx /= denom;
    d_cov_xy /= denom;
    d_cov_yy /= denom;

    let denom = (d_cov_xx.sqrt() * d_cov_yy.sqrt()).sqrt();
    if denom == 0.0 {
        return Err(CodependenceError::ZeroDistanceVariance);
    }

    Ok(d_cov_xy.sqrt() / denom)
}

/// Histogram bin count that minimises the bias of entropy estimates (Hacine-Gharbi et al.,
/// 2012).
///
/// With `corr_coef = None` uses the univariate (marginal entropy) rule; with the sample
/// correlation `rho` of two series uses the bivariate (joint entropy) rule
/// `round(sqrt(1 + sqrt(1 + 24 N / (1 - rho^2))) / sqrt(2))`. A correlation within `1e-4` of
/// `+-1` falls back to the univariate rule.
///
/// # Errors
///
/// - [`CodependenceError::InputTooShort`] if `num_obs` is zero.
/// - [`CodependenceError::InvalidBins`] if the rule does not yield a positive count (a `NaN`
///   correlation).
///
/// ```
/// use openquant::codependence::get_optimal_number_of_bins;
///
/// # fn main() -> Result<(), openquant::codependence::CodependenceError> {
/// assert_eq!(get_optimal_number_of_bins(1_000, None)?, 15);
/// assert_eq!(get_optimal_number_of_bins(1_000, Some(0.9))?, 13);
/// # Ok(())
/// # }
/// ```
pub fn get_optimal_number_of_bins(
    num_obs: usize,
    corr_coef: Option<f64>,
) -> CodependenceResult<usize> {
    if num_obs == 0 {
        return Err(CodependenceError::InputTooShort);
    }

    let n = num_obs as f64;
    let univariate = || {
        let z = (8.0 + 324.0 * n + 12.0 * (36.0 * n + 729.0 * n * n).sqrt()).cbrt();
        (z / 6.0 + 2.0 / (3.0 * z) + 1.0 / 3.0).round()
    };
    // Arm order keeps a NaN correlation on the bivariate branch, as before.
    // At |corr| = 1 the bivariate formula divides by zero, so both signs fall back.
    let bins = match corr_coef {
        None => univariate(),
        Some(corr) if (corr.abs() - 1.0).abs() <= 1e-4 => univariate(),
        Some(corr) => {
            let inner = (1.0 + 24.0 * n / (1.0 - corr * corr)).sqrt();
            (2.0_f64).powf(-0.5) * (1.0 + inner).sqrt()
        }
    };

    let bins = bins.round() as isize;
    if bins <= 0 {
        return Err(CodependenceError::InvalidBins);
    }
    Ok(bins as usize)
}

/// Mutual information `I[X;Y] = H[X] + H[Y] - H[X,Y]` of `x` and `y`, estimated from an
/// `n_bins x n_bins` equal-width histogram.
///
/// `n_bins = None` uses [`get_optimal_number_of_bins`] with the sample correlation. With
/// `normalize = true` the result is divided by `min(H[X], H[Y])` and lies in `[0, 1]`. The
/// histogram estimate is biased upward on small samples; compare values only at equal length
/// and binning. Normalised mutual information is not a distance; use
/// [`variation_of_information_score`] for clustering.
///
/// # Errors
///
/// - [`CodependenceError::InputLengthMismatch`] if `x` and `y` differ in length.
/// - [`CodependenceError::InputTooShort`] if they are empty, or (with `n_bins = None`) have
///   fewer than two observations.
/// - [`CodependenceError::InvalidBins`] if `n_bins` is `Some(0)` or the bin rule fails.
/// - [`CodependenceError::ZeroVariance`] if `n_bins = None` and a series is constant, or
///   `normalize` is true and either marginal entropy is zero.
pub fn get_mutual_info(
    x: &[f64],
    y: &[f64],
    n_bins: Option<usize>,
    normalize: bool,
) -> CodependenceResult<f64> {
    if x.len() != y.len() {
        return Err(CodependenceError::InputLengthMismatch);
    }
    if x.is_empty() {
        return Err(CodependenceError::InputTooShort);
    }

    let bins = if let Some(bins) = n_bins {
        bins
    } else {
        let corr = corrcoef(x, y)?;
        get_optimal_number_of_bins(x.len(), Some(corr))?
    };

    let contingency = histogram2d(x, y, bins)?;
    let total: usize = contingency.iter().map(|row| row.iter().sum::<usize>()).sum();
    if total == 0 {
        return Err(CodependenceError::InputTooShort);
    }
    let total_f = total as f64;

    let mut row_sums = vec![0.0; bins];
    let mut col_sums = vec![0.0; bins];
    for i in 0..bins {
        for (j, col_sum) in col_sums.iter_mut().enumerate() {
            let value = contingency[i][j] as f64;
            row_sums[i] += value;
            *col_sum += value;
        }
    }

    let mut mutual_info = 0.0;
    for i in 0..bins {
        for (j, col_sum) in col_sums.iter().enumerate() {
            let value = contingency[i][j] as f64;
            if value == 0.0 {
                continue;
            }
            let p_ij = value / total_f;
            let p_i = row_sums[i] / total_f;
            let p_j = col_sum / total_f;
            mutual_info += p_ij * (p_ij / (p_i * p_j)).ln();
        }
    }

    if normalize {
        let marginal_x = entropy(&histogram(x, bins)?)?;
        let marginal_y = entropy(&histogram(y, bins)?)?;
        let denom = marginal_x.min(marginal_y);
        if denom == 0.0 {
            return Err(CodependenceError::ZeroVariance);
        }
        mutual_info /= denom;
    }

    Ok(mutual_info)
}

/// Variation of information `VI[X;Y] = H[X] + H[Y] - 2 I[X;Y]` of `x` and `y` (Meilă, 2007),
/// estimated from equal-width histograms.
///
/// A true metric: the uncertainty left in each variable once the other is known. With
/// `normalize = true` it is divided by the joint entropy `H[X,Y]` and lies in `[0, 1]`, with 0
/// meaning each variable determines the other. `n_bins = None` uses
/// [`get_optimal_number_of_bins`] with the sample correlation.
///
/// # Errors
///
/// - [`CodependenceError::InputLengthMismatch`] if `x` and `y` differ in length.
/// - [`CodependenceError::InputTooShort`] if they are empty, or (with `n_bins = None`) have
///   fewer than two observations.
/// - [`CodependenceError::InvalidBins`] if `n_bins` is `Some(0)` or the bin rule fails.
/// - [`CodependenceError::ZeroVariance`] if `n_bins = None` and a series is constant, or
///   `normalize` is true and the joint entropy is zero.
pub fn variation_of_information_score(
    x: &[f64],
    y: &[f64],
    n_bins: Option<usize>,
    normalize: bool,
) -> CodependenceResult<f64> {
    if x.len() != y.len() {
        return Err(CodependenceError::InputLengthMismatch);
    }
    if x.is_empty() {
        return Err(CodependenceError::InputTooShort);
    }

    let bins = if let Some(bins) = n_bins {
        bins
    } else {
        let corr = corrcoef(x, y)?;
        get_optimal_number_of_bins(x.len(), Some(corr))?
    };

    let contingency = histogram2d(x, y, bins)?;
    let total: usize = contingency.iter().map(|row| row.iter().sum::<usize>()).sum();
    if total == 0 {
        return Err(CodependenceError::InputTooShort);
    }
    let total_f = total as f64;

    let mut row_sums = vec![0.0; bins];
    let mut col_sums = vec![0.0; bins];
    for i in 0..bins {
        for (j, col_sum) in col_sums.iter_mut().enumerate() {
            let value = contingency[i][j] as f64;
            row_sums[i] += value;
            *col_sum += value;
        }
    }

    let mut mutual_info = 0.0;
    for i in 0..bins {
        for (j, col_sum) in col_sums.iter().enumerate() {
            let value = contingency[i][j] as f64;
            if value == 0.0 {
                continue;
            }
            let p_ij = value / total_f;
            let p_i = row_sums[i] / total_f;
            let p_j = col_sum / total_f;
            mutual_info += p_ij * (p_ij / (p_i * p_j)).ln();
        }
    }

    let marginal_x = entropy(&histogram(x, bins)?)?;
    let marginal_y = entropy(&histogram(y, bins)?)?;
    let mut score = marginal_x + marginal_y - 2.0 * mutual_info;

    if normalize {
        let joint_dist = marginal_x + marginal_y - mutual_info;
        if joint_dist == 0.0 {
            return Err(CodependenceError::ZeroVariance);
        }
        score /= joint_dist;
    }

    Ok(score)
}
