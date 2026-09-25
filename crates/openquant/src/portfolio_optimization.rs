//! Mean-variance portfolio allocation with weight bounds: inverse variance, minimum
//! volatility, maximum Sharpe ratio, and minimum risk for a target return.
//!
//! References: Markowitz (1952), *Portfolio selection*; AFML Chapter 16, §16.2 (the problem
//! with convex portfolio optimisation) and §16.3 (Markowitz's curse) for why these portfolios
//! are fragile; Stellato et al. (2020), OSQP, for the solver formulation; Michaud (1989) on
//! error maximisation.
//!
//! | `solution` | Problem |
//! | --- | --- |
//! | `"inverse_variance"` | `w_i ∝ 1 / Sigma_ii` (correlation ignored), then projected onto the bounds |
//! | `"min_volatility"` | `min w'Σw` s.t. `1'w = 1`, `l <= w <= u` |
//! | `"max_sharpe"` | maximise `(mu'w - rf) / sqrt(w'Σw)` s.t. `1'w = 1`, `l <= w <= u` |
//! | `"efficient_risk"` | `min w'Σw` s.t. `mu'w >= target_return`, `1'w = 1`, `l <= w <= u` |
//!
//! The three optimisations are solved as quadratic programmes by an internal dense ADMM
//! solver followed by an exact solve on the active set; **bounds are part of the problem**,
//! not applied afterwards. Maximum Sharpe uses the substitution `y = kappa w`: minimise `y'Σy`
//! subject to `(mu - rf)'y = 1` and `l_i 1'y <= y_i <= u_i 1'y`, then `w = y / 1'y`.
//!
//! Conventions:
//! - Weights are fully invested (`sum w = 1`) and long-only by default (`0 <= w_i <= 1`).
//!   [`AllocationOptions::tuple_bounds`] sets `(lo, hi)` for every asset and
//!   [`AllocationOptions::bounds`] overrides it per asset index. Upper bounds above 1 are
//!   treated as 1.
//! - Price matrices are `T x N`: rows are dates in ascending order, columns are assets. From
//!   prices, returns are **simple** returns `p_t / p_{t-1} - 1`, and the expected returns
//!   **and** the covariance are annualised by `252 / step` (`step` = 1, 5 or 21 with
//!   `resample_by`), so `risk_free_rate` and `target_return` are annual figures and
//!   `portfolio_sharpe` is an annual Sharpe ratio.
//! - With [`allocate_from_inputs`] the units are the caller's: `mu`, `Σ` and
//!   `risk_free_rate` must agree (an annual `mu` with a daily `Σ` overstates the Sharpe ratio
//!   by `sqrt(252)`). Weights do not depend on the scale of `Σ`.
//! - Everything is in-sample: a maximum-Sharpe portfolio's `portfolio_sharpe` is the best ratio
//!   on the history it was fitted to. The covariance is the plain sample covariance; shrink or
//!   denoise it for many assets. The solver is dense and meant for tens of assets.
//!
//! ```
//! use nalgebra::DMatrix;
//! use openquant::portfolio_optimization::{allocate_from_inputs, AllocError, AllocationOptions};
//!
//! # fn main() -> Result<(), AllocError> {
//! let mu = [0.03, 0.07, 0.09, 0.04];
//! let vol = [0.05, 0.16, 0.22, 0.15];
//! let rho = [[1.0, 0.1, 0.1, 0.1], [0.1, 1.0, 0.8, 0.0], [0.1, 0.8, 1.0, 0.0], [0.1, 0.0, 0.0, 1.0]];
//! let cov = DMatrix::from_fn(4, 4, |i, j| rho[i][j] * vol[i] * vol[j]);
//!
//! let min_vol = allocate_from_inputs(&mu, &cov, "min_volatility", &AllocationOptions::default())?;
//! assert!((min_vol.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
//! assert!(min_vol.weights.iter().all(|w| *w > -1e-9));
//! assert!((min_vol.portfolio_risk - 0.0476).abs() < 1e-4);
//!
//! // A return target above the minimum-variance return binds exactly, and costs risk.
//! let target = AllocationOptions { target_return: 0.065, ..AllocationOptions::default() };
//! let efficient = allocate_from_inputs(&mu, &cov, "efficient_risk", &target)?;
//! assert!((efficient.portfolio_return - 0.065).abs() < 1e-7);
//! assert!(efficient.portfolio_risk > min_vol.portfolio_risk);
//!
//! // With a 35% cap the most any portfolio can return is 6.8%, so 7% is infeasible...
//! let capped = AllocationOptions {
//!     target_return: 0.07,
//!     tuple_bounds: Some((0.0, 0.35)),
//!     ..AllocationOptions::default()
//! };
//! assert!(matches!(
//!     allocate_from_inputs(&mu, &cov, "efficient_risk", &capped),
//!     Err(AllocError::OptimizationFailed(_))
//! ));
//! // ...and bounds that cannot sum to one are rejected before solving.
//! let impossible = AllocationOptions { tuple_bounds: Some((0.3, 0.4)), ..AllocationOptions::default() };
//! assert!(matches!(
//!     allocate_from_inputs(&mu, &cov, "min_volatility", &impossible),
//!     Err(AllocError::InfeasibleBounds { .. })
//! ));
//! # Ok(())
//! # }
//! ```
#![deny(missing_docs)]

use nalgebra::{DMatrix, DVector};

use crate::util::qp::{solve_qp, QpError};
use crate::util::resample::{freq_step, resample_prices};
use std::collections::HashMap;

/// Errors returned by the allocation functions.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum AllocError {
    /// The price matrix has fewer than two rows (after resampling), so no return can be formed.
    #[error("no data: supply asset prices, or expected returns and a covariance matrix")]
    NoData,
    /// `solution` is not one of `"inverse_variance"`, `"min_volatility"`, `"max_sharpe"`,
    /// `"efficient_risk"`.
    #[error("unknown solution: {0}")]
    UnknownSolution(String),
    /// [`returns_method_from_str`] was given an unknown name (the lower-cased name is carried).
    #[error("unknown returns method: {0}")]
    UnknownReturns(String),
    /// The lower bounds sum above 1 or the (capped) upper bounds sum below 1, so no fully
    /// invested portfolio satisfies them. Also returned for an empty asset set, or when the
    /// inverse-variance projection cannot place the weight.
    #[error("weight bounds cannot sum to 1: lower bounds sum to {lower_sum}, upper bounds to {upper_sum}")]
    InfeasibleBounds {
        /// Sum of the lower bounds.
        lower_sum: f64,
        /// Sum of the upper bounds, each capped at 1.
        upper_sum: f64,
    },
    /// Asset `asset`'s bounds are unusable: a bound is `NaN`, or the lower bound exceeds the
    /// upper bound (after capping it at 1). Checked before any solution runs.
    #[error("invalid weight bounds for asset {asset}: lower {lower}, upper {upper}")]
    InvalidBounds {
        /// Column index of the asset.
        asset: usize,
        /// Its lower bound.
        lower: f64,
        /// Its upper bound, as given.
        upper: f64,
    },
    /// The solver or a solution-specific precondition failed; the message says which:
    /// `"no portfolio satisfies the constraints"` (infeasible constraints such as an
    /// unreachable `target_return`, or no convergence), `"covariance is not positive
    /// definite"` (malformed problem),
    /// `"no asset has a return above the risk-free rate"` (`"max_sharpe"`), or a zero
    /// covariance diagonal (`"inverse_variance"`).
    #[error("optimization failed: {0}")]
    OptimizationFailed(&'static str),
    /// `expected_returns` and `covariance` disagree on the number of assets, or the covariance
    /// is not square.
    #[error("inputs disagree on the number of assets")]
    DimensionMismatch,
    /// A non-finite result: a zero price in a return denominator, a non-finite portfolio risk,
    /// or a degenerate maximum-Sharpe solution (`1'y` not positive).
    #[error("result is NaN: {0}")]
    NaNResult(&'static str),
}

/// How expected returns are estimated from a price history.
#[derive(Clone, Copy, Default)]
pub enum ReturnsMethod {
    /// Arithmetic mean of the per-period simple returns, annualised.
    #[default]
    Mean,
    /// Exponentially weighted mean of the per-period simple returns, newest weighted most, with
    /// decay `alpha = 2 / (span + 1)`, annualised.
    Exponential {
        /// Span of the exponential weighting, in periods (after resampling). Not validated:
        /// `span = 0` gives `alpha = 2` and alternating-sign weights.
        span: usize,
    },
}

/// Options shared by the allocation functions.
///
/// [`Default`] gives a zero risk-free rate, a 1% target return, long-only `[0, 1]` bounds, no
/// resampling and [`ReturnsMethod::Mean`].
#[derive(Clone)]
pub struct AllocationOptions<'a> {
    /// Risk-free rate for the Sharpe ratio and for `"max_sharpe"`; annual when allocating from
    /// prices, otherwise in the units of the expected returns.
    pub risk_free_rate: f64,
    /// Minimum expected return for `"efficient_risk"` (a floor, not an equality); annual when
    /// allocating from prices. Ignored by the other solutions.
    pub target_return: f64,
    /// Per-asset `(lower, upper)` weight bounds keyed by column index; takes precedence over
    /// `tuple_bounds`. Indices not present fall back to `tuple_bounds`; out-of-range keys are
    /// ignored.
    pub bounds: Option<HashMap<usize, (f64, f64)>>,
    /// `(lower, upper)` weight bounds for every asset not in `bounds`; `None` means `(0, 1)`.
    /// Upper bounds above 1 are treated as 1; negative lower bounds allow shorting.
    pub tuple_bounds: Option<(f64, f64)>,
    /// Resampling of prices before returns are formed: `"W"`/`"week"`/`"weekly"` keeps every
    /// 5th row, `"M"`/`"month"`/`"monthly"` every 21st (the last row of each block;
    /// case-insensitive). Anything else, or `None`, keeps every row. Ignored by
    /// [`allocate_from_inputs`].
    pub resample_by: Option<&'a str>,
    /// Expected-return estimator used when allocating from prices.
    pub returns_method: ReturnsMethod,
}

impl Default for AllocationOptions<'_> {
    fn default() -> Self {
        AllocationOptions {
            risk_free_rate: 0.0,
            target_return: 0.01,
            bounds: None,
            tuple_bounds: None,
            resample_by: None,
            returns_method: ReturnsMethod::Mean,
        }
    }
}

/// An allocation and its statistics under the inputs it was solved with.
///
/// From prices, every figure is annualised by the same factor, `252 / step` periods a year:
/// `portfolio_return` is `mu'w` with `mu` the annualised mean simple return, `portfolio_risk` is
/// `sqrt(w' Sigma w)` with `Sigma` the annualised covariance, and `portfolio_sharpe` is
/// `(portfolio_return - risk_free_rate) / portfolio_risk`, so `risk_free_rate` is an annual rate.
/// From `allocate_from_inputs` the units are the caller's. `portfolio_sharpe` is computed for
/// every solution, and is 0 only when the risk is 0.
#[derive(Debug, Clone)]
pub struct MeanVariance {
    /// Portfolio weights, one per asset in column order, summing to 1.
    pub weights: Vec<f64>,
    /// Portfolio volatility `sqrt(w' Sigma w)`.
    pub portfolio_risk: f64,
    /// Portfolio expected return `mu' w`.
    pub portfolio_return: f64,
    /// `(portfolio_return - risk_free_rate) / portfolio_risk`, or 0 when the risk is 0.
    pub portfolio_sharpe: f64,
}

/// Parse an expected-returns method name (case-insensitive): `"mean"` or `"mean_historical"`
/// give [`ReturnsMethod::Mean`]; `"exponential"` or `"exponential_historical"` give
/// [`ReturnsMethod::Exponential`] with `span = 500`.
///
/// # Errors
///
/// [`AllocError::UnknownReturns`] for any other name.
///
/// ```
/// use openquant::portfolio_optimization::{returns_method_from_str, AllocError, ReturnsMethod};
///
/// assert!(matches!(returns_method_from_str("Mean"), Ok(ReturnsMethod::Mean)));
/// assert!(matches!(
///     returns_method_from_str("exponential_historical"),
///     Ok(ReturnsMethod::Exponential { span: 500 })
/// ));
/// assert!(matches!(
///     returns_method_from_str("median"),
///     Err(AllocError::UnknownReturns(name)) if name == "median"
/// ));
/// ```
pub fn returns_method_from_str(name: &str) -> Result<ReturnsMethod, AllocError> {
    match name.to_lowercase().as_str() {
        "mean" | "mean_historical" => Ok(ReturnsMethod::Mean),
        "exponential" | "exponential_historical" => Ok(ReturnsMethod::Exponential { span: 500 }),
        other => Err(AllocError::UnknownReturns(other.to_string())),
    }
}

/// Simple returns `p_t / p_{t-1} - 1`, the convention of `cla`, `hrp` and `hcaa`. A portfolio's
/// simple return is the weighted sum of its assets' simple returns, which is what a one-period
/// mean-variance problem on weights assumes; the same is not true of log returns.
fn returns_from_prices(prices: &DMatrix<f64>) -> Result<DMatrix<f64>, AllocError> {
    let rows = prices.nrows();
    let cols = prices.ncols();
    if rows < 2 {
        return Err(AllocError::NoData);
    }
    let mut out = DMatrix::<f64>::zeros(rows - 1, cols);
    for r in 1..rows {
        for c in 0..cols {
            let prev = prices[(r - 1, c)];
            if prev == 0.0 {
                return Err(AllocError::NaNResult("price contained zero"));
            }
            out[(r - 1, c)] = prices[(r, c)] / prev - 1.0;
        }
    }
    Ok(out)
}

/// Annualised expected simple returns and their annualised sample covariance: the inputs the
/// price-based allocators solve with. Both are scaled by `252 / step`, so passing the pair to
/// [`allocate_from_inputs`] gives the same result as [`allocate_with_solution`] on the prices.
///
/// `prices` is `T x N` (rows ascending in time). Returns `(mu, Sigma)`: `mu` has `N` entries
/// from `returns_method`, `Sigma` is the `N x N` sample covariance (denominator `T' - 1` for
/// `T'` returns; all zeros when there is only one return).
///
/// # Errors
///
/// - [`AllocError::NoData`] if fewer than two price rows remain after resampling.
/// - [`AllocError::NaNResult`] if a price used as a return denominator is zero.
///
/// ```
/// use nalgebra::DMatrix;
/// use openquant::portfolio_optimization::{compute_expected_and_covariance, ReturnsMethod};
///
/// // Asset 0 returns +10%, +10%; asset 1 returns +20%, -10%.
/// let prices = DMatrix::from_row_slice(3, 2, &[100.0, 100.0, 110.0, 120.0, 121.0, 108.0]);
/// let (mu, cov) = compute_expected_and_covariance(&prices, ReturnsMethod::Mean, None).unwrap();
/// assert!((mu[0] - 0.10 * 252.0).abs() < 1e-9);
/// assert!((mu[1] - 0.05 * 252.0).abs() < 1e-9);
/// // Sample variance of (0.2, -0.1) is 0.045, annualised.
/// assert!((cov[(1, 1)] - 0.045 * 252.0).abs() < 1e-9);
/// assert!(cov[(0, 0)].abs() < 1e-12);
/// ```
pub fn compute_expected_and_covariance(
    prices: &DMatrix<f64>,
    returns_method: ReturnsMethod,
    resample_by: Option<&str>,
) -> Result<(Vec<f64>, DMatrix<f64>), AllocError> {
    let opts = AllocationOptions { returns_method, resample_by, ..Default::default() };
    returns_and_means(prices, &opts)
}

fn returns_and_means(
    prices: &DMatrix<f64>,
    opts: &AllocationOptions,
) -> Result<(Vec<f64>, DMatrix<f64>), AllocError> {
    let step = freq_step(opts.resample_by);
    let sampled_prices = resample_prices(prices, step);
    let returns = returns_from_prices(&sampled_prices)?;
    let rows = returns.nrows();
    if rows == 0 {
        return Err(AllocError::NoData);
    }
    let cols = returns.ncols();
    let freq = 252.0 / step as f64;
    let mut expected = vec![0.0; cols];
    match opts.returns_method {
        ReturnsMethod::Mean => {
            for (c, slot) in expected.iter_mut().enumerate() {
                *slot = (returns.column(c).sum() / rows as f64) * freq;
            }
        }
        ReturnsMethod::Exponential { span } => {
            let alpha = 2.0 / (span as f64 + 1.0);
            for c in 0..cols {
                let mut weight = 1.0;
                let mut num = 0.0;
                let mut denom = 0.0;
                for r in (0..rows).rev() {
                    num += weight * returns[(r, c)];
                    denom += weight;
                    weight *= 1.0 - alpha;
                }
                if denom > 0.0 {
                    expected[c] = (num / denom) * freq;
                }
            }
        }
    }
    // The covariance is annualised by the same factor as the means, so the reported risk and
    // Sharpe ratio are in one unit. Scaling it by a constant moves none of the optimisers.
    Ok((expected, covariance(&returns) * freq))
}

fn covariance(returns: &DMatrix<f64>) -> DMatrix<f64> {
    let rows = returns.nrows();
    let cols = returns.ncols();
    if rows < 2 {
        return DMatrix::<f64>::zeros(cols, cols);
    }
    let mut cov = DMatrix::<f64>::zeros(cols, cols);
    let means: Vec<f64> = (0..cols).map(|c| returns.column(c).sum() / rows as f64).collect();
    for i in 0..cols {
        for j in i..cols {
            let mut s = 0.0;
            for r in 0..rows {
                let di = returns[(r, i)] - means[i];
                let dj = returns[(r, j)] - means[j];
                s += di * dj;
            }
            s /= (rows - 1) as f64;
            cov[(i, j)] = s;
            cov[(j, i)] = s;
        }
    }
    cov
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn quad_risk(cov: &DMatrix<f64>, w: &[f64]) -> f64 {
    let wv = DVector::from_vec(w.to_vec());
    (wv.transpose() * cov * wv)[(0, 0)]
}

fn build_bounds(
    n: usize,
    bounds: &Option<HashMap<usize, (f64, f64)>>,
    tuple_bounds: Option<(f64, f64)>,
) -> Vec<(f64, f64)> {
    let default = tuple_bounds.unwrap_or((0.0, 1.0));
    (0..n).map(|i| bounds.as_ref().and_then(|m| m.get(&i)).copied().unwrap_or(default)).collect()
}

fn check_bounds_feasible(bounds: &[(f64, f64)]) -> Result<(), AllocError> {
    for (asset, &(lo, hi)) in bounds.iter().enumerate() {
        // `f64::min` drops a NaN `hi`, so test it first; `!(lo <= upper)` also catches NaN `lo`.
        let upper = hi.min(1.0);
        if hi.is_nan() || !(lo <= upper) {
            return Err(AllocError::InvalidBounds { asset, lower: lo, upper: hi });
        }
    }
    let lower: f64 = bounds.iter().map(|b| b.0).sum();
    let upper: f64 = bounds.iter().map(|b| b.1.min(1.0)).sum();
    if lower - 1.0 > 1e-9 || upper + 1e-9 < 1.0 {
        Err(AllocError::InfeasibleBounds { lower_sum: lower, upper_sum: upper })
    } else {
        Ok(())
    }
}

#[allow(dead_code)]
fn initial_feasible(bounds: &[(f64, f64)]) -> Result<Vec<f64>, AllocError> {
    check_bounds_feasible(bounds)?;
    let mut weights: Vec<f64> = bounds.iter().map(|b| b.0).collect();
    let remaining = 1.0 - weights.iter().sum::<f64>();
    if remaining < -1e-9 {
        return Err(AllocError::InfeasibleBounds {
            lower_sum: 1.0 - remaining,
            upper_sum: 1.0 - remaining,
        });
    }
    if remaining > 0.0 {
        let capacities: Vec<f64> =
            bounds.iter().zip(weights.iter()).map(|(b, w)| b.1.min(1.0) - *w).collect();
        let total_cap: f64 = capacities.iter().sum();
        if total_cap <= 0.0 {
            return Err(AllocError::InfeasibleBounds { lower_sum: 1.0, upper_sum: 0.0 });
        }
        for i in 0..weights.len() {
            weights[i] += remaining * capacities[i] / total_cap;
        }
    }
    Ok(weights)
}

fn project_to_bounds(weights: &mut [f64], bounds: &[(f64, f64)]) -> Result<(), AllocError> {
    for (w, (lo, hi)) in weights.iter_mut().zip(bounds.iter()) {
        *w = w.clamp(*lo, hi.min(1.0));
    }
    let sum: f64 = weights.iter().sum();
    if (sum - 1.0).abs() < 1e-12 {
        return Ok(());
    }
    if sum < 1.0 {
        let deficit = 1.0 - sum;
        let capacities: Vec<f64> =
            bounds.iter().zip(weights.iter()).map(|(b, w)| b.1.min(1.0) - *w).collect();
        let total_cap: f64 = capacities.iter().sum();
        if total_cap <= 1e-12 {
            return Err(AllocError::InfeasibleBounds { lower_sum: 1.0, upper_sum: 0.0 });
        }
        for i in 0..weights.len() {
            weights[i] += deficit * capacities[i] / total_cap;
        }
    } else {
        let excess = sum - 1.0;
        let removable: Vec<f64> =
            bounds.iter().zip(weights.iter()).map(|(b, w)| (w - b.0).max(0.0)).collect();
        let total_rm: f64 = removable.iter().sum();
        if total_rm <= 1e-12 {
            return Err(AllocError::InfeasibleBounds { lower_sum: 1.0, upper_sum: 0.0 });
        }
        for i in 0..weights.len() {
            weights[i] -= excess * removable[i] / total_rm;
        }
    }
    Ok(())
}

fn inverse_variance(cov: &DMatrix<f64>, bounds: &[(f64, f64)]) -> Result<Vec<f64>, AllocError> {
    check_bounds_feasible(bounds)?;
    let diag = cov.diagonal();
    if diag.iter().any(|v| *v == 0.0) {
        return Err(AllocError::OptimizationFailed("covariance contained zero on diagonal"));
    }
    let mut ivp: Vec<f64> = diag.iter().map(|v| 1.0 / v).collect();
    let sum: f64 = ivp.iter().sum();
    if sum == 0.0 {
        return Err(AllocError::OptimizationFailed("zero inverse variance sum"));
    }
    for v in ivp.iter_mut() {
        *v /= sum;
    }
    project_to_bounds(&mut ivp, bounds)?;
    Ok(ivp)
}

/// Budget row `1'w = 1` followed by one box row per asset.
fn budget_and_box(bounds: &[(f64, f64)]) -> (DMatrix<f64>, Vec<f64>, Vec<f64>) {
    let n = bounds.len();
    let mut a = DMatrix::zeros(n + 1, n);
    let (mut lower, mut upper) = (vec![1.0], vec![1.0]);
    for (j, (lo, hi)) in bounds.iter().enumerate() {
        a[(0, j)] = 1.0;
        a[(j + 1, j)] = 1.0;
        lower.push(*lo);
        upper.push(hi.min(1.0));
    }
    (a, lower, upper)
}

fn qp_failure(err: QpError) -> AllocError {
    match err {
        QpError::Malformed => AllocError::OptimizationFailed("covariance is not positive definite"),
        QpError::NotConverged => {
            AllocError::OptimizationFailed("no portfolio satisfies the constraints")
        }
    }
}

/// `min w'Cw` subject to the budget and the bounds. The bounds are part of the problem: the
/// unconstrained closed form shorts assets, and clamping it afterwards is not the long-only
/// optimum.
fn solve_min_vol(cov: &DMatrix<f64>, bounds: &[(f64, f64)]) -> Result<Vec<f64>, AllocError> {
    check_bounds_feasible(bounds)?;
    if cov.nrows() == 0 {
        return Err(AllocError::NoData);
    }
    let (a, lower, upper) = budget_and_box(bounds);
    solve_qp(cov, &a, &lower, &upper).map_err(qp_failure)
}

/// Maximum Sharpe ratio by the usual homogenising substitution `y = kappa * w`, `kappa > 0`:
/// `min y'Cy` subject to `(mu - rf)'y = 1` and `lo_i * 1'y <= y_i <= hi_i * 1'y`, which keeps
/// the bounds linear in `y`; then `w = y / 1'y`.
fn solve_max_sharpe(
    cov: &DMatrix<f64>,
    exp_ret: &[f64],
    risk_free: f64,
    bounds: &[(f64, f64)],
) -> Result<Vec<f64>, AllocError> {
    check_bounds_feasible(bounds)?;
    let n = cov.nrows();
    if n == 0 || exp_ret.len() != n {
        return Err(AllocError::DimensionMismatch);
    }
    let excess: Vec<f64> = exp_ret.iter().map(|r| r - risk_free).collect();
    if excess.iter().all(|e| *e <= 0.0) {
        return Err(AllocError::OptimizationFailed(
            "no asset has a return above the risk-free rate",
        ));
    }

    let mut a = DMatrix::zeros(2 * n + 1, n);
    let (mut lower, mut upper) = (vec![1.0], vec![1.0]);
    for j in 0..n {
        a[(0, j)] = excess[j];
    }
    for (i, (lo, hi)) in bounds.iter().enumerate() {
        for j in 0..n {
            let unit = if i == j { 1.0 } else { 0.0 };
            a[(1 + i, j)] = unit - lo; // y_i - lo * 1'y >= 0
            a[(1 + n + i, j)] = unit - hi.min(1.0); // y_i - hi * 1'y <= 0
        }
    }
    lower.extend(std::iter::repeat_n(0.0, n));
    upper.extend(std::iter::repeat_n(f64::INFINITY, n));
    lower.extend(std::iter::repeat_n(f64::NEG_INFINITY, n));
    upper.extend(std::iter::repeat_n(0.0, n));

    let y = solve_qp(cov, &a, &lower, &upper).map_err(qp_failure)?;
    let kappa: f64 = y.iter().sum();
    if kappa.is_nan() || kappa <= 1e-12 {
        return Err(AllocError::NaNResult("weights not finite"));
    }
    Ok(y.iter().map(|v| v / kappa).collect())
}

/// `min w'Cw` subject to the budget, the bounds and `mu'w >= target_return`. The inequality
/// keeps the answer on the efficient branch: a target below the minimum-variance portfolio's
/// return yields that portfolio, not a dominated one.
fn efficient_risk_from_inputs(
    exp_ret: &[f64],
    cov: &DMatrix<f64>,
    target_return: f64,
    bounds: &[(f64, f64)],
    _risk_free: f64,
) -> Result<Vec<f64>, AllocError> {
    check_bounds_feasible(bounds)?;
    let n = cov.nrows();
    if n == 0 || exp_ret.len() != n {
        return Err(AllocError::DimensionMismatch);
    }
    let (box_rows, mut lower, mut upper) = budget_and_box(bounds);
    let mut a = DMatrix::zeros(n + 2, n);
    a.view_mut((0, 0), (n + 1, n)).copy_from(&box_rows);
    for j in 0..n {
        a[(n + 1, j)] = exp_ret[j];
    }
    lower.push(target_return);
    upper.push(f64::INFINITY);
    solve_qp(cov, &a, &lower, &upper).map_err(qp_failure)
}

/// Risk, return and Sharpe ratio of `weights`, all in the units of `exp_ret` and `cov`.
fn summarise(
    weights: Vec<f64>,
    exp_ret: &[f64],
    cov: &DMatrix<f64>,
    risk_free: f64,
) -> Result<MeanVariance, AllocError> {
    let risk = quad_risk(cov, &weights).max(0.0).sqrt();
    if !risk.is_finite() {
        return Err(AllocError::NaNResult("risk not finite"));
    }
    let port_ret = dot(exp_ret, &weights);
    let sharpe = if risk > 0.0 { (port_ret - risk_free) / risk } else { 0.0 };
    Ok(MeanVariance {
        weights,
        portfolio_risk: risk,
        portfolio_return: port_ret,
        portfolio_sharpe: sharpe,
    })
}

/// Inverse-variance portfolio from prices with default options (long-only, no resampling):
/// `w_i ∝ 1 / Sigma_ii`, ignoring correlation.
///
/// See [`allocate_with_solution`] for the price conventions.
///
/// # Errors
///
/// As [`allocate_with_solution`] with `"inverse_variance"`; in particular
/// [`AllocError::OptimizationFailed`] if an asset's return variance is zero.
///
/// ```
/// use nalgebra::DMatrix;
/// use openquant::portfolio_optimization::allocate_inverse_variance;
///
/// // Asset 1's returns are twice asset 0's, so its variance is four times as large.
/// let r = [0.01, -0.01, 0.01, -0.01];
/// let mut p = [100.0, 100.0];
/// let mut data = vec![p[0], p[1]];
/// for x in r {
///     p = [p[0] * (1.0 + x), p[1] * (1.0 + 2.0 * x)];
///     data.extend(p);
/// }
/// let prices = DMatrix::from_row_slice(5, 2, &data);
/// let result = allocate_inverse_variance(&prices).unwrap();
/// assert!((result.weights[0] - 0.8).abs() < 1e-9);
/// assert!((result.weights[1] - 0.2).abs() < 1e-9);
/// ```
pub fn allocate_inverse_variance(prices: &DMatrix<f64>) -> Result<MeanVariance, AllocError> {
    allocate_inverse_variance_with(prices, &AllocationOptions::default())
}

/// Inverse-variance portfolio from prices with explicit [`AllocationOptions`] (bounds,
/// resampling); the weights are projected onto the bounds after normalising.
///
/// # Errors
///
/// As [`allocate_with_solution`] with `"inverse_variance"`.
pub fn allocate_inverse_variance_with(
    prices: &DMatrix<f64>,
    opts: &AllocationOptions,
) -> Result<MeanVariance, AllocError> {
    allocate_with_solution(prices, "inverse_variance", opts)
}

/// Minimum-volatility portfolio from prices, `min w'Σw` subject to the budget and bounds.
///
/// `bounds` / `tuple_bounds` are as in [`AllocationOptions`]; other options take their
/// defaults. See [`allocate_with_solution`] for the price conventions.
///
/// # Errors
///
/// As [`allocate_with_solution`] with `"min_volatility"`.
pub fn allocate_min_vol(
    prices: &DMatrix<f64>,
    bounds: Option<HashMap<usize, (f64, f64)>>,
    tuple_bounds: Option<(f64, f64)>,
) -> Result<MeanVariance, AllocError> {
    let opts = AllocationOptions { bounds, tuple_bounds, ..Default::default() };
    allocate_min_vol_with(prices, &opts)
}

/// Minimum-volatility portfolio from prices with explicit [`AllocationOptions`].
///
/// # Errors
///
/// As [`allocate_with_solution`] with `"min_volatility"`.
pub fn allocate_min_vol_with(
    prices: &DMatrix<f64>,
    opts: &AllocationOptions,
) -> Result<MeanVariance, AllocError> {
    allocate_with_solution(prices, "min_volatility", opts)
}

/// Maximum-Sharpe portfolio from prices, with `risk_free` an **annual** rate.
///
/// `bounds` / `tuple_bounds` are as in [`AllocationOptions`]; other options take their
/// defaults. See [`allocate_with_solution`] for the price conventions.
///
/// # Errors
///
/// As [`allocate_with_solution`] with `"max_sharpe"`; in particular
/// [`AllocError::OptimizationFailed`] if no asset's annualised expected return exceeds
/// `risk_free`.
pub fn allocate_max_sharpe(
    prices: &DMatrix<f64>,
    risk_free: f64,
    bounds: Option<HashMap<usize, (f64, f64)>>,
    tuple_bounds: Option<(f64, f64)>,
) -> Result<MeanVariance, AllocError> {
    let opts =
        AllocationOptions { risk_free_rate: risk_free, bounds, tuple_bounds, ..Default::default() };
    allocate_max_sharpe_with(prices, &opts)
}

/// Maximum-Sharpe portfolio from prices with explicit [`AllocationOptions`].
///
/// # Errors
///
/// As [`allocate_with_solution`] with `"max_sharpe"`.
pub fn allocate_max_sharpe_with(
    prices: &DMatrix<f64>,
    opts: &AllocationOptions,
) -> Result<MeanVariance, AllocError> {
    allocate_with_solution(prices, "max_sharpe", opts)
}

/// Least-risk portfolio from prices whose expected return is at least `target_return` (an
/// **annual** figure). A target below the minimum-variance portfolio's return yields the
/// minimum-variance portfolio.
///
/// `bounds` / `tuple_bounds` are as in [`AllocationOptions`]; other options take their
/// defaults. See [`allocate_with_solution`] for the price conventions.
///
/// # Errors
///
/// As [`allocate_with_solution`] with `"efficient_risk"`; an unreachable target is reported
/// as [`AllocError::OptimizationFailed`].
pub fn allocate_efficient_risk(
    prices: &DMatrix<f64>,
    target_return: f64,
    bounds: Option<HashMap<usize, (f64, f64)>>,
    tuple_bounds: Option<(f64, f64)>,
) -> Result<MeanVariance, AllocError> {
    let opts = AllocationOptions { target_return, bounds, tuple_bounds, ..Default::default() };
    allocate_efficient_risk_with(prices, &opts)
}

/// Least-risk portfolio for `opts.target_return` from prices with explicit
/// [`AllocationOptions`].
///
/// # Errors
///
/// As [`allocate_with_solution`] with `"efficient_risk"`.
pub fn allocate_efficient_risk_with(
    prices: &DMatrix<f64>,
    opts: &AllocationOptions,
) -> Result<MeanVariance, AllocError> {
    allocate_with_solution(prices, "efficient_risk", opts)
}

/// Allocate from given expected returns and covariance with the named `solution` (see the
/// [module table](self)).
///
/// `expected_returns` has `N` entries and `covariance` is `N x N` in the same asset order; the
/// units are the caller's and must agree with `opts.risk_free_rate` and `opts.target_return`.
/// `opts.resample_by` and `opts.returns_method` are ignored.
///
/// # Errors
///
/// - [`AllocError::DimensionMismatch`] if `covariance` is not square or its size differs from
///   `expected_returns.len()`.
/// - [`AllocError::UnknownSolution`] for an unsupported `solution`.
/// - [`AllocError::InvalidBounds`] if an asset's lower bound is above its (capped) upper
///   bound, or a bound is `NaN`.
/// - [`AllocError::InfeasibleBounds`] if the bounds cannot sum to 1 (including an empty asset
///   set), or the inverse-variance projection cannot place the weight.
/// - [`AllocError::OptimizationFailed`] if the solver finds no feasible portfolio (e.g. an
///   unreachable `target_return`) or does not converge, the problem is malformed (reported as
///   "covariance is not positive definite"), no asset beats `risk_free_rate` (`"max_sharpe"`), or a covariance diagonal entry
///   is zero (`"inverse_variance"`).
/// - [`AllocError::NaNResult`] if the portfolio risk is not finite or the maximum-Sharpe
///   solution is degenerate.
///
/// ```
/// use nalgebra::DMatrix;
/// use openquant::portfolio_optimization::{allocate_from_inputs, AllocationOptions};
///
/// let opts = AllocationOptions::default();
///
/// // Two assets: min-vol weight on asset 0 is (s2^2 - s12) / (s1^2 + s2^2 - 2 s12).
/// let cov = DMatrix::from_row_slice(2, 2, &[0.04, 0.006, 0.006, 0.09]);
/// let mv = allocate_from_inputs(&[0.05, 0.10], &cov, "min_volatility", &opts).unwrap();
/// assert!((mv.weights[0] - 0.084 / 0.118).abs() < 1e-6);
///
/// // Uncorrelated assets: max-Sharpe weights are proportional to mu_i / s_i^2 = (2.5, 1.25).
/// let cov = DMatrix::from_row_slice(2, 2, &[0.04, 0.0, 0.0, 0.16]);
/// let ms = allocate_from_inputs(&[0.10, 0.20], &cov, "max_sharpe", &opts).unwrap();
/// assert!((ms.weights[0] - 2.0 / 3.0).abs() < 1e-6);
///
/// // A binding target: 0.05 w + 0.15 (1 - w) = 0.12 gives w = 0.3.
/// let target = AllocationOptions { target_return: 0.12, ..AllocationOptions::default() };
/// let er = allocate_from_inputs(&[0.05, 0.15], &cov, "efficient_risk", &target).unwrap();
/// assert!((er.weights[0] - 0.3).abs() < 1e-6);
/// assert!((er.portfolio_return - 0.12).abs() < 1e-6);
/// ```
pub fn allocate_from_inputs(
    expected_returns: &[f64],
    covariance: &DMatrix<f64>,
    solution: &str,
    opts: &AllocationOptions,
) -> Result<MeanVariance, AllocError> {
    if expected_returns.len() != covariance.nrows() || covariance.nrows() != covariance.ncols() {
        return Err(AllocError::DimensionMismatch);
    }
    let bounds = build_bounds(covariance.nrows(), &opts.bounds, opts.tuple_bounds);
    let weights = match solution {
        "inverse_variance" => inverse_variance(covariance, &bounds)?,
        "min_volatility" => solve_min_vol(covariance, &bounds)?,
        "max_sharpe" => {
            solve_max_sharpe(covariance, expected_returns, opts.risk_free_rate, &bounds)?
        }
        "efficient_risk" => efficient_risk_from_inputs(
            expected_returns,
            covariance,
            opts.target_return,
            &bounds,
            opts.risk_free_rate,
        )?,
        other => return Err(AllocError::UnknownSolution(other.to_string())),
    };
    summarise(weights, expected_returns, covariance, opts.risk_free_rate)
}

/// Allocate from a price matrix with the named `solution`: estimate annualised `mu` and `Σ`
/// with [`compute_expected_and_covariance`], then call [`allocate_from_inputs`].
///
/// `prices` is `T x N`, rows ascending in time, columns in asset order. Returns are simple
/// returns after optional resampling; `mu` and `Σ` are both annualised by `252 / step`, so
/// `opts.risk_free_rate` and `opts.target_return` are annual and the reported risk and Sharpe
/// ratio are annual.
///
/// # Errors
///
/// - [`AllocError::NoData`] if fewer than two price rows remain after resampling.
/// - [`AllocError::NaNResult`] if a price used as a return denominator is zero.
/// - Otherwise as [`allocate_from_inputs`].
///
/// ```
/// use nalgebra::DMatrix;
/// use openquant::portfolio_optimization::{
///     allocate_from_inputs, allocate_with_solution, compute_expected_and_covariance,
///     AllocationOptions, ReturnsMethod,
/// };
///
/// let prices = DMatrix::from_fn(40, 3, |t, j| {
///     100.0 * (1.0 + 0.002 * (j as f64 + 1.0)).powi(t as i32)
///         * (1.0 + 0.01 * ((t * (j + 2)) as f64).sin())
/// });
/// let opts = AllocationOptions::default();
/// let from_prices = allocate_with_solution(&prices, "min_volatility", &opts).unwrap();
/// assert!((from_prices.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
///
/// // Same answer from the annualised inputs.
/// let (mu, cov) = compute_expected_and_covariance(&prices, ReturnsMethod::Mean, None).unwrap();
/// let from_inputs = allocate_from_inputs(&mu, &cov, "min_volatility", &opts).unwrap();
/// assert_eq!(from_prices.weights, from_inputs.weights);
/// assert_eq!(from_prices.portfolio_risk, from_inputs.portfolio_risk);
/// ```
pub fn allocate_with_solution(
    prices: &DMatrix<f64>,
    solution: &str,
    opts: &AllocationOptions,
) -> Result<MeanVariance, AllocError> {
    let (exp_ret, cov) = returns_and_means(prices, opts)?;
    allocate_from_inputs(&exp_ret, &cov, solution, opts)
}
