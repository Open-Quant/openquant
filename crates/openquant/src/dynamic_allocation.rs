//! Dynamic portfolio allocation by exhaustive integer search (AFML Chapter 21).
//!
//! López de Prado, *Advances in Financial Machine Learning* (2018), §21.3–21.5, poses the
//! following problem. Over `H` horizons we hold a portfolio of `N` assets. For each horizon `h`
//! we have a forecast of the mean return vector `μ_h`, of the covariance matrix `V_h`, and a
//! vector of transaction-cost coefficients `c_h`. Trading from `ω_{h-1}` to `ω_h` costs
//!
//! ```text
//! τ_h[ω] = Σ_n c_{n,h} · sqrt(|ω_{n,h} − ω_{n,h-1}|)
//! ```
//!
//! where `ω_0` is the portfolio held before the first horizon. The square root makes the cost
//! concave in the size of the trade, so the problem is not convex and a continuous optimiser
//! can get stuck. The chapter's objective is the Sharpe ratio of the whole trajectory,
//!
//! ```text
//! SR[ω] = Σ_h (μ_hᵀ ω_h − τ_h[ω]) / sqrt(Σ_h ω_hᵀ V_h ω_h),
//! ```
//!
//! and the chapter's answer is to discretise and enumerate:
//!
//! 1. [`pigeonhole_partitions`] (Snippet 21.1): every way to put `K` indivisible units of
//!    capital into `N` assets. There are `C(K+N−1, N−1)` of them ([`partition_count`]).
//! 2. [`all_weights`] (Snippet 21.2): every sign pattern of every partition, divided by `K`, so
//!    each weight vector has gross exposure `Σ_n |ω_n| = 1`. This is the set `Ω`.
//! 3. [`dynamic_optimal_portfolio`] (Snippet 21.3): every trajectory in `Ω^H`, scored by
//!    [`trajectory_sharpe_ratio`] net of [`transaction_costs`]; the best one wins.
//!
//! The search is exact but grows as `|Ω|^H`, so it takes a cap
//! ([`DynamicAllocationConfig::max_trajectories`]) and returns
//! [`DynamicAllocationError::TooManyTrajectories`] instead of starting a search it cannot
//! finish.
//!
//! This is a different problem from [`crate::combinatorial_optimization`], whose trajectory
//! tools model a single instrument's inventory path with linear impact and a fixed ticket cost.
//!
//! ```
//! use nalgebra::DMatrix;
//! use openquant::dynamic_allocation::{
//!     all_weights, dynamic_optimal_portfolio, pigeonhole_partitions, weight_count,
//!     DynamicAllocationConfig, HorizonForecast,
//! };
//!
//! assert_eq!(pigeonhole_partitions(2, 2), vec![vec![2, 0], vec![1, 1], vec![0, 2]]);
//! assert_eq!(
//!     all_weights(1, 2).unwrap(),
//!     vec![vec![-1.0, 0.0], vec![1.0, 0.0], vec![0.0, -1.0], vec![0.0, 1.0]]
//! );
//! assert_eq!(weight_count(3, 3), Some(38));
//!
//! // One horizon, two uncorrelated assets with 20% volatility, K = 1 unit of capital.
//! let horizon = HorizonForecast {
//!     mean: vec![0.10, -0.05],
//!     covariance: DMatrix::from_diagonal_element(2, 2, 0.04),
//!     cost: vec![0.01, 0.01],
//! };
//! let best = dynamic_optimal_portfolio(&[horizon], &DynamicAllocationConfig::new(1)).unwrap();
//! // Long asset 0 beats short asset 1: (0.10 - 0.01 * sqrt(1)) / 0.2 = 0.45 against 0.20.
//! assert_eq!(best.weights, vec![vec![1.0, 0.0]]);
//! assert!((best.sharpe_ratio - 0.45).abs() < 1e-12);
//! assert!((best.transaction_costs[0] - 0.01).abs() < 1e-12);
//! assert_eq!(best.trajectories_evaluated, 4);
//! ```

use nalgebra::DMatrix;

/// Default for [`DynamicAllocationConfig::max_trajectories`]: one million trajectories.
pub const DEFAULT_MAX_TRAJECTORIES: usize = 1_000_000;

/// Errors from the Chapter 21 dynamic allocation search.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DynamicAllocationError {
    /// A count that must be positive was zero: the units of capital `K`, the number of
    /// assets `N`, or the number of horizons `H`.
    #[error("{0} must be at least 1")]
    ZeroCount(&'static str),
    /// An input had the wrong number of entries (or rows, or columns).
    #[error("{name} has {got} entries, expected {expected}")]
    DimensionMismatch {
        /// Which input, e.g. `"mean of horizon 2"`.
        name: String,
        /// The number of entries it should have.
        expected: usize,
        /// The number of entries it has.
        got: usize,
    },
    /// An input contained a NaN or an infinity.
    #[error("{0} must contain only finite values")]
    NonFinite(String),
    /// A transaction-cost coefficient was negative.
    #[error("cost coefficients must be non-negative (horizon {horizon})")]
    NegativeCost {
        /// The horizon, counted from 0.
        horizon: usize,
    },
    /// A covariance matrix was not symmetric positive definite, so some portfolio would have
    /// zero or negative variance and its Sharpe ratio would be undefined.
    #[error("the covariance matrix of horizon {horizon} must be symmetric positive definite")]
    CovarianceNotPositiveDefinite {
        /// The horizon, counted from 0.
        horizon: usize,
    },
    /// A trajectory's variance `Σ_h ω_hᵀ V_h ω_h` was not positive (only reachable through
    /// round-off, since the covariances are checked to be positive definite).
    #[error("trajectory variance must be positive, got {0}")]
    NonPositiveVariance(f64),
    /// The search would evaluate more trajectories than the configured cap.
    #[error(
        "the search would evaluate {trajectories} trajectories ({weights} weight vectors over \
         {horizons} horizons), more than max_trajectories = {max_trajectories}"
    )]
    TooManyTrajectories {
        /// `|Ω|^H`, saturated at `u128::MAX`.
        trajectories: u128,
        /// `|Ω|`, the number of distinct weight vectors, saturated at `u128::MAX`.
        weights: u128,
        /// `H`, the number of horizons.
        horizons: usize,
        /// The cap that was exceeded.
        max_trajectories: usize,
    },
}

/// The forecasts for one horizon `h`: the inputs of Snippet 21.3's `params[h]`.
#[derive(Debug, Clone, PartialEq)]
pub struct HorizonForecast {
    /// `μ_h`, the expected return of each asset over the horizon (length `N`).
    pub mean: Vec<f64>,
    /// `V_h`, the `N × N` covariance of the assets' returns over the horizon. Must be
    /// symmetric positive definite.
    pub covariance: DMatrix<f64>,
    /// `c_h`, the cost coefficient of each asset (length `N`, non-negative): trading `x` of
    /// gross weight in asset `n` at horizon `h` costs `c_{n,h} · sqrt(|x|)`.
    pub cost: Vec<f64>,
}

/// Settings for [`dynamic_optimal_portfolio`].
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicAllocationConfig {
    /// `K`, the number of indivisible units of capital. Weights are multiples of `1/K`. The
    /// book's default is `K = N`.
    pub units: usize,
    /// `ω_0`, the portfolio held before the first horizon (length `N`). `None` starts from
    /// cash, all zeros, as Snippet 21.3 does.
    pub initial_weights: Option<Vec<f64>>,
    /// The largest number of trajectories `|Ω|^H` the search may evaluate. A larger problem
    /// returns [`DynamicAllocationError::TooManyTrajectories`] before any work is done.
    pub max_trajectories: usize,
}

impl DynamicAllocationConfig {
    /// `K = units`, starting from cash, with the [`DEFAULT_MAX_TRAJECTORIES`] cap.
    pub fn new(units: usize) -> Self {
        Self { units, initial_weights: None, max_trajectories: DEFAULT_MAX_TRAJECTORIES }
    }
}

/// The best trajectory found by [`dynamic_optimal_portfolio`].
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicAllocation {
    /// The optimal weights, one vector of length `N` per horizon, in horizon order. (The book
    /// returns the transpose, an `N × H` array.)
    pub weights: Vec<Vec<f64>>,
    /// The trajectory's Sharpe ratio net of transaction costs, `SR[ω]`.
    pub sharpe_ratio: f64,
    /// `τ_h[ω]` of the optimal trajectory, one per horizon.
    pub transaction_costs: Vec<f64>,
    /// How many trajectories were scored: `|Ω|^H`.
    pub trajectories_evaluated: usize,
}

/// Every way to place `k` indivisible units into `n` slots (AFML Snippet 21.1, `pigeonHole`).
///
/// Each partition is a vector of `n` non-negative counts summing to `k`. They come in the
/// book's order, the order Python's `combinations_with_replacement(range(n), k)` implies:
/// `[k, 0, …, 0]` first and `[0, …, 0, k]` last. There are `C(k+n−1, n−1)` of them
/// ([`partition_count`]). With `n = 0` there is one (empty) partition of `k = 0` and none of
/// any larger `k`, as in the book.
pub fn pigeonhole_partitions(k: usize, n: usize) -> Vec<Vec<usize>> {
    fn fill(slot: usize, left: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if slot + 1 == current.len() {
            current[slot] = left;
            out.push(current.clone());
            return;
        }
        for units in (0..=left).rev() {
            current[slot] = units;
            fill(slot + 1, left - units, current, out);
        }
    }

    if n == 0 {
        return if k == 0 { vec![Vec::new()] } else { Vec::new() };
    }
    let mut out = Vec::new();
    fill(0, k, &mut vec![0; n], &mut out);
    out
}

/// The number of pigeonhole partitions of `k` units into `n` slots, `C(k+n−1, n−1)` (stars and
/// bars), or `None` if it does not fit in a `u128`.
pub fn partition_count(k: usize, n: usize) -> Option<u128> {
    if n == 0 {
        return Some(u128::from(k == 0));
    }
    binomial(k as u128 + n as u128 - 1, n as u128 - 1)
}

/// The set `Ω` of signed weight vectors (AFML Snippet 21.2, `getAllWeights`).
///
/// Every partition from [`pigeonhole_partitions`] is divided by `k`, so its absolute weights sum
/// to 1, and then given every combination of signs. The book loops over all `2^n` sign patterns
/// of every partition, so a partition with zero entries appears more than once (flipping the sign
/// of a zero changes nothing). This function keeps the first occurrence of each vector and drops
/// the repeats. The set is the same, the order of first occurrences is the book's, and so the
/// trajectory search picks the same trajectory; it just evaluates fewer duplicates.
///
/// There are `Σ_j C(n, j) · C(k−1, j−1) · 2^j` vectors ([`weight_count`]), `j` counting the
/// non-zero entries.
///
/// # Errors
///
/// [`DynamicAllocationError::ZeroCount`] if `k` or `n` is zero.
pub fn all_weights(k: usize, n: usize) -> Result<Vec<Vec<f64>>, DynamicAllocationError> {
    if k == 0 {
        return Err(DynamicAllocationError::ZeroCount("the number of units k"));
    }
    if n == 0 {
        return Err(DynamicAllocationError::ZeroCount("the number of assets n"));
    }
    let mut out = Vec::new();
    for partition in pigeonhole_partitions(k, n) {
        let nonzero: Vec<usize> = (0..n).filter(|&i| partition[i] > 0).collect();
        let j = nonzero.len();
        // `itertools.product([-1, 1], repeat=n)` order: the first asset's sign varies slowest
        // and -1 comes before +1. Only the signs of non-zero entries matter; with the zeros'
        // signs held at -1 this visits each vector at its first occurrence in the book's list.
        // Bit `j-1-m` of `pattern` set means the m-th non-zero entry is positive.
        for pattern in 0..(1u128 << j) {
            let mut weights = vec![0.0; n];
            for (m, &i) in nonzero.iter().enumerate() {
                let magnitude = partition[i] as f64 / k as f64;
                let positive = (pattern >> (j - 1 - m)) & 1 == 1;
                weights[i] = if positive { magnitude } else { -magnitude };
            }
            out.push(weights);
        }
    }
    Ok(out)
}

/// `|Ω|`, the number of distinct vectors [`all_weights`] returns for `k ≥ 1` units and `n ≥ 1`
/// assets: `Σ_{j=1}^{min(k,n)} C(n, j) · C(k−1, j−1) · 2^j`. `None` if it does not fit in a
/// `u128`; `Some(0)` if `k` or `n` is zero. (The book's list, which keeps the duplicates, has
/// `2^n · C(k+n−1, n−1)` entries.)
pub fn weight_count(k: usize, n: usize) -> Option<u128> {
    if k == 0 || n == 0 {
        return Some(0);
    }
    let mut total: u128 = 0;
    for j in 1..=k.min(n) as u128 {
        let signs = 1u128.checked_shl(u32::try_from(j).ok()?)?;
        let term = binomial(n as u128, j)?
            .checked_mul(binomial(k as u128 - 1, j - 1)?)?
            .checked_mul(signs)?;
        total = total.checked_add(term)?;
    }
    Some(total)
}

/// `τ_h[ω]` for every horizon of a trajectory (AFML Snippet 21.3, `evalTCosts`).
///
/// `τ_h = Σ_n c_{n,h} · sqrt(|ω_{n,h} − ω_{n,h−1}|)`, where `ω_{·,0}` is `initial_weights` (all
/// zeros in the book). `trajectory` holds one weight vector per horizon, in the same order as
/// `horizons`.
///
/// # Errors
///
/// - [`DynamicAllocationError::ZeroCount`] if `horizons` is empty or its first mean vector is
///   empty.
/// - [`DynamicAllocationError::DimensionMismatch`] if a forecast's mean, cost or covariance,
///   `initial_weights`, the number of weight vectors in `trajectory`, or one of them, does not
///   match `N` (the length of the first horizon's mean) or `H`.
/// - [`DynamicAllocationError::NonFinite`] if any of those inputs holds a NaN or infinity.
/// - [`DynamicAllocationError::NegativeCost`] if a cost coefficient is negative.
/// - [`DynamicAllocationError::CovarianceNotPositiveDefinite`] if a covariance is not
///   symmetric positive definite.
pub fn transaction_costs(
    trajectory: &[Vec<f64>],
    horizons: &[HorizonForecast],
    initial_weights: &[f64],
) -> Result<Vec<f64>, DynamicAllocationError> {
    let n = validate_horizons(horizons)?;
    validate_vector("initial weights", initial_weights, n)?;
    validate_trajectory(trajectory, horizons.len(), n)?;
    let mut previous = initial_weights;
    let mut costs = Vec::with_capacity(horizons.len());
    for (weights, forecast) in trajectory.iter().zip(horizons) {
        costs.push(trade_cost(previous, weights, &forecast.cost));
        previous = weights;
    }
    Ok(costs)
}

/// The Sharpe ratio of a trajectory net of transaction costs (AFML Snippet 21.3, `evalSR`):
///
/// ```text
/// SR[ω] = Σ_h (μ_hᵀ ω_h − τ_h[ω]) / sqrt(Σ_h ω_hᵀ V_h ω_h)
/// ```
///
/// with `τ_h` from [`transaction_costs`]. Costs are subtracted from the mean only; the variance
/// is that of the gross returns, as in the book.
///
/// # Errors
///
/// Every error of [`transaction_costs`], and
/// [`DynamicAllocationError::NonPositiveVariance`] if the trajectory's total variance is not
/// positive and finite (an all-zero trajectory, for example).
pub fn trajectory_sharpe_ratio(
    trajectory: &[Vec<f64>],
    horizons: &[HorizonForecast],
    initial_weights: &[f64],
) -> Result<f64, DynamicAllocationError> {
    let costs = transaction_costs(trajectory, horizons, initial_weights)?;
    let mut mean = 0.0;
    let mut variance = 0.0;
    for ((weights, forecast), cost) in trajectory.iter().zip(horizons).zip(costs) {
        mean += dot(weights, &forecast.mean) - cost;
        variance += quadratic_form(weights, &forecast.covariance);
    }
    sharpe(mean, variance)
}

/// The trajectory in `Ω^H` with the highest net Sharpe ratio (AFML Snippet 21.3, `dynOptPort`).
///
/// `Ω` is [`all_weights`]`(config.units, N)`. Every trajectory, one element of `Ω` per horizon,
/// is scored by [`trajectory_sharpe_ratio`], in the order of the book's
/// `itertools.product(Ω, repeat=H)`; the first trajectory with the highest score is kept, as the
/// book's strict `sr < sr_` comparison does.
///
/// The number of trajectories is `|Ω|^H` and grows fast: 38 weight vectors for `N = K = 3`, so
/// 54,872 trajectories over three horizons, but 23 million over five. When `|Ω|^H` exceeds
/// `config.max_trajectories` this returns [`DynamicAllocationError::TooManyTrajectories`]
/// without evaluating anything. Use [`weight_count`] to size a problem in advance.
///
/// # Errors
///
/// - The forecast-validation errors of [`transaction_costs`] (for `horizons` and
///   `config.initial_weights`).
/// - [`DynamicAllocationError::ZeroCount`] if `config.units` is zero.
/// - [`DynamicAllocationError::TooManyTrajectories`] if `|Ω|^H > config.max_trajectories`.
/// - [`DynamicAllocationError::NonPositiveVariance`] if a trajectory's variance is not
///   positive, which positive definite covariances only allow through round-off.
pub fn dynamic_optimal_portfolio(
    horizons: &[HorizonForecast],
    config: &DynamicAllocationConfig,
) -> Result<DynamicAllocation, DynamicAllocationError> {
    let n = validate_horizons(horizons)?;
    if config.units == 0 {
        return Err(DynamicAllocationError::ZeroCount("the number of units k"));
    }
    let zeros = vec![0.0; n];
    let initial = config.initial_weights.as_deref().unwrap_or(&zeros);
    validate_vector("initial weights", initial, n)?;

    let h = horizons.len();
    let omega_size = weight_count(config.units, n).unwrap_or(u128::MAX);
    let trajectories =
        u32::try_from(h).ok().and_then(|exp| omega_size.checked_pow(exp)).unwrap_or(u128::MAX);
    if trajectories > config.max_trajectories as u128 {
        return Err(DynamicAllocationError::TooManyTrajectories {
            trajectories,
            weights: omega_size,
            horizons: h,
            max_trajectories: config.max_trajectories,
        });
    }

    let omega = all_weights(config.units, n)?;
    // Each horizon's gross return and variance depend on that horizon's weights alone, so they
    // are computed once per (horizon, weight vector) rather than once per trajectory.
    let returns: Vec<Vec<f64>> =
        horizons.iter().map(|f| omega.iter().map(|w| dot(w, &f.mean)).collect()).collect();
    let variances: Vec<Vec<f64>> = horizons
        .iter()
        .map(|f| omega.iter().map(|w| quadratic_form(w, &f.covariance)).collect())
        .collect();

    let mut search = Search {
        horizons,
        omega: &omega,
        returns: &returns,
        variances: &variances,
        initial,
        path: vec![0; h],
        best: None,
        evaluated: 0,
    };
    search.descend(0, 0.0, 0.0)?;

    let (sharpe_ratio, best_path) = search.best.expect("Ω is non-empty for k, n ≥ 1");
    let weights: Vec<Vec<f64>> = best_path.iter().map(|&i| omega[i].clone()).collect();
    let transaction_costs = transaction_costs(&weights, horizons, initial)?;
    Ok(DynamicAllocation {
        weights,
        sharpe_ratio,
        transaction_costs,
        trajectories_evaluated: search.evaluated,
    })
}

/// Depth-first walk over `Ω^H` in lexicographic (book) order, carrying partial sums.
struct Search<'a> {
    horizons: &'a [HorizonForecast],
    omega: &'a [Vec<f64>],
    returns: &'a [Vec<f64>],
    variances: &'a [Vec<f64>],
    initial: &'a [f64],
    path: Vec<usize>,
    best: Option<(f64, Vec<usize>)>,
    evaluated: usize,
}

impl Search<'_> {
    fn descend(
        &mut self,
        horizon: usize,
        mean: f64,
        variance: f64,
    ) -> Result<(), DynamicAllocationError> {
        if horizon == self.horizons.len() {
            self.evaluated += 1;
            let sr = sharpe(mean, variance)?;
            if self.best.as_ref().is_none_or(|(best, _)| *best < sr) {
                self.best = Some((sr, self.path.clone()));
            }
            return Ok(());
        }
        let cost = &self.horizons[horizon].cost;
        for i in 0..self.omega.len() {
            let previous =
                if horizon == 0 { self.initial } else { &self.omega[self.path[horizon - 1]] };
            let tau = trade_cost(previous, &self.omega[i], cost);
            self.path[horizon] = i;
            self.descend(
                horizon + 1,
                mean + self.returns[horizon][i] - tau,
                variance + self.variances[horizon][i],
            )?;
        }
        Ok(())
    }
}

fn trade_cost(previous: &[f64], current: &[f64], cost: &[f64]) -> f64 {
    previous.iter().zip(current).zip(cost).map(|((p, w), c)| c * (w - p).abs().sqrt()).sum()
}

fn sharpe(mean: f64, variance: f64) -> Result<f64, DynamicAllocationError> {
    if variance > 0.0 && variance.is_finite() {
        Ok(mean / variance.sqrt())
    } else {
        Err(DynamicAllocationError::NonPositiveVariance(variance))
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn quadratic_form(w: &[f64], v: &DMatrix<f64>) -> f64 {
    let mut total = 0.0;
    for (i, wi) in w.iter().enumerate() {
        for (j, wj) in w.iter().enumerate() {
            total += wi * v[(i, j)] * wj;
        }
    }
    total
}

fn binomial(n: u128, k: u128) -> Option<u128> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut acc: u128 = 1;
    for i in 0..k {
        // acc = C(n, i), and C(n, i) * (n - i) = C(n, i + 1) * (i + 1), so this is exact.
        acc = acc.checked_mul(n - i)? / (i + 1);
    }
    Some(acc)
}

fn validate_vector(name: &str, values: &[f64], n: usize) -> Result<(), DynamicAllocationError> {
    if values.len() != n {
        return Err(DynamicAllocationError::DimensionMismatch {
            name: name.to_string(),
            expected: n,
            got: values.len(),
        });
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(DynamicAllocationError::NonFinite(name.to_string()));
    }
    Ok(())
}

/// Checks every forecast and returns the number of assets `N`.
fn validate_horizons(horizons: &[HorizonForecast]) -> Result<usize, DynamicAllocationError> {
    let first =
        horizons.first().ok_or(DynamicAllocationError::ZeroCount("the number of horizons"))?;
    let n = first.mean.len();
    if n == 0 {
        return Err(DynamicAllocationError::ZeroCount("the number of assets n"));
    }
    for (h, forecast) in horizons.iter().enumerate() {
        validate_vector(&format!("mean of horizon {h}"), &forecast.mean, n)?;
        validate_vector(&format!("cost of horizon {h}"), &forecast.cost, n)?;
        if forecast.cost.iter().any(|&c| c < 0.0) {
            return Err(DynamicAllocationError::NegativeCost { horizon: h });
        }
        let v = &forecast.covariance;
        for (dim, got) in [("rows", v.nrows()), ("columns", v.ncols())] {
            if got != n {
                return Err(DynamicAllocationError::DimensionMismatch {
                    name: format!("covariance {dim} of horizon {h}"),
                    expected: n,
                    got,
                });
            }
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(DynamicAllocationError::NonFinite(format!("covariance of horizon {h}")));
        }
        let scale = v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let symmetric =
            (0..n).all(|i| (0..i).all(|j| (v[(i, j)] - v[(j, i)]).abs() <= 1e-12 * scale));
        // nalgebra's Cholesky accepts a zero pivot, so a singular matrix can pass it; require
        // every pivot to be clearly positive relative to the matrix's scale.
        let positive_definite = v.clone().cholesky().is_some_and(|c| {
            let l = c.l();
            (0..n).all(|i| l[(i, i)] * l[(i, i)] > 1e-12 * scale)
        });
        if !symmetric || !positive_definite {
            return Err(DynamicAllocationError::CovarianceNotPositiveDefinite { horizon: h });
        }
    }
    Ok(n)
}

fn validate_trajectory(
    trajectory: &[Vec<f64>],
    horizons: usize,
    n: usize,
) -> Result<(), DynamicAllocationError> {
    if trajectory.len() != horizons {
        return Err(DynamicAllocationError::DimensionMismatch {
            name: "trajectory".to_string(),
            expected: horizons,
            got: trajectory.len(),
        });
    }
    for (h, weights) in trajectory.iter().enumerate() {
        validate_vector(&format!("weights of horizon {h}"), weights, n)?;
    }
    Ok(())
}
