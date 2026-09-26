//! The Critical Line Algorithm (Markowitz, 1956; Bailey and López de Prado, 2013), the
//! mean-variance benchmark of AFML chapter 16.
//!
//! [`CLA::allocate`] computes every turning point of the efficient frontier under box bounds
//! and the budget constraint (weights sum to 1), exactly, walking from the maximum-return
//! portfolio (`lambda = inf`) down to the minimum-variance portfolio (`lambda = 0`). Between
//! two consecutive turning points every efficient portfolio is a convex combination of
//! them. From the turning points it can also return the minimum-variance portfolio, the
//! maximum-Sharpe portfolio (risk-free rate zero), or about 100 points along the frontier.
//!
//! Conventions:
//!
//! - Price matrices have one row per observation (oldest first) and one column per asset.
//!   From prices, returns are **simple** returns, and expected returns are annualised with
//!   `252 / step` periods per year, where `step` is 1 (daily), 5 (`resample_by = "W"`) or
//!   21 (`"M"`). Resampling is positional: every 5th or 21st row. The 252 assumes **daily**
//!   rows and is intentional: for intraday rows the "annual" returns are per 252 rows. It is
//!   a constant scale of the expected returns only, so the turning-point weights, the
//!   minimum-variance and the maximum-Sharpe solution do not depend on it.
//! - Expected returns and covariance supplied directly are used as given (no annualisation).
//! - Weights are returned in the column order of the inputs.
//!
//! Methods whose names begin with an underscore are public for parity testing with the
//! reference implementation; [`CLA::allocate`] already calls them.
//!
//! ```
//! use nalgebra::DMatrix;
//! use openquant::cla::{WeightBounds, CLA};
//!
//! # fn main() -> Result<(), openquant::cla::ClaError> {
//! let mu = DMatrix::from_column_slice(4, 1, &[0.03, 0.07, 0.09, 0.04]);
//! let vol = [0.05, 0.16, 0.22, 0.15];
//! let rho = [
//!     [1.0, 0.1, 0.1, 0.1],
//!     [0.1, 1.0, 0.8, 0.0],
//!     [0.1, 0.8, 1.0, 0.0],
//!     [0.1, 0.0, 0.0, 1.0],
//! ];
//! let cov = DMatrix::from_fn(4, 4, |i, j| rho[i][j] * vol[i] * vol[j]);
//!
//! let mut cla = CLA::new(WeightBounds::Tuple(0.0, 1.0), "mean");
//! cla.allocate(None, Some(&mu), Some(&cov), None, None)?;
//!
//! // The walk starts fully in the highest-return asset and ends at minimum variance.
//! assert_eq!(cla.weights.first().unwrap(), &vec![0.0, 0.0, 1.0, 0.0]);
//! assert_eq!(*cla.lambdas.last().unwrap(), 0.0);
//! for w in &cla.weights {
//!     assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
//! }
//! # Ok(())
//! # }
//! ```

use crate::util::resample::{freq_step, resample_prices};
use chrono::NaiveDate;
use nalgebra::{DMatrix, DVector};

#[derive(Debug, PartialEq, thiserror::Error)]
/// Errors returned by [`CLA`] and [`ReturnsEstimation`].
pub enum ClaError {
    /// Neither prices nor both expected returns and a covariance matrix were supplied.
    #[error("supply asset prices, or expected returns and a covariance matrix")]
    MissingInputs,
    /// An [`AssetPrices`] index is empty or does not have one date per row of prices.
    #[error(
        "asset price index has {dates} dates for {rows} rows of prices; it needs one date per row"
    )]
    InvalidPriceIndex {
        /// Rows of the price matrix.
        rows: usize,
        /// Entries in the date index.
        dates: usize,
    },
    /// No portfolio satisfies the bounds and the budget: a bound is not finite, an asset's
    /// lower bound exceeds its upper bound, the lower bounds sum to more than 1, or the
    /// upper bounds (capped at 1) sum to less than 1.
    #[error("infeasible weight bounds: {0}")]
    InfeasibleBounds(&'static str),
    /// The covariance matrix of the assets free at a step of the walk cannot be inverted,
    /// e.g. two free assets are perfectly correlated, or there are more assets than
    /// observations.
    #[error("covariance of the free assets is singular")]
    SingularCovariance,
    /// The walk along the critical line did not reach `lambda = 0` within its step limit
    /// (`4 n^2 + 100` steps).
    #[error("the critical line did not terminate")]
    NoTermination,
    /// Every turning point was removed as numerically invalid, so there is no portfolio to
    /// choose from.
    #[error("no valid turning points remain")]
    NoTurningPoints,
    /// The expected-returns method is not `"mean"` or `"exponential"`.
    #[error("unknown returns method: {0}")]
    UnknownReturns(String),
    /// The solution name is not one of the four supported.
    #[error("unknown solution: {0}")]
    UnknownSolution(String),
    /// Inputs disagree on the number of assets, or expected returns are not a vector.
    #[error("inputs disagree on the number of assets")]
    DimensionMismatch,
    /// Internal bookkeeping vectors disagree in length.
    #[error("asset index out of range")]
    IndexError,
    /// Too few observations to form a return, a zero price, or no assets.
    #[error("no data")]
    NoData,
}

/// A price matrix with a date index.
#[derive(Clone)]
pub struct AssetPrices {
    /// Prices, one row per date (oldest first) and one column per asset.
    pub data: DMatrix<f64>,
    /// The date of each row; must have one entry per row of `data`.
    pub index: Vec<NaiveDate>,
}

impl AssetPrices {
    /// Wraps a price matrix and its date index. Nothing is validated here; [`CLA::allocate`]
    /// checks that the index is non-empty and matches the number of rows
    /// ([`ClaError::InvalidPriceIndex`]).
    pub fn new(data: DMatrix<f64>, index: Vec<NaiveDate>) -> Self {
        AssetPrices { data, index }
    }
}

/// Price input to [`CLA::allocate`].
pub enum AssetPricesInput<'a> {
    /// Prices with a date index.
    Prices(&'a AssetPrices),
    /// A bare price matrix, one row per observation (at least two rows).
    RawMatrix(&'a DMatrix<f64>),
}

/// Box bounds on the portfolio weights.
#[derive(Clone)]
pub enum WeightBounds {
    /// The same `(lower, upper)` bound for every asset.
    Tuple(f64, f64),
    /// Per-asset `(lowers, uppers)`, each with one entry per asset.
    Lists(Vec<f64>, Vec<f64>),
}

/// Expected-return and return estimators used by [`CLA`] when it is given prices.
///
/// All take a price matrix with one row per observation (oldest first), optionally
/// resampled positionally with `resample_by` (`"W"`/`"week"`/`"weekly"` keeps every 5th row,
/// `"M"`/`"month"`/`"monthly"` every 21st; anything else is daily), and compute simple
/// returns.
pub struct ReturnsEstimation;

impl ReturnsEstimation {
    /// Annualised mean simple return per asset: the mean periodic return times
    /// `252 / step`.
    ///
    /// # Errors
    ///
    /// [`ClaError::NoData`] if fewer than two (resampled) rows remain, or a price used as a
    /// denominator is zero.
    pub fn calculate_mean_historical_returns(
        asset_prices: &DMatrix<f64>,
        resample_by: Option<&str>,
    ) -> Result<Vec<f64>, ClaError> {
        let (returns, freq) = returns_and_frequency(asset_prices, resample_by)?;
        let rows = returns.nrows();
        let cols = returns.ncols();
        let mut out = vec![0.0; cols];
        for (c, slot) in out.iter_mut().enumerate() {
            *slot = (returns.column(c).sum() / rows as f64) * freq;
        }
        Ok(out)
    }

    /// Annualised exponentially weighted mean simple return per asset, with smoothing
    /// `alpha = 2 / (span + 1)` seeded at the first return (no bias adjustment), times
    /// `252 / step`.
    ///
    /// # Errors
    ///
    /// [`ClaError::NoData`] if fewer than two (resampled) rows remain, or a price used as a
    /// denominator is zero.
    pub fn calculate_exponential_historical_returns(
        asset_prices: &DMatrix<f64>,
        resample_by: Option<&str>,
        span: usize,
    ) -> Result<Vec<f64>, ClaError> {
        let (returns, freq) = returns_and_frequency(asset_prices, resample_by)?;
        let rows = returns.nrows();
        let cols = returns.ncols();
        let alpha = 2.0 / (span as f64 + 1.0);
        let mut out = vec![0.0; cols];
        for c in 0..cols {
            let mut ema = returns[(0, c)];
            for r in 1..rows {
                ema = alpha * returns[(r, c)] + (1.0 - alpha) * ema;
            }
            out[c] = ema * freq;
        }
        Ok(out)
    }

    /// Periodic simple returns, one row per (resampled) period after the first.
    ///
    /// # Errors
    ///
    /// [`ClaError::NoData`] if fewer than two (resampled) rows remain, or a price used as a
    /// denominator is zero.
    pub fn calculate_returns(
        asset_prices: &DMatrix<f64>,
        resample_by: Option<&str>,
    ) -> Result<DMatrix<f64>, ClaError> {
        let (returns, _freq) = returns_and_frequency(asset_prices, resample_by)?;
        Ok(returns)
    }
}

/// Critical Line Algorithm solver and its results.
///
/// Build with [`CLA::new`], call [`CLA::allocate`], then read the public fields.
pub struct CLA {
    /// Bounds on the weights.
    pub weight_bounds: WeightBounds,
    /// How expected returns are estimated from prices: `"mean"` or `"exponential"`.
    pub calculate_expected_returns: String,
    /// The requested solution: every turning point (maximum return first) for
    /// `"cla_turning_points"`, one portfolio for `"min_volatility"` and `"max_sharpe"`, or the
    /// frontier points for `"efficient_frontier"`. Each inner vector has one weight per asset.
    pub weights: Vec<Vec<f64>>,
    /// Risk-aversion parameter at each turning point, falling to 0 at minimum variance. The
    /// first entry is infinite (the starting portfolio, which the first proper turning point
    /// repeats).
    pub lambdas: Vec<f64>,
    /// Budget-constraint multiplier at each turning point.
    pub gammas: Vec<f64>,
    /// Indices of the assets strictly inside their bounds at each turning point.
    pub free_weights: Vec<Vec<usize>>,
    /// Expected returns as an `n x 1` column.
    pub expected_returns: DMatrix<f64>,
    /// The `n x n` covariance matrix.
    pub cov_matrix: DMatrix<f64>,
    /// Per-asset lower bounds.
    pub lower_bounds: Vec<f64>,
    /// Per-asset upper bounds.
    pub upper_bounds: Vec<f64>,
    /// Expected return of each frontier point (filled only for `"efficient_frontier"`).
    pub efficient_frontier_means: Vec<f64>,
    /// Volatility of each frontier point (filled only for `"efficient_frontier"`).
    pub efficient_frontier_sigma: Vec<f64>,
}

impl Default for CLA {
    fn default() -> Self {
        CLA::new(WeightBounds::Tuple(0.0, 1.0), "mean")
    }
}

impl CLA {
    /// Creates a solver with the given bounds and expected-return method (`"mean"` or
    /// `"exponential"`; checked only when [`CLA::allocate`] estimates from prices).
    pub fn new(weight_bounds: WeightBounds, calculate_expected_returns: &str) -> Self {
        CLA {
            weight_bounds,
            calculate_expected_returns: calculate_expected_returns.to_string(),
            weights: Vec::new(),
            lambdas: Vec::new(),
            gammas: Vec::new(),
            free_weights: Vec::new(),
            expected_returns: DMatrix::<f64>::zeros(0, 0),
            cov_matrix: DMatrix::<f64>::zeros(0, 0),
            lower_bounds: Vec::new(),
            upper_bounds: Vec::new(),
            efficient_frontier_means: Vec::new(),
            efficient_frontier_sigma: Vec::new(),
        }
    }

    /// Runs the Critical Line Algorithm and stores the requested solution in
    /// [`CLA::weights`].
    ///
    /// Supply either prices (`asset_prices`, from which missing expected returns and
    /// covariance are estimated) or both `expected_asset_returns` (an `n x 1` or `1 x n`
    /// matrix) and `covariance_matrix` (`n x n`). `resample_by` only applies to prices.
    /// `solution` is one of `"cla_turning_points"` (default), `"min_volatility"`,
    /// `"max_sharpe"` (maximises `mu'w / sigma`, i.e. a zero risk-free rate) or
    /// `"efficient_frontier"`. [`CLA::lambdas`], [`CLA::gammas`] and [`CLA::free_weights`]
    /// always describe the turning points.
    ///
    /// If every expected return is identical, `1e-5` is added to the last one so the walk has
    /// a starting asset.
    ///
    /// # Errors
    ///
    /// - [`ClaError::MissingInputs`] if no prices are given and either expected returns or
    ///   covariance is missing.
    /// - [`ClaError::InvalidPriceIndex`] if an [`AssetPrices`] index is empty or does not
    ///   have one date per row.
    /// - [`ClaError::NoData`] if a price matrix has fewer than two rows, contains a zero
    ///   price, or there are no assets.
    /// - [`ClaError::UnknownReturns`] if expected returns must be estimated and the method is
    ///   not `"mean"` or `"exponential"`.
    /// - [`ClaError::DimensionMismatch`] if the inputs disagree on the number of assets or
    ///   expected returns are not a vector.
    /// - [`ClaError::InfeasibleBounds`] if a bound is not finite, a lower bound exceeds its
    ///   upper bound, or the bounds cannot sum to one.
    /// - [`ClaError::SingularCovariance`] if the covariance of the free assets cannot be
    ///   inverted at some step of the walk.
    /// - [`ClaError::NoTermination`] if the walk does not reach `lambda = 0`.
    /// - [`ClaError::NoTurningPoints`] if every turning point is removed as numerically
    ///   invalid (not seen with valid inputs; reported rather than panicking).
    /// - [`ClaError::UnknownSolution`] for any other `solution` name.
    pub fn allocate(
        &mut self,
        asset_prices: Option<AssetPricesInput<'_>>,
        expected_asset_returns: Option<&DMatrix<f64>>,
        covariance_matrix: Option<&DMatrix<f64>>,
        resample_by: Option<&str>,
        solution: Option<&str>,
    ) -> Result<(), ClaError> {
        if asset_prices.is_none() && expected_asset_returns.is_none() && covariance_matrix.is_none()
        {
            return Err(ClaError::MissingInputs);
        }
        match asset_prices {
            Some(AssetPricesInput::Prices(prices)) => {
                if prices.index.len() != prices.data.nrows() || prices.index.is_empty() {
                    return Err(ClaError::InvalidPriceIndex {
                        rows: prices.data.nrows(),
                        dates: prices.index.len(),
                    });
                }
                self._initialise(
                    &prices.data,
                    resample_by,
                    expected_asset_returns,
                    covariance_matrix,
                )?;
            }
            // A bare matrix of prices, one row per observation. mlfinlab rejects anything that
            // is not a DataFrame; that check has no meaning here, and refusing a matrix made
            // the price path unreachable from Python.
            Some(AssetPricesInput::RawMatrix(prices)) => {
                if prices.nrows() < 2 {
                    return Err(ClaError::NoData);
                }
                self._initialise(prices, resample_by, expected_asset_returns, covariance_matrix)?;
            }
            None => {
                let (Some(exp), Some(cov)) = (expected_asset_returns, covariance_matrix) else {
                    return Err(ClaError::MissingInputs);
                };
                self.expected_returns = normalize_expected_returns(exp)?;
                self.cov_matrix = cov.clone_owned();
                let bounds = build_bounds(self.expected_returns.nrows(), &self.weight_bounds)?;
                self.lower_bounds = bounds.iter().map(|b| b.0).collect();
                self.upper_bounds = bounds.iter().map(|b| b.1).collect();
            }
        }
        let n = self.expected_returns.nrows();
        if self.cov_matrix.nrows() != n || self.cov_matrix.ncols() != n {
            return Err(ClaError::DimensionMismatch);
        }
        let bounds: Vec<(f64, f64)> =
            self.lower_bounds.iter().copied().zip(self.upper_bounds.iter().copied()).collect();
        check_bounds_feasible(&bounds)?;

        let mean: Vec<f64> = self.expected_returns.column(0).iter().copied().collect();
        let points =
            critical_line(&mean, &self.cov_matrix, &self.lower_bounds, &self.upper_bounds)?;
        self.weights = points.iter().map(|p| p.weights.clone()).collect();
        self.lambdas = points.iter().map(|p| p.lambda).collect();
        self.gammas = points.iter().map(|p| p.gamma).collect();
        self.free_weights = points.iter().map(|p| p.free.clone()).collect();
        self._purge_num_err(1e-9)?;
        self._purge_excess()?;
        if self.weights.is_empty() {
            return Err(ClaError::NoTurningPoints);
        }
        self.efficient_frontier_means.clear();
        self.efficient_frontier_sigma.clear();

        // `lambdas`, `gammas` and `free_weights` always describe the turning points. `weights`
        // holds the requested solution, which for "cla_turning_points" is those same points.
        let turning_points = self.weights.clone();
        match solution.unwrap_or("cla_turning_points") {
            "cla_turning_points" => {}
            "min_volatility" => {
                let best = turning_points
                    .iter()
                    .min_by(|a, b| {
                        quad_risk(&self.cov_matrix, a).total_cmp(&quad_risk(&self.cov_matrix, b))
                    })
                    .ok_or(ClaError::NoTurningPoints)?;
                self.weights = vec![best.clone()];
            }
            "max_sharpe" => {
                self.weights =
                    vec![max_sharpe_on_frontier(&turning_points, &mean, &self.cov_matrix)?];
            }
            "efficient_frontier" => {
                self.weights = frontier_points(&turning_points, 100);
                for w in &self.weights {
                    self.efficient_frontier_means.push(dot(w, &mean));
                    self.efficient_frontier_sigma.push(quad_risk(&self.cov_matrix, w).sqrt());
                }
            }
            other => return Err(ClaError::UnknownSolution(other.to_string())),
        }
        Ok(())
    }

    /// Sets expected returns, covariance and bounds from prices (called by
    /// [`CLA::allocate`]; public for parity testing).
    ///
    /// Supplied `expected_asset_returns` or `covariance_matrix` take precedence over the
    /// estimates from `asset_prices`. Clears previous results.
    ///
    /// # Errors
    ///
    /// [`ClaError::NoData`], [`ClaError::UnknownReturns`] or [`ClaError::DimensionMismatch`]
    /// under the same conditions as [`CLA::allocate`]. Bounds are checked by
    /// [`CLA::allocate`], not here.
    pub fn _initialise(
        &mut self,
        asset_prices: &DMatrix<f64>,
        resample_by: Option<&str>,
        expected_asset_returns: Option<&DMatrix<f64>>,
        covariance_matrix: Option<&DMatrix<f64>>,
    ) -> Result<(), ClaError> {
        if let Some(exp) = expected_asset_returns {
            self.expected_returns = normalize_expected_returns(exp)?;
        } else if self.calculate_expected_returns == "mean" {
            let exp =
                ReturnsEstimation::calculate_mean_historical_returns(asset_prices, resample_by)?;
            self.expected_returns =
                normalize_expected_returns(&DMatrix::from_column_slice(exp.len(), 1, &exp))?;
        } else if self.calculate_expected_returns == "exponential" {
            let exp = ReturnsEstimation::calculate_exponential_historical_returns(
                asset_prices,
                resample_by,
                500,
            )?;
            self.expected_returns =
                normalize_expected_returns(&DMatrix::from_column_slice(exp.len(), 1, &exp))?;
        } else {
            return Err(ClaError::UnknownReturns(self.calculate_expected_returns.clone()));
        }

        if let Some(covariance_matrix) = covariance_matrix {
            self.cov_matrix = covariance_matrix.clone_owned();
        } else {
            let returns = ReturnsEstimation::calculate_returns(asset_prices, resample_by)?;
            self.cov_matrix = covariance(&returns);
        }

        let bounds = build_bounds(self.expected_returns.nrows(), &self.weight_bounds)?;
        self.lower_bounds = bounds.iter().map(|b| b.0).collect();
        self.upper_bounds = bounds.iter().map(|b| b.1).collect();
        self.weights.clear();
        self.lambdas.clear();
        self.gammas.clear();
        self.free_weights.clear();
        Ok(())
    }

    /// Removes turning points whose weights do not sum to 1 or break a bound by more than
    /// `tol` (called by [`CLA::allocate`]; public for parity testing).
    ///
    /// # Errors
    ///
    /// [`ClaError::IndexError`] if the result vectors or bounds disagree in length.
    pub fn _purge_num_err(&mut self, tol: f64) -> Result<(), ClaError> {
        if self.weights.len() != self.lambdas.len()
            || self.weights.len() != self.gammas.len()
            || self.weights.len() != self.free_weights.len()
        {
            return Err(ClaError::IndexError);
        }
        let mut i = 0;
        while i < self.weights.len() {
            let weights = &self.weights[i];
            let mut flag = (weights.iter().sum::<f64>() - 1.0).abs() > tol;
            if !flag {
                for (j, w) in weights.iter().enumerate() {
                    if j >= self.lower_bounds.len() || j >= self.upper_bounds.len() {
                        return Err(ClaError::IndexError);
                    }
                    if w - self.lower_bounds[j] < -tol || w - self.upper_bounds[j] > tol {
                        flag = true;
                        break;
                    }
                }
            }
            if flag {
                self.weights.remove(i);
                if i >= self.lambdas.len() || i >= self.gammas.len() || i >= self.free_weights.len()
                {
                    return Err(ClaError::IndexError);
                }
                self.lambdas.remove(i);
                self.gammas.remove(i);
                self.free_weights.remove(i);
            } else {
                i += 1;
            }
        }
        Ok(())
    }

    /// Removes turning points whose expected return is below that of a later point
    /// (called by [`CLA::allocate`]; public for parity testing).
    ///
    /// # Errors
    ///
    /// [`ClaError::IndexError`] if the result vectors disagree in length.
    pub fn _purge_excess(&mut self) -> Result<(), ClaError> {
        if self.weights.len() != self.lambdas.len()
            || self.weights.len() != self.gammas.len()
            || self.weights.len() != self.free_weights.len()
        {
            return Err(ClaError::IndexError);
        }
        let mut index_1: usize = 0;
        let mut repeat = false;
        loop {
            if !repeat {
                index_1 += 1;
            }
            if index_1 >= self.weights.len().saturating_sub(1) {
                break;
            }
            let mean = dot(&self.weights[index_1], self.expected_returns.column(0).as_slice());
            let mut index_2 = index_1 + 1;
            repeat = false;
            while index_2 < self.weights.len() {
                let mean_ = dot(&self.weights[index_2], self.expected_returns.column(0).as_slice());
                if mean < mean_ {
                    self.weights.remove(index_1);
                    self.lambdas.remove(index_1);
                    self.gammas.remove(index_1);
                    self.free_weights.remove(index_1);
                    repeat = true;
                    break;
                }
                index_2 += 1;
            }
        }
        Ok(())
    }
}

fn returns_and_frequency(
    prices: &DMatrix<f64>,
    resample_by: Option<&str>,
) -> Result<(DMatrix<f64>, f64), ClaError> {
    let step = freq_step(resample_by);
    let sampled_prices = resample_prices(prices, step);
    let returns = pct_change(&sampled_prices)?;
    if returns.nrows() == 0 {
        return Err(ClaError::NoData);
    }
    let freq = 252.0 / step as f64;
    Ok((returns, freq))
}

fn pct_change(prices: &DMatrix<f64>) -> Result<DMatrix<f64>, ClaError> {
    let rows = prices.nrows();
    let cols = prices.ncols();
    if rows < 2 {
        return Err(ClaError::NoData);
    }
    let mut out = DMatrix::<f64>::zeros(rows - 1, cols);
    for r in 1..rows {
        for c in 0..cols {
            let prev = prices[(r - 1, c)];
            if prev == 0.0 {
                return Err(ClaError::NoData);
            }
            out[(r - 1, c)] = prices[(r, c)] / prev - 1.0;
        }
    }
    Ok(out)
}

/// Sample covariance (ddof = 1) of a returns matrix with one row per observation and one
/// column per asset. Fewer than two rows give a zero matrix.
pub fn covariance(returns: &DMatrix<f64>) -> DMatrix<f64> {
    let rows = returns.nrows();
    let cols = returns.ncols();
    if rows < 2 {
        return DMatrix::<f64>::zeros(cols, cols);
    }
    let means: Vec<f64> = (0..cols).map(|c| returns.column(c).sum() / rows as f64).collect();
    let mut cov = DMatrix::<f64>::zeros(cols, cols);
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

fn normalize_expected_returns(exp: &DMatrix<f64>) -> Result<DMatrix<f64>, ClaError> {
    let n = exp.nrows().max(exp.ncols());
    if n == 0 {
        return Err(ClaError::NoData);
    }
    let mut out = if exp.ncols() == 1 {
        exp.clone_owned()
    } else if exp.nrows() == 1 {
        let row: Vec<f64> = exp.row(0).iter().copied().collect();
        DMatrix::from_column_slice(exp.ncols(), 1, &row)
    } else {
        return Err(ClaError::DimensionMismatch);
    };
    let mean = out.iter().sum::<f64>() / n as f64;
    if out.iter().all(|v| (*v - mean).abs() < 1e-12) {
        let last = out.nrows() - 1;
        out[(last, 0)] += 1e-5;
    }
    Ok(out)
}

fn build_bounds(n: usize, bounds: &WeightBounds) -> Result<Vec<(f64, f64)>, ClaError> {
    match bounds {
        WeightBounds::Tuple(lo, hi) => Ok(vec![(*lo, *hi); n]),
        WeightBounds::Lists(low, high) => {
            if low.len() != n || high.len() != n {
                return Err(ClaError::DimensionMismatch);
            }
            Ok(low.iter().copied().zip(high.iter().copied()).collect())
        }
    }
}

fn check_bounds_feasible(bounds: &[(f64, f64)]) -> Result<(), ClaError> {
    if bounds.iter().any(|(lo, hi)| !lo.is_finite() || !hi.is_finite()) {
        return Err(ClaError::InfeasibleBounds("every bound must be finite"));
    }
    if bounds.iter().any(|(lo, hi)| lo > hi) {
        return Err(ClaError::InfeasibleBounds("a lower bound exceeds its upper bound"));
    }
    let lower: f64 = bounds.iter().map(|b| b.0).sum();
    let upper: f64 = bounds.iter().map(|b| b.1.min(1.0)).sum();
    if lower - 1.0 > 1e-9 {
        return Err(ClaError::InfeasibleBounds("the lower bounds sum to more than 1"));
    }
    if upper + 1e-9 < 1.0 {
        return Err(ClaError::InfeasibleBounds("the upper bounds sum to less than 1"));
    }
    Ok(())
}

/// One corner of the efficient frontier: the portfolio at which the set of assets strictly
/// inside their bounds (`free`) changes.
struct TurningPoint {
    weights: Vec<f64>,
    lambda: f64,
    gamma: f64,
    free: Vec<usize>,
}

/// What an asset's weight is pinned to when `compute_lambda` evaluates it.
enum Pin {
    /// Leaving the free set: it lands on whichever bound the sign of the derivative picks.
    Bounds(f64, f64),
    /// Entering the free set: it starts from its current, bounded weight.
    Value(f64),
}

fn select(m: &DMatrix<f64>, rows: &[usize], cols: &[usize]) -> DMatrix<f64> {
    DMatrix::from_fn(rows.len(), cols.len(), |i, j| m[(rows[i], cols[j])])
}

struct Blocks {
    covar_f_inv: DMatrix<f64>,
    covar_fb: DMatrix<f64>,
    mean_f: DVector<f64>,
    w_b: DVector<f64>,
}

fn blocks(
    free: &[usize],
    mean: &[f64],
    cov: &DMatrix<f64>,
    weights: &[f64],
) -> Result<Blocks, ClaError> {
    let bounded: Vec<usize> = (0..mean.len()).filter(|i| !free.contains(i)).collect();
    let covar_f_inv = select(cov, free, free).try_inverse().ok_or(ClaError::SingularCovariance)?;
    Ok(Blocks {
        covar_f_inv,
        covar_fb: select(cov, free, &bounded),
        mean_f: DVector::from_iterator(free.len(), free.iter().map(|&i| mean[i])),
        w_b: DVector::from_iterator(bounded.len(), bounded.iter().map(|&i| weights[i])),
    })
}

/// The value of lambda at which free asset `i` (a position within the free set) reaches `pin`.
/// `None` when the weight does not depend on lambda. Bailey and Lopez de Prado (2013), eq. 4.
fn compute_lambda(b: &Blocks, i: usize, pin: Pin) -> Option<(f64, f64)> {
    let ones = DVector::from_element(b.mean_f.len(), 1.0);
    let c1 = (ones.transpose() * &b.covar_f_inv * &ones)[(0, 0)];
    let c2 = &b.covar_f_inv * &b.mean_f;
    let c3 = (ones.transpose() * &c2)[(0, 0)];
    let c4 = &b.covar_f_inv * &ones;
    let c = -c1 * c2[i] + c3 * c4[i];
    if c == 0.0 || !c.is_finite() {
        return None;
    }
    let bi = match pin {
        Pin::Bounds(lower, upper) => {
            if c > 0.0 {
                upper
            } else {
                lower
            }
        }
        Pin::Value(v) => v,
    };
    let lambda = if b.w_b.is_empty() {
        (c4[i] - c1 * bi) / c
    } else {
        let l1 = b.w_b.sum();
        let l3 = &b.covar_f_inv * &b.covar_fb * &b.w_b;
        let l2 = l3.sum();
        ((1.0 - l1 + l2) * c4[i] - c1 * (bi + l3[i])) / c
    };
    Some((lambda, bi))
}

/// The free weights, and gamma, on the critical line at `lambda`. Eq. 3 of the same paper.
fn compute_w(b: &Blocks, lambda: f64, mean_f: &DVector<f64>) -> (DVector<f64>, f64) {
    let ones = DVector::from_element(mean_f.len(), 1.0);
    let g1 = (ones.transpose() * &b.covar_f_inv * mean_f)[(0, 0)];
    let g2 = (ones.transpose() * &b.covar_f_inv * &ones)[(0, 0)];
    let (w1, gamma) = if b.w_b.is_empty() {
        (DVector::zeros(mean_f.len()), -lambda * g1 / g2 + 1.0 / g2)
    } else {
        let w1 = &b.covar_f_inv * &b.covar_fb * &b.w_b;
        let gamma = -lambda * g1 / g2 + (1.0 - b.w_b.sum() + w1.sum()) / g2;
        (w1, gamma)
    };
    let w = -w1 + (&b.covar_f_inv * &ones) * gamma + (&b.covar_f_inv * mean_f) * lambda;
    (w, gamma)
}

/// The Critical Line Algorithm: every turning point of the efficient frontier under box bounds
/// and the budget constraint, from the maximum-return corner down to minimum variance.
fn critical_line(
    mean: &[f64],
    cov: &DMatrix<f64>,
    lower: &[f64],
    upper: &[f64],
) -> Result<Vec<TurningPoint>, ClaError> {
    let n = mean.len();
    if n == 0 {
        return Err(ClaError::NoData);
    }

    // Start at the maximum-return corner: everything at its lower bound, then fill the
    // highest-mean assets to their upper bounds until the budget is spent. The asset the
    // budget runs out on is the only free one.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| mean[a].total_cmp(&mean[b]).then(a.cmp(&b)));
    let mut w = lower.to_vec();
    let mut at = n;
    while w.iter().sum::<f64>() < 1.0 && at > 0 {
        at -= 1;
        w[order[at]] = upper[order[at]];
    }
    if at == n {
        // The lower bounds already spend the budget: there is one feasible portfolio.
        return Ok(vec![TurningPoint { weights: w, lambda: 0.0, gamma: 0.0, free: Vec::new() }]);
    }
    w[order[at]] += 1.0 - w.iter().sum::<f64>();
    let mut free = vec![order[at]];

    // lambda is +infinity at the first corner: only return matters there.
    let mut points = vec![TurningPoint {
        weights: w.clone(),
        lambda: f64::INFINITY,
        gamma: 0.0,
        free: free.clone(),
    }];

    for _ in 0..(4 * n * n + 100) {
        let last_lambda = points.last().map(|p| p.lambda).unwrap_or(f64::INFINITY);

        // (a) A free asset hits a bound.
        let mut going_in: Option<(f64, usize, f64)> = None;
        if free.len() > 1 {
            let b = blocks(&free, mean, cov, &w)?;
            for (j, &i) in free.iter().enumerate() {
                if let Some((l, bi)) = compute_lambda(&b, j, Pin::Bounds(lower[i], upper[i]))
                    && going_in.is_none_or(|(best, _, _)| l > best)
                {
                    going_in = Some((l, i, bi));
                }
            }
        }

        // (b) A bounded asset becomes free.
        let mut going_out: Option<(f64, usize)> = None;
        if free.len() < n {
            for i in (0..n).filter(|i| !free.contains(i)) {
                let mut candidate = free.clone();
                candidate.push(i);
                let b = blocks(&candidate, mean, cov, &w)?;
                if let Some((l, _)) = compute_lambda(&b, candidate.len() - 1, Pin::Value(w[i]))
                    && l < last_lambda
                    && going_out.is_none_or(|(best, _)| l > best)
                {
                    going_out = Some((l, i));
                }
            }
        }

        let l_in = going_in.map(|g| g.0).filter(|l| *l >= 0.0);
        let l_out = going_out.map(|g| g.0).filter(|l| *l >= 0.0);
        let (lambda, mean_for_w) = match (l_in, l_out) {
            // Neither event happens at a non-negative lambda: finish at minimum variance.
            (None, None) => (0.0, None),
            (Some(a), b) if b.is_none_or(|b| a > b) => {
                let (_, i, bi) = going_in.expect("l_in came from going_in");
                free.retain(|&x| x != i);
                w[i] = bi;
                (a, Some(()))
            }
            _ => {
                let (l, i) = going_out.expect("l_out came from going_out");
                free.push(i);
                (l, Some(()))
            }
        };

        let b = blocks(&free, mean, cov, &w)?;
        let mean_f =
            if mean_for_w.is_some() { b.mean_f.clone() } else { DVector::zeros(free.len()) };
        let (w_f, gamma) = compute_w(&b, lambda, &mean_f);
        for (j, &i) in free.iter().enumerate() {
            w[i] = w_f[j];
        }
        points.push(TurningPoint { weights: w.clone(), lambda, gamma, free: free.clone() });
        if lambda == 0.0 {
            return Ok(points);
        }
    }
    Err(ClaError::NoTermination)
}

/// The frontier is piecewise linear in the weights between turning points, and the Sharpe ratio
/// is quasi-concave along each piece, so a golden-section search per segment finds the maximum.
///
/// [`ClaError::NoTurningPoints`] if `points` is empty. `allocate` checks that first, so this is
/// a second guard; it replaced a `points[0]` that would have panicked.
fn max_sharpe_on_frontier(
    points: &[Vec<f64>],
    mean: &[f64],
    cov: &DMatrix<f64>,
) -> Result<Vec<f64>, ClaError> {
    let sharpe = |w: &[f64]| {
        let sigma = quad_risk(cov, w).sqrt();
        if sigma > 0.0 {
            dot(w, mean) / sigma
        } else {
            f64::NEG_INFINITY
        }
    };
    let blend = |w0: &[f64], w1: &[f64], a: f64| -> Vec<f64> {
        w0.iter().zip(w1).map(|(x, y)| a * x + (1.0 - a) * y).collect()
    };

    let mut best = points.first().ok_or(ClaError::NoTurningPoints)?.clone();
    for pair in points.windows(2) {
        let ratio = (5.0f64.sqrt() - 1.0) / 2.0;
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..200 {
            let (x1, x2) = (hi - ratio * (hi - lo), lo + ratio * (hi - lo));
            if sharpe(&blend(&pair[0], &pair[1], x1)) > sharpe(&blend(&pair[0], &pair[1], x2)) {
                hi = x2;
            } else {
                lo = x1;
            }
        }
        let candidate = blend(&pair[0], &pair[1], 0.5 * (lo + hi));
        if sharpe(&candidate) > sharpe(&best) {
            best = candidate;
        }
    }
    Ok(best)
}

/// About `points` portfolios spread evenly along each segment between turning points, from
/// maximum return to minimum variance.
fn frontier_points(turning_points: &[Vec<f64>], points: usize) -> Vec<Vec<f64>> {
    if turning_points.len() < 2 {
        return turning_points.to_vec();
    }
    let per_segment = (points / turning_points.len()).max(2);
    let segments = turning_points.len() - 1;
    let mut out = Vec::new();
    for (s, pair) in turning_points.windows(2).enumerate() {
        // Each segment leaves its end to the next one; the last keeps it.
        let steps = if s + 1 == segments { per_segment } else { per_segment - 1 };
        for k in 0..steps {
            let j = k as f64 / (per_segment - 1) as f64;
            out.push(
                pair[0].iter().zip(&pair[1]).map(|(w0, w1)| w1 * j + (1.0 - j) * w0).collect(),
            );
        }
    }
    out
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn quad_risk(cov: &DMatrix<f64>, w: &[f64]) -> f64 {
    let wv = DVector::from_vec(w.to_vec());
    (wv.transpose() * cov * wv)[(0, 0)]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #168: `max_sharpe_on_frontier` indexed `points[0]` and panicked on an empty frontier.
    /// `allocate` now returns `NoTurningPoints` before it gets here, so the guard is tested
    /// directly.
    #[test]
    fn max_sharpe_on_an_empty_frontier_is_an_error() {
        let cov = DMatrix::from_row_slice(2, 2, &[0.04, 0.0, 0.0, 0.01]);
        assert_eq!(max_sharpe_on_frontier(&[], &[0.1, 0.05], &cov), Err(ClaError::NoTurningPoints));
        let one = vec![vec![0.5, 0.5]];
        assert_eq!(max_sharpe_on_frontier(&one, &[0.1, 0.05], &cov), Ok(vec![0.5, 0.5]));
    }
}
