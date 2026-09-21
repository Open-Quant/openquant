use crate::util::resample::{freq_step, resample_prices};
use chrono::NaiveDate;
use nalgebra::{DMatrix, DVector};

#[derive(Debug, PartialEq, thiserror::Error)]
pub enum ClaError {
    #[error("supply asset prices, or expected returns and a covariance matrix")]
    MissingInputs,
    #[error("invalid asset prices: {0}")]
    InvalidAssetPrices(&'static str),
    #[error("unknown returns method: {0}")]
    UnknownReturns(String),
    #[error("unknown solution: {0}")]
    UnknownSolution(String),
    #[error("inputs disagree on the number of assets")]
    DimensionMismatch,
    #[error("asset index out of range")]
    IndexError,
    #[error("no data")]
    NoData,
}

#[derive(Clone)]
pub struct AssetPrices {
    pub data: DMatrix<f64>,
    pub index: Vec<NaiveDate>,
}

impl AssetPrices {
    pub fn new(data: DMatrix<f64>, index: Vec<NaiveDate>) -> Self {
        AssetPrices { data, index }
    }
}

pub enum AssetPricesInput<'a> {
    Prices(&'a AssetPrices),
    RawMatrix(&'a DMatrix<f64>),
}

#[derive(Clone)]
pub enum WeightBounds {
    Tuple(f64, f64),
    Lists(Vec<f64>, Vec<f64>),
}

pub struct ReturnsEstimation;

impl ReturnsEstimation {
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

    pub fn calculate_returns(
        asset_prices: &DMatrix<f64>,
        resample_by: Option<&str>,
    ) -> Result<DMatrix<f64>, ClaError> {
        let (returns, _freq) = returns_and_frequency(asset_prices, resample_by)?;
        Ok(returns)
    }
}

pub struct CLA {
    pub weight_bounds: WeightBounds,
    pub calculate_expected_returns: String,
    pub weights: Vec<Vec<f64>>,
    pub lambdas: Vec<f64>,
    pub gammas: Vec<f64>,
    pub free_weights: Vec<Vec<usize>>,
    pub expected_returns: DMatrix<f64>,
    pub cov_matrix: DMatrix<f64>,
    pub lower_bounds: Vec<f64>,
    pub upper_bounds: Vec<f64>,
    pub efficient_frontier_means: Vec<f64>,
    pub efficient_frontier_sigma: Vec<f64>,
}

impl Default for CLA {
    fn default() -> Self {
        CLA::new(WeightBounds::Tuple(0.0, 1.0), "mean")
    }
}

impl CLA {
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
                    return Err(ClaError::InvalidAssetPrices(
                        "Asset prices index must be datetime",
                    ));
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
                    .ok_or(ClaError::NoData)?;
                self.weights = vec![best.clone()];
            }
            "max_sharpe" => {
                self.weights =
                    vec![max_sharpe_on_frontier(&turning_points, &mean, &self.cov_matrix)];
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
    let lower: f64 = bounds.iter().map(|b| b.0).sum();
    let upper: f64 = bounds.iter().map(|b| b.1.min(1.0)).sum();
    if lower - 1.0 > 1e-9 || upper + 1e-9 < 1.0 {
        Err(ClaError::DimensionMismatch)
    } else {
        Ok(())
    }
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
    let covar_f_inv = select(cov, free, free)
        .try_inverse()
        .ok_or(ClaError::InvalidAssetPrices("covariance of the free assets is singular"))?;
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
                if let Some((l, bi)) = compute_lambda(&b, j, Pin::Bounds(lower[i], upper[i])) {
                    if going_in.is_none_or(|(best, _, _)| l > best) {
                        going_in = Some((l, i, bi));
                    }
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
                if let Some((l, _)) = compute_lambda(&b, candidate.len() - 1, Pin::Value(w[i])) {
                    if l < last_lambda && going_out.is_none_or(|(best, _)| l > best) {
                        going_out = Some((l, i));
                    }
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
    Err(ClaError::InvalidAssetPrices("the critical line did not terminate"))
}

/// The frontier is piecewise linear in the weights between turning points, and the Sharpe ratio
/// is quasi-concave along each piece, so a golden-section search per segment finds the maximum.
fn max_sharpe_on_frontier(points: &[Vec<f64>], mean: &[f64], cov: &DMatrix<f64>) -> Vec<f64> {
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

    let mut best = points[0].clone();
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
    best
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
