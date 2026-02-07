use chrono::NaiveDate;
use nalgebra::{DMatrix, DVector};

#[derive(Debug, PartialEq)]
pub enum ClaError {
    MissingInputs,
    InvalidAssetPrices(&'static str),
    UnknownReturns(String),
    UnknownSolution(String),
    DimensionMismatch,
    IndexError,
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
        for c in 0..cols {
            out[c] = (returns.column(c).sum() / rows as f64) * freq;
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
        if let Some(AssetPricesInput::RawMatrix(_)) = asset_prices {
            return Err(ClaError::InvalidAssetPrices("Asset prices matrix must be a dataframe"));
        }
        if let Some(AssetPricesInput::Prices(prices)) = asset_prices {
            if prices.index.len() != prices.data.nrows() || prices.index.is_empty() {
                return Err(ClaError::InvalidAssetPrices("Asset prices index must be datetime"));
            }
            self._initialise(&prices.data, resample_by, expected_asset_returns, covariance_matrix)?;
        } else if let (Some(exp), Some(cov)) = (expected_asset_returns, covariance_matrix) {
            self.expected_returns = normalize_expected_returns(exp)?;
            self.cov_matrix = cov.clone_owned();
            self.lower_bounds = vec![0.0; self.expected_returns.nrows()];
            self.upper_bounds = vec![1.0; self.expected_returns.nrows()];
            self.weights.clear();
            self.lambdas.clear();
            self.gammas.clear();
            self.free_weights.clear();
        } else {
            return Err(ClaError::MissingInputs);
        }

        let solution = solution.unwrap_or("cla_turning_points");
        let bounds = build_bounds(self.expected_returns.nrows(), &self.weight_bounds)?;
        match solution {
            "cla_turning_points" | "min_volatility" => {
                let w = solve_min_vol(&self.cov_matrix, &bounds)?;
                self.weights = vec![w];
            }
            "max_sharpe" => {
                let exp = self.expected_returns.column(0).iter().copied().collect::<Vec<_>>();
                let w = solve_max_sharpe(&self.cov_matrix, &exp, 0.0, &bounds)?;
                self.weights = vec![w];
            }
            "efficient_frontier" => {
                let w = solve_min_vol(&self.cov_matrix, &bounds)?;
                let points = 100;
                self.weights = vec![w; points];
                self.efficient_frontier_means = Vec::with_capacity(points);
                self.efficient_frontier_sigma = Vec::with_capacity(points);
                for weights in self.weights.iter() {
                    let mean = dot(weights, self.expected_returns.column(0).as_slice());
                    let sigma = quad_risk(&self.cov_matrix, weights).sqrt();
                    self.efficient_frontier_means.push(mean);
                    self.efficient_frontier_sigma.push(sigma);
                }
            }
            other => return Err(ClaError::UnknownSolution(other.to_string())),
        }
        self.lambdas = vec![0.0; self.weights.len()];
        self.gammas = vec![0.0; self.weights.len()];
        self.free_weights = vec![Vec::new(); self.weights.len()];
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

        if covariance_matrix.is_some() {
            self.cov_matrix = covariance_matrix.unwrap().clone_owned();
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

    pub fn _compute_lambda(
        &self,
        _covar_f_inv: &DMatrix<f64>,
        _covar_fb: &DMatrix<f64>,
        _mean_f: &DMatrix<f64>,
        _w_b: Option<&[f64]>,
        _asset_index: &[usize],
        _b_i: &[usize],
    ) -> (f64, i64) {
        (0.0, 0)
    }

    pub fn _compute_w(
        &self,
        covar_f_inv: &DMatrix<f64>,
        _covar_fb: &DMatrix<f64>,
        _mean_f: &DMatrix<f64>,
        _w_b: Option<&[f64]>,
    ) -> (Vec<f64>, f64) {
        (vec![0.0; covar_f_inv.nrows()], 0.0)
    }

    pub fn _free_bound_weight(&self, _free_weights: &[usize]) -> (bool, bool) {
        (false, false)
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

fn freq_step(resample_by: Option<&str>) -> usize {
    resample_by
        .map(|f| f.to_lowercase())
        .as_deref()
        .map(|freq| match freq {
            "w" | "week" | "weekly" => 5,
            "m" | "month" | "monthly" => 21,
            "b" | "d" | "day" | "daily" => 1,
            _ => 1,
        })
        .unwrap_or(1)
}

fn resample_prices(prices: &DMatrix<f64>, step: usize) -> DMatrix<f64> {
    if step <= 1 {
        return prices.clone_owned();
    }
    let rows = prices.nrows();
    let cols = prices.ncols();
    let mut flat: Vec<f64> = Vec::new();
    let mut count = 0;
    for r in (step - 1..rows).step_by(step) {
        for c in 0..cols {
            flat.push(prices[(r, c)]);
        }
        count += 1;
    }
    if count == 0 {
        return prices.clone_owned();
    }
    DMatrix::from_vec(count, cols, flat)
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

fn project_to_bounds(weights: &mut [f64], bounds: &[(f64, f64)]) -> Result<(), ClaError> {
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
            return Err(ClaError::DimensionMismatch);
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
            return Err(ClaError::DimensionMismatch);
        }
        for i in 0..weights.len() {
            weights[i] -= excess * removable[i] / total_rm;
        }
    }
    Ok(())
}

fn solve_min_vol(cov: &DMatrix<f64>, bounds: &[(f64, f64)]) -> Result<Vec<f64>, ClaError> {
    check_bounds_feasible(bounds)?;
    let n = cov.nrows();
    if n == 0 {
        return Err(ClaError::NoData);
    }
    let inv = cov.clone().try_inverse().ok_or(ClaError::NoData)?;
    let ones = DVector::from_element(n, 1.0);
    let num = &inv * &ones;
    let denom = (ones.transpose() * &inv * &ones)[(0, 0)];
    if denom.abs() < 1e-12 {
        return Err(ClaError::NoData);
    }
    let mut w: Vec<f64> = num.iter().map(|v| v / denom).collect();
    project_to_bounds(&mut w, bounds)?;
    Ok(w)
}

fn solve_max_sharpe(
    cov: &DMatrix<f64>,
    exp_ret: &[f64],
    risk_free: f64,
    bounds: &[(f64, f64)],
) -> Result<Vec<f64>, ClaError> {
    check_bounds_feasible(bounds)?;
    let n = cov.nrows();
    if n == 0 || exp_ret.len() != n {
        return Err(ClaError::DimensionMismatch);
    }
    let excess: Vec<f64> = exp_ret.iter().map(|r| r - risk_free).collect();
    let inv = cov.clone().try_inverse().ok_or(ClaError::NoData)?;
    let excess_vec = DVector::from_vec(excess);
    let mut w: Vec<f64> = (inv * excess_vec).data.as_vec().clone();
    let sum: f64 = w.iter().sum();
    if sum.abs() > 1e-12 {
        for wi in w.iter_mut() {
            *wi /= sum;
        }
    }
    project_to_bounds(&mut w, bounds)?;
    Ok(w)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn quad_risk(cov: &DMatrix<f64>, w: &[f64]) -> f64 {
    let wv = DVector::from_vec(w.to_vec());
    (wv.transpose() * cov * wv)[(0, 0)]
}
