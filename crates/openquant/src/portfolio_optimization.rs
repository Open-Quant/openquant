use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

#[derive(Debug, PartialEq)]
pub enum AllocError {
    NoData,
    UnknownSolution(String),
    UnknownReturns(String),
    InfeasibleBounds { lower_sum: f64, upper_sum: f64 },
    OptimizationFailed(&'static str),
    DimensionMismatch,
    NaNResult(&'static str),
}

#[derive(Clone, Copy)]
pub enum ReturnsMethod {
    Mean,
    Exponential { span: usize },
}

impl Default for ReturnsMethod {
    fn default() -> Self {
        ReturnsMethod::Mean
    }
}

#[derive(Clone)]
pub struct AllocationOptions<'a> {
    pub risk_free_rate: f64,
    pub target_return: f64,
    pub bounds: Option<HashMap<usize, (f64, f64)>>,
    pub tuple_bounds: Option<(f64, f64)>,
    pub resample_by: Option<&'a str>,
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

#[derive(Debug, Clone)]
pub struct MeanVariance {
    pub weights: Vec<f64>,
    pub portfolio_risk: f64,
    pub portfolio_return: f64,
    pub portfolio_sharpe: f64,
}

pub fn returns_method_from_str(name: &str) -> Result<ReturnsMethod, AllocError> {
    match name.to_lowercase().as_str() {
        "mean" | "mean_historical" => Ok(ReturnsMethod::Mean),
        "exponential" | "exponential_historical" => Ok(ReturnsMethod::Exponential { span: 500 }),
        other => Err(AllocError::UnknownReturns(other.to_string())),
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
            out[(r - 1, c)] = (prices[(r, c)] / prev).ln();
        }
    }
    Ok(out)
}

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
            for c in 0..cols {
                expected[c] = (returns.column(c).sum() / rows as f64) * freq;
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
    Ok((expected, covariance(&returns)))
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

fn ones(n: usize) -> DVector<f64> {
    DVector::from_element(n, 1.0)
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

fn solve_min_vol(cov: &DMatrix<f64>, bounds: &[(f64, f64)]) -> Result<Vec<f64>, AllocError> {
    check_bounds_feasible(bounds)?;
    let n = cov.nrows();
    if n == 0 {
        return Err(AllocError::NoData);
    }
    // Closed form: w = inv(C)1 / (1^T inv(C) 1)
    let inv = cov
        .clone()
        .try_inverse()
        .ok_or(AllocError::OptimizationFailed("covariance not invertible"))?;
    let ones = ones(n);
    let num = &inv * &ones;
    let denom = (ones.transpose() * &inv * &ones)[(0, 0)];
    if denom.abs() < 1e-12 {
        return Err(AllocError::OptimizationFailed("degenerate covariance"));
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
) -> Result<Vec<f64>, AllocError> {
    check_bounds_feasible(bounds)?;
    let n = cov.nrows();
    if n == 0 || exp_ret.len() != n {
        return Err(AllocError::DimensionMismatch);
    }
    let excess: Vec<f64> = exp_ret.iter().map(|r| r - risk_free).collect();
    let excess_vec = DVector::from_vec(excess.clone());
    let inv = cov
        .clone()
        .try_inverse()
        .ok_or(AllocError::OptimizationFailed("covariance not invertible"))?;
    let mut w: Vec<f64> = (inv.clone() * excess_vec).data.as_vec().clone();
    // normalize to sum 1
    let sum: f64 = w.iter().sum();
    if sum.abs() > 1e-12 {
        for wi in w.iter_mut() {
            *wi /= sum;
        }
    }
    if w.iter().all(|v| !v.is_finite()) {
        return Err(AllocError::NaNResult("weights not finite"));
    }
    project_to_bounds(&mut w, bounds)?;
    Ok(w)
}

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
    let inv = cov
        .clone()
        .try_inverse()
        .ok_or(AllocError::OptimizationFailed("covariance not invertible"))?;
    let ones = ones(n);
    let mu = DVector::from_vec(exp_ret.to_vec());
    let a = (ones.transpose() * &inv * &ones)[(0, 0)];
    let b = (ones.transpose() * &inv * &mu)[(0, 0)];
    let c = (mu.transpose() * &inv * &mu)[(0, 0)];
    let denom = a * c - b * b;
    if denom.abs() < 1e-12 {
        return Err(AllocError::OptimizationFailed("degenerate frontier"));
    }
    let lambda = (c - b * target_return) / denom;
    let gamma = (a * target_return - b) / denom;
    let w_vec = (&inv * (&mu * lambda + &ones * gamma)).data.as_vec().clone();
    let mut w = w_vec;
    project_to_bounds(&mut w, bounds)?;
    Ok(w)
}

pub fn allocate_inverse_variance(prices: &DMatrix<f64>) -> Result<MeanVariance, AllocError> {
    allocate_inverse_variance_with(prices, &AllocationOptions::default())
}

pub fn allocate_inverse_variance_with(
    prices: &DMatrix<f64>,
    opts: &AllocationOptions,
) -> Result<MeanVariance, AllocError> {
    let (exp_ret, cov) = returns_and_means(prices, opts)?;
    let bounds = build_bounds(cov.nrows(), &opts.bounds, opts.tuple_bounds);
    let w = inverse_variance(&cov, &bounds)?;
    let risk = quad_risk(&cov, &w).sqrt();
    Ok(MeanVariance {
        weights: w.clone(),
        portfolio_risk: risk,
        portfolio_return: dot(&exp_ret, &w),
        portfolio_sharpe: 0.0,
    })
}

pub fn allocate_min_vol(
    prices: &DMatrix<f64>,
    bounds: Option<HashMap<usize, (f64, f64)>>,
    tuple_bounds: Option<(f64, f64)>,
) -> Result<MeanVariance, AllocError> {
    let opts = AllocationOptions { bounds, tuple_bounds, ..Default::default() };
    allocate_min_vol_with(prices, &opts)
}

pub fn allocate_min_vol_with(
    prices: &DMatrix<f64>,
    opts: &AllocationOptions,
) -> Result<MeanVariance, AllocError> {
    let (exp_ret, cov) = returns_and_means(prices, opts)?;
    let bounds = build_bounds(cov.nrows(), &opts.bounds, opts.tuple_bounds);
    let w = solve_min_vol(&cov, &bounds)?;
    let risk = quad_risk(&cov, &w).sqrt();
    Ok(MeanVariance {
        weights: w.clone(),
        portfolio_risk: risk,
        portfolio_return: dot(&exp_ret, &w),
        portfolio_sharpe: 0.0,
    })
}

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

pub fn allocate_max_sharpe_with(
    prices: &DMatrix<f64>,
    opts: &AllocationOptions,
) -> Result<MeanVariance, AllocError> {
    let (exp_ret, cov) = returns_and_means(prices, opts)?;
    let bounds = build_bounds(cov.nrows(), &opts.bounds, opts.tuple_bounds);
    let weights = solve_max_sharpe(&cov, &exp_ret, opts.risk_free_rate, &bounds)?;
    let risk = quad_risk(&cov, &weights).sqrt();
    if !risk.is_finite() {
        return Err(AllocError::NaNResult("risk not finite"));
    }
    let port_ret = dot(&exp_ret, &weights);
    let sharpe = if risk > 0.0 { (port_ret - opts.risk_free_rate) / risk } else { 0.0 };
    Ok(MeanVariance {
        weights,
        portfolio_risk: risk,
        portfolio_return: port_ret,
        portfolio_sharpe: sharpe,
    })
}

pub fn allocate_efficient_risk(
    prices: &DMatrix<f64>,
    target_return: f64,
    bounds: Option<HashMap<usize, (f64, f64)>>,
    tuple_bounds: Option<(f64, f64)>,
) -> Result<MeanVariance, AllocError> {
    let opts = AllocationOptions { target_return, bounds, tuple_bounds, ..Default::default() };
    allocate_efficient_risk_with(prices, &opts)
}

pub fn allocate_efficient_risk_with(
    prices: &DMatrix<f64>,
    opts: &AllocationOptions,
) -> Result<MeanVariance, AllocError> {
    let (exp_ret, cov) = returns_and_means(prices, opts)?;
    let bounds = build_bounds(cov.nrows(), &opts.bounds, opts.tuple_bounds);
    let weights = efficient_risk_from_inputs(
        &exp_ret,
        &cov,
        opts.target_return,
        &bounds,
        opts.risk_free_rate,
    )?;
    let risk = quad_risk(&cov, &weights).sqrt();
    Ok(MeanVariance {
        weights: weights.clone(),
        portfolio_risk: risk,
        portfolio_return: dot(&exp_ret, &weights),
        portfolio_sharpe: 0.0,
    })
}

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
    match solution {
        "inverse_variance" => {
            let w = inverse_variance(covariance, &bounds)?;
            Ok(MeanVariance {
                portfolio_risk: quad_risk(covariance, &w).sqrt(),
                portfolio_return: dot(expected_returns, &w),
                portfolio_sharpe: 0.0,
                weights: w,
            })
        }
        "min_volatility" => {
            let w = solve_min_vol(covariance, &bounds)?;
            Ok(MeanVariance {
                portfolio_risk: quad_risk(covariance, &w).sqrt(),
                portfolio_return: dot(expected_returns, &w),
                portfolio_sharpe: 0.0,
                weights: w,
            })
        }
        "max_sharpe" => {
            let w = solve_max_sharpe(covariance, expected_returns, opts.risk_free_rate, &bounds)?;
            let risk = quad_risk(covariance, &w).sqrt();
            Ok(MeanVariance {
                portfolio_risk: risk,
                portfolio_return: dot(expected_returns, &w),
                portfolio_sharpe: if risk > 0.0 {
                    (dot(expected_returns, &w) - opts.risk_free_rate) / risk
                } else {
                    0.0
                },
                weights: w,
            })
        }
        "efficient_risk" => {
            let w = efficient_risk_from_inputs(
                expected_returns,
                covariance,
                opts.target_return,
                &bounds,
                opts.risk_free_rate,
            )?;
            Ok(MeanVariance {
                portfolio_risk: quad_risk(covariance, &w).sqrt(),
                portfolio_return: dot(expected_returns, &w),
                portfolio_sharpe: 0.0,
                weights: w,
            })
        }
        other => Err(AllocError::UnknownSolution(other.to_string())),
    }
}

pub fn allocate_with_solution(
    prices: &DMatrix<f64>,
    solution: &str,
    opts: &AllocationOptions,
) -> Result<MeanVariance, AllocError> {
    let (exp_ret, cov) = returns_and_means(prices, opts)?;
    allocate_from_inputs(&exp_ret, &cov, solution, opts)
}
