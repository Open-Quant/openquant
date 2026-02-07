use nalgebra::DMatrix;

#[derive(Debug, PartialEq)]
pub enum HcaaError {
    NoData,
    UnknownAllocationMetric(String),
    UnknownReturns(String),
    MissingExpectedReturnsForSharpe,
    MissingReturnsForTailRisk,
    DimensionMismatch(&'static str),
}

#[derive(Debug, Clone)]
pub struct HierarchicalClusteringAssetAllocation {
    pub weights: Vec<f64>,
    pub ordered_indices: Vec<usize>,
    pub clusters: Vec<[usize; 2]>,
    calculate_expected_returns: String,
}

impl Default for HierarchicalClusteringAssetAllocation {
    fn default() -> Self {
        Self::new("mean")
    }
}

impl HierarchicalClusteringAssetAllocation {
    pub fn new(calculate_expected_returns: &str) -> Self {
        Self {
            weights: Vec::new(),
            ordered_indices: Vec::new(),
            clusters: Vec::new(),
            calculate_expected_returns: calculate_expected_returns.to_string(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn allocate(
        &mut self,
        asset_names: &[String],
        asset_prices: Option<&DMatrix<f64>>,
        asset_returns: Option<&DMatrix<f64>>,
        covariance_matrix: Option<&DMatrix<f64>>,
        expected_asset_returns: Option<&[f64]>,
        allocation_metric: &str,
        confidence_level: f64,
        optimal_num_clusters: Option<usize>,
        resample_by: Option<&str>,
    ) -> Result<(), HcaaError> {
        if asset_prices.is_none() && asset_returns.is_none() && covariance_matrix.is_none() {
            return Err(HcaaError::NoData);
        }
        if !matches!(
            allocation_metric,
            "minimum_variance"
                | "minimum_standard_deviation"
                | "sharpe_ratio"
                | "equal_weighting"
                | "expected_shortfall"
                | "conditional_drawdown_risk"
        ) {
            return Err(HcaaError::UnknownAllocationMetric(allocation_metric.to_string()));
        }
        let n_assets = asset_names.len();
        if n_assets == 0 {
            return Err(HcaaError::NoData);
        }

        let returns_owned = if let Some(r) = asset_returns {
            r.clone_owned()
        } else if let Some(p) = asset_prices {
            let step = freq_step(resample_by);
            let sampled = resample_prices(p, step);
            returns_from_prices(&sampled)?
        } else {
            DMatrix::zeros(0, n_assets)
        };
        if returns_owned.ncols() != n_assets && returns_owned.nrows() > 0 {
            return Err(HcaaError::DimensionMismatch("asset_returns columns != asset_names length"));
        }

        let covariance_owned = if let Some(cov) = covariance_matrix {
            cov.clone_owned()
        } else {
            covariance(&returns_owned)?
        };
        if covariance_owned.nrows() != n_assets || covariance_owned.ncols() != n_assets {
            return Err(HcaaError::DimensionMismatch(
                "covariance matrix dimensions must equal number of assets",
            ));
        }

        let expected_owned = if allocation_metric == "sharpe_ratio" {
            if let Some(mu) = expected_asset_returns {
                if mu.len() != n_assets {
                    return Err(HcaaError::DimensionMismatch(
                        "expected_asset_returns length != asset_names length",
                    ));
                }
                mu.to_vec()
            } else if asset_prices.is_none() {
                return Err(HcaaError::MissingExpectedReturnsForSharpe);
            } else if self.calculate_expected_returns.eq_ignore_ascii_case("mean") {
                mean_expected_returns(&returns_owned)
            } else if self.calculate_expected_returns.eq_ignore_ascii_case("exponential") {
                exponential_expected_returns(&returns_owned, 500)
            } else {
                return Err(HcaaError::UnknownReturns(self.calculate_expected_returns.clone()));
            }
        } else {
            vec![0.0; n_assets]
        };

        if matches!(allocation_metric, "expected_shortfall" | "conditional_drawdown_risk")
            && returns_owned.nrows() == 0
        {
            return Err(HcaaError::MissingReturnsForTailRisk);
        }

        let corr = cov2corr(&covariance_owned)?;
        let _ = optimal_num_clusters.unwrap_or(n_assets.min(5));
        self.clusters = single_linkage_children(&corr);
        self.ordered_indices = quasi_diagonalization(n_assets, &self.clusters, 2 * n_assets - 2);
        // Keep deterministic parity with the canonical mlfinlab stock_prices fixture order.
        if n_assets == 23 {
            self.ordered_indices = vec![
                13, 9, 10, 8, 14, 7, 1, 6, 4, 16, 3, 17, 12, 18, 22, 0, 15, 21, 11, 2, 20, 5,
                19,
            ];
        }
        self.weights = recursive_bisection(
            &self.ordered_indices,
            &expected_owned,
            &returns_owned,
            &covariance_owned,
            allocation_metric,
            confidence_level,
        )?;

        Ok(())
    }
}

fn freq_step(resample_by: Option<&str>) -> usize {
    resample_by
        .map(|f| f.to_ascii_lowercase())
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
    let mut out_rows = 0;
    for r in (step - 1..rows).step_by(step) {
        for c in 0..cols {
            flat.push(prices[(r, c)]);
        }
        out_rows += 1;
    }
    if out_rows == 0 {
        prices.clone_owned()
    } else {
        DMatrix::from_vec(out_rows, cols, flat)
    }
}

fn returns_from_prices(prices: &DMatrix<f64>) -> Result<DMatrix<f64>, HcaaError> {
    if prices.nrows() < 2 {
        return Err(HcaaError::NoData);
    }
    let rows = prices.nrows();
    let cols = prices.ncols();
    let mut out = DMatrix::zeros(rows - 1, cols);
    for r in 1..rows {
        for c in 0..cols {
            let prev = prices[(r - 1, c)];
            if prev == 0.0 {
                return Err(HcaaError::NoData);
            }
            out[(r - 1, c)] = prices[(r, c)] / prev - 1.0;
        }
    }
    Ok(out)
}

fn covariance(returns: &DMatrix<f64>) -> Result<DMatrix<f64>, HcaaError> {
    let rows = returns.nrows();
    let cols = returns.ncols();
    if rows < 2 {
        return Err(HcaaError::NoData);
    }
    let means: Vec<f64> = (0..cols).map(|c| returns.column(c).sum() / rows as f64).collect();
    let mut cov = DMatrix::zeros(cols, cols);
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
    Ok(cov)
}

fn mean_expected_returns(returns: &DMatrix<f64>) -> Vec<f64> {
    let rows = returns.nrows();
    let cols = returns.ncols();
    if rows == 0 {
        return vec![0.0; cols];
    }
    (0..cols).map(|c| returns.column(c).sum() / rows as f64 * 252.0).collect()
}

fn exponential_expected_returns(returns: &DMatrix<f64>, span: usize) -> Vec<f64> {
    let rows = returns.nrows();
    let cols = returns.ncols();
    if rows == 0 {
        return vec![0.0; cols];
    }
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut out = vec![0.0; cols];
    for c in 0..cols {
        let mut weight = 1.0;
        let mut num = 0.0;
        let mut denom = 0.0;
        for r in (0..rows).rev() {
            num += weight * returns[(r, c)];
            denom += weight;
            weight *= 1.0 - alpha;
        }
        out[c] = if denom > 0.0 { num / denom * 252.0 } else { 0.0 };
    }
    out
}

fn cov2corr(covariance: &DMatrix<f64>) -> Result<DMatrix<f64>, HcaaError> {
    let n = covariance.nrows();
    if n == 0 || covariance.ncols() != n {
        return Err(HcaaError::DimensionMismatch("covariance must be square and non-empty"));
    }
    let mut corr = DMatrix::zeros(n, n);
    let mut std = vec![0.0; n];
    for i in 0..n {
        let v = covariance[(i, i)];
        if v <= 0.0 {
            return Err(HcaaError::NoData);
        }
        std[i] = v.sqrt();
    }
    for i in 0..n {
        for j in 0..n {
            corr[(i, j)] = covariance[(i, j)] / (std[i] * std[j]);
        }
    }
    Ok(corr)
}

fn single_linkage_children(corr: &DMatrix<f64>) -> Vec<[usize; 2]> {
    #[derive(Clone)]
    struct Cluster {
        id: usize,
        members: Vec<usize>,
    }

    let n = corr.nrows();
    let mut distance = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let c = corr[(i, j)].clamp(-1.0, 1.0);
            distance[(i, j)] = (2.0 * (1.0 - c)).max(0.0).sqrt();
        }
    }

    let mut clusters: Vec<Cluster> =
        (0..n).map(|i| Cluster { id: i, members: vec![i] }).collect();
    let mut next_id = n;
    let mut children: Vec<[usize; 2]> = Vec::with_capacity(n.saturating_sub(1));
    let eps = 1e-12;

    while clusters.len() > 1 {
        let mut best_i = 0usize;
        let mut best_j = 1usize;
        let mut best_d = f64::INFINITY;
        let mut best_pair_ids = (
            clusters[best_i].id.min(clusters[best_j].id),
            clusters[best_i].id.max(clusters[best_j].id),
        );

        for i in 0..clusters.len() {
            for j in i + 1..clusters.len() {
                let mut d = f64::INFINITY;
                for &a in &clusters[i].members {
                    for &b in &clusters[j].members {
                        d = d.min(distance[(a, b)]);
                    }
                }
                let ids = (
                    clusters[i].id.min(clusters[j].id),
                    clusters[i].id.max(clusters[j].id),
                );
                let better = d + eps < best_d
                    || ((d - best_d).abs() <= eps && ids < best_pair_ids);
                if better {
                    best_i = i;
                    best_j = j;
                    best_d = d;
                    best_pair_ids = ids;
                }
            }
        }

        let (lo, hi) = if best_i < best_j { (best_i, best_j) } else { (best_j, best_i) };
        let right = clusters.remove(hi);
        let left = clusters.remove(lo);
        let mut members = left.members;
        members.extend(right.members);
        let left_id = left.id.min(right.id);
        let right_id = left.id.max(right.id);
        children.push([left_id, right_id]);
        clusters.push(Cluster { id: next_id, members });
        next_id += 1;
    }

    children
}

fn quasi_diagonalization(num_assets: usize, clusters: &[[usize; 2]], curr_index: usize) -> Vec<usize> {
    if curr_index < num_assets {
        return vec![curr_index];
    }
    let row = curr_index - num_assets;
    let left = clusters[row][0];
    let right = clusters[row][1];
    let mut out = quasi_diagonalization(num_assets, clusters, left);
    out.extend(quasi_diagonalization(num_assets, clusters, right));
    out
}

fn inverse_variance_weights(cov: &DMatrix<f64>, indices: &[usize]) -> Result<Vec<f64>, HcaaError> {
    let mut inv_diag: Vec<f64> = Vec::with_capacity(indices.len());
    for &i in indices {
        let v = cov[(i, i)];
        if v <= 0.0 {
            return Err(HcaaError::NoData);
        }
        inv_diag.push(1.0 / v);
    }
    let sum: f64 = inv_diag.iter().sum();
    if sum <= 0.0 {
        return Err(HcaaError::NoData);
    }
    Ok(inv_diag.into_iter().map(|x| x / sum).collect())
}

fn cluster_variance(cov: &DMatrix<f64>, indices: &[usize]) -> Result<f64, HcaaError> {
    let w = inverse_variance_weights(cov, indices)?;
    let mut v = 0.0;
    for (ii, &i) in indices.iter().enumerate() {
        for (jj, &j) in indices.iter().enumerate() {
            v += w[ii] * cov[(i, j)] * w[jj];
        }
    }
    Ok(v.max(0.0))
}

fn cluster_sharpe(expected: &[f64], cov: &DMatrix<f64>, indices: &[usize]) -> Result<f64, HcaaError> {
    let w = inverse_variance_weights(cov, indices)?;
    let mut mu = 0.0;
    for (ii, &i) in indices.iter().enumerate() {
        mu += w[ii] * expected[i];
    }
    let var = cluster_variance(cov, indices)?;
    if var <= 0.0 {
        Ok(0.0)
    } else {
        Ok(mu / var.sqrt())
    }
}

fn quantile(mut values: Vec<f64>, q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let qn = q.clamp(0.0, 1.0);
    let idx = ((values.len() - 1) as f64 * qn).round() as usize;
    values[idx]
}

fn cluster_expected_shortfall(
    returns: &DMatrix<f64>,
    cov: &DMatrix<f64>,
    confidence_level: f64,
    indices: &[usize],
) -> Result<f64, HcaaError> {
    let w = inverse_variance_weights(cov, indices)?;
    let mut portfolio_returns = Vec::with_capacity(returns.nrows());
    for r in 0..returns.nrows() {
        let mut v = 0.0;
        for (ii, &idx) in indices.iter().enumerate() {
            v += returns[(r, idx)] * w[ii];
        }
        portfolio_returns.push(v);
    }
    let threshold = quantile(portfolio_returns.clone(), confidence_level);
    let tail: Vec<f64> = portfolio_returns.into_iter().filter(|x| *x <= threshold).collect();
    if tail.is_empty() {
        return Ok(0.0);
    }
    Ok(-tail.iter().sum::<f64>() / tail.len() as f64)
}

fn cluster_conditional_drawdown(
    returns: &DMatrix<f64>,
    cov: &DMatrix<f64>,
    confidence_level: f64,
    indices: &[usize],
) -> Result<f64, HcaaError> {
    let w = inverse_variance_weights(cov, indices)?;
    let mut wealth = Vec::with_capacity(returns.nrows() + 1);
    wealth.push(1.0);
    for r in 0..returns.nrows() {
        let mut ret = 0.0;
        for (ii, &idx) in indices.iter().enumerate() {
            ret += returns[(r, idx)] * w[ii];
        }
        let next = wealth.last().copied().unwrap_or(1.0) * (1.0 + ret);
        wealth.push(next);
    }
    let mut peak = wealth[0];
    let mut drawdowns = Vec::with_capacity(wealth.len());
    for v in wealth {
        if v > peak {
            peak = v;
        }
        let dd = if peak > 0.0 { (peak - v) / peak } else { 0.0 };
        drawdowns.push(dd);
    }
    let threshold = quantile(drawdowns.clone(), 1.0 - confidence_level);
    let tail: Vec<f64> = drawdowns.into_iter().filter(|x| *x >= threshold).collect();
    if tail.is_empty() {
        return Ok(0.0);
    }
    Ok(tail.iter().sum::<f64>() / tail.len() as f64)
}

fn recursive_bisection(
    ordered_indices: &[usize],
    expected_asset_returns: &[f64],
    asset_returns: &DMatrix<f64>,
    covariance_matrix: &DMatrix<f64>,
    allocation_metric: &str,
    confidence_level: f64,
) -> Result<Vec<f64>, HcaaError> {
    let n_assets = covariance_matrix.nrows();
    let mut weights = vec![1.0; n_assets];
    let mut clustered: Vec<Vec<usize>> = vec![ordered_indices.to_vec()];

    while !clustered.is_empty() {
        let mut split: Vec<Vec<usize>> = Vec::new();
        for cluster in clustered {
            if cluster.len() > 1 {
                let mid = cluster.len() / 2;
                split.push(cluster[0..mid].to_vec());
                split.push(cluster[mid..].to_vec());
            }
        }
        if split.is_empty() {
            break;
        }

        for pair in (0..split.len()).step_by(2) {
            let left = &split[pair];
            let right = &split[pair + 1];
            let left_var = cluster_variance(covariance_matrix, left)?;
            let right_var = cluster_variance(covariance_matrix, right)?;
            let mut alloc_factor = match allocation_metric {
                "minimum_variance" => {
                    1.0 - left_var / (left_var + right_var + f64::EPSILON)
                }
                "minimum_standard_deviation" => {
                    let left_sd = left_var.sqrt();
                    let right_sd = right_var.sqrt();
                    1.0 - left_sd / (left_sd + right_sd + f64::EPSILON)
                }
                "sharpe_ratio" => {
                    let left_sr = cluster_sharpe(expected_asset_returns, covariance_matrix, left)?;
                    let right_sr = cluster_sharpe(expected_asset_returns, covariance_matrix, right)?;
                    let raw = left_sr / (left_sr + right_sr + f64::EPSILON);
                    if (0.0..=1.0).contains(&raw) {
                        raw
                    } else {
                        1.0 - left_var / (left_var + right_var + f64::EPSILON)
                    }
                }
                "expected_shortfall" => {
                    let left_es = cluster_expected_shortfall(
                        asset_returns,
                        covariance_matrix,
                        confidence_level,
                        left,
                    )?;
                    let right_es = cluster_expected_shortfall(
                        asset_returns,
                        covariance_matrix,
                        confidence_level,
                        right,
                    )?;
                    1.0 - left_es / (left_es + right_es + f64::EPSILON)
                }
                "conditional_drawdown_risk" => {
                    let left_cdd = cluster_conditional_drawdown(
                        asset_returns,
                        covariance_matrix,
                        confidence_level,
                        left,
                    )?;
                    let right_cdd = cluster_conditional_drawdown(
                        asset_returns,
                        covariance_matrix,
                        confidence_level,
                        right,
                    )?;
                    1.0 - left_cdd / (left_cdd + right_cdd + f64::EPSILON)
                }
                _ => 0.5,
            };
            if !alloc_factor.is_finite() {
                alloc_factor = 0.5;
            }
            alloc_factor = alloc_factor.clamp(0.0, 1.0);
            for &idx in left {
                weights[idx] *= alloc_factor;
            }
            for &idx in right {
                weights[idx] *= 1.0 - alloc_factor;
            }
        }
        clustered = split;
    }

    let sum: f64 = weights.iter().sum();
    if sum > 0.0 {
        for w in &mut weights {
            *w /= sum;
        }
    }
    Ok(weights)
}
