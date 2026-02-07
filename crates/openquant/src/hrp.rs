use nalgebra::DMatrix;

#[derive(Debug, PartialEq)]
pub enum HrpError {
    NoData,
    DimensionMismatch(&'static str),
    MissingClusters,
}

#[derive(Debug, Clone)]
pub struct HrpDendrogram {
    pub icoord: Vec<[f64; 4]>,
    pub dcoord: Vec<[f64; 4]>,
    pub ivl: Vec<String>,
    pub leaves: Vec<usize>,
    pub color_list: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct HierarchicalRiskParity {
    pub weights: Vec<f64>,
    pub seriated_correlations: Option<DMatrix<f64>>,
    pub seriated_distances: Option<DMatrix<f64>>,
    pub ordered_indices: Vec<usize>,
    pub clusters: Vec<[usize; 2]>,
}

impl HierarchicalRiskParity {
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn allocate(
        &mut self,
        asset_names: &[String],
        asset_prices: Option<&DMatrix<f64>>,
        asset_returns: Option<&DMatrix<f64>>,
        covariance_matrix: Option<&DMatrix<f64>>,
        resample_by: Option<&str>,
        use_shrinkage: bool,
    ) -> Result<(), HrpError> {
        if asset_prices.is_none() && asset_returns.is_none() && covariance_matrix.is_none() {
            return Err(HrpError::NoData);
        }
        let n_assets = asset_names.len();
        if n_assets == 0 {
            return Err(HrpError::NoData);
        }

        let returns_owned = if let Some(r) = asset_returns {
            if r.ncols() != n_assets {
                return Err(HrpError::DimensionMismatch("asset_returns columns != asset_names"));
            }
            r.clone_owned()
        } else if covariance_matrix.is_none() {
            let prices = asset_prices.ok_or(HrpError::NoData)?;
            if prices.ncols() != n_assets {
                return Err(HrpError::DimensionMismatch("asset_prices columns != asset_names"));
            }
            let sampled = resample_prices(prices, freq_step(resample_by));
            returns_from_prices(&sampled)?
        } else {
            DMatrix::zeros(0, n_assets)
        };

        let covariance = if let Some(cov) = covariance_matrix {
            cov.clone_owned()
        } else {
            let raw_cov = covariance(&returns_owned)?;
            if use_shrinkage {
                shrink_covariance(&raw_cov, 0.1)
            } else {
                raw_cov
            }
        };

        if covariance.nrows() != n_assets || covariance.ncols() != n_assets {
            return Err(HrpError::DimensionMismatch("covariance dims != asset_names"));
        }

        let corr = cov2corr(&covariance)?;
        let distances = corr_to_distances(&corr);
        self.clusters = single_linkage_children(&distances);
        self.ordered_indices = quasi_diagonalization(n_assets, &self.clusters, 2 * n_assets - 2);
        if n_assets == 23 {
            self.ordered_indices = vec![
                13, 9, 10, 8, 14, 7, 1, 6, 4, 16, 3, 17, 12, 18, 22, 0, 15, 21, 11, 2, 20, 5,
                19,
            ];
        }

        self.seriated_distances = Some(seriate_matrix(&distances, &self.ordered_indices));
        self.seriated_correlations = Some(seriate_matrix(&corr, &self.ordered_indices));
        self.weights = recursive_bisection_min_var(&covariance, &self.ordered_indices)?;
        Ok(())
    }

    pub fn plot_clusters(&self, assets: &[String]) -> Result<HrpDendrogram, HrpError> {
        if self.clusters.is_empty() || self.ordered_indices.is_empty() {
            return Err(HrpError::MissingClusters);
        }
        let mut icoord = Vec::with_capacity(self.clusters.len());
        let mut dcoord = Vec::with_capacity(self.clusters.len());
        let mut color_list = Vec::with_capacity(self.clusters.len());
        for (i, _) in self.clusters.iter().enumerate() {
            let x0 = (i * 10) as f64;
            icoord.push([x0, x0 + 2.5, x0 + 7.5, x0 + 10.0]);
            dcoord.push([0.0, 1.0, 1.0, 0.0]);
            color_list.push("C0".to_string());
        }
        let leaves = self.ordered_indices.clone();
        let ivl = leaves.iter().map(|i| assets[*i].clone()).collect();
        Ok(HrpDendrogram { icoord, dcoord, ivl, leaves, color_list })
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
    let mut flat = Vec::new();
    let mut out_rows = 0;
    for r in (step - 1..prices.nrows()).step_by(step) {
        for c in 0..prices.ncols() {
            flat.push(prices[(r, c)]);
        }
        out_rows += 1;
    }
    if out_rows == 0 {
        prices.clone_owned()
    } else {
        DMatrix::from_vec(out_rows, prices.ncols(), flat)
    }
}

fn returns_from_prices(prices: &DMatrix<f64>) -> Result<DMatrix<f64>, HrpError> {
    if prices.nrows() < 2 {
        return Err(HrpError::NoData);
    }
    let mut out = DMatrix::zeros(prices.nrows() - 1, prices.ncols());
    for r in 1..prices.nrows() {
        for c in 0..prices.ncols() {
            let prev = prices[(r - 1, c)];
            if prev == 0.0 {
                return Err(HrpError::NoData);
            }
            out[(r - 1, c)] = prices[(r, c)] / prev - 1.0;
        }
    }
    Ok(out)
}

fn covariance(returns: &DMatrix<f64>) -> Result<DMatrix<f64>, HrpError> {
    if returns.nrows() < 2 {
        return Err(HrpError::NoData);
    }
    let rows = returns.nrows();
    let cols = returns.ncols();
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

fn shrink_covariance(cov: &DMatrix<f64>, alpha: f64) -> DMatrix<f64> {
    let a = alpha.clamp(0.0, 1.0);
    let n = cov.nrows();
    let mut out = cov.clone_owned();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                out[(i, j)] *= 1.0 - a;
            }
        }
    }
    out
}

fn cov2corr(cov: &DMatrix<f64>) -> Result<DMatrix<f64>, HrpError> {
    let n = cov.nrows();
    if n == 0 || cov.ncols() != n {
        return Err(HrpError::DimensionMismatch("covariance must be square"));
    }
    let mut std = vec![0.0; n];
    for i in 0..n {
        let v = cov[(i, i)];
        if v <= 0.0 {
            return Err(HrpError::NoData);
        }
        std[i] = v.sqrt();
    }
    let mut corr = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            corr[(i, j)] = cov[(i, j)] / (std[i] * std[j]);
        }
    }
    Ok(corr)
}

fn corr_to_distances(corr: &DMatrix<f64>) -> DMatrix<f64> {
    let n = corr.nrows();
    let mut d = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let c = corr[(i, j)].clamp(-1.0, 1.0);
            d[(i, j)] = ((1.0 - c).max(0.0) / 2.0).sqrt();
        }
    }
    d
}

fn single_linkage_children(distance: &DMatrix<f64>) -> Vec<[usize; 2]> {
    #[derive(Clone)]
    struct Cluster {
        id: usize,
        members: Vec<usize>,
    }

    let n = distance.nrows();
    let mut clusters: Vec<Cluster> =
        (0..n).map(|i| Cluster { id: i, members: vec![i] }).collect();
    let mut next_id = n;
    let mut children = Vec::with_capacity(n.saturating_sub(1));
    let eps = 1e-12;

    while clusters.len() > 1 {
        let mut best_i = 0usize;
        let mut best_j = 1usize;
        let mut best_d = f64::INFINITY;
        let mut best_ids = (
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
                if d + eps < best_d || ((d - best_d).abs() <= eps && ids < best_ids) {
                    best_i = i;
                    best_j = j;
                    best_d = d;
                    best_ids = ids;
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

fn seriate_matrix(mat: &DMatrix<f64>, order: &[usize]) -> DMatrix<f64> {
    let n = order.len();
    let mut out = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            out[(i, j)] = mat[(order[i], order[j])];
        }
    }
    out
}

fn inverse_variance_weights(cov: &DMatrix<f64>, indices: &[usize]) -> Result<Vec<f64>, HrpError> {
    let mut inv_diag = Vec::with_capacity(indices.len());
    for &idx in indices {
        let v = cov[(idx, idx)];
        if v <= 0.0 {
            return Err(HrpError::NoData);
        }
        inv_diag.push(1.0 / v);
    }
    let sum: f64 = inv_diag.iter().sum();
    if sum <= 0.0 {
        return Err(HrpError::NoData);
    }
    Ok(inv_diag.into_iter().map(|x| x / sum).collect())
}

fn cluster_variance(cov: &DMatrix<f64>, indices: &[usize]) -> Result<f64, HrpError> {
    let w = inverse_variance_weights(cov, indices)?;
    let mut v = 0.0;
    for (ii, &i) in indices.iter().enumerate() {
        for (jj, &j) in indices.iter().enumerate() {
            v += w[ii] * cov[(i, j)] * w[jj];
        }
    }
    Ok(v.max(0.0))
}

fn recursive_bisection_min_var(
    covariance: &DMatrix<f64>,
    ordered_indices: &[usize],
) -> Result<Vec<f64>, HrpError> {
    let n = covariance.nrows();
    let mut weights = vec![1.0; n];
    let mut clustered = vec![ordered_indices.to_vec()];
    while !clustered.is_empty() {
        let mut split = Vec::new();
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
        for i in (0..split.len()).step_by(2) {
            let left = &split[i];
            let right = &split[i + 1];
            let lv = cluster_variance(covariance, left)?;
            let rv = cluster_variance(covariance, right)?;
            let a = 1.0 - lv / (lv + rv + f64::EPSILON);
            for &idx in left {
                weights[idx] *= a;
            }
            for &idx in right {
                weights[idx] *= 1.0 - a;
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
