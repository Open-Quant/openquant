#[derive(Debug)]
pub enum CodependenceError {
    InputLengthMismatch,
    InputTooShort,
    InvalidBins,
    ZeroVariance,
    ZeroDistanceVariance,
}

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

pub fn angular_distance(x: &[f64], y: &[f64]) -> CodependenceResult<f64> {
    let corr_coef = corrcoef(x, y)?;
    Ok((0.5 * (1.0 - corr_coef)).sqrt())
}

pub fn absolute_angular_distance(x: &[f64], y: &[f64]) -> CodependenceResult<f64> {
    let corr_coef = corrcoef(x, y)?;
    Ok((0.5 * (1.0 - corr_coef.abs())).sqrt())
}

pub fn squared_angular_distance(x: &[f64], y: &[f64]) -> CodependenceResult<f64> {
    let corr_coef = corrcoef(x, y)?;
    Ok((0.5 * (1.0 - corr_coef.powi(2))).sqrt())
}

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

pub fn get_optimal_number_of_bins(
    num_obs: usize,
    corr_coef: Option<f64>,
) -> CodependenceResult<usize> {
    if num_obs == 0 {
        return Err(CodependenceError::InputTooShort);
    }

    let n = num_obs as f64;
    let bins = if corr_coef.is_none() || (corr_coef.unwrap() - 1.0).abs() <= 1e-4 {
        let z = (8.0 + 324.0 * n + 12.0 * (36.0 * n + 729.0 * n * n).sqrt()).cbrt();
        (z / 6.0 + 2.0 / (3.0 * z) + 1.0 / 3.0).round()
    } else {
        let corr = corr_coef.unwrap();
        let inner = (1.0 + 24.0 * n / (1.0 - corr * corr)).sqrt();
        (2.0_f64).powf(-0.5) * (1.0 + inner).sqrt()
    };

    let bins = bins.round() as isize;
    if bins <= 0 {
        return Err(CodependenceError::InvalidBins);
    }
    Ok(bins as usize)
}

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
        for j in 0..bins {
            let value = contingency[i][j] as f64;
            row_sums[i] += value;
            col_sums[j] += value;
        }
    }

    let mut mutual_info = 0.0;
    for i in 0..bins {
        for j in 0..bins {
            let value = contingency[i][j] as f64;
            if value == 0.0 {
                continue;
            }
            let p_ij = value / total_f;
            let p_i = row_sums[i] / total_f;
            let p_j = col_sums[j] / total_f;
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
        for j in 0..bins {
            let value = contingency[i][j] as f64;
            row_sums[i] += value;
            col_sums[j] += value;
        }
    }

    let mut mutual_info = 0.0;
    for i in 0..bins {
        for j in 0..bins {
            let value = contingency[i][j] as f64;
            if value == 0.0 {
                continue;
            }
            let p_ij = value / total_f;
            let p_i = row_sums[i] / total_f;
            let p_j = col_sums[j] / total_f;
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
