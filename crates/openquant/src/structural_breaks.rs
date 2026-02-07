#[derive(Debug)]
pub enum StructuralBreakError {
    InvalidTestType(String),
    InvalidModel(String),
    NotImplemented(&'static str),
    InputTooShort,
    IndexOutOfBounds,
}

pub type StructuralBreakResult<T> = Result<T, StructuralBreakError>;

#[derive(Debug, Clone)]
pub struct ChuStinchcombeWhiteResult {
    pub critical_value: Vec<f64>,
    pub stat: Vec<f64>,
}

#[derive(Debug, Clone)]
pub enum SadfLags {
    Fixed(usize),
    Array(Vec<usize>),
}

pub fn get_chow_type_stat(
    _log_prices: &[f64],
    _min_length: usize,
) -> StructuralBreakResult<Vec<f64>> {
    let series_len = _log_prices.len();
    if series_len < _min_length * 2 {
        return Ok(Vec::new());
    }

    let mut diffs = Vec::with_capacity(series_len.saturating_sub(1));
    let mut lags = Vec::with_capacity(series_len.saturating_sub(1));
    for i in 1..series_len {
        diffs.push(_log_prices[i] - _log_prices[i - 1]);
        lags.push(_log_prices[i - 1]);
    }

    let mut stats = Vec::with_capacity(series_len - _min_length * 2);
    for index in _min_length..(series_len - _min_length) {
        let mut x = lags.clone();
        for lag_value in x.iter_mut().take(index) {
            *lag_value = 0.0;
        }
        let x_matrix = x.into_iter().map(|v| vec![v]).collect::<Vec<_>>();
        let y_matrix = diffs.iter().map(|v| vec![*v]).collect::<Vec<_>>();
        let (coefs, coef_vars) = _get_betas(&x_matrix, &y_matrix)?;
        let b_estimate = coefs[0];
        let b_var = coef_vars[0][0];
        stats.push(b_estimate / b_var.sqrt());
    }

    Ok(stats)
}

pub fn get_chu_stinchcombe_white_statistics(
    _log_prices: &[f64],
    _test_type: &str,
) -> StructuralBreakResult<ChuStinchcombeWhiteResult> {
    let series_len = _log_prices.len();
    if series_len < 3 {
        return Err(StructuralBreakError::InputTooShort);
    }

    let mut critical_value = Vec::with_capacity(series_len - 2);
    let mut stat = Vec::with_capacity(series_len - 2);

    for index in 2..series_len {
        let mut squared_diff_sum = 0.0;
        for i in 1..=index {
            let diff = _log_prices[i] - _log_prices[i - 1];
            squared_diff_sum += diff * diff;
        }
        let sigma_sq_t = (1.0 / (index as f64 - 1.0)) * squared_diff_sum;

        let mut max_s_n_value = f64::NEG_INFINITY;
        let mut max_s_n_critical_value: Option<f64> = None;

        for ind in 0..index {
            let values_diff = _get_values_diff(_test_type, _log_prices, index, ind)?;
            let distance = (index - ind) as f64;
            let s_n_t = (1.0 / (sigma_sq_t * distance.sqrt())) * values_diff;

            if s_n_t > max_s_n_value {
                max_s_n_value = s_n_t;
                max_s_n_critical_value = Some((4.6 + distance.ln()).sqrt());
            }
        }

        stat.push(max_s_n_value);
        critical_value.push(max_s_n_critical_value.unwrap_or(f64::NAN));
    }

    Ok(ChuStinchcombeWhiteResult { critical_value, stat })
}

pub fn get_sadf(
    _series: &[f64],
    _model: &str,
    _add_const: bool,
    _min_length: usize,
    _lags: SadfLags,
) -> StructuralBreakResult<Vec<f64>> {
    let (x, y, indices) = get_y_x(_series, _model, _lags, _add_const)?;
    if y.len() <= _min_length {
        return Ok(Vec::new());
    }

    let mut sadf_values = Vec::with_capacity(y.len().saturating_sub(_min_length));
    for (pos, _) in indices.iter().enumerate().skip(_min_length) {
        let x_subset = x[..=pos].to_vec();
        let y_subset = y[..=pos].to_vec();
        let value = get_sadf_at_t(&x_subset, &y_subset, _min_length)?;
        sadf_values.push(value);
    }

    Ok(sadf_values)
}

pub fn _get_values_diff(
    test_type: &str,
    series: &[f64],
    index: usize,
    ind: usize,
) -> StructuralBreakResult<f64> {
    let left = series.get(index).ok_or(StructuralBreakError::IndexOutOfBounds)?;
    let right = series.get(ind).ok_or(StructuralBreakError::IndexOutOfBounds)?;
    match test_type {
        "one_sided" => Ok(left - right),
        "two_sided" => Ok((left - right).abs()),
        _ => Err(StructuralBreakError::InvalidTestType(test_type.to_string())),
    }
}

pub fn _get_betas(
    _x: &[Vec<f64>],
    _y: &[Vec<f64>],
) -> StructuralBreakResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let x_matrix = to_matrix(_x)?;
    let y_matrix = to_matrix(_y)?;

    let rows = x_matrix.nrows();
    let cols = x_matrix.ncols();
    let y_cols = y_matrix.ncols();

    let xy = x_matrix.transpose() * &y_matrix;
    let xx = x_matrix.transpose() * &x_matrix;

    let Some(xx_inv) = xx.try_inverse() else {
        let b_mean = vec![f64::NAN; cols];
        let b_var = vec![vec![f64::NAN; cols]; cols];
        return Ok((b_mean, b_var));
    };

    let b_mean = &xx_inv * xy;
    let err = y_matrix - x_matrix * &b_mean;
    let err_t_err = err.transpose() * err;
    let denom = rows as f64 - cols as f64;
    let scale = err_t_err / denom;

    let b_var_matrix = if y_cols == 1 {
        let scalar = scale[(0, 0)];
        xx_inv * scalar
    } else if scale.nrows() == cols && scale.ncols() == cols {
        xx_inv.component_mul(&scale)
    } else {
        let scalar = scale[(0, 0)];
        xx_inv * scalar
    };

    let mut b_mean_vec = Vec::with_capacity(cols);
    for i in 0..cols {
        b_mean_vec.push(b_mean[(i, 0)]);
    }

    Ok((b_mean_vec, matrix_to_vec(b_var_matrix)))
}

fn get_y_x(
    series: &[f64],
    model: &str,
    lags: SadfLags,
    add_const: bool,
) -> StructuralBreakResult<(Vec<Vec<f64>>, Vec<f64>, Vec<usize>)> {
    let series_len = series.len();
    if series_len < 2 {
        return Err(StructuralBreakError::InputTooShort);
    }

    let mut series_diff = Vec::with_capacity(series_len - 1);
    for i in 1..series_len {
        series_diff.push(series[i] - series[i - 1]);
    }

    let lag_values = match lags {
        SadfLags::Fixed(value) => (1..=value).collect::<Vec<_>>(),
        SadfLags::Array(values) => values.into_iter().map(|v| v as usize).collect(),
    };
    let max_lag = *lag_values.iter().max().unwrap_or(&0);
    let start_index = max_lag + 1;
    if series_len <= start_index {
        return Err(StructuralBreakError::InputTooShort);
    }

    let mut indices = Vec::new();
    let mut x_rows = Vec::new();
    let mut y_values = Vec::new();

    for idx in start_index..=series_len - 1 {
        let mut row = Vec::with_capacity(lag_values.len());
        for lag in &lag_values {
            let pos = idx - lag - 1;
            row.push(series_diff[pos]);
        }
        x_rows.push(row);
        indices.push(idx);
        y_values.push(series_diff[idx - 1]);
    }

    let mut x = x_rows;
    let mut y = y_values;

    match model {
        "linear" | "quadratic" => {
            let mut updated = Vec::with_capacity(x.len());
            for (i, row) in x.into_iter().enumerate() {
                let mut new_row = Vec::with_capacity(row.len() + 3);
                new_row.push(series[indices[i] - 1]);
                new_row.extend(row);
                if add_const {
                    new_row.push(1.0);
                }
                let trend = i as f64;
                let trend_value = if model == "quadratic" { trend * trend } else { trend };
                new_row.push(trend_value);
                updated.push(new_row);
            }
            x = updated;
        }
        "sm_poly_1" => {
            y = indices.iter().map(|&idx| series[idx]).collect();
            let mut updated = Vec::with_capacity(y.len());
            for i in 0..y.len() {
                let trend = i as f64;
                let row = vec![trend * trend, 1.0, trend];
                updated.push(row);
            }
            x = updated;
        }
        "sm_poly_2" => {
            y = indices.iter().map(|&idx| series[idx].ln()).collect();
            let mut updated = Vec::with_capacity(y.len());
            for i in 0..y.len() {
                let trend = i as f64;
                let row = vec![trend * trend, 1.0, trend];
                updated.push(row);
            }
            x = updated;
        }
        "sm_exp" => {
            y = indices.iter().map(|&idx| series[idx].ln()).collect();
            let mut updated = Vec::with_capacity(y.len());
            for (i, _) in y.iter().enumerate() {
                let trend = i as f64;
                let row = vec![trend, 1.0];
                updated.push(row);
            }
            x = updated;
        }
        "sm_power" => {
            y = indices.iter().map(|&idx| series[idx].ln()).collect();
            let mut updated = Vec::with_capacity(y.len());
            for (i, _) in y.iter().enumerate() {
                let trend = (i as f64).ln();
                let row = vec![trend, 1.0];
                updated.push(row);
            }
            x = updated;
        }
        _ => {
            return Err(StructuralBreakError::InvalidModel(model.to_string()));
        }
    }

    Ok((x, y, indices))
}

fn get_sadf_at_t(x: &[Vec<f64>], y: &[f64], min_length: usize) -> StructuralBreakResult<f64> {
    let y_len = y.len();
    if y_len < min_length {
        return Ok(f64::NEG_INFINITY);
    }

    let mut bsadf = f64::NEG_INFINITY;
    let start_points = 0..=(y_len - min_length);
    for start in start_points {
        let y_subset = y[start..].iter().map(|v| vec![*v]).collect::<Vec<_>>();
        let x_subset = x[start..].to_vec();

        let (b_mean, b_var) = _get_betas(&x_subset, &y_subset)?;
        if b_mean.get(0).map(|v| v.is_nan()).unwrap_or(true) {
            continue;
        }

        let b_estimate = b_mean[0];
        let b_std = b_var[0][0].sqrt();
        let all_adf = b_estimate / b_std;
        if all_adf > bsadf {
            bsadf = all_adf;
        }
    }

    Ok(bsadf)
}

fn to_matrix(data: &[Vec<f64>]) -> StructuralBreakResult<DMatrix<f64>> {
    let rows = data.len();
    if rows == 0 {
        return Err(StructuralBreakError::InputTooShort);
    }
    let cols = data[0].len();
    if cols == 0 {
        return Err(StructuralBreakError::InputTooShort);
    }
    if data.iter().any(|row| row.len() != cols) {
        return Err(StructuralBreakError::InputTooShort);
    }
    let flat = data.iter().flat_map(|row| row.iter().cloned()).collect::<Vec<_>>();
    Ok(DMatrix::from_row_slice(rows, cols, &flat))
}

fn matrix_to_vec(matrix: DMatrix<f64>) -> Vec<Vec<f64>> {
    let mut rows = Vec::with_capacity(matrix.nrows());
    for i in 0..matrix.nrows() {
        let mut row = Vec::with_capacity(matrix.ncols());
        for j in 0..matrix.ncols() {
            row.push(matrix[(i, j)]);
        }
        rows.push(row);
    }
    rows
}
use nalgebra::DMatrix;
