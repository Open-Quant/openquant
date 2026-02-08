use chrono::{Duration, NaiveDateTime};

/// Compute daily volatility via exponentially weighted std of daily returns.
/// Mirrors mlfinlab.util.volatility.get_daily_vol with span `lookback`.
pub fn get_daily_vol(close: &[(NaiveDateTime, f64)], lookback: usize) -> Vec<(NaiveDateTime, f64)> {
    if close.len() < 2 || lookback == 0 {
        return Vec::new();
    }

    let alpha = 2.0 / (lookback as f64 + 1.0);
    let one_minus = 1.0 - alpha;

    let mut out = Vec::new();
    let mut mean = 0.0f64;
    let mut var = 0.0f64;
    let mut initialized = false;

    for i in 0..close.len() {
        let (ts_i, price_i) = close[i];
        let target_time = ts_i - Duration::days(1);

        // searchsorted equivalent: find insertion point for target_time
        let mut j_opt = None;
        for j in 0..i {
            if close[j].0 <= target_time {
                j_opt = Some(j);
            }
        }
        if let Some(j) = j_opt {
            let (_, price_prev) = close[j];
            let ret = price_i / price_prev - 1.0;

            if !initialized {
                mean = ret;
                var = 0.0;
                initialized = true;
            } else {
                let prev_mean = mean;
                mean = alpha * ret + one_minus * mean;
                var = one_minus * (var + alpha * (ret - prev_mean).powi(2));
            }
            let std = var.max(0.0).sqrt();
            out.push((ts_i, std));
        }
    }

    out
}

/// Parkinson volatility estimator.
/// Mirrors mlfinlab.util.volatility.get_parksinson_vol.
pub fn get_parksinson_vol(high: &[f64], low: &[f64], window: usize) -> Vec<f64> {
    assert_eq!(high.len(), low.len(), "high/low length mismatch");
    let estimator: Vec<f64> = high
        .iter()
        .zip(low.iter())
        .map(|(&h, &l)| {
            let ret = (h / l).ln();
            (ret * ret) / (4.0 * 2.0f64.ln())
        })
        .collect();
    rolling_sqrt_mean(&estimator, window)
}

/// Garman-Klass volatility estimator.
/// Mirrors mlfinlab.util.volatility.get_garman_class_vol.
pub fn get_garman_class_vol(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    window: usize,
) -> Vec<f64> {
    assert_eq!(open.len(), high.len(), "open/high length mismatch");
    assert_eq!(open.len(), low.len(), "open/low length mismatch");
    assert_eq!(open.len(), close.len(), "open/close length mismatch");

    let c = 2.0 * 2.0f64.ln() - 1.0;
    let estimator: Vec<f64> = open
        .iter()
        .zip(high.iter())
        .zip(low.iter())
        .zip(close.iter())
        .map(|(((&o, &h), &l), &c_px)| {
            let hl = (h / l).ln();
            let co = (c_px / o).ln();
            0.5 * hl * hl - c * co * co
        })
        .collect();
    rolling_sqrt_mean(&estimator, window)
}

/// Yang-Zhang volatility estimator.
/// Mirrors mlfinlab.util.volatility.get_yang_zhang_vol.
pub fn get_yang_zhang_vol(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    window: usize,
) -> Vec<f64> {
    assert_eq!(open.len(), high.len(), "open/high length mismatch");
    assert_eq!(open.len(), low.len(), "open/low length mismatch");
    assert_eq!(open.len(), close.len(), "open/close length mismatch");

    let n = open.len();
    if n == 0 {
        return Vec::new();
    }
    if window < 2 {
        return vec![f64::NAN; n];
    }

    let k = 0.34 / (1.34 + ((window + 1) as f64 / (window - 1) as f64));

    let mut open_prev_close_ret = vec![f64::NAN; n];
    let mut close_prev_open_ret = vec![f64::NAN; n];
    let mut rs_component = vec![f64::NAN; n];

    for i in 1..n {
        open_prev_close_ret[i] = (open[i] / close[i - 1]).ln();
        close_prev_open_ret[i] = (close[i] / open[i - 1]).ln();
    }
    for i in 0..n {
        let high_close_ret = (high[i] / close[i]).ln();
        let high_open_ret = (high[i] / open[i]).ln();
        let low_close_ret = (low[i] / close[i]).ln();
        let low_open_ret = (low[i] / open[i]).ln();
        rs_component[i] = high_close_ret * high_open_ret + low_close_ret * low_open_ret;
    }

    let sigma_open_sq = rolling_sum_with_min_periods(
        &open_prev_close_ret.iter().map(|v| v * v).collect::<Vec<_>>(),
        window,
        window,
    );
    let sigma_close_sq = rolling_sum_with_min_periods(
        &close_prev_open_ret.iter().map(|v| v * v).collect::<Vec<_>>(),
        window,
        window,
    );
    let sigma_rs_sq = rolling_sum_with_min_periods(&rs_component, window, window);

    sigma_open_sq
        .iter()
        .zip(sigma_close_sq.iter())
        .zip(sigma_rs_sq.iter())
        .map(|((&o_sq, &c_sq), &rs_sq)| {
            if o_sq.is_nan() || c_sq.is_nan() || rs_sq.is_nan() {
                f64::NAN
            } else {
                (o_sq / (window - 1) as f64
                    + k * c_sq / (window - 1) as f64
                    + (1.0 - k) * rs_sq / (window - 1) as f64)
                    .sqrt()
            }
        })
        .collect()
}

fn rolling_sqrt_mean(values: &[f64], window: usize) -> Vec<f64> {
    rolling_sum_with_min_periods(values, window, window)
        .iter()
        .map(|&x| if x.is_nan() { f64::NAN } else { (x / window as f64).sqrt() })
        .collect()
}

fn rolling_sum_with_min_periods(values: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    if window == 0 {
        return vec![f64::NAN; n];
    }

    let mut out = vec![f64::NAN; n];
    let mut sum = 0.0;
    let mut valid = 0usize;

    for i in 0..n {
        let x = values[i];
        if !x.is_nan() {
            sum += x;
            valid += 1;
        }
        if i >= window {
            let old = values[i - window];
            if !old.is_nan() {
                sum -= old;
                valid -= 1;
            }
        }
        if i + 1 >= window && valid >= min_periods {
            out[i] = sum;
        }
    }
    out
}
