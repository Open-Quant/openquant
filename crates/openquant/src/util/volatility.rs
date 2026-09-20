use chrono::{Duration, NaiveDateTime};

/// Compute daily volatility via exponentially weighted std of daily returns.
/// Mirrors mlfinlab.util.volatility.get_daily_vol with span `lookback`.
pub fn get_daily_vol(close: &[(NaiveDateTime, f64)], lookback: usize) -> Vec<(NaiveDateTime, f64)> {
    if close.len() < 2 || lookback == 0 {
        return Vec::new();
    }

    // AFML snippet 3.1: the return over (at least) the previous day, then
    // `ewm(span=lookback).std()`. pandas' defaults are `adjust=True, bias=False`: weights
    // (1 - alpha)^k over the whole history and the unbiased weighted variance. The mean and
    // variance are updated incrementally, as pandas does, rather than as
    // `sum(w x^2)/sum(w) - mean^2`, which cancels catastrophically when returns barely vary.
    let decay = 1.0 - 2.0 / (lookback as f64 + 1.0);
    let (mut sum_wt, mut sum_wt2, mut old_wt) = (0.0f64, 0.0f64, 0.0f64);
    let (mut mean, mut var) = (0.0f64, 0.0f64);

    let mut out = Vec::new();
    for (i, &(ts_i, price_i)) in close.iter().enumerate() {
        // `searchsorted(t - 1 day) - 1`: the last bar strictly before one day ago.
        let target_time = ts_i - Duration::days(1);
        let Some(j) = close[..i].iter().rposition(|(ts_j, _)| *ts_j < target_time) else {
            continue;
        };
        let ret = price_i / close[j].1 - 1.0;

        sum_wt *= decay;
        sum_wt2 *= decay * decay;
        old_wt *= decay;

        let old_mean = mean;
        let total = old_wt + 1.0;
        mean = (old_wt * old_mean + ret) / total;
        var = (old_wt * (var + (old_mean - mean).powi(2)) + (ret - mean).powi(2)) / total;

        sum_wt += 1.0;
        sum_wt2 += 1.0;
        old_wt += 1.0;

        // One observation has no sample variance: pandas reports NaN there and so does this,
        // so the result lines up with `ewm().std()` row for row. `get_events` drops a NaN
        // target, exactly as `target[target > min_ret]` does.
        let denom = sum_wt * sum_wt - sum_wt2;
        if denom <= 0.0 {
            out.push((ts_i, f64::NAN));
            continue;
        }
        out.push((ts_i, (var.max(0.0) * sum_wt * sum_wt / denom).sqrt()));
    }

    out
}

/// Parkinson volatility estimator.
/// Mirrors mlfinlab's `get_parksinson_vol` — note that upstream misspells
/// "Parkinson"; this crate spells it correctly.
pub fn get_parkinson_vol(high: &[f64], low: &[f64], window: usize) -> Vec<f64> {
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
