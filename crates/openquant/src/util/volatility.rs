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
