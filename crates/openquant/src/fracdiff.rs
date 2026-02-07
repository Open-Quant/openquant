/// Fractional differentiation utilities (AFML chapter 5 style).

pub fn get_weights(diff_amt: f64, size: usize) -> Vec<f64> {
    if size == 0 {
        return Vec::new();
    }
    let mut weights = Vec::with_capacity(size);
    weights.push(1.0);
    for k in 1..size {
        let w = -weights[k - 1] * (diff_amt - k as f64 + 1.0) / k as f64;
        weights.push(w);
    }
    weights.reverse();
    weights
}

pub fn get_weights_ffd(diff_amt: f64, thresh: f64, lim: usize) -> Vec<f64> {
    if lim == 0 {
        return Vec::new();
    }
    let mut weights = vec![1.0];
    let mut k = 1usize;
    let mut ctr = 0usize;
    loop {
        let next = -weights[weights.len() - 1] * (diff_amt - k as f64 + 1.0) / k as f64;
        if next.abs() < thresh {
            break;
        }
        weights.push(next);
        k += 1;
        ctr += 1;
        if ctr == lim - 1 {
            break;
        }
    }
    weights.reverse();
    weights
}

pub fn frac_diff(series: &[f64], diff_amt: f64, thresh: f64) -> Vec<f64> {
    let n = series.len();
    if n == 0 {
        return Vec::new();
    }
    let weights = get_weights(diff_amt, n);

    let mut cum = Vec::with_capacity(n);
    let mut s = 0.0;
    for w in &weights {
        s += w.abs();
        cum.push(s);
    }
    let total = *cum.last().unwrap_or(&1.0);
    if total != 0.0 {
        for v in &mut cum {
            *v /= total;
        }
    }
    let skip = cum.iter().filter(|v| **v > thresh).count();

    let mut out = vec![f64::NAN; n];
    for iloc in skip..n {
        let w_start = n - (iloc + 1);
        let mut acc = 0.0;
        for j in 0..=iloc {
            acc += weights[w_start + j] * series[j];
        }
        out[iloc] = acc;
    }
    out
}

pub fn frac_diff_ffd(series: &[f64], diff_amt: f64, thresh: f64) -> Vec<f64> {
    let n = series.len();
    if n == 0 {
        return Vec::new();
    }
    let weights = get_weights_ffd(diff_amt, thresh, n);
    if weights.is_empty() {
        return vec![f64::NAN; n];
    }
    let width = weights.len() - 1;
    let mut out = vec![f64::NAN; n];
    for iloc in width..n {
        let loc0 = iloc - width;
        let mut acc = 0.0;
        for (k, w) in weights.iter().enumerate() {
            acc += *w * series[loc0 + k];
        }
        out[iloc] = acc;
    }
    out
}
