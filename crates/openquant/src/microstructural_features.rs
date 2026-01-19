use std::f64::NAN;
use statrs::distribution::{ContinuousCDF, Normal};
use chrono::NaiveDateTime;

fn rolling_cov(x: &[f64], y: &[f64], window: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![NAN; n];
    if window < 2 {
        return out;
    }
    for i in 0..n {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let slice_x = &x[start..=i];
        let slice_y = &y[start..=i];
        let mean_x: f64 = slice_x.iter().sum::<f64>() / window as f64;
        let mean_y: f64 = slice_y.iter().sum::<f64>() / window as f64;
        let mut s = 0.0;
        for j in 0..window {
            s += (slice_x[j] - mean_x) * (slice_y[j] - mean_y);
        }
        out[i] = s / (window as f64 - 1.0);
    }
    out
}

pub fn get_roll_measure(close: &[f64], window: usize) -> Vec<f64> {
    if close.len() < 2 {
        return vec![NAN; close.len()];
    }
    let mut diff = vec![NAN; close.len()];
    for i in 1..close.len() {
        diff[i] = close[i] - close[i - 1];
    }
    let mut diff_lag = vec![NAN; close.len()];
    for i in 1..diff.len() {
        diff_lag[i] = diff[i - 1];
    }
    let cov = rolling_cov(&diff, &diff_lag, window);
    cov.iter()
        .map(|c| if c.is_nan() { NAN } else { 2.0 * (c.abs()).sqrt() })
        .collect()
}

pub fn get_roll_impact(close: &[f64], dollar_volume: &[f64], window: usize) -> Vec<f64> {
    let roll = get_roll_measure(close, window);
    roll.iter()
        .zip(dollar_volume.iter())
        .map(|(r, dv)| if r.is_nan() || *dv == 0.0 { NAN } else { r / dv })
        .collect()
}

fn rolling_max(arr: &[f64], window: usize) -> Vec<f64> {
    let mut out = vec![NAN; arr.len()];
    if window == 0 {
        return out;
    }
    for i in 0..arr.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let max_v = arr[start..=i].iter().fold(f64::NEG_INFINITY, |m, v| m.max(*v));
        out[i] = max_v;
    }
    out
}

fn rolling_min(arr: &[f64], window: usize) -> Vec<f64> {
    let mut out = vec![NAN; arr.len()];
    if window == 0 {
        return out;
    }
    for i in 0..arr.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let min_v = arr[start..=i].iter().fold(f64::INFINITY, |m, v| m.min(*v));
        out[i] = min_v;
    }
    out
}

fn _get_beta(high: &[f64], low: &[f64], window: usize) -> Vec<f64> {
    let mut ret_sq = vec![NAN; high.len()];
    for i in 0..high.len() {
        if low[i] == 0.0 {
            continue;
        }
        ret_sq[i] = (high[i] / low[i]).ln().powi(2);
    }
    // rolling sum over 2
    let mut two_sum = vec![NAN; ret_sq.len()];
    for i in 1..ret_sq.len() {
        if ret_sq[i].is_nan() || ret_sq[i - 1].is_nan() {
            continue;
        }
        two_sum[i] = ret_sq[i] + ret_sq[i - 1];
    }
    // rolling mean over window
    let mut beta = vec![NAN; ret_sq.len()];
    if window == 0 {
        return beta;
    }
    for i in 0..two_sum.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let slice = &two_sum[start..=i];
        if slice.iter().any(|v| v.is_nan()) {
            continue;
        }
        let mean = slice.iter().sum::<f64>() / window as f64;
        beta[i] = mean;
    }
    beta
}

fn _get_gamma(high: &[f64], low: &[f64]) -> Vec<f64> {
    let high_max = rolling_max(high, 2);
    let low_min = rolling_min(low, 2);
    high_max
        .iter()
        .zip(low_min.iter())
        .map(|(h, l)| {
            if h.is_nan() || l.is_nan() || *l == 0.0 {
                NAN
            } else {
                (h / l).ln().powi(2)
            }
        })
        .collect()
}

fn _get_alpha(beta: &[f64], gamma: &[f64]) -> Vec<f64> {
    let den = 3.0 - 2.0 * 2.0_f64.sqrt();
    beta.iter()
        .zip(gamma.iter())
        .map(|(b, g)| {
            if b.is_nan() || g.is_nan() {
                return NAN;
            }
            let mut alpha = (2.0_f64.sqrt() - 1.0) * b.sqrt() / den;
            alpha -= (g / den).sqrt();
            if alpha < 0.0 {
                0.0
            } else {
                alpha
            }
        })
        .collect()
}

pub fn get_corwin_schultz_estimator(high: &[f64], low: &[f64], window: usize) -> Vec<f64> {
    let beta = _get_beta(high, low, window);
    let gamma = _get_gamma(high, low);
    let alpha = _get_alpha(&beta, &gamma);
    alpha
        .iter()
        .map(|a| {
            if a.is_nan() {
                NAN
            } else {
                let ea = a.exp();
                2.0 * (ea - 1.0) / (1.0 + ea)
            }
        })
        .collect()
}

pub fn get_bekker_parkinson_vol(high: &[f64], low: &[f64], window: usize) -> Vec<f64> {
    let beta = _get_beta(high, low, window);
    let gamma = _get_gamma(high, low);
    let k2 = (8.0 / std::f64::consts::PI).sqrt();
    let den = 3.0 - 2.0 * 2.0_f64.sqrt();
    beta.iter()
        .zip(gamma.iter())
        .map(|(b, g)| {
            if b.is_nan() || g.is_nan() {
                return NAN;
            }
            let mut sigma = (2.0_f64.powf(-0.5) - 1.0) * b.sqrt() / (k2 * den);
            sigma += (g / (k2 * k2 * den)).sqrt();
            if sigma < 0.0 {
                0.0
            } else {
                sigma
            }
        })
        .collect()
}

pub fn get_bar_based_kyle_lambda(close: &[f64], volume: &[f64], window: usize) -> Vec<f64> {
    let mut diff = vec![NAN; close.len()];
    for i in 1..close.len() {
        diff[i] = close[i] - close[i - 1];
    }
    let mut sign = vec![NAN; diff.len()];
    for i in 0..diff.len() {
        let s = diff[i].signum();
        sign[i] = if s == 0.0 && i > 0 { sign[i - 1] } else { s };
    }
    let ratio: Vec<f64> = diff
        .iter()
        .zip(volume.iter())
        .zip(sign.iter())
        .map(|((d, v), s)| if *v == 0.0 || s.is_nan() { NAN } else { d / (v * s) })
        .collect();
    let mut out = vec![NAN; close.len()];
    for i in 0..close.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let slice = &ratio[start..=i];
        if slice.iter().any(|v| v.is_nan()) {
            continue;
        }
        out[i] = slice.iter().sum::<f64>() / window as f64;
    }
    out
}

pub fn get_bar_based_amihud_lambda(close: &[f64], dollar_volume: &[f64], window: usize) -> Vec<f64> {
    let mut ret_abs = vec![NAN; close.len()];
    for i in 1..close.len() {
        if close[i - 1] == 0.0 {
            continue;
        }
        ret_abs[i] = (close[i] / close[i - 1]).ln().abs();
    }
    let mut out = vec![NAN; close.len()];
    for i in 0..close.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let r = &ret_abs[start..=i];
        let dv = &dollar_volume[start..=i];
        if r.iter().any(|v| v.is_nan()) || dv.iter().any(|v| v.is_nan()) {
            continue;
        }
        let mut sum = 0.0;
        for (a, b) in r.iter().zip(dv.iter()) {
            if *b != 0.0 {
                sum += a / b;
            }
        }
        out[i] = sum / window as f64;
    }
    out
}

pub fn get_bar_based_hasbrouck_lambda(close: &[f64], dollar_volume: &[f64], window: usize) -> Vec<f64> {
    let mut log_ret = vec![NAN; close.len()];
    for i in 1..close.len() {
        if close[i - 1] == 0.0 {
            continue;
        }
        log_ret[i] = (close[i] / close[i - 1]).ln();
    }
    let mut sign = vec![NAN; log_ret.len()];
    for i in 0..log_ret.len() {
        let s = log_ret[i].signum();
        sign[i] = if s == 0.0 && i > 0 { sign[i - 1] } else { s };
    }
    let signed_sqrt: Vec<f64> = sign
        .iter()
        .zip(dollar_volume.iter())
        .map(|(s, dv)| if s.is_nan() || *dv < 0.0 { NAN } else { s * dv.sqrt() })
        .collect();
    let mut out = vec![NAN; close.len()];
    for i in 0..close.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let lr = &log_ret[start..=i];
        let sdv = &signed_sqrt[start..=i];
        if lr.iter().any(|v| v.is_nan()) || sdv.iter().any(|v| v.is_nan()) {
            continue;
        }
        let mut sum = 0.0;
        for (r, s) in lr.iter().zip(sdv.iter()) {
            if *s != 0.0 {
                sum += r / s;
            }
        }
        out[i] = sum / window as f64;
    }
    out
}

pub fn get_trades_based_kyle_lambda(price_diff: &[f64], volume: &[f64], aggressor_flags: &[f64]) -> f64 {
    let signed: Vec<f64> = volume.iter().zip(aggressor_flags.iter()).map(|(v, a)| v * a).collect();
    let num: f64 = signed.iter().zip(price_diff.iter()).map(|(x, y)| x * y).sum();
    let den: f64 = signed.iter().map(|x| x * x).sum();
    if den == 0.0 { NAN } else { num / den }
}

pub fn get_trades_based_amihud_lambda(log_ret: &[f64], dollar_volume: &[f64]) -> f64 {
    let num: f64 = dollar_volume.iter().zip(log_ret.iter()).map(|(x, y)| x * y.abs()).sum();
    let den: f64 = dollar_volume.iter().map(|x| x * x).sum();
    if den == 0.0 { NAN } else { num / den }
}

pub fn get_trades_based_hasbrouck_lambda(log_ret: &[f64], dollar_volume: &[f64], aggressor_flags: &[f64]) -> f64 {
    let signed: Vec<f64> = dollar_volume
        .iter()
        .zip(aggressor_flags.iter())
        .map(|(v, a)| v.sqrt() * a)
        .collect();
    let num: f64 = signed.iter().zip(log_ret.iter()).map(|(x, y)| x * y.abs()).sum();
    let den: f64 = signed.iter().map(|x| x * x).sum();
    if den == 0.0 { NAN } else { num / den }
}

// Misc helpers
pub fn vwap(dollar_volume: &[f64], volume: &[f64]) -> f64 {
    let sum_v: f64 = volume.iter().sum();
    if sum_v == 0.0 {
        return NAN;
    }
    dollar_volume.iter().sum::<f64>() / sum_v
}

pub fn get_avg_tick_size(tick_sizes: &[f64]) -> f64 {
    if tick_sizes.is_empty() {
        return NAN;
    }
    tick_sizes.iter().sum::<f64>() / tick_sizes.len() as f64
}

pub fn get_vpin(volume: &[f64], buy_volume: &[f64], window: usize) -> Vec<f64> {
    let sell_volume: Vec<f64> = volume.iter().zip(buy_volume.iter()).map(|(v, b)| v - b).collect();
    let imbalance: Vec<f64> = buy_volume
        .iter()
        .zip(sell_volume.iter())
        .map(|(b, s)| (b - s).abs())
        .collect();
    let mut out = vec![NAN; volume.len()];
    for i in 0..volume.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let imb_slice = &imbalance[start..=i];
        let vol = volume[i];
        if vol == 0.0 || imb_slice.iter().any(|v| v.is_nan()) {
            continue;
        }
        let mean_imb = imb_slice.iter().sum::<f64>() / window as f64;
        out[i] = mean_imb / vol;
    }
    out
}

pub fn get_bvc_buy_volume(close: &[f64], volume: &[f64], window: usize) -> Vec<f64> {
    let mut out = vec![NAN; close.len()];
    let norm = Normal::new(0.0, 1.0).unwrap();
    let mut diff = vec![NAN; close.len()];
    for i in 1..close.len() {
        diff[i] = close[i] - close[i - 1];
    }
    let mut rolling_std = vec![NAN; close.len()];
    for i in 0..close.len() {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let slice = &diff[start..=i];
        if slice.iter().any(|v| v.is_nan()) {
            continue;
        }
        let mean = slice.iter().sum::<f64>() / window as f64;
        let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (window as f64 - 1.0);
        rolling_std[i] = var.sqrt();
    }
    for i in 0..close.len() {
        if diff[i].is_nan() || rolling_std[i].is_nan() {
            continue;
        }
        let z = diff[i] / rolling_std[i].max(1e-12);
        out[i] = volume[i] * norm.cdf(z);
    }
    out
}

// Encoding utilities
pub fn encode_tick_rule_array(arr: &[i32]) -> Result<String, String> {
    let mut s = String::new();
    for v in arr {
        match *v {
            1 => s.push('a'),
            -1 => s.push('b'),
            0 => s.push('c'),
            other => return Err(format!("Unknown value for tick rule: {}", other)),
        }
    }
    Ok(s)
}

fn ascii_table() -> Vec<char> {
    (0..256).map(|i| char::from_u32(i).unwrap()).collect()
}

pub fn quantile_mapping(array: &[f64], num_letters: usize) -> Result<Vec<(f64, char)>, String> {
    if num_letters == 0 || num_letters > 256 {
        return Err("num_letters out of range".into());
    }
    let table = ascii_table();
    let alphabet = &table[..num_letters];
    let mut sorted = array.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut out: Vec<(f64, char)> = Vec::new();
    for (q, letter) in linspace(0.01, 1.0, alphabet.len()).iter().zip(alphabet.iter()) {
        let idx = ((*q) * (sorted.len() as f64 - 1.0)).round() as usize;
        out.push((sorted[idx], *letter));
    }
    Ok(out)
}

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n as f64 - 1.0);
    (0..n).map(|i| start + step * i as f64).collect()
}

pub fn sigma_mapping(array: &[f64], step: f64) -> Result<Vec<(f64, char)>, String> {
    if step <= 0.0 {
        return Err("step must be positive".into());
    }
    let table = ascii_table();
    let mut out: Vec<(f64, char)> = Vec::new();
    let mut i = 0usize;
    let mut val = array.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = array.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    while val < max_val {
        if i >= table.len() {
            return Err("Length of dictionary exceeds ASCII table".into());
        }
        out.push((val, table[i]));
        i += 1;
        val += step;
    }
    Ok(out)
}

fn find_nearest(enc: &[(f64, char)], value: f64) -> Option<char> {
    let mut best = None;
    let mut dist = f64::INFINITY;
    for (k, c) in enc {
        let d = (k - value).abs();
        if d < dist {
            dist = d;
            best = Some(*c);
        }
    }
    best
}

pub fn encode_array(array: &[f64], encoding: &[(f64, char)]) -> String {
    let mut s = String::new();
    for v in array {
        if let Some(c) = find_nearest(encoding, *v) {
            s.push(c);
        }
    }
    s
}

fn parse_datetime(s: &str) -> Result<NaiveDateTime, chrono::ParseError> {
    NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f")
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S"))
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y/%m/%d %H:%M:%S%.f"))
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y/%m/%d %H:%M:%S"))
}

// Entropy helpers
pub fn get_shannon_entropy(message: &str) -> f64 {
    let mut counts = std::collections::HashMap::new();
    for ch in message.chars() {
        *counts.entry(ch).or_insert(0usize) += 1;
    }
    let len = message.len() as f64;
    let mut ent = 0.0;
    for v in counts.values() {
        let freq = *v as f64 / len;
        ent += freq * freq.log2();
    }
    -ent
}

pub fn get_lempel_ziv_entropy(message: &str) -> f64 {
    if message.is_empty() {
        return 0.0;
    }
    let mut i = 1usize;
    let mut lib: Vec<String> = vec![message[0..1].to_string()];
    while i < message.len() {
        let mut j = i;
        while j < message.len() {
            let substr = &message[i..=j];
            if !lib.contains(&substr.to_string()) {
                lib.push(substr.to_string());
                break;
            }
            j += 1;
        }
        i = j + 1;
    }
    lib.len() as f64 / message.len() as f64
}

fn prob_mass_function(message: &str, word_length: usize) -> std::collections::HashMap<String, f64> {
    let mut lib: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
    for i in word_length..message.len() {
        let sub = &message[i - word_length..i];
        lib.entry(sub.to_string()).or_default().push(i - word_length);
    }
    let total = (message.len() - word_length) as f64;
    lib.into_iter()
        .map(|(k, v)| (k, v.len() as f64 / total))
        .collect()
}

pub fn get_plug_in_entropy(message: &str, word_length: usize) -> f64 {
    let pmf = prob_mass_function(message, word_length);
    -pmf.values().map(|p| p * p.log2()).sum::<f64>() / word_length as f64
}

fn match_length(message: &str, start: usize, window: usize) -> usize {
    let mut matched = 0usize;
    let start_window = start.saturating_sub(window);
    for length in 0..window {
        let end1 = start + length + 1;
        if end1 > message.len() {
            break;
        }
        let msg1 = &message[start..end1];
        for j in start_window..start {
            let end0 = j + length + 1;
            if end0 > message.len() {
                continue;
            }
            let msg0 = &message[j..end0];
            if msg0.len() != msg1.len() {
                continue;
            }
            if msg0 == msg1 {
                matched = msg1.len();
                break;
            }
        }
    }
    matched + 1
}

pub fn get_konto_entropy(message: &str, window: usize) -> f64 {
    if message.len() < 2 {
        return 0.0;
    }
    let points: Vec<usize> = if window == 0 {
        (1..=message.len() / 2).collect()
    } else {
        let w = window.min(message.len() / 2);
        (w..=message.len() - w).collect()
    };
    let mut sum = 0.0;
    let mut num = 0.0;
    for i in points {
        let l = match_length(message, i, if window == 0 { i } else { window });
        let denom = if window == 0 { (i + 1) as f64 } else { (window + 1) as f64 };
        sum += denom.log2() / l as f64;
        num += 1.0;
    }
    if num == 0.0 {
        0.0
    } else {
        sum / num
    }
}

pub struct MicrostructuralFeaturesGenerator {
    tick_num_iter: std::vec::IntoIter<usize>,
    current_bar_tick: usize,
    price_diff: Vec<f64>,
    trade_size: Vec<f64>,
    tick_rule: Vec<f64>,
    dollar_size: Vec<f64>,
    log_ret: Vec<f64>,
    prev_price: Option<f64>,
    prev_tick_rule: f64,
    volume_encoding: Option<Vec<(f64, char)>>,
    pct_encoding: Option<Vec<(f64, char)>>,
    entropy_types: Vec<&'static str>,
}

impl MicrostructuralFeaturesGenerator {
    pub fn new_from_csv(
        trades_path: &str,
        tick_num_series: &[usize],
        volume_encoding: Option<Vec<(f64, char)>>,
        pct_encoding: Option<Vec<(f64, char)>>,
    ) -> Result<Self, String> {
        // validate header
        let mut rdr = csv::ReaderBuilder::new().has_headers(true).from_path(trades_path).map_err(|e| e.to_string())?;
        if let Some(result) = rdr.records().next() {
            let rec = result.map_err(|e| e.to_string())?;
            if rec.len() != 3 {
                return Err("Must have only 3 columns in csv: date_time, price, & volume.".into());
            }
            rec[1].parse::<f64>().map_err(|_| "price column in csv not float.".to_string())?;
            rec[2].parse::<f64>().map_err(|_| "volume column in csv not int or float.".to_string())?;
            // Try multiple datetime formats (with/without fractional seconds)
            let _ = parse_datetime(&rec[0]).map_err(|_| "column 0 not datetime".to_string())?;
        }
        Ok(Self {
            tick_num_iter: tick_num_series.to_vec().into_iter(),
            current_bar_tick: tick_num_series.get(0).copied().unwrap_or(0),
            price_diff: Vec::new(),
            trade_size: Vec::new(),
            tick_rule: Vec::new(),
            dollar_size: Vec::new(),
            log_ret: Vec::new(),
            prev_price: None,
            prev_tick_rule: 0.0,
            volume_encoding,
            pct_encoding,
            entropy_types: vec!["shannon", "plug_in", "lempel_ziv", "konto"],
        })
    }

    fn reset_cache(&mut self) {
        self.price_diff.clear();
        self.trade_size.clear();
        self.tick_rule.clear();
        self.dollar_size.clear();
        self.log_ret.clear();
    }

    fn apply_tick_rule(&mut self, price: f64) -> f64 {
        let tick_diff = if let Some(prev) = self.prev_price { price - prev } else { 0.0 };
        let signed_tick = if tick_diff != 0.0 {
            let s = tick_diff.signum();
            self.prev_tick_rule = s;
            s
        } else {
            self.prev_tick_rule
        };
        signed_tick
    }

    fn get_price_diff(&self, price: f64) -> f64 {
        if let Some(prev) = self.prev_price {
            price - prev
        } else {
            0.0
        }
    }

    fn get_log_ret(&self, price: f64) -> f64 {
        if let Some(prev) = self.prev_price {
            (price / prev).ln()
        } else {
            0.0
        }
    }

    fn encode_entropy_features(&self, message: &str, out: &mut Vec<f64>) {
        out.push(get_shannon_entropy(message));
        out.push(get_plug_in_entropy(message, 1));
        out.push(get_lempel_ziv_entropy(message));
        out.push(get_konto_entropy(message, 0));
    }

    fn bar_features(&self, date_time: NaiveDateTime) -> Vec<f64> {
        let mut features = Vec::new();
        features.push(date_time.timestamp_millis() as f64);
        features.push(get_avg_tick_size(&self.trade_size));
        features.push(self.tick_rule.iter().sum::<f64>());
        features.push(vwap(&self.dollar_size, &self.trade_size));
        features.push(get_trades_based_kyle_lambda(&self.price_diff, &self.trade_size, &self.tick_rule));
        features.push(get_trades_based_amihud_lambda(&self.log_ret, &self.dollar_size));
        features.push(get_trades_based_hasbrouck_lambda(&self.log_ret, &self.dollar_size, &self.tick_rule));

        let tick_msg = encode_tick_rule_array(&self.tick_rule.iter().map(|v| *v as i32).collect::<Vec<_>>()).unwrap_or_default();
        self.encode_entropy_features(&tick_msg, &mut features);

        if let Some(enc) = &self.volume_encoding {
            let msg = encode_array(&self.trade_size, enc);
            self.encode_entropy_features(&msg, &mut features);
        }
        if let Some(enc) = &self.pct_encoding {
            let msg = encode_array(&self.log_ret, enc);
            self.encode_entropy_features(&msg, &mut features);
        }
        features
    }

    pub fn get_features_from_csv(&mut self, trades_path: &str) -> Result<Vec<Vec<f64>>, String> {
        let mut rdr = csv::ReaderBuilder::new().has_headers(true).from_path(trades_path).map_err(|e| e.to_string())?;
        let mut bars: Vec<Vec<f64>> = Vec::new();
        let mut tick_num = 0usize;
        for rec in rdr.records() {
            let rec = rec.map_err(|e| e.to_string())?;
            let ts = parse_datetime(&rec[0]).map_err(|e| e.to_string())?;
            let price = rec[1].parse::<f64>().map_err(|e| e.to_string())?;
            let volume = rec[2].parse::<f64>().map_err(|e| e.to_string())?;
            let dollar_value = price * volume;
            let signed_tick = self.apply_tick_rule(price);
            tick_num += 1;
            self.price_diff.push(self.get_price_diff(price));
            self.trade_size.push(volume);
            self.tick_rule.push(signed_tick);
            self.dollar_size.push(dollar_value);
            self.log_ret.push(self.get_log_ret(price));
            self.prev_price = Some(price);

            if self.current_bar_tick > 0 && tick_num >= self.current_bar_tick {
                bars.push(self.bar_features(ts));
                if let Some(next) = self.tick_num_iter.next() {
                    self.current_bar_tick = next;
                } else {
                    break;
                }
                self.reset_cache();
            }
        }
        Ok(bars)
    }
}
