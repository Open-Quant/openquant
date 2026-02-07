use chrono::NaiveDateTime;
use rand::Rng;
use statrs::distribution::{ContinuousCDF, Normal};

pub fn bet_size_sigmoid(w_param: f64, price_div: f64) -> f64 {
    price_div * (w_param + price_div * price_div).powf(-0.5)
}

pub fn bet_size_power(w_param: f64, price_div: f64) -> f64 {
    if price_div < -1.0 || price_div > 1.0 {
        panic!(
            "Price divergence must be between -1 and 1, inclusive. Found price divergence value: {}",
            price_div
        );
    }
    if price_div == 0.0 {
        return 0.0;
    }
    price_div.signum() * price_div.abs().powf(w_param)
}

pub fn bet_size(w_param: f64, price_div: f64, func: &str) -> f64 {
    match func {
        "sigmoid" => bet_size_sigmoid(w_param, price_div),
        "power" => bet_size_power(w_param, price_div),
        _ => panic!("Invalid bet size function: {}", func),
    }
}

pub fn inv_price_sigmoid(forecast_price: f64, w_param: f64, m_bet_size: f64) -> f64 {
    forecast_price - m_bet_size * (w_param / (1.0 - m_bet_size * m_bet_size)).sqrt()
}

pub fn inv_price_power(forecast_price: f64, w_param: f64, m_bet_size: f64) -> f64 {
    if m_bet_size == 0.0 {
        return forecast_price;
    }
    forecast_price - m_bet_size.signum() * m_bet_size.abs().powf(1.0 / w_param)
}

pub fn inv_price(forecast_price: f64, w_param: f64, m_bet_size: f64, func: &str) -> f64 {
    match func {
        "sigmoid" => inv_price_sigmoid(forecast_price, w_param, m_bet_size),
        "power" => inv_price_power(forecast_price, w_param, m_bet_size),
        _ => panic!("Invalid inv_price function: {}", func),
    }
}

pub fn get_signal(prob: &[f64], num_classes: usize, pred: Option<&[f64]>) -> Vec<f64> {
    if prob.is_empty() {
        return Vec::new();
    }
    let norm = Normal::new(0.0, 1.0).unwrap();
    let mapped: Vec<f64> = prob
        .iter()
        .map(|p| {
            let z = (p - 1.0 / num_classes as f64) / (p * (1.0 - p)).sqrt();
            2.0 * norm.cdf(z) - 1.0
        })
        .collect();
    match pred {
        Some(side) => mapped.iter().zip(side.iter()).map(|(m, s)| m * s).collect(),
        None => mapped,
    }
}

pub fn discrete_signal(signal0: &[f64], step_size: f64) -> Vec<f64> {
    if step_size <= 0.0 {
        return signal0.to_vec();
    }
    signal0
        .iter()
        .map(|s| {
            let mut v = (s / step_size).round() * step_size;
            if v > 1.0 {
                v = 1.0;
            }
            if v < -1.0 {
                v = -1.0;
            }
            v
        })
        .collect()
}

pub fn avg_active_signals(
    signal: &[(NaiveDateTime, f64)],
    t1: &[NaiveDateTime],
) -> Vec<(NaiveDateTime, f64)> {
    let mut t_points: Vec<NaiveDateTime> = t1.iter().copied().collect();
    t_points.extend(signal.iter().map(|(ts, _)| *ts));
    t_points.sort();
    t_points.dedup();
    mp_avg_active_signals(signal, t1, &t_points)
}

pub fn mp_avg_active_signals(
    signal: &[(NaiveDateTime, f64)],
    t1: &[NaiveDateTime],
    molecule: &[NaiveDateTime],
) -> Vec<(NaiveDateTime, f64)> {
    let mut out = Vec::new();
    for loc in molecule {
        let mut sum = 0.0;
        let mut count = 0.0;
        for ((s_ts, s_val), end) in signal.iter().zip(t1.iter()) {
            if *s_ts <= *loc && (*loc < *end) {
                sum += *s_val;
                count += 1.0;
            }
        }
        if count > 0.0 {
            out.push((*loc, sum / count));
        } else {
            out.push((*loc, 0.0));
        }
    }
    out
}

pub fn bet_size_probability(
    events: &[(NaiveDateTime, NaiveDateTime, f64, f64)], // (start, t1, prob, side)
    num_classes: usize,
    step_size: f64,
    average_active: bool,
) -> Vec<(NaiveDateTime, f64)> {
    let prob: Vec<f64> = events.iter().map(|(_, _, p, _)| *p).collect();
    let side: Vec<f64> = events.iter().map(|(_, _, _, s)| *s).collect();
    let signal0 = get_signal(&prob, num_classes, Some(&side));
    let mut signals: Vec<(NaiveDateTime, f64)> =
        events.iter().map(|(ts, _, _, _)| *ts).zip(signal0.into_iter()).collect();
    if average_active {
        let t1: Vec<NaiveDateTime> = events.iter().map(|(_, t1, _, _)| *t1).collect();
        signals = avg_active_signals(&signals, &t1);
    }
    let discretized: Vec<f64> =
        discrete_signal(&signals.iter().map(|(_, v)| *v).collect::<Vec<_>>(), step_size.abs());
    signals.iter().zip(discretized.iter()).map(|(t, v)| (t.0, *v)).collect()
}

pub fn confirm_and_cast_to_df(
    pos: &[f64],
    max_pos: &[f64],
    m_p: &[f64],
    f: &[f64],
) -> Vec<(f64, f64, f64, f64)> {
    let lengths = [pos.len(), max_pos.len(), m_p.len(), f.len()];
    let target_len = lengths.into_iter().max().unwrap_or(0);
    assert!(target_len > 0, "Inputs must be non-empty.");

    fn broadcast(values: &[f64], len: usize, name: &str) -> Vec<f64> {
        if values.len() == len {
            values.to_vec()
        } else if values.len() == 1 {
            vec![values[0]; len]
        } else {
            panic!(
                "Input '{}' has length {}, expected 1 or {} for broadcast.",
                name,
                values.len(),
                len
            );
        }
    }

    let pos_v = broadcast(pos, target_len, "pos");
    let max_pos_v = broadcast(max_pos, target_len, "max_pos");
    let m_p_v = broadcast(m_p, target_len, "m_p");
    let f_v = broadcast(f, target_len, "f");

    (0..target_len).map(|i| (pos_v[i], max_pos_v[i], m_p_v[i], f_v[i])).collect()
}

pub fn get_w(price_div: f64, m_bet_size: f64, func: &str) -> f64 {
    match func {
        "sigmoid" => get_w_sigmoid(price_div, m_bet_size),
        "power" => get_w_power(price_div, m_bet_size),
        _ => panic!("Invalid get_w function: {}", func),
    }
}

pub fn get_target_pos(w: f64, f: f64, m_p: f64, max_pos: f64, func: &str) -> f64 {
    match func {
        "sigmoid" => get_target_pos_sigmoid(w, f, m_p, max_pos),
        "power" => get_target_pos_power(w, f, m_p, max_pos),
        _ => panic!("Invalid get_target_pos function: {}", func),
    }
}

pub fn limit_price(t_pos: f64, pos: f64, f: f64, w: f64, max_pos: f64, func: &str) -> f64 {
    match func {
        "sigmoid" => limit_price_sigmoid(t_pos, pos, f, w, max_pos),
        "power" => limit_price_power(t_pos, pos, f, w, max_pos),
        _ => panic!("Invalid limit_price function: {}", func),
    }
}

pub fn get_w_sigmoid(price_div: f64, m_bet_size: f64) -> f64 {
    (price_div * price_div) * ((1.0 / (m_bet_size * m_bet_size)) - 1.0)
}

pub fn get_w_power(price_div: f64, m_bet_size: f64) -> f64 {
    if price_div < -1.0 || price_div > 1.0 {
        panic!(
            "Price divergence argument 'x' must be between -1 and 1, inclusive when using function 'power'."
        );
    }
    let w_calc = (m_bet_size / price_div.signum()).ln() / price_div.abs().ln();
    if w_calc < 0.0 {
        return 0.0;
    }
    w_calc
}

pub fn get_target_pos_sigmoid(
    w_param: f64,
    forecast_price: f64,
    market_price: f64,
    max_pos: f64,
) -> f64 {
    (bet_size_sigmoid(w_param, forecast_price - market_price) * max_pos).trunc()
}

pub fn get_target_pos_power(
    w_param: f64,
    forecast_price: f64,
    market_price: f64,
    max_pos: f64,
) -> f64 {
    (bet_size_power(w_param, forecast_price - market_price) * max_pos).trunc()
}

pub fn limit_price_sigmoid(t_pos: f64, pos: f64, f: f64, w: f64, max_pos: f64) -> f64 {
    let target = t_pos.trunc() as i64;
    let current = pos.trunc() as i64;
    if target == current {
        return f64::NAN;
    }
    let sgn = (target - current).signum();
    let mut l_p = 0.0;
    let start = (current + sgn).abs();
    let end = target.abs();
    for j in start..=end {
        let m_bet = j as f64 / max_pos;
        l_p += inv_price_sigmoid(f, w, m_bet);
    }
    l_p / (target - current).abs() as f64
}

pub fn limit_price_power(t_pos: f64, pos: f64, f: f64, w: f64, max_pos: f64) -> f64 {
    let target = t_pos.trunc() as i64;
    let current = pos.trunc() as i64;
    if target == current {
        return f64::NAN;
    }
    let sgn = (target - current).signum();
    let mut l_p = 0.0;
    let start = (current + sgn).abs();
    let end = target.abs();
    for j in start..=end {
        let m_bet = j as f64 / max_pos;
        l_p += inv_price_power(f, w, m_bet);
    }
    l_p / (target - current).abs() as f64
}

pub fn bet_size_dynamic(
    pos: &[f64],
    max_pos: &[f64],
    m_p: &[f64],
    f: &[f64],
) -> Vec<(f64, f64, f64)> {
    let w_param = get_w(10.0, 0.95, "sigmoid");
    confirm_and_cast_to_df(pos, max_pos, m_p, f)
        .into_iter()
        .map(|(p, m, mp, forecast)| {
            let t_pos = get_target_pos(w_param, forecast, mp, m, "sigmoid");
            let l_p = limit_price(t_pos, p, forecast, w_param, m, "sigmoid");
            let b = bet_size(w_param, forecast - mp, "sigmoid");
            (b, t_pos, l_p)
        })
        .collect()
}

pub fn get_concurrent_sides(
    t1: &[(NaiveDateTime, NaiveDateTime)],
    side: &[f64],
) -> Vec<(NaiveDateTime, f64, f64)> {
    // returns (index, active_long, active_short)
    let mut out = Vec::new();
    for (start, _end) in t1.iter() {
        let mut long = 0.0;
        let mut short = 0.0;
        for (j, (s, e)) in t1.iter().enumerate() {
            if *s <= *start && *e > *start {
                if side[j] > 0.0 {
                    long += 1.0;
                } else {
                    short += 1.0;
                }
            }
        }
        out.push((*start, long, short));
    }
    out
}

pub fn bet_size_budget(
    t1: &[(NaiveDateTime, NaiveDateTime)],
    side: &[f64],
) -> Vec<(NaiveDateTime, f64)> {
    let conc = get_concurrent_sides(t1, side);
    let max_long = conc.iter().map(|(_, l, _)| *l).fold(0.0, f64::max);
    let max_short = conc.iter().map(|(_, _, s)| *s).fold(0.0, f64::max);
    conc.iter()
        .map(|(ts, l, s)| {
            let avg_long = if max_long > 0.0 { l / max_long } else { 0.0 };
            let avg_short = if max_short > 0.0 { s / max_short } else { 0.0 };
            (*ts, avg_long - avg_short)
        })
        .collect()
}

pub fn cdf_mixture(mu1: f64, mu2: f64, sigma1: f64, sigma2: f64, p1: f64, x: f64) -> f64 {
    let n1 = Normal::new(mu1, sigma1.max(1e-8)).unwrap();
    let n2 = Normal::new(mu2, sigma2.max(1e-8)).unwrap();
    p1 * n1.cdf(x) + (1.0 - p1) * n2.cdf(x)
}

fn normal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let s = sigma.max(1e-8);
    let z = (x - mu) / s;
    (-0.5 * z * z).exp() / (s * (2.0 * std::f64::consts::PI).sqrt())
}

fn fit_two_normal_mixture_em(
    samples: &[f64],
    fit_runs: usize,
    epsilon: f64,
    max_iter: usize,
) -> [f64; 5] {
    assert!(!samples.is_empty(), "samples must be non-empty");
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n.max(1.0);
    let std = var.sqrt().max(1e-3);
    let mut rng = rand::thread_rng();

    let mut best_ll = f64::NEG_INFINITY;
    let mut best = [mean - std, mean + std, std, std, 0.5];

    for _ in 0..fit_runs.max(1) {
        let x1 = samples[rng.gen_range(0..samples.len())];
        let x2 = samples[rng.gen_range(0..samples.len())];
        let mut mu1 = x1.min(x2);
        let mut mu2 = x1.max(x2);
        let mut sigma1 = std;
        let mut sigma2 = std;
        let mut p1 = rng.gen_range(0.25..0.75);
        let mut prev_ll = f64::NEG_INFINITY;

        for _ in 0..max_iter.max(1) {
            let mut gammas = Vec::with_capacity(samples.len());
            let mut ll = 0.0;

            for &x in samples {
                let w1 = p1 * normal_pdf(x, mu1, sigma1);
                let w2 = (1.0 - p1) * normal_pdf(x, mu2, sigma2);
                let denom = (w1 + w2).max(1e-16);
                gammas.push(w1 / denom);
                ll += denom.ln();
            }

            let n1 = gammas.iter().sum::<f64>().max(1e-8);
            let n2 = (n - n1).max(1e-8);
            p1 = (n1 / n).clamp(1e-6, 1.0 - 1e-6);

            mu1 = gammas.iter().zip(samples.iter()).map(|(g, x)| g * x).sum::<f64>() / n1;
            mu2 = gammas.iter().zip(samples.iter()).map(|(g, x)| (1.0 - g) * x).sum::<f64>() / n2;

            sigma1 = (gammas
                .iter()
                .zip(samples.iter())
                .map(|(g, x)| g * (x - mu1).powi(2))
                .sum::<f64>()
                / n1)
                .sqrt()
                .max(1e-6);

            sigma2 = (gammas
                .iter()
                .zip(samples.iter())
                .map(|(g, x)| (1.0 - g) * (x - mu2).powi(2))
                .sum::<f64>()
                / n2)
                .sqrt()
                .max(1e-6);

            if (ll - prev_ll).abs() < epsilon {
                prev_ll = ll;
                break;
            }
            prev_ll = ll;
        }

        if prev_ll > best_ll {
            best_ll = prev_ll;
            best = [mu1, mu2, sigma1, sigma2, p1];
        }
    }

    best
}

pub fn single_bet_size_mixed(c: f64, fit: &[f64; 5]) -> f64 {
    let c0 = cdf_mixture(fit[0], fit[1], fit[2], fit[3], fit[4], 0.0);
    let cdf = cdf_mixture(fit[0], fit[1], fit[2], fit[3], fit[4], c);
    if c >= 0.0 {
        (cdf - c0) / (1.0 - c0)
    } else {
        (cdf - c0) / c0
    }
}

pub fn bet_size_reserve_with_fit(
    t1: &[(NaiveDateTime, NaiveDateTime)],
    side: &[f64],
    fit: &[f64; 5],
) -> Vec<(NaiveDateTime, f64, f64, f64, f64)> {
    get_concurrent_sides(t1, side)
        .into_iter()
        .map(|(ts, l, s)| {
            let c_t = l - s;
            let b = single_bet_size_mixed(c_t, fit);
            (ts, l, s, c_t, b)
        })
        .collect()
}

pub fn bet_size_reserve_full(
    t1: &[(NaiveDateTime, NaiveDateTime)],
    side: &[f64],
    fit_runs: usize,
    epsilon: f64,
    max_iter: usize,
    return_parameters: bool,
) -> (Vec<(NaiveDateTime, f64, f64, f64, f64)>, Option<[f64; 5]>) {
    let concurrent = get_concurrent_sides(t1, side);
    let c_t: Vec<f64> = concurrent.iter().map(|(_, l, s)| l - s).collect();
    let fit = fit_two_normal_mixture_em(&c_t, fit_runs, epsilon, max_iter);
    let events = concurrent
        .into_iter()
        .zip(c_t.into_iter())
        .map(|((ts, l, s), c)| {
            let b = single_bet_size_mixed(c, &fit);
            (ts, l, s, c, b)
        })
        .collect();
    let params = if return_parameters { Some(fit) } else { None };
    (events, params)
}

pub fn bet_size_reserve(
    t1: &[(NaiveDateTime, NaiveDateTime)],
    side: &[f64],
    fit: &[f64; 5],
) -> Vec<(NaiveDateTime, f64, f64, f64)> {
    bet_size_reserve_with_fit(t1, side, fit)
        .into_iter()
        .map(|(ts, l, s, _c_t, b)| (ts, l, s, b))
        .collect()
}
