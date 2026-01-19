use chrono::NaiveDateTime;
use statrs::distribution::{Normal, ContinuousCDF};

fn bet_size_sigmoid(w_param: f64, price_div: f64) -> f64 {
    price_div * (w_param + price_div * price_div).powf(-0.5)
}

fn inv_price_sigmoid(forecast_price: f64, w_param: f64, m_bet_size: f64) -> f64 {
    forecast_price - m_bet_size * (w_param / (1.0 - m_bet_size * m_bet_size)).sqrt()
}

pub fn get_signal(prob: &[f64], num_classes: usize, pred: &[f64]) -> Vec<f64> {
    prob.iter()
        .zip(pred.iter())
        .map(|(p, s)| {
            let z = (p - 1.0 / num_classes as f64) / (p * (1.0 - p)).sqrt();
            let norm = Normal::new(0.0, 1.0).unwrap();
            let size = 2.0 * norm.cdf(z) - 1.0;
            size * s.signum()
        })
        .collect()
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

pub fn avg_active_signals(signal: &[(NaiveDateTime, f64)], t1: &[NaiveDateTime]) -> Vec<(NaiveDateTime, f64)> {
    // Average bet sizes across active signals at each change point.
    let mut t_points: Vec<NaiveDateTime> = t1.iter().copied().collect();
    t_points.extend(signal.iter().map(|(ts, _)| *ts));
    t_points.sort();
    t_points.dedup();
    let mut out = Vec::new();
    for loc in t_points {
        let mut sum = 0.0;
        let mut count = 0.0;
        for ((s_ts, s_val), end) in signal.iter().zip(t1.iter()) {
            if *s_ts <= loc && (loc < *end) {
                sum += *s_val;
                count += 1.0;
            }
        }
        if count > 0.0 {
            out.push((loc, sum / count));
        } else {
            out.push((loc, 0.0));
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
    let signal0 = get_signal(&prob, num_classes, &side);
    let mut signals: Vec<(NaiveDateTime, f64)> = events.iter().map(|(ts, _, _, _)| *ts).zip(signal0.into_iter()).collect();
    if average_active {
        let t1: Vec<NaiveDateTime> = events.iter().map(|(_, t1, _, _)| *t1).collect();
        signals = avg_active_signals(&signals, &t1);
    }
    let discretized: Vec<f64> = discrete_signal(&signals.iter().map(|(_, v)| *v).collect::<Vec<_>>(), step_size);
    signals
        .iter()
        .zip(discretized.iter())
        .map(|(t, v)| (t.0, *v))
        .collect()
}

pub fn confirm_and_cast_to_df(pos: &[f64], max_pos: &[f64], m_p: &[f64], f: &[f64]) -> Vec<(f64, f64, f64, f64)> {
    pos.iter().zip(max_pos.iter()).zip(m_p.iter()).zip(f.iter()).map(|(((p, m), mp), f)| (*p, *m, *mp, *f)).collect()
}

pub fn get_w(price_div: f64, m_bet_size: f64, func: &str) -> f64 {
    match func {
        "sigmoid" => (price_div * price_div) * ((1.0 / (m_bet_size * m_bet_size)) - 1.0),
        _ => price_div,
    }
}

pub fn get_target_pos(w: f64, f: f64, m_p: f64, max_pos: f64, func: &str) -> f64 {
    let price_div = f - m_p;
    match func {
        "sigmoid" => {
            let w_param = w;
            (bet_size_sigmoid(w_param, price_div) * max_pos).trunc()
        }
        _ => price_div,
    }
}

pub fn limit_price(t_pos: f64, pos: f64, f: f64, w: f64, max_pos: f64, func: &str) -> f64 {
    if (t_pos - pos).abs() < f64::EPSILON {
        return f64::NAN;
    }
    match func {
        "sigmoid" => {
            let w_param = w;
            let sgn = (t_pos - pos).signum();
            let mut l_p = 0.0;
            for j in ((pos as i32 + sgn as i32).abs())..=(t_pos as i32).abs() {
                let m_bet = j as f64 / max_pos;
                l_p += inv_price_sigmoid(f, w_param, m_bet);
            }
            l_p / (t_pos - pos).abs()
        }
        _ => f,
    }
}

pub fn bet_size(w: f64, x: f64, func: &str) -> f64 {
    match func {
        "sigmoid" => bet_size_sigmoid(w, x),
        _ => x,
    }
}

pub fn bet_size_dynamic(
    pos: &[f64],
    max_pos: &[f64],
    m_p: &[f64],
    f: &[f64],
) -> Vec<(f64, f64, f64)> {
    let w_param = get_w(10.0, 0.95, "sigmoid");
    let mut out = Vec::new();
    for i in 0..pos.len() {
        let t_pos = get_target_pos(w_param, f[i], m_p[i], max_pos[i], "sigmoid");
        let l_p = limit_price(t_pos, pos[i], f[i], w_param, max_pos[i], "sigmoid");
        let b = bet_size(w_param, f[i] - m_p[i], "sigmoid");
        out.push((b, t_pos, l_p));
    }
    out
}

pub fn get_concurrent_sides(t1: &[(NaiveDateTime, NaiveDateTime)], side: &[f64]) -> Vec<(NaiveDateTime, f64, f64)> {
    // returns (index, active_long, active_short)
    let mut out = Vec::new();
    for (start, end) in t1.iter() {
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

pub fn bet_size_budget(t1: &[(NaiveDateTime, NaiveDateTime)], side: &[f64]) -> Vec<(NaiveDateTime, f64)> {
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

pub fn single_bet_size_mixed(c: f64, fit: &[f64; 5]) -> f64 {
    let c0 = cdf_mixture(fit[0], fit[1], fit[2], fit[3], fit[4], 0.0);
    let cdf = cdf_mixture(fit[0], fit[1], fit[2], fit[3], fit[4], c);
    if c >= 0.0 {
        (cdf - c0) / (1.0 - c0)
    } else {
        (cdf - c0) / c0
    }
}

pub fn bet_size_reserve(
    t1: &[(NaiveDateTime, NaiveDateTime)],
    side: &[f64],
    fit: &[f64; 5],
) -> Vec<(NaiveDateTime, f64, f64, f64)> {
    get_concurrent_sides(t1, side)
        .into_iter()
        .map(|(ts, l, s)| {
            let c_t = l - s;
            let b = single_bet_size_mixed(c_t, fit);
            (ts, l, s, b)
        })
        .collect()
}
