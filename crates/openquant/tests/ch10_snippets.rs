use chrono::{Duration, NaiveDate, NaiveDateTime};
use openquant::bet_sizing::*;
use statrs::distribution::{ContinuousCDF, Normal};

struct Ch10Setup {
    prob: Vec<f64>,
    side: Vec<f64>,
    bet_size: Vec<f64>,
    bet_size_d: Vec<f64>,
    t1: Vec<NaiveDateTime>,
    signal: Vec<(NaiveDateTime, f64)>,
    t_pnts: Vec<NaiveDateTime>,
    avg_active: Vec<(NaiveDateTime, f64)>,
}

fn build_ch10_setup() -> Ch10Setup {
    let prob: Vec<f64> = vec![0.711, 0.898, 0.992, 0.595, 0.544, 0.775];
    let side: Vec<f64> = vec![1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
    let base = NaiveDate::from_ymd_opt(2000, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    let dates: Vec<NaiveDateTime> = (0..6).map(|i| base + Duration::days(i)).collect();
    let shift_list: Vec<f64> = vec![0.5, 1.0, 2.0, 1.5, 0.8, 0.2];
    let shift_dt: Vec<Duration> =
        shift_list.iter().map(|d| Duration::seconds((d * 86400.0).round() as i64)).collect();
    let t1: Vec<NaiveDateTime> = dates.iter().zip(shift_dt.iter()).map(|(d, s)| *d + *s).collect();

    let norm = Normal::new(0.0, 1.0).unwrap();
    let bet_size: Vec<f64> = prob
        .iter()
        .zip(side.iter())
        .map(|(p, s)| {
            let z = (p - 0.5) / (p * (1.0 - p)).sqrt();
            let size = 2.0 * norm.cdf(z) - 1.0;
            size * s
        })
        .collect();

    let bet_size_d: Vec<f64> = bet_size
        .iter()
        .map(|m| {
            let mut v = (m / 0.1).round() * 0.1;
            if v > 1.0 {
                v = 1.0;
            }
            if v < -1.0 {
                v = -1.0;
            }
            v
        })
        .collect();

    let signal: Vec<(NaiveDateTime, f64)> =
        dates.iter().copied().zip(bet_size.iter().copied()).collect();
    let mut t_pnts: Vec<NaiveDateTime> = t1.iter().copied().collect();
    t_pnts.extend(dates.iter().copied());
    t_pnts.sort();
    t_pnts.dedup();

    let mut avg_active = Vec::new();
    for loc in t_pnts.iter() {
        let mut sum = 0.0;
        let mut count = 0.0;
        for ((start, signal_val), end) in signal.iter().zip(t1.iter()) {
            if *start <= *loc && (*loc < *end) {
                sum += *signal_val;
                count += 1.0;
            }
        }
        if count > 0.0 {
            avg_active.push((*loc, sum / count));
        } else {
            avg_active.push((*loc, 0.0));
        }
    }

    Ch10Setup { prob, side, bet_size, bet_size_d, t1, signal, t_pnts, avg_active }
}

#[test]
fn test_get_signal() {
    let setup = build_ch10_setup();
    let test_bet_size = get_signal(&setup.prob, 2, Some(&setup.side));
    for (val, exp) in test_bet_size.iter().zip(setup.bet_size.iter()) {
        assert!((val - exp).abs() < 1e-7);
    }

    let test_bet_size_abs = get_signal(&setup.prob, 2, None);
    for (val, exp) in test_bet_size_abs.iter().zip(setup.bet_size.iter().map(|v| v.abs())) {
        assert!((val - exp).abs() < 1e-7);
    }

    let empty = get_signal(&[], 2, None);
    assert!(empty.is_empty());
}

#[test]
fn test_discrete_signal() {
    let setup = build_ch10_setup();
    let test_discrete = discrete_signal(&setup.bet_size, 0.1);
    for (val, exp) in test_discrete.iter().zip(setup.bet_size_d.iter()) {
        assert!((val - exp).abs() < 1e-7);
    }
}

#[test]
fn test_avg_active_signals() {
    let setup = build_ch10_setup();
    let test_avg = avg_active_signals(&setup.signal, &setup.t1);
    assert_eq!(test_avg.len(), setup.avg_active.len());
    for ((t, val), (exp_t, exp_val)) in test_avg.iter().zip(setup.avg_active.iter()) {
        assert_eq!(t, exp_t);
        assert!((val - exp_val).abs() < 1e-7);
    }
}

#[test]
fn test_mp_avg_active_signals() {
    let setup = build_ch10_setup();
    let test_avg = mp_avg_active_signals(&setup.signal, &setup.t1, &setup.t_pnts);
    assert_eq!(test_avg.len(), setup.avg_active.len());
    for ((t, val), (exp_t, exp_val)) in test_avg.iter().zip(setup.avg_active.iter()) {
        assert_eq!(t, exp_t);
        assert!((val - exp_val).abs() < 1e-7);
    }
}

#[test]
fn test_bet_size_sigmoid() {
    let x_div: f64 = 15.0;
    let w_param: f64 = 7.5;
    let m_test = x_div / (w_param + x_div * x_div).sqrt();
    let res = bet_size_sigmoid(w_param, x_div);
    assert!((res - m_test).abs() < 1e-7);
}

#[test]
fn test_bet_size_power() {
    let x_div: f64 = 0.4;
    let w_param: f64 = 1.5;
    let m_test = x_div.signum() * x_div.abs().powf(w_param);
    let res = bet_size_power(w_param, x_div);
    assert!((res - m_test).abs() < 1e-7);
}

#[test]
#[should_panic]
fn test_bet_size_power_value_error() {
    let _ = bet_size_power(2.0, 1.5);
}

#[test]
fn test_bet_size_power_return_zero() {
    let res = bet_size_power(2.0, 0.0);
    assert_eq!(res, 0.0);
}

#[test]
fn test_bet_size() {
    let x_div_sig: f64 = 25.0;
    let w_param_sig: f64 = 3.5;
    let m_test_sig = x_div_sig / (w_param_sig + x_div_sig * x_div_sig).sqrt();
    let res_sig = bet_size(w_param_sig, x_div_sig, "sigmoid");
    assert!((res_sig - m_test_sig).abs() < 1e-7);

    let x_div_pow: f64 = 0.7;
    let w_param_pow: f64 = 2.1;
    let m_test_pow = x_div_pow.signum() * x_div_pow.abs().powf(w_param_pow);
    let res_pow = bet_size(w_param_pow, x_div_pow, "power");
    assert!((res_pow - m_test_pow).abs() < 1e-7);
}

#[test]
#[should_panic]
fn test_bet_size_key_error() {
    let _ = bet_size(2.0, 3.0, "NotAFunction");
}

#[test]
fn test_get_target_pos_sigmoid() {
    let f_i: f64 = 34.6;
    let m_p: f64 = 21.9;
    let x_div = f_i - m_p;
    let w_param: f64 = 2.5;
    let max_pos: f64 = 200.0;
    let pos_test = (max_pos * x_div / (w_param + x_div * x_div).sqrt()).trunc();
    let res = get_target_pos_sigmoid(w_param, f_i, m_p, max_pos);
    assert!((res - pos_test).abs() < 1e-7);
}

#[test]
fn test_get_target_pos_power() {
    let f_i: f64 = 34.6;
    let m_p: f64 = 34.1;
    let x_div = f_i - m_p;
    let w_param: f64 = 2.1;
    let max_pos: f64 = 100.0;
    let pos_test = (max_pos * x_div.signum() * x_div.abs().powf(w_param)).trunc();
    let res = get_target_pos_power(w_param, f_i, m_p, max_pos);
    assert!((res - pos_test).abs() < 1e-7);
}

#[test]
fn test_get_target_pos() {
    let f_i_sig: f64 = 31.6;
    let m_p_sig: f64 = 22.9;
    let x_div_sig = f_i_sig - m_p_sig;
    let w_param_sig: f64 = 2.6;
    let max_pos_sig: f64 = 220.0;
    let pos_test_sig =
        (max_pos_sig * x_div_sig / (w_param_sig + x_div_sig * x_div_sig).sqrt()).trunc();
    let res_sig = get_target_pos(w_param_sig, f_i_sig, m_p_sig, max_pos_sig, "sigmoid");
    assert!((res_sig - pos_test_sig).abs() < 1e-7);

    let f_i_pow: f64 = 34.8;
    let m_p_pow: f64 = 34.1;
    let x_div_pow = f_i_pow - m_p_pow;
    let w_param_pow: f64 = 2.9;
    let max_pos_pow: f64 = 175.0;
    let pos_test_pow =
        (max_pos_pow * x_div_pow.signum() * x_div_pow.abs().powf(w_param_pow)).trunc();
    let res_pow = get_target_pos(w_param_pow, f_i_pow, m_p_pow, max_pos_pow, "power");
    assert!((res_pow - pos_test_pow).abs() < 1e-7);
}

#[test]
#[should_panic]
fn test_get_target_pos_key_error() {
    let _ = get_target_pos(1.0, 2.0, 1.0, 5.0, "NotAFunction");
}

#[test]
fn test_get_w_sigmoid() {
    let x_sig: f64 = 24.2;
    let m_sig: f64 = 0.98;
    let w_sig = x_sig * x_sig * (m_sig.powf(-2.0) - 1.0);
    let res = get_w_sigmoid(x_sig, m_sig);
    assert!((res - w_sig).abs() < 1e-7);
}

#[test]
fn test_get_w_power() {
    let x_pow: f64 = 0.9;
    let m_pow: f64 = 0.76;
    let w_pow = (m_pow / x_pow.signum()).ln() / x_pow.abs().ln();
    let res = get_w_power(x_pow, m_pow);
    assert!((res - w_pow).abs() < 1e-7);
}

#[test]
#[should_panic]
fn test_get_w_power_value_error() {
    let _ = get_w_power(1.2, 0.8);
}

#[test]
fn test_get_w_power_warning() {
    let res = get_w_power(0.1, 2.0);
    assert_eq!(res, 0.0);
}

#[test]
#[should_panic]
fn test_get_w_key_error() {
    let _ = get_w(0.6, 0.9, "NotAFunction");
}

#[test]
fn test_inv_price_sigmoid() {
    let f_i: f64 = 35.19;
    let w_sig: f64 = 9.32;
    let m_sig: f64 = 0.72;
    let inv_p = f_i - m_sig * (w_sig / (1.0 - m_sig * m_sig)).sqrt();
    let res = inv_price_sigmoid(f_i, w_sig, m_sig);
    assert!((res - inv_p).abs() < 1e-7);
}

#[test]
fn test_inv_price_power() {
    let f_i: f64 = 35.19;
    let w_pow: f64 = 3.32;
    let m_pow: f64 = 0.72;
    let inv_p = f_i - m_pow.signum() * m_pow.abs().powf(1.0 / w_pow);
    let res = inv_price_power(f_i, w_pow, m_pow);
    assert!((res - inv_p).abs() < 1e-7);
    assert_eq!(inv_price_power(f_i, w_pow, 0.0), f_i);
}

#[test]
fn test_inv_price() {
    let f_i_sig: f64 = 87.19;
    let w_sig: f64 = 7.34;
    let m_sig: f64 = 0.82;
    let inv_sig = f_i_sig - m_sig * (w_sig / (1.0 - m_sig * m_sig)).sqrt();
    let res_sig = inv_price(f_i_sig, w_sig, m_sig, "sigmoid");
    assert!((res_sig - inv_sig).abs() < 1e-7);

    let f_i_pow: f64 = 129.19;
    let w_pow: f64 = 4.02;
    let m_pow: f64 = 0.81;
    let inv_pow = f_i_pow - m_pow.signum() * m_pow.abs().powf(1.0 / w_pow);
    let res_pow = inv_price(f_i_pow, w_pow, m_pow, "power");
    assert!((res_pow - inv_pow).abs() < 1e-7);
}

#[test]
#[should_panic]
fn test_inv_price_key_error() {
    let _ = inv_price(12.0, 1.5, 0.7, "NotAFunction");
}

#[test]
fn test_limit_price_sigmoid() {
    let t_pos: f64 = 124.0;
    let pos: f64 = 112.0;
    let f: f64 = 165.5;
    let w: f64 = 8.44;
    let max_pos: f64 = 150.0;
    let sgn = (t_pos - pos).signum();
    let start = (pos + sgn).abs() as i64;
    let end = t_pos.abs() as i64;
    let mut sum_inv = 0.0;
    for j in start..=end {
        sum_inv += inv_price_sigmoid(f, w, j as f64 / max_pos);
    }
    let limit_p = sum_inv / (t_pos - pos).abs();
    let res = limit_price_sigmoid(t_pos, pos, f, w, max_pos);
    assert!((res - limit_p).abs() < 1e-7);
}

#[test]
fn test_limit_price_sigmoid_return_nan() {
    let res = limit_price_sigmoid(1.0, 1.0, 123.0, 21.0, 234.0);
    assert!(res.is_nan());
}

#[test]
fn test_limit_price_power() {
    let t_pos: f64 = 101.0;
    let pos: f64 = 95.0;
    let f: f64 = 195.7;
    let w: f64 = 3.44;
    let max_pos: f64 = 130.0;
    let sgn = (t_pos - pos).signum();
    let start = (pos + sgn).abs() as i64;
    let end = t_pos.abs() as i64;
    let mut sum_inv = 0.0;
    for j in start..=end {
        sum_inv += inv_price_power(f, w, j as f64 / max_pos);
    }
    let limit_p = sum_inv / (t_pos - pos).abs();
    let res = limit_price_power(t_pos, pos, f, w, max_pos);
    assert!((res - limit_p).abs() < 1e-7);
}

#[test]
#[should_panic]
fn test_limit_price_key_error() {
    let _ = limit_price(231.0, 221.0, 110.0, 3.4, 250.0, "NotAFunction");
}
