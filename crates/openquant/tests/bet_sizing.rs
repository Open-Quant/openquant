use chrono::{Duration, NaiveDateTime};
use openquant::bet_sizing::*;
use serde_json::Value;
use std::fs::File;
use std::path::Path;

fn dates(n: usize, start: &str, days: i64) -> Vec<NaiveDateTime> {
    let base =
        chrono::NaiveDate::parse_from_str(start, "%Y-%m-%d").unwrap().and_hms_opt(0, 0, 0).unwrap();
    (0..n).map(|i| base + Duration::days(i as i64 * days)).collect()
}

fn load_prob_dynamic_budget_fixture() -> Value {
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/bet_sizing/prob_dynamic_budget.json");
    let fixture_file = File::open(fixture_path).unwrap();
    serde_json::from_reader(fixture_file).unwrap()
}

fn standard_events_with_sides(side: [f64; 5]) -> Vec<(NaiveDateTime, NaiveDateTime, f64, f64)> {
    let shift: Vec<Duration> =
        (0..5).map(|i| Duration::hours((24.0 * (0.5 * i as f64 + 1.0)) as i64)).collect();
    let dates_vec = dates(5, "2000-01-01", 1);
    let t1: Vec<(NaiveDateTime, NaiveDateTime)> =
        dates_vec.iter().zip(shift.iter()).map(|(d, s)| (*d, *d + *s)).collect();
    let prob = [0.55, 0.7, 0.95, 0.65, 0.85];
    dates_vec
        .iter()
        .zip(t1.iter())
        .zip(prob.iter())
        .zip(side.iter())
        .map(|(((s, (_st, en)), p), si)| (*s, *en, *p, *si))
        .collect()
}

#[test]
fn test_bet_size_probability_defaults() {
    let events = standard_events_with_sides([1.0, -1.0, 1.0, -1.0, 1.0]);
    let res = bet_size_probability(&events, 2, 0.0, false);
    assert_eq!(res.len(), events.len());
    let fixture = load_prob_dynamic_budget_fixture();
    let expected: Vec<f64> =
        fixture["prob_default"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    for (r, e) in res.iter().map(|(_, v)| *v).zip(expected.iter()) {
        // statrs and scipy evaluate the normal CDF differently in the last ~1e-11.
        assert!((r - e).abs() < 1e-9, "got {r}, reference {e}");
    }
}

#[test]
fn test_bet_size_probability_avg_active() {
    let events = standard_events_with_sides([1.0, -1.0, 1.0, -1.0, 1.0]);
    let res = bet_size_probability(&events, 2, 0.0, true);
    let fixture = load_prob_dynamic_budget_fixture();
    let expected: Vec<f64> =
        fixture["prob_avg"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    assert_eq!(res.len(), expected.len());
    for (r, e) in res.iter().map(|(_, v)| *v).zip(expected.iter()) {
        assert!((r - e).abs() < 1e-9, "got {r}, reference {e}");
    }
}

#[test]
fn test_bet_size_probability_stepsize() {
    let events = standard_events_with_sides([1.0, -1.0, 1.0, -1.0, 1.0]);
    let res = bet_size_probability(&events, 2, 0.1, false);
    let fixture = load_prob_dynamic_budget_fixture();
    let expected: Vec<f64> =
        fixture["prob_step"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    assert_eq!(res.len(), expected.len());
    for (r, e) in res.iter().map(|(_, v)| *v).zip(expected.iter()) {
        assert!((r - e).abs() < 1e-12);
    }
}

#[test]
fn test_bet_size_probability_negative_stepsize_uses_abs() {
    let events = standard_events_with_sides([1.0, -1.0, 1.0, -1.0, 1.0]);
    let pos = bet_size_probability(&events, 2, 0.1, false);
    let neg = bet_size_probability(&events, 2, -0.1, false);
    assert_eq!(pos.len(), neg.len());
    for ((_, p), (_, n)) in pos.iter().zip(neg.iter()) {
        assert!((p - n).abs() < 1e-12);
    }
}

#[test]
fn test_bet_size_dynamic() {
    let _dates = dates(5, "2000-01-01", 1);
    let pos = [25.0, 35.0, 45.0, 40.0, 30.0];
    let max_pos = [55.0; 5];
    let m_p = [75.5, 76.9, 74.1, 67.75, 62.0];
    let f = [80.0, 75.0, 72.5, 65.0, 70.8];
    let res = bet_size_dynamic(&pos, &max_pos, &m_p, &f).unwrap();
    assert_eq!(res.len(), pos.len());
    let fixture = load_prob_dynamic_budget_fixture();
    let exp_bs: Vec<f64> = fixture["dynamic"]["bet_size"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let exp_tpos: Vec<f64> = fixture["dynamic"]["t_pos"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let exp_lp: Vec<f64> =
        fixture["dynamic"]["l_p"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    for (i, (b, tp, lp)) in res.iter().enumerate() {
        assert!((b - exp_bs[i]).abs() < 1e-12, "row {i} bet size");
        assert!((tp - exp_tpos[i]).abs() < 1e-12, "row {i} target position");
        // Rows 1-3 cross zero, where AFML snippet 10.4's limit-price loop is empty and gives 0.
        assert!((lp - exp_lp[i]).abs() < 1e-9, "row {i} limit price {lp}, reference {}", exp_lp[i]);
    }
}

#[test]
fn test_bet_size_budget() {
    let shift: Vec<Duration> =
        (0..5).map(|i| Duration::hours((24.0 * (0.5 * i as f64 + 1.0)) as i64)).collect();
    let t1: Vec<(NaiveDateTime, NaiveDateTime)> =
        dates(5, "2000-01-01", 1).iter().zip(shift.iter()).map(|(d, s)| (*d, *d + *s)).collect();
    let side = [1.0, -1.0, 1.0, -1.0, 1.0];
    let res = bet_size_budget(&t1, &side).unwrap();
    assert_eq!(res.len(), t1.len());
    let fixture = load_prob_dynamic_budget_fixture();
    let exp = fixture["budget"]["bet_size"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect::<Vec<_>>();
    for (r, e) in res.iter().map(|(_, v)| *v).zip(exp.iter()) {
        assert!((r - e).abs() < 1e-12);
    }
}

#[test]
fn test_bet_size_budget_div_zero_short_side() {
    let shift: Vec<Duration> =
        (0..5).map(|i| Duration::hours((24.0 * (0.5 * i as f64 + 1.0)) as i64)).collect();
    let t1: Vec<(NaiveDateTime, NaiveDateTime)> =
        dates(5, "2000-01-01", 1).iter().zip(shift.iter()).map(|(d, s)| (*d, *d + *s)).collect();
    let side = [1.0, 1.0, 1.0, 1.0, 1.0];
    let res = bet_size_budget(&t1, &side).unwrap();
    assert_eq!(res.len(), t1.len());
    for (_, v) in res {
        assert!(v >= 0.0);
    }
}

#[test]
fn test_get_concurrent_sides_counts() {
    let dates_vec = dates(3, "2000-01-01", 1);
    let t1 = vec![
        (dates_vec[0], dates_vec[0] + Duration::days(3)),
        (dates_vec[1], dates_vec[1] + Duration::days(3)),
        (dates_vec[2], dates_vec[2] + Duration::days(1)),
    ];
    let side = [1.0, -1.0, 1.0];
    let res = get_concurrent_sides(&t1, &side).unwrap();
    assert_eq!(res.len(), 3);
    assert_eq!(res[0].1, 1.0);
    assert_eq!(res[0].2, 0.0);
    assert_eq!(res[1].1, 1.0);
    assert_eq!(res[1].2, 1.0);
    assert_eq!(res[2].1, 2.0);
    assert_eq!(res[2].2, 1.0);
}

#[test]
fn test_confirm_and_cast_to_df_all_arrays() {
    let pos = [25.0, 35.0, 45.0];
    let max_pos = [55.0, 55.0, 55.0];
    let m_p = [75.5, 76.9, 74.1];
    let f = [80.0, 75.0, 72.5];
    let res = confirm_and_cast_to_df(&pos, &max_pos, &m_p, &f).unwrap();
    assert_eq!(res.len(), 3);
    assert_eq!(res[0], (25.0, 55.0, 75.5, 80.0));
    assert_eq!(res[2], (45.0, 55.0, 74.1, 72.5));
}

#[test]
fn test_confirm_and_cast_to_df_scalar_like() {
    let pos = [35.0];
    let max_pos = [55.0];
    let m_p = [75.0];
    let f = [80.0];
    let res = confirm_and_cast_to_df(&pos, &max_pos, &m_p, &f).unwrap();
    assert_eq!(res, vec![(35.0, 55.0, 75.0, 80.0)]);
}

#[test]
fn test_confirm_and_cast_to_df_one_series_broadcast() {
    let pos = [25.0, 35.0, 45.0, 40.0, 30.0];
    let max_pos = [55.0];
    let m_p = [75.0];
    let f = [80.0];
    let res = confirm_and_cast_to_df(&pos, &max_pos, &m_p, &f).unwrap();
    assert_eq!(res.len(), 5);
    assert_eq!(res[0], (25.0, 55.0, 75.0, 80.0));
    assert_eq!(res[4], (30.0, 55.0, 75.0, 80.0));
}

#[test]
fn test_confirm_and_cast_to_df_checked_shape_mismatch_error() {
    let pos = [25.0, 35.0, 45.0];
    let max_pos = [55.0, 56.0];
    let m_p = [75.0];
    let f = [80.0];
    let err = confirm_and_cast_to_df(&pos, &max_pos, &m_p, &f)
        .expect_err("shape mismatch should return typed error");
    assert_eq!(err, BetSizingError::ShapeMismatch { name: "max_pos", len: 2, expected: 3 });
    assert!(err.to_string().contains("expected 1 or 3"));
}

#[test]
fn test_cdf_mixture_and_single_above_zero() {
    let fit = [0.0, 1.0, 1.0, 2.0, 0.5];
    let cdf = cdf_mixture(fit[0], fit[1], fit[2], fit[3], fit[4], 0.5);
    assert!(cdf > 0.0 && cdf < 1.0);
    let b = single_bet_size_mixed(0.5, &fit);
    assert!((-1.0..=1.0).contains(&b));
}

#[test]
fn test_single_bet_size_mixed_below_zero() {
    let fit = [-1.0, 4.0, 2.0, 1.5, 0.4];
    let c0 = cdf_mixture(fit[0], fit[1], fit[2], fit[3], fit[4], 0.0);
    let cm = cdf_mixture(fit[0], fit[1], fit[2], fit[3], fit[4], -4.0);
    let expected = (cm - c0) / c0;
    let got = single_bet_size_mixed(-4.0, &fit);
    assert!((expected - got).abs() < 1e-12);
}

#[test]
fn test_power_helpers_and_limit_price_equal_pos() {
    let b = bet_size_power(2.0, 0.5).unwrap();
    assert!((b - 0.25).abs() < 1e-12);
    let m = inv_price_power(100.0, 2.0, 0.25);
    assert!((m - 99.5).abs() < 1e-12);
    let p = limit_price_power(10.0, 10.0, 100.0, 2.0, 50.0);
    assert!(p.is_nan());
}

/// AFML 10.2 "reserve" sizing against tests/fixtures/bet_sizing/generate_reserve.py, which
/// counts concurrent long and short bets and maps c_t through a fitted two-Gaussian mixture in
/// scipy. The fit is taken as given (the library fits by EM with a random start, so its own fit
/// is not reproducible); every one of the 500 rows is compared.
#[test]
fn test_bet_size_reserve_matches_reference() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/bet_sizing/reserve_fixture.json");
    let file = File::open(path).expect("fixture");
    let v: Value = serde_json::from_reader(file).expect("json");
    let floats = |key: &str| -> Vec<f64> {
        v["events_active"][key].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
    };
    let fit_vec: Vec<f64> =
        v["fit"].as_array().expect("fit").iter().map(|x| x.as_f64().unwrap()).collect();
    let fit: [f64; 5] = fit_vec.try_into().unwrap();
    let parse = |s: &Value| {
        NaiveDateTime::parse_from_str(s.as_str().unwrap(), "%Y-%m-%d %H:%M:%S%.f").unwrap()
    };
    let t1_vec: Vec<(NaiveDateTime, NaiveDateTime)> = v["events_active"]["index"]
        .as_array()
        .unwrap()
        .iter()
        .zip(v["events_active"]["t1"].as_array().unwrap())
        .map(|(s, e)| (parse(s), parse(e)))
        .collect();
    let side = floats("side");
    let (want_long, want_short, want_c, want_bet) =
        (floats("active_long"), floats("active_short"), floats("c_t"), floats("bet_size"));
    assert_eq!(t1_vec.len(), 500);

    let rows = bet_size_reserve_with_fit(&t1_vec, &side, &fit).unwrap();
    assert_eq!(rows.len(), t1_vec.len());
    for (i, (ts, long, short, c_t, bet)) in rows.iter().enumerate() {
        assert_eq!(*ts, t1_vec[i].0, "row {i} timestamp");
        assert_eq!(*long, want_long[i], "row {i} active_long");
        assert_eq!(*short, want_short[i], "row {i} active_short");
        assert_eq!(*c_t, want_c[i], "row {i} c_t");
        // The mixture CDF is statrs here and scipy there; they differ by ~3e-11.
        assert!((bet - want_bet[i]).abs() < 1e-9, "row {i}: bet {bet}, reference {}", want_bet[i]);
        assert!((single_bet_size_mixed(*c_t, &fit) - bet).abs() < 1e-15);
    }
}

#[test]
fn test_bet_size_reserve_fit_and_return_parameters() {
    let dates_vec = dates(8, "2000-01-01", 1);
    let t1 = vec![
        (dates_vec[0], dates_vec[0] + Duration::days(3)),
        (dates_vec[1], dates_vec[1] + Duration::days(3)),
        (dates_vec[2], dates_vec[2] + Duration::days(3)),
        (dates_vec[3], dates_vec[3] + Duration::days(2)),
        (dates_vec[4], dates_vec[4] + Duration::days(2)),
        (dates_vec[5], dates_vec[5] + Duration::days(2)),
        (dates_vec[6], dates_vec[6] + Duration::days(1)),
        (dates_vec[7], dates_vec[7] + Duration::days(1)),
    ];
    let side = [1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0];

    let (rows, params_opt) = bet_size_reserve_full(&t1, &side, 8, 1e-6, 500, true).unwrap();
    assert_eq!(rows.len(), t1.len());
    let params = params_opt.expect("expected fit parameters");
    assert!(params[2] > 0.0);
    assert!(params[3] > 0.0);
    assert!(params[4] > 0.0 && params[4] < 1.0);
    for (_, _, _, _, b) in rows {
        assert!((-1.0..=1.0).contains(&b));
    }
}
