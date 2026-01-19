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

#[test]
fn test_bet_size_probability_defaults() {
    let shift: Vec<Duration> =
        (0..5).map(|i| Duration::hours((24.0 * (0.5 * i as f64 + 1.0)) as i64)).collect();
    let dates_vec = dates(5, "2000-01-01", 1);
    let t1: Vec<(NaiveDateTime, NaiveDateTime)> =
        dates_vec.iter().zip(shift.iter()).map(|(d, s)| (*d, *d + *s)).collect();
    let prob = [0.55, 0.7, 0.95, 0.65, 0.85];
    let side = [1.0, -1.0, 1.0, -1.0, 1.0];
    let events: Vec<_> = dates_vec
        .iter()
        .zip(t1.iter())
        .zip(prob.iter())
        .zip(side.iter())
        .map(|(((s, (_st, en)), p), si)| (*s, *en, *p, *si))
        .collect();
    let res = bet_size_probability(&events, 2, 0.0, false);
    assert_eq!(res.len(), events.len());
    // compare to Python fixture
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/bet_sizing/prob_dynamic_budget.json");
    let fixture_file = File::open(fixture_path).unwrap();
    let fixture: Value = serde_json::from_reader(fixture_file).unwrap();
    let expected: Vec<f64> =
        fixture["prob_default"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    for (r, e) in res.iter().map(|(_, v)| *v).zip(expected.iter()) {
        assert!((r - e).abs() < 1e-6);
    }
}

#[test]
fn test_bet_size_dynamic() {
    let _dates = dates(5, "2000-01-01", 1);
    let pos = [25.0, 35.0, 45.0, 40.0, 30.0];
    let max_pos = [55.0; 5];
    let m_p = [75.5, 76.9, 74.1, 67.75, 62.0];
    let f = [80.0, 75.0, 72.5, 65.0, 70.8];
    let res = bet_size_dynamic(&pos, &max_pos, &m_p, &f);
    assert_eq!(res.len(), pos.len());
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/bet_sizing/prob_dynamic_budget.json");
    let fixture_file = File::open(fixture_path).unwrap();
    let fixture: Value = serde_json::from_reader(fixture_file).unwrap();
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
    for ((b, tp, _), (eb, etp)) in res.iter().zip(exp_bs.iter().zip(exp_tpos.iter())) {
        assert!((b - *eb).abs() < 1e-6);
        assert!((tp - *etp).abs() < 1e-6);
    }
}

#[test]
fn test_bet_size_budget() {
    let shift: Vec<Duration> =
        (0..5).map(|i| Duration::hours((24.0 * (0.5 * i as f64 + 1.0)) as i64)).collect();
    let t1: Vec<(NaiveDateTime, NaiveDateTime)> =
        dates(5, "2000-01-01", 1).iter().zip(shift.iter()).map(|(d, s)| (*d, *d + *s)).collect();
    let side = [1.0, -1.0, 1.0, -1.0, 1.0];
    let res = bet_size_budget(&t1, &side);
    assert_eq!(res.len(), t1.len());
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/bet_sizing/prob_dynamic_budget.json");
    let fixture_file = File::open(fixture_path).unwrap();
    let fixture: Value = serde_json::from_reader(fixture_file).unwrap();
    let exp = fixture["budget"]["bet_size"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect::<Vec<_>>();
    for (r, e) in res.iter().map(|(_, v)| *v).zip(exp.iter()) {
        assert!((r - e).abs() < 1e-6);
    }
}

#[test]
fn test_cdf_mixture_and_single() {
    let fit = [0.0, 1.0, 1.0, 2.0, 0.5];
    let cdf = cdf_mixture(fit[0], fit[1], fit[2], fit[3], fit[4], 0.5);
    assert!(cdf > 0.0 && cdf < 1.0);
    let b = single_bet_size_mixed(0.5, &fit);
    assert!(b >= -1.0 && b <= 1.0);
}

#[test]
fn test_bet_size_reserve_stub() {
    // load fixture from Python run
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/bet_sizing/reserve_fixture.json");
    let file = File::open(path).expect("fixture");
    let v: Value = serde_json::from_reader(file).expect("json");
    let fit_arr = v["fit"].as_array().expect("fit");
    let fit: [f64; 5] = [
        fit_arr[0].as_f64().unwrap(),
        fit_arr[1].as_f64().unwrap(),
        fit_arr[2].as_f64().unwrap(),
        fit_arr[3].as_f64().unwrap(),
        fit_arr[4].as_f64().unwrap(),
    ];
    let _t1_vec = v["events_active"]["index"]
        .as_array()
        .unwrap()
        .iter()
        .zip(v["events_active"]["t1"].as_array().unwrap().iter())
        .map(|(s, e)| {
            let st =
                NaiveDateTime::parse_from_str(s.as_str().unwrap(), "%Y-%m-%d %H:%M:%S%.f").unwrap();
            let en =
                NaiveDateTime::parse_from_str(e.as_str().unwrap(), "%Y-%m-%d %H:%M:%S%.f").unwrap();
            (st, en)
        })
        .collect::<Vec<_>>();
    let c_t_vals: Vec<f64> =
        v["events_active"]["c_t"].as_array().unwrap().iter().map(|c| c.as_f64().unwrap()).collect();
    // compute bet sizes directly from c_t and fit parameters
    let rust_bets: Vec<f64> = c_t_vals.iter().map(|c| single_bet_size_mixed(*c, &fit)).collect();
    let expected_bets: Vec<f64> = v["events_active"]["bet_size"]
        .as_array()
        .unwrap()
        .iter()
        .take(5)
        .map(|x| x.as_f64().unwrap())
        .collect();
    for (rb, eb) in rust_bets.iter().take(5).zip(expected_bets.iter()) {
        assert!((rb - eb).abs() < 1e-6);
    }
}
