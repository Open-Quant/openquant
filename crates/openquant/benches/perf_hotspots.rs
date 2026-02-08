use chrono::{Duration, NaiveDate, NaiveDateTime};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use openquant::bet_sizing::{bet_size_reserve, bet_size_reserve_full};
use openquant::structural_breaks::{get_sadf, SadfLags};

fn fixture_series(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            100.0 + 0.2 * (t / 9.0).sin() + 0.05 * (t / 17.0).cos() + 0.001 * t
        })
        .collect()
}

fn bench_sadf_sm_power(c: &mut Criterion) {
    let prices = fixture_series(320);
    let logs: Vec<f64> = prices.into_iter().map(f64::ln).collect();

    c.bench_function("structural_breaks/get_sadf_sm_power", |b| {
        b.iter(|| {
            let _ = get_sadf(&logs, "sm_power", true, 20, SadfLags::Fixed(5)).expect("sadf");
        });
    });
}

fn bench_bet_sizing_reserve_fit(c: &mut Criterion) {
    let (t1, sides) = synthetic_events(500);

    c.bench_function("bet_sizing/bet_size_reserve_fit", |b| {
        b.iter_batched(
            || (t1.clone(), sides.clone()),
            |(events, dirs)| {
                let (out, params) = bet_size_reserve_full(&events, &dirs, 100, 1e-5, 200, true);
                assert_eq!(out.len(), dirs.len());
                assert!(params.is_some());
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_bet_sizing_reserve_reuse_fit(c: &mut Criterion) {
    let (t1, sides) = synthetic_events(500);

    let (_, params) = bet_size_reserve_full(&t1, &sides, 100, 1e-5, 200, true);
    let params = params.expect("fit params");

    c.bench_function("bet_sizing/bet_size_reserve_reuse_fit", |b| {
        b.iter(|| {
            let out = bet_size_reserve(&t1, &sides, &params);
            assert_eq!(out.len(), sides.len());
        });
    });
}

fn synthetic_events(n: usize) -> (Vec<(NaiveDateTime, NaiveDateTime)>, Vec<f64>) {
    let start = NaiveDate::from_ymd_opt(2020, 1, 1)
        .expect("valid date")
        .and_hms_opt(0, 0, 0)
        .expect("valid time");

    let mut events = Vec::with_capacity(n);
    let mut side = Vec::with_capacity(n);
    for i in 0..n {
        let s = start + Duration::minutes(i as i64);
        let e = s + Duration::minutes(5 + (i % 7) as i64);
        events.push((s, e));
        side.push(if i % 2 == 0 { 1.0 } else { -1.0 });
    }
    (events, side)
}

criterion_group!(
    perf_hotspots,
    bench_sadf_sm_power,
    bench_bet_sizing_reserve_fit,
    bench_bet_sizing_reserve_reuse_fit
);
criterion_main!(perf_hotspots);
