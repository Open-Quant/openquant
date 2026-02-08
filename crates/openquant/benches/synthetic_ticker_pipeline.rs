use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use nalgebra::DMatrix;
use openquant::risk_metrics::RiskMetrics;
use openquant::sampling::{get_ind_matrix, seq_bootstrap};
use openquant::util::fast_ewma::ewma;

fn synthetic_prices(n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    let mut p = 100.0;
    for i in 0..n {
        let t = i as f64;
        let drift = 0.0001;
        let seasonal = 0.0008 * (t / 64.0).sin() + 0.0004 * (t / 17.0).cos();
        let shock = 0.0006 * (t / 7.0).sin() * (t / 11.0).cos();
        let ret = drift + seasonal + shock;
        p *= 1.0 + ret;
        out.push(p);
    }
    out
}

fn returns_from_prices(prices: &[f64]) -> Vec<f64> {
    prices.windows(2).map(|w| w[1] / w[0] - 1.0).collect()
}

fn synthetic_indicator_matrix(n_bars: usize, n_labels: usize) -> Vec<Vec<u8>> {
    let bar_index: Vec<usize> = (0..n_bars).collect();
    let mut events = Vec::with_capacity(n_labels);
    for i in 0..n_labels {
        let start = (i * 3) % (n_bars - 20);
        let end = (start + 5 + (i % 17)).min(n_bars - 1);
        events.push((start, end));
    }
    get_ind_matrix(&events, &bar_index)
}

fn bench_ewma_on_ticker(c: &mut Criterion) {
    let prices = synthetic_prices(100_000);
    c.bench_function("synthetic_ticker/ewma_100k", |b| {
        b.iter(|| {
            let v = ewma(black_box(&prices), black_box(64));
            black_box(v);
        });
    });
}

fn bench_risk_metrics_on_ticker(c: &mut Criterion) {
    let prices = synthetic_prices(120_000);
    let rets = returns_from_prices(&prices);
    let rm = RiskMetrics;

    c.bench_function("synthetic_ticker/risk_metrics_var_es_cdar", |b| {
        b.iter(|| {
            let var = rm.calculate_value_at_risk(black_box(&rets), 0.05).unwrap();
            let es = rm.calculate_expected_shortfall(black_box(&rets), 0.05).unwrap();
            let cdar = rm
                .calculate_conditional_drawdown_risk(black_box(&rets), 0.05)
                .unwrap();
            black_box((var, es, cdar));
        });
    });
}

fn bench_seq_bootstrap_on_ticker_events(c: &mut Criterion) {
    let ind = synthetic_indicator_matrix(2000, 600);
    c.bench_function("synthetic_ticker/seq_bootstrap_2k_600", |b| {
        b.iter(|| {
            let sampled = seq_bootstrap(black_box(&ind), Some(240), None);
            black_box(sampled);
        });
    });
}

fn bench_end_to_end_ticker_pipeline(c: &mut Criterion) {
    c.bench_function("synthetic_ticker/pipeline_end_to_end", |b| {
        b.iter_batched(
            || {
                let prices = synthetic_prices(20_000);
                let rets = returns_from_prices(&prices);
                let ind = synthetic_indicator_matrix(1200, 360);
                (prices, rets, ind)
            },
            |(prices, rets, ind)| {
                let _ewma = ewma(&prices, 32);
                let rm = RiskMetrics;
                let _ = rm.calculate_value_at_risk(&rets, 0.05).unwrap();
                let _ = rm.calculate_expected_shortfall(&rets, 0.05).unwrap();
                let _ = rm.calculate_conditional_drawdown_risk(&rets, 0.05).unwrap();

                let n = rets.len().min(1000);
                let mut cov = DMatrix::zeros(3, 3);
                let mut cols = [Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n)];
                for i in 0..n {
                    let r = rets[i];
                    cols[0].push(r);
                    cols[1].push((1.0 + r).ln());
                    cols[2].push(r * r.signum());
                }
                let means: Vec<f64> = cols.iter().map(|v| v.iter().sum::<f64>() / v.len() as f64).collect();
                for i in 0..3 {
                    for j in 0..3 {
                        let mut s = 0.0;
                        for k in 0..n {
                            s += (cols[i][k] - means[i]) * (cols[j][k] - means[j]);
                        }
                        cov[(i, j)] = s / (n - 1) as f64;
                    }
                }
                let _ = rm.calculate_variance(&cov, &[0.4, 0.3, 0.3]).unwrap();

                let sampled = seq_bootstrap(&ind, Some(200), None);
                black_box(sampled);
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    synthetic_ticker_pipeline,
    bench_ewma_on_ticker,
    bench_risk_metrics_on_ticker,
    bench_seq_bootstrap_on_ticker_events,
    bench_end_to_end_ticker_pipeline
);
criterion_main!(synthetic_ticker_pipeline);
