---
title: Rust Core Workflow
description: One runnable Rust program from event sampling to portfolio risk, with its output.
status: reviewed
last_validated: '2026-08-30'
audience:
  - quant-dev
  - platform-engineering
afml_chapter:
  - CHAPTER 2
  - CHAPTER 3
  - CHAPTER 7
  - CHAPTER 11
  - CHAPTER 12
  - CHAPTER 14
  - CHAPTER 16
sidebar:
  order: 1
---

The AFML pipeline is five stages, and the interesting part is how they
constrain each other: how you sample events decides what a label can
mean, and how labels overlap in time decides whether your validation is
honest. This page is one program that threads all five, and its actual
output.

Save it as `crates/openquant/examples/afml_workflow.rs` and run:

```bash
cargo run -p openquant --example afml_workflow
```

That is the command used to produce the output below. Add `--release` if
you want it fast; the first build of either profile compiles `polars` and
`nalgebra` and takes a while.

## The program

```rust
//! End-to-end AFML workflow: sampling -> labeling -> sizing -> purged CV -> risk.
use chrono::{Duration, NaiveDate, NaiveDateTime};
use nalgebra::DMatrix;
use openquant::backtest_statistics::sharpe_ratio;
use openquant::bet_sizing::{discrete_signal, get_signal};
use openquant::cross_validation::PurgedKFold;
use openquant::filters::{cusum_filter_indices, Threshold};
use openquant::labeling::{
    add_vertical_barrier, triple_barrier_events, triple_barrier_labels, TripleBarrierConfig,
};
use openquant::portfolio_optimization::allocate_max_sharpe;
use openquant::risk_metrics::RiskMetrics;

const N_BARS: usize = 480;
const N_ASSETS: usize = 4;

/// Deterministic synthetic minute bars. No RNG, so the printed numbers below
/// are reproducible on any machine.
fn synthetic_series() -> (Vec<NaiveDateTime>, Vec<f64>) {
    let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap().and_hms_opt(9, 30, 0).unwrap();
    let mut ts = Vec::with_capacity(N_BARS);
    let mut close = Vec::with_capacity(N_BARS);
    for i in 0..N_BARS {
        let x = i as f64;
        ts.push(start + Duration::minutes(i as i64));
        close.push(80.0 + 0.004 * x + 0.45 * (x / 9.0).sin() + 0.25 * (x / 37.0).sin());
    }
    (ts, close)
}

fn main() {
    let (ts, close) = synthetic_series();
    let bars: Vec<(NaiveDateTime, f64)> = ts.iter().copied().zip(close.iter().copied()).collect();

    // ---- Stage 1: event sampling -------------------------------------------
    // Sample on cumulative 0.1% moves instead of on the clock, so every event
    // carries comparable information (AFML ch. 2/3).
    let event_idx = cusum_filter_indices(&close, Threshold::Scalar(0.001));
    let t_events: Vec<NaiveDateTime> = event_idx.iter().map(|&i| ts[i]).collect();
    println!("stage 1  {:>4} bars -> {:>3} CUSUM events", N_BARS, t_events.len());

    // ---- Stage 2: triple-barrier labels + bet sizing ------------------------
    // Barrier width is a multiple of a per-event volatility target, so profit
    // taking and stop loss scale with the regime rather than being fixed ticks.
    let window = 20usize;
    let target: Vec<(NaiveDateTime, f64)> = (0..N_BARS)
        .map(|i| {
            let lo = i.saturating_sub(window);
            let rets: Vec<f64> =
                (lo + 1..=i).map(|k| close[k] / close[k - 1] - 1.0).collect();
            let n = rets.len().max(1) as f64;
            let mean = rets.iter().sum::<f64>() / n;
            let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
            (ts[i], var.sqrt().max(1e-4))
        })
        .collect();

    let vbars = add_vertical_barrier(&t_events, &bars, 0, 0, 45, 0);
    let cfg = TripleBarrierConfig {
        pt: 1.0,                              // profit-take barrier = 1.0 x target vol
        sl: 1.0,                              // stop-loss barrier  = 1.0 x target vol
        min_ret: 0.0002,                      // skip events too small to trade
        vertical_barrier_times: Some(&vbars), // 45-minute holding-period cap
    };
    let events = triple_barrier_events(&bars, &t_events, &target, cfg, None);
    let labeled = triple_barrier_labels(&events, &bars);
    let wins = labeled.iter().filter(|l| l.label > 0).count();
    println!(
        "stage 2  {:>3} labeled events, {:>3} positive ({:.0}%)",
        labeled.len(),
        wins,
        100.0 * wins as f64 / labeled.len().max(1) as f64
    );

    // A real model goes here. We stand in a calibrated-probability vector so the
    // sizing stage has something to consume; `get_signal` maps p -> [-1, 1] via
    // a z-test against 1/num_classes, and `discrete_signal` quantises to
    // tradable steps to suppress churn.
    let probs: Vec<f64> = (0..labeled.len())
        .map(|i| 0.5 + 0.18 * ((i as f64) / 7.0).sin())
        .collect();
    let raw = get_signal(&probs, 2, None);
    let sized = discrete_signal(&raw, 0.25);
    println!(
        "stage 3  sizes in [{:.2}, {:.2}], {} distinct steps",
        sized.iter().cloned().fold(f64::INFINITY, f64::min),
        sized.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        {
            let mut v: Vec<i64> = sized.iter().map(|s| (s * 100.0).round() as i64).collect();
            v.sort_unstable();
            v.dedup();
            v.len()
        }
    );

    // ---- Stage 4: purged, embargoed cross-validation ------------------------
    // Labels overlap in time, so a plain K-fold leaks: a training label whose
    // barrier closes inside the test window has already seen the test outcome.
    // PurgedKFold drops those, then embargoes a fraction after each test fold.
    let info_sets: Vec<(NaiveDateTime, NaiveDateTime)> = events
        .iter()
        .map(|(t0, ev)| (*t0, ev.t1.unwrap_or(*t0)))
        .collect();
    let cv = PurgedKFold::new(5, info_sets.clone(), 0.01).expect("purged kfold");
    let splits = cv.split(info_sets.len()).expect("split");
    let naive_train = info_sets.len() - info_sets.len() / 5;
    println!(
        "stage 4  5 folds, embargo 1%: train {} -> {} rows ({} purged by overlap)",
        naive_train,
        splits[0].0.len(),
        naive_train - splits[0].0.len()
    );

    // ---- Stage 5: strategy risk and portfolio allocation --------------------
    let strat_returns: Vec<f64> =
        labeled.iter().zip(sized.iter()).map(|(l, s)| l.ret * s).collect();
    let sr = sharpe_ratio(&strat_returns, 252.0 * 390.0, 0.0);
    let rm = RiskMetrics;
    let var = rm.calculate_value_at_risk(&strat_returns, 0.05).expect("var");
    let es = rm.calculate_expected_shortfall(&strat_returns, 0.05).expect("es");
    println!("stage 5  sharpe {:+.3}   VaR(5%) {:+.5}   ES(5%) {:+.5}", sr, var, es);

    let mut px = Vec::with_capacity(N_BARS * N_ASSETS);
    for i in 0..N_BARS {
        for j in 0..N_ASSETS {
            let lag = i.saturating_sub(j + 1);
            px.push(close[lag] * (1.0 + 0.0015 * j as f64)
                + (0.45 + 0.07 * j as f64) * ((i + 3 * j) as f64 / (9.5 + j as f64)).sin());
        }
    }
    let prices = DMatrix::from_row_slice(N_BARS, N_ASSETS, &px);
    let alloc = allocate_max_sharpe(&prices, 0.0, None, None).expect("max sharpe");
    println!(
        "stage 5  weights {:?}  portfolio sharpe {:.3}",
        alloc.weights.iter().map(|w| (w * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        alloc.portfolio_sharpe
    );
}
```

## Output

```text
stage 1   480 bars -> 147 CUSUM events
stage 2  106 labeled events,  57 positive (54%)
stage 3  sizes in [-0.25, 0.25], 3 distinct steps
stage 4  5 folds, embargo 1%: train 85 -> 82 rows (3 purged by overlap)
stage 5  sharpe +47.975   VaR(5%) -0.00015   ES(5%) -0.00016
stage 5  weights [0.262, 0.415, 0.0, 0.324]  portfolio sharpe 27.548
```

:::caution[Those Sharpe numbers are nonsense, and that is the point]
`sharpe +47.975` is not a result. The synthetic series is a sum of two
sine waves with no noise term, so it is very nearly deterministic, and
the program annualises by `sqrt(252 × 390)` ≈ 313. A near-riskless series
divided by a near-zero standard deviation and multiplied by 313 produces
exactly this.

The numbers to actually read on this run are the structural ones — 480
bars gave 147 events, 106 of which cleared `min_ret` and got labels, and
purging removed 3 of 85 training rows. Those describe the pipeline. The
Sharpe describes the toy.

If a real dataset ever hands you a number like this, the first thing to
check is your `entries_per_year`, and the second is whether your returns
series has any variance in it.
:::

## What each stage is doing, and why

### Stage 1 — sample on information, not on the clock

`cusum_filter_indices` accumulates signed returns and fires when the
running sum breaches a threshold, resetting on each event. Time bars
give you many uninformative observations in quiet periods and too few in
active ones; CUSUM events arrive at a rate set by the market. Everything
downstream inherits this: a label is a statement about *an event*, so
sampling badly is not something later stages can repair.

`Threshold::Scalar(0.001)` is a flat 0.1%. `Threshold::Dynamic(Vec<f64>)`
takes a per-bar threshold if you want to scale it by a volatility
estimate, which is usually the better choice on real data.

### Stage 2 — label with barriers scaled to volatility

`triple_barrier_events` places three barriers around each event: a
profit-take at `pt × target`, a stop-loss at `sl × target`, and a
vertical (time) barrier. `target` is per-event, so a 1.0 multiplier means
"one unit of that event's own volatility" rather than a fixed number of
ticks — the same configuration stays meaningful across regimes.

The three knobs and what they cost you:

| Field | In the program | What moving it does |
|---|---|---|
| `pt` / `sl` | `1.0` / `1.0` | Symmetric barriers. Asymmetry here is a directional prior; make it deliberately, not by accident. |
| `min_ret` | `0.0002` | Events whose target is below this are dropped as untradeable. Raising it discards more events; each one you keep must clear costs. |
| `vertical_barrier_times` | 45 minutes | The holding-period cap. This is what bounds label overlap, and therefore how much stage 4 has to purge. |

`triple_barrier_labels` then returns `{-1, 0, 1}` labels. Pass
`side_prediction` to `triple_barrier_events` instead and the regime
becomes `{0, 1}` — meta-labeling, where the label answers "should I take
the bet my primary model proposed?" rather than "which way does it go?".

### Stage 3 — turn probabilities into positions

`get_signal` maps a calibrated probability to a size in `[-1, 1]` by a
z-test against `1/num_classes`, so a probability barely above chance
produces a size near zero rather than a full-size bet.
`discrete_signal(&raw, 0.25)` then quantises to steps of 0.25. That is
not cosmetic: without it, a signal that drifts by 0.01 produces a trade,
and turnover — not gross return — is what a mid-frequency strategy
usually dies of.

### Stage 4 — purge and embargo, or your validation lies

This is the stage the rest of the library exists to make possible.

Triple-barrier labels **overlap in time**: a label starting at 10:00 with
a 45-minute vertical barrier is still open at 10:30. Under plain K-fold,
that 10:00 training label has already observed the outcome of a 10:30
test event. The model does not need to generalise; it can recall.

`PurgedKFold::new(n_splits, samples_info_sets, pct_embargo)` takes the
`(start, end)` interval of every label — that is what `samples_info_sets`
is — and drops from each training fold every label whose interval
overlaps the test fold. It then embargoes a further `pct_embargo`
fraction of samples after the test window, because serial correlation
leaks across an exact boundary even without overlap.

:::caution
`PurgedKFold::new` takes **three** arguments, and the middle one is the
interval list, not a number. The `PurgedKFold::new(5, 0.01)` form that
appears on some module pages does not compile. `pct_embargo` is the
third argument; `0.01` means 1% of the sample count.
:::

The stage 4 line is the point: `train 85 -> 82 rows (3 purged by
overlap)`. The purge removes real training rows. If your purged and
unpurged fold sizes come out identical, your labels do not overlap — and
with a vertical barrier set, that is worth checking rather than
celebrating.

### Stage 5 — separate strategy risk from portfolio risk

`sharpe_ratio(&returns, entries_per_year, risk_free_rate)` annualises by
`sqrt(entries_per_year)`. The program passes `252.0 * 390.0` because the
synthetic series is minute bars; passing the wrong figure here is the
easiest way to report a Sharpe that is off by an order of magnitude.

`RiskMetrics` covers the loss tail — `calculate_value_at_risk`,
`calculate_expected_shortfall`, `calculate_conditional_drawdown_risk` —
and answers "how bad is a bad day for this return stream".
`allocate_max_sharpe` answers a different question: how to weight several
assets. Do not read one as the other. A well-diversified allocation with
a broken strategy is still broken.

A Sharpe from a single backtest is also not evidence on its own. When you
report one, report how many configurations you tried to get it —
`backtest_statistics::deflated_sharpe_ratio` and
`probabilistic_sharpe_ratio` exist for that, and
`minimum_track_record_length` tells you how much data the claim needs.

## Where to go next

- [Python Core Workflow](/workflows/python-core-workflow/) — the same ground in Python
- [Modules by AFML chapter](/module-reference/by-afml-chapter/) — the module behind each stage
- [Module Reference Index](/modules/) — every module, by subject and by language surface
