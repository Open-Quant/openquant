---
title: Rust Core Workflow
description: One runnable Rust program from event sampling to portfolio risk, with its output.
status: authored
last_authored: '2026-11-07'
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

This synthetic program connects event sampling, labeling, position sizing, purged splits
and risk APIs. It demonstrates their data contracts; it does not fit a predictive model,
calibrate probabilities or produce an out-of-sample backtest. The portfolio allocation is
a separate in-sample calculation, not the portfolio used to earn the event outcomes.

Save the complete program as `crates/openquant/examples/afml_workflow.rs` and run from
the repository root:

```bash
cargo run -p openquant --example afml_workflow
```

That example file is not preinstalled. Add `--release` for an optimized build. The first
build compiles the crate and its dependencies. The output below was checked by compiling
and executing this full program against the current Rust crate; it uses no random draws,
but floating-point details can vary between platforms.

## The program

```rust
//! Synthetic API demonstration: sampling -> labeling -> sizing -> splits -> diagnostics.
use std::collections::BTreeMap;
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

/// Synthetic minute bars without random draws. Floating-point results can vary
/// slightly with platform/compiler math implementations.
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
    // Sample after a CUSUM accumulator crosses 0.001 log-return units.
    // This does not establish equal information content across events.
    let event_idx = cusum_filter_indices(&close, Threshold::Scalar(0.001)).expect("CUSUM");
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
    // A missing vertical barrier means no time cap, not an earlier end-of-sample cap.
    // Keep only events for which the full 45-minute horizon is available.
    let eligible: Vec<NaiveDateTime> = vbars.iter().map(|(start, _)| *start).collect();
    let cfg = TripleBarrierConfig {
        pt: 1.0,                              // profit-take barrier = 1.0 x target vol
        sl: 1.0,                              // stop-loss barrier  = 1.0 x target vol
        min_ret: 0.0002,                      // require target > this value; not a cost test
        vertical_barrier_times: Some(&vbars), // 45-minute holding-period cap
    };
    let events = triple_barrier_events(&bars, &eligible, &target, cfg, None);
    let labeled = triple_barrier_labels(&events, &bars);
    let wins = labeled.iter().filter(|l| l.label > 0).count();
    println!(
        "stage 2  {:>3} labeled events, {:>3} positive ({:.0}%)",
        labeled.len(),
        wins,
        100.0 * wins as f64 / labeled.len().max(1) as f64
    );

    // Artificial probability-shaped inputs: no model is fitted or calibrated.
    // They exist only to demonstrate the normal-CDF sizing map and quantization.
    // A research evaluation would generate predictions from each training fold.
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
    // Construct split intervals in exactly the same row order as the labels/sizes.
    // Labeling can omit unresolved events; do not assume raw event rows still align.
    // This demo inspects splits only: it does not fit or score a model on them.
    let event_ends: BTreeMap<NaiveDateTime, NaiveDateTime> = events
        .iter()
        .filter_map(|(start, event)| event.t1.map(|end| (*start, end)))
        .collect();
    let info_sets: Vec<(NaiveDateTime, NaiveDateTime)> = labeled
        .iter()
        .map(|label| (label.timestamp, event_ends[&label.timestamp]))
        .collect();
    assert_eq!(info_sets.len(), sized.len());
    assert!(info_sets.windows(2).all(|w| w[0].0 < w[1].0));
    let cv = PurgedKFold::new(5, info_sets.clone(), 0.01).expect("purged kfold");
    let splits = cv.split_with_diagnostics(info_sets.len()).expect("split");
    let first = &splits[0];
    let naive_train = info_sets.len() - first.test_indices.len();
    println!(
        "stage 4  5 folds, embargo 1%: train {} -> {} rows ({} purged, {} embargoed)",
        naive_train,
        first.train_indices.len(),
        first.diagnostics.purged_indices.len(),
        first.diagnostics.embargo_indices.len()
    );

    // ---- Stage 5: descriptive event outcomes and a separate allocation -------
    let event_outcomes: Vec<f64> =
        labeled.iter().zip(sized.iter()).map(|(l, s)| l.ret * s).collect();
    let sr = sharpe_ratio(&event_outcomes, 1.0, 0.0);
    let rm = RiskMetrics;
    let var = rm.calculate_value_at_risk(&event_outcomes, 0.05).expect("var");
    let es = rm.calculate_expected_shortfall(&event_outcomes, 0.05).expect("es");
    println!("stage 5  per-event mean/std {:+.3}   VaR(5%) {:+.5}   ES(5%) {:+.5}", sr, var, es);

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
        "stage 5  weights {:?}  allocation SR(252 rows/year) {:.3}",
        alloc.weights.iter().map(|w| (w * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        alloc.portfolio_sharpe
    );
}
```

## Output

```text
stage 1   480 bars -> 147 CUSUM events
stage 2   97 labeled events,  50 positive (52%)
stage 3  sizes in [-0.25, 0.25], 3 distinct steps
stage 4  5 folds, embargo 1%: train 77 -> 76 rows (0 purged, 1 embargoed)
stage 5  per-event mean/std +0.088   VaR(5%) -0.00015   ES(5%) -0.00016
stage 5  weights [0.21, 0.451, -0.0, 0.339]  allocation SR(252 rows/year) 1.744
```

## What the output means

The stage 5 event statistic uses `sharpe_ratio(..., 1.0, 0.0)`: mean divided by sample
standard deviation, with no annualization or risk-free adjustment. Each observation is
one label's return multiplied by its artificial size. These returns have different holding
times and can overlap. They are not one-minute account returns, so multiplying their ratio
by `sqrt(252 * 390)` would give the wrong clock. There is no account-level capital allocation,
concurrency aggregation, turnover or transaction-cost model here.

The separate allocator estimates moments from the entire synthetic price panel and uses
its built-in convention of 252 rows per year. Its printed Sharpe retains that convention
even though these rows are minute bars. It is an in-sample objective, not comparable to the
unannualized event statistic or an out-of-sample performance estimate.

## Stage 1 — sample events

CUSUM accumulates log returns separately in positive and negative accumulators. It fires
when either strictly crosses its threshold and resets only that side. A threshold of
`0.001` is in log-return units; it is not a guarantee that each event contains equal
information. A dynamic threshold must be aligned with the price rows and computed from
information available by that bar.

## Stage 2 — label with volatility-scaled barriers

The example uses the population deviation of up to 20 trailing simple returns, floored
at `1e-4`, as its target. Targets must be strictly above `min_ret=0.0002` to survive.
This is a target-size filter, not evidence that a trade clears costs.

Profit-taking and stop-loss widths are each one target unit. The target is a per-bar
volatility proxy, not a fitted forecast of the return over 45 minutes. The program first
keeps events with an available full 45-minute vertical barrier; otherwise
`add_vertical_barrier` omits the pair and the event API would have no time cap for that
start. `triple_barrier_labels` can omit unresolved or unaligned events, so later stages
join interval endpoints by each retained label's timestamp.

Without a side, labels are the sign of the realized return (`-1`, `0`, `1`); zero means
exactly zero return, not necessarily a vertical-barrier exit. With a supplied side,
labels become `0` or `1` according to whether the side-adjusted return is positive.

## Stage 3 — demonstrate sizing

The sinusoidal values between 0.32 and 0.68 are probability-shaped placeholders, not
calibrated predictions. `get_signal` transforms them through its normal-CDF sizing
formula relative to `1 / num_classes`; `discrete_signal(..., 0.25)` rounds to size steps.
Quantization can reduce position changes, but this program neither estimates calibration
nor measures trading costs. The placeholder rows follow the retained labels only to
illustrate the API alignment.

## Stage 4 — inspect purged splits

`PurgedKFold::new(5, info_sets, 0.01)` receives a `(start, end)` interval for every row,
in increasing start-time order. The intervals, labels and sizes here have the same row
identity. In research, features, predictions and sample weights must use that same order.
A shared timestamp is not permission to use future observations in a feature.

Purging removes training labels that overlap the test interval; the embargo then excludes up to
`ceil(0.01 * n)` available sample positions after the test labels resolve. The program
prints the diagnostic exclusion lists separately. Those lists can overlap, so their
lengths should not be added to infer the number of removed rows. This first fold has
zero purged rows and one embargoed row; it does not demonstrate overlap removal.
Equal training counts in two split schemes do not by themselves prove either leakage
or independence.

The code constructs splits and prints their sizes only. It never trains or evaluates a
model within them. A real evaluation must fit preprocessing and the model on each training
fold and use only held-out predictions, while accounting for feature lookbacks as well as
label intervals. Purging and embargo do not certify the entire workflow as leakage-free.

## Stage 5 — keep the return clocks explicit

`RiskMetrics` receives the same per-event outcomes. VaR is their lower-tail quantile;
ES here averages observations strictly below that quantile and preserves the return sign.
They are descriptive event-tail estimates, not daily account risk. Ties can change the
included tail fraction, and an empty strict tail returns `NaN`.

For an account-level backtest, first build a time-indexed position and return path with
explicit capital and costs. Annualize according to that path's observation interval;
`backtest_statistics::sharpe_ratio` takes a **per-observation** risk-free rate, whereas
`allocate_max_sharpe` takes an annual rate under its own 252-row convention. Zero is used
for both in this example. Check variance and finite outputs before interpreting ratios.

Report selection history and appropriate uncertainty alongside any later performance
estimate. PSR, deflated Sharpe and minimum-track-record helpers depend on their own
sampling and moment assumptions; they do not turn this synthetic API demonstration into
research evidence.

## Where to go next

- [Python Core Workflow](/workflows/python-core-workflow/) — wrapper conventions and research steps
- [Cross-validation](/modules/cross-validation/) — purging, embargo and ordering requirements
- [Risk metrics](/modules/risk-metrics/) — tail estimator conventions
- [Modules by AFML chapter](/module-reference/by-afml-chapter/) — the module behind each stage
