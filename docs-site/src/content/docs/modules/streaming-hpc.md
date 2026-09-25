---
title: "streaming_hpc"
description: "VPIN and a venue-concentration HHI updated event by event in constant memory, with an alert when both cross their thresholds, and a synthetic flash crash to calibrate on."
status: authored
last_authored: '2026-09-24'
audience:
  - quant-dev
  - platform-engineering
module: "streaming_hpc"
api_surface: "both"
afml_chapter:
  - "22"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 22 (H. Simon and K. Wu): §22.6.4 The Flash Crash of 2010; §22.6.5 Volume-synchronized Probability of Informed Trading Calibration."
  - "Easley, D., López de Prado, M. and O'Hara, M. (2011). The microstructure of the 'Flash Crash': Flow toxicity, liquidity crashes and the probability of informed trading. Journal of Portfolio Management 37(2), 118–128."
  - "Easley, D., López de Prado, M. and O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. Review of Financial Studies 25(5), 1457–1493."
  - "Wu, K., Bethel, E. W., Gu, M., Leinweber, D. and Rübel, O. (2013). A big data approach to analyzing market volatility. Algorithmic Finance 2(3–4), 241–267."
  - "Hirschman, A. O. (1980). National Power and the Structure of Foreign Trade. University of California Press."
rust_api:
  - "StreamEvent"
  - "VpinConfig"
  - "VpinState"
  - "HhiConfig"
  - "HhiState"
  - "AlertThresholds"
  - "StreamingPipelineConfig"
  - "StreamingEarlyWarningEngine"
  - "EarlyWarningSnapshot"
  - "run_streaming_pipeline"
  - "StreamingRunReport"
  - "StreamingRunMetrics"
  - "run_streaming_pipeline_parallel"
  - "ParallelStreamingReport"
  - "StreamSummary"
  - "SyntheticStreamConfig"
  - "generate_synthetic_flash_crash_stream"
  - "StreamingHpcError"
python_api:
  - "streaming_hpc.run_streaming_pipeline"
  - "streaming_hpc.generate_synthetic_flash_crash_stream"
sidebar:
  badge: Module
---

An early warning is worth something only if it arrives before the event it warns of. AFML's
Chapter 22, written by Horst Simon and Kesheng Wu, tells how their group at Lawrence Berkeley
computed two indicators that had shown warning signs ahead of the Flash Crash of 6 May 2010 —
VPIN, the volume-synchronised probability of informed trading, and a variant of the
Herfindahl–Hirschman index of market fragmentation — over ten years of S&P 500 trades,
and made the computation fast enough to matter.

This module keeps both indicators as incremental state. Each event updates them in constant
time, memory stays bounded however long the stream runs, and an alert fires when both are
above their thresholds at once.

## The two indicators

**VPIN** measures how one-sided the order flow is, on a volume clock (Easley, López de Prado
and O'Hara, 2011, 2012). Trades are poured into buckets of a fixed volume $V$; an event that
overflows a bucket is split between it and the next, keeping its buy/sell ratio. For bucket
$\tau$ with buy volume $V^B_\tau$ and sell volume $V^S_\tau$,

$$
\mathrm{VPIN} = \frac{1}{n}\sum_{\tau=1}^{n}\frac{\lvert V^B_\tau - V^S_\tau\rvert}{V}
$$

over the last $n$ = `support_buckets` full buckets. It is 0 when every bucket is balanced and
1 when every bucket is all buying or all selling. Because buckets are equal-volume, a burst of
trading fills buckets faster and VPIN reacts in fewer events, which is the point of the
volume clock.

**HHI** measures how concentrated trading is across venues (Hirschman, 1980). With $n_v$ the
number of the last $L$ = `lookback_events` events that happened on venue $v$,

$$
\mathrm{HHI} = \sum_v \left(\frac{n_v}{L}\right)^2 .
$$

It is $1/K$ when $K$ venues share the flow evenly and 1 when one venue has all of it.

**The alert** fires when $\mathrm{VPIN}\ge\tau_V$ *and* $\mathrm{HHI}\ge\tau_H$.
`normalized_risk_score` is $\tfrac12(\mathrm{VPIN}/\tau_V + \mathrm{HHI}/\tau_H)$, a single
number for a dashboard. Both are `None` until VPIN has $n$ full buckets and HHI has $L$ events.

## A synthetic flash crash

`generate_synthetic_flash_crash_stream` produces a deterministic stream for calibration.
Before the crash, events rotate over `calm_venues` venues with 120 bought and 130 sold, and the
price drifts up by 0.01% per event. From the crash on, every event is on `shock_venue` with 80
bought and 320 sold, and the price falls 0.25% per event.

```python
from openquant import streaming_hpc

# 1,000 events, 1 ms apart. Calm flow rotates over four venues; from event 700 all
# flow is on venue 0 and four-fifths of it is selling.
events = streaming_hpc.generate_synthetic_flash_crash_stream(
    events=1000, crash_start_fraction=0.7, calm_venues=4, shock_venue=0
)
report = streaming_hpc.run_streaming_pipeline(
    events,
    bucket_volume=1000.0,  # VPIN bucket size, in units of volume
    support_buckets=10,  # VPIN averages the last 10 full buckets
    lookback_events=50,  # HHI counts venues over the last 50 events
    vpin_threshold=0.3,
    hhi_threshold=0.5,
)

def show(x):
    return "   none" if x is None else f"{x:7.3f}"

print("event    price     vpin      hhi    score  alert")
for i in (0, 39, 49, 699, 705, 715, 725, 729, 750, 999):
    ts, price, vpin, hhi, score, alert = report["snapshots"][i]
    print(f"{i:5d}  {price:7.2f}  {show(vpin)}  {show(hhi)}  {show(score)}  {alert}")

first = next(i for i, s in enumerate(report["snapshots"]) if s[5])
print("first alert at event", first, "- the crash began at event 700")
print("alerts:", report["alert_count"])
```

```text
event    price     vpin      hhi    score  alert
    0   100.01     none     none     none  False
   39   100.40    0.040     none     none  False
   49   100.50    0.040    0.250    0.317  False
  699   107.25    0.040    0.250    0.317  False
  705   105.65    0.152    0.261    0.514  False
  715   103.04    0.376    0.321    0.947  False
  725   100.49    0.600    0.453    1.453  False
  729    99.49    0.600    0.520    1.520  True
  750    94.40    0.600    1.000    2.000  True
  999    50.61    0.600    1.000    2.000  True
first alert at event 729 - the crash began at event 700
alerts: 271
```

The calm stream has VPIN $|120-130|/250 = 0.04$ and HHI $0.25$, one quarter per venue. VPIN
needs 10,000 units of volume, 40 events, before it reports; HHI needs 50 events. Once the
crash starts, VPIN crosses 0.3 at event 712, because crash events carry more
volume and fill buckets faster, and settles at $|80-320|/400 = 0.6$. HHI is slower: it takes
until event 729 for venue 0 to hold enough of the 50-event window, and the alert waits for
it. At event 725 the score is 1.45 and there is still no alert.

<figure>
<img class="dark:sl-hidden" src="/figures/ch22-early-warning-light.svg" alt="VPIN and HHI from event 650 to 800 of the synthetic stream. Both are flat before the crash at event 700, VPIN at 0.04 and HHI at 0.25. After it, VPIN climbs steeply to 0.6 by event 724 and HHI climbs more slowly to 1.0 by event 749. The alert region begins at event 729, where HHI crosses its threshold of 0.5; VPIN crossed its threshold of 0.3 at event 712, 17 events earlier." />
<img class="light:sl-hidden" src="/figures/ch22-early-warning-dark.svg" alt="VPIN and HHI from event 650 to 800 of the synthetic stream. Both are flat before the crash at event 700, VPIN at 0.04 and HHI at 0.25. After it, VPIN climbs steeply to 0.6 by event 724 and HHI climbs more slowly to 1.0 by event 749. The alert region begins at event 729, where HHI crosses its threshold of 0.5; VPIN crossed its threshold of 0.3 at event 712, 17 events earlier." />
<figcaption>The example's two indicators around the crash, with their thresholds. The alert needs both, so it fires when the slower one, HHI, crosses.</figcaption>
</figure>

## From Rust

`StreamingEarlyWarningEngine::on_event` is the incremental interface; `run_streaming_pipeline`
loops it over a slice and times each call. `run_streaming_pipeline_parallel` runs many
independent streams on [`hpc_parallel`](/modules/hpc-parallel/), one stream per atom, and
keeps only a summary of each.

```rust
use openquant::hpc_parallel::{ExecutionMode, HpcParallelConfig, PartitionStrategy};
use openquant::streaming_hpc::{
    generate_synthetic_flash_crash_stream, run_streaming_pipeline_parallel, AlertThresholds,
    HhiConfig, StreamEvent, StreamingEarlyWarningEngine, StreamingHpcError,
    StreamingPipelineConfig, SyntheticStreamConfig, VpinConfig,
};

let cfg = StreamingPipelineConfig {
    vpin: VpinConfig { bucket_volume: 1_000.0, support_buckets: 10 },
    hhi: HhiConfig { lookback_events: 50 },
    thresholds: AlertThresholds { vpin: 0.3, hhi: 0.5 },
};
let stream_with_crash_at = |fraction: f64| {
    generate_synthetic_flash_crash_stream(SyntheticStreamConfig {
        events: 1_000,
        crash_start_fraction: fraction,
        calm_venues: 4,
        shock_venue: 0,
    })
};

// Event by event: the engine holds only its two rolling windows.
let stream = stream_with_crash_at(0.7)?;
let mut engine = StreamingEarlyWarningEngine::new(cfg)?;
let mut first_alert = None;
for (i, event) in stream.iter().enumerate() {
    if engine.on_event(*event)?.is_alert && first_alert.is_none() {
        first_alert = Some(i);
    }
}
assert_eq!(first_alert, Some(729));

// A bad tick is rejected before it touches the state.
let bad = StreamEvent { price: f64::NAN, ..stream[0] };
assert!(matches!(engine.on_event(bad), Err(StreamingHpcError::InvalidEvent(_))));

// Eight streams, crashing at events 100, 200, ..., 800, on four threads.
let streams =
    (1..=8).map(|k| stream_with_crash_at(k as f64 / 10.0)).collect::<Result<Vec<_>, _>>()?;
let parallel = HpcParallelConfig {
    mode: ExecutionMode::Threaded { num_threads: 4 },
    partition: PartitionStrategy::Linear,
    mp_batches: 1,
    progress_every: 1,
};
let report = run_streaming_pipeline_parallel(&streams, cfg, parallel)?;
// One summary per stream, in input order. Each alerts from 29 events after its crash to the end.
let alerts: Vec<usize> = report.stream_summaries.iter().map(|s| s.alert_count).collect();
assert_eq!(alerts, [871, 771, 671, 571, 471, 371, 271, 171]);
assert_eq!(report.stream_summaries[0].latest_hhi, Some(1.0));
```

## What to watch for

- **The thresholds are on raw VPIN, not on its distribution.** The chapter's calibrated
  setting alerts when VPIN's *cumulative distribution function* passes 0.99, which adapts to
  each instrument's normal level of toxicity. Here the threshold is an absolute number, so a
  value that suits one contract is wrong for another. Calibrate per instrument, or feed
  quantiles in yourself ([#113](https://github.com/Open-Quant/openquant/issues/113)).
- **Buy and sell volume are inputs.** VPIN needs every trade signed, and the chapter's
  best parameters use bulk volume classification over bars. This module classifies nothing;
  it trusts `buy_volume` and `sell_volume` as given. See
  [`microstructural_features`](/modules/microstructural-features/) for bulk volume
  classification (`get_bvc_buy_volume`) and a bar-based VPIN over a finished history.
- **HHI counts events, not volume.** A venue that prints many small trades dominates the index
  even if a quiet venue carries most of the volume. The chapter describes its HHI only as a
  variant of the fragmentation index and does not say which weighting it uses.
- **The two windows run on different clocks.** VPIN's window is `support_buckets × bucket_volume`
  units of volume; HHI's is `lookback_events` events. When volume per event changes, as in
  the crash above, one reacts faster than the other, and the AND rule makes the alert as slow
  as the slower.
- **A score above 1 is not an alert.** The score averages the two ratios, so one indicator far
  above its threshold can lift it past 1 while the other is still below. The example's event
  725 has a score of 1.45 and no alert.
- **The latency metrics time the engine, not your pipeline.** `avg_event_latency_micros` and
  `max_event_latency_micros` are wall-clock times of `on_event` calls, which take around a
  microsecond each; they vary from run to run and exclude parsing, queueing and I/O.
- **The parallel runner returns summaries only.** `run_streaming_pipeline_parallel` keeps the
  alert count and the last snapshot of each stream; to see the path of each stream, call
  `run_streaming_pipeline` per stream. Parallelism is across streams; a single stream is
  sequential by nature. It is Rust-only.
- **The synthetic stream is a switch, not a model.** Two fixed regimes and a hard change
  between them make it good for checking wiring and reaction time, and useless for estimating
  false-positive rates. The chapter's point about calibration — 20% false positives cut to 7%
  over a hundred futures contracts by choosing parameters on the whole universe — needs real
  data. If `shock_venue` is one of the calm venues, as here, HHI starts from that venue's
  existing share and crosses sooner than for a new venue.

## Related modules

- [`microstructural-features`](/modules/microstructural-features/) — VPIN and other flow
  features over bars, and trade classification.
- [`hpc-parallel`](/modules/hpc-parallel/) — the engine behind
  `run_streaming_pipeline_parallel`.
- [`structural-breaks`](/modules/structural-breaks/) — tests for regime change in a price
  series, the other family of early-warning statistics in the book.
