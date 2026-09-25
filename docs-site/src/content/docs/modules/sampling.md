---
title: "sampling"
description: "Label concurrency, average uniqueness and the sequential bootstrap, for training sets whose labels overlap in time."
status: authored
last_authored: '2026-09-25'
audience:
  - quant-dev
  - platform-engineering
module: "sampling"
api_surface: "both"
afml_chapter:
  - "4"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 4: §4.2 Overlapping Outcomes; §4.3 Number of Concurrent Labels (Snippet 4.1); §4.4 Average Uniqueness of a Label (Snippet 4.2); §4.5 Bagging Classifiers and Uniqueness; §4.5.2 Implementation of Sequential Bootstrap (Snippets 4.3–4.5); §4.5.3 A Numerical Example (Snippet 4.6); §4.5.4 Monte Carlo Experiments (Snippets 4.7–4.9)."
rust_api:
  - "get_ind_matrix"
  - "get_ind_mat_average_uniqueness"
  - "get_ind_mat_label_uniqueness"
  - "get_av_uniqueness_from_triple_barrier"
  - "num_concurrent_events"
  - "bootstrap_loop_run"
  - "seq_bootstrap"
python_api:
  - "sampling.get_ind_matrix"
  - "sampling.get_ind_mat_average_uniqueness"
  - "sampling.get_ind_mat_label_uniqueness"
  - "sampling.get_av_uniqueness_from_triple_barrier"
  - "sampling.num_concurrent_events"
  - "sampling.bootstrap_loop_run"
  - "sampling.seq_bootstrap"
sidebar:
  badge: Module
---

A triple-barrier label is a statement about a stretch of prices, from the event bar to the
bar where a barrier was touched. Two labels whose stretches overlap are statements about some
of the same returns, so they are not independent observations — and most of machine learning,
from the bootstrap inside a random forest to the standard error of a cross-validation score,
assumes they are (AFML §4.2). This module measures the overlap and provides a bootstrap that
draws around it. [`sample-weights`](/modules/sample-weights/) turns the same measurement into
weights.

Everything here works in **bar positions**, not timestamps: a label is a pair
`(start, end)` of inclusive indices into the bar series.

## Concurrency and uniqueness

Let $\mathbb{1}_{t,i}$ be 1 when label $i$ spans bar $t$. The number of labels alive at bar
$t$ is $c_t=\sum_i \mathbb{1}_{t,i}$, and the uniqueness of label $i$ at that bar is
$u_{t,i}=\mathbb{1}_{t,i}/c_t$: a label that shares a bar with one other owns half of it. The
**average uniqueness** of a label is that quantity averaged over its own lifespan,

$$
\bar u_i \;=\; \frac{\sum_{t} u_{t,i}}{\sum_{t} \mathbb{1}_{t,i}}
$$

which is 1 for a label that overlaps nothing and falls toward $1/k$ for a label that shares
every bar with $k-1$ others.

The example is the one AFML works by hand in §4.5.3: three labels over six bars.

```python
from openquant import sampling

spans = [(0, 2), (2, 3), (4, 5)]  # (first bar, last bar) of each label, inclusive
ind = sampling.get_ind_matrix(spans, list(range(6)))
for bar, row in enumerate(ind):
    print(f"bar {bar}: {row}  concurrent = {sum(row)}")

print("average uniqueness per label:",
      [round(u, 3) for u in sampling.get_av_uniqueness_from_triple_barrier(spans, 6)])

# Sequential bootstrap, second draw, given that label 1 was drawn first.
concurrency = [float(row[1]) for row in ind]
u = sampling.bootstrap_loop_run(ind, concurrency)
print("after drawing label 1:", [f"{v / sum(u):.3f}" for v in u])
```

```text
bar 0: [1, 0, 0]  concurrent = 1
bar 1: [1, 0, 0]  concurrent = 1
bar 2: [1, 1, 0]  concurrent = 2
bar 3: [0, 1, 0]  concurrent = 1
bar 4: [0, 0, 1]  concurrent = 1
bar 5: [0, 0, 1]  concurrent = 1
average uniqueness per label: [0.833, 0.75, 1.0]
after drawing label 1: ['0.357', '0.214', '0.429']
```

Labels 0 and 1 meet at bar 2. Label 0 owns two of its three bars outright and half of the
third, $(1+1+\tfrac12)/3=0.833$; label 1 owns one and a half of two, $0.75$. The last line is
the book's $\{5/14,\,3/14,\,6/14\}$.

<figure>
<img class="dark:sl-hidden" src="/figures/ch4-concurrency-light.svg" alt="Three labels drawn as horizontal spans over six bars. Label 0 covers bars 0 to 2, label 1 covers bars 2 and 3, label 2 covers bars 4 and 5. Beneath, the count of concurrent labels is 1 at every bar except bar 2, where it is 2. Average uniqueness is 0.833, 0.750 and 1.000." />
<img class="light:sl-hidden" src="/figures/ch4-concurrency-dark.svg" alt="Three labels drawn as horizontal spans over six bars. Label 0 covers bars 0 to 2, label 1 covers bars 2 and 3, label 2 covers bars 4 and 5. Beneath, the count of concurrent labels is 1 at every bar except bar 2, where it is 2. Average uniqueness is 0.833, 0.750 and 1.000." />
<figcaption>The indicator matrix, drawn. One shared bar is enough to cost both labels some uniqueness.</figcaption>
</figure>

`get_ind_matrix(label_endtime, bar_index)` returns the matrix itself, one row per bar and one
column per label. `get_ind_mat_label_uniqueness` returns $u_{t,i}$ for every bar and label,
`get_ind_mat_average_uniqueness` the mean of $\bar u_i$ over labels as a single number, and
`num_concurrent_events(n_bars, spans, _)` just $c_t$.

## The sequential bootstrap

A standard bootstrap draws $I$ labels uniformly with replacement. When labels overlap, that
sample is more redundant than it looks: the in-bag observations resemble each other, and the
out-of-bag ones resemble the in-bag ones, so bagged trees are more correlated and out-of-bag
accuracy is inflated (§4.5).

The sequential bootstrap (§4.5.1–2) draws one label at a time and, before each draw, lowers
the probability of every label that overlaps what has already been drawn. With $\varphi$ the
draws so far, the probability of drawing $j$ next is proportional to the average uniqueness
$j$ *would* have if it were added:

$$
\bar u_j^{(\varphi)} \;=\; \frac{1}{\sum_t \mathbb{1}_{t,j}}
\sum_{t}\frac{\mathbb{1}_{t,j}}{1+\sum_{k\in\varphi}\mathbb{1}_{t,k}},
\qquad
\delta_j \;=\; \frac{\bar u_j^{(\varphi)}}{\sum_k \bar u_k^{(\varphi)}}
$$

`bootstrap_loop_run(ind_mat, concurrency)` computes the $\bar u_j^{(\varphi)}$ for one step,
given the per-bar concurrency of the draws so far; `seq_bootstrap(ind_mat, sample_length,
warmup_samples)` runs the whole loop and returns the drawn label indices. In the example
above, having drawn label 1, the label that does not touch it becomes twice as likely as
drawing label 1 again — but no probability goes to zero. Repeats remain possible, only less
likely.

How much does it buy? The measure is the average uniqueness *of the drawn sample*, computed
on the indicator matrix restricted to the drawn columns, repeats included.

```python
import random

from openquant import sampling

# 40 labels of 10 bars each, a new one every 3 bars: every label overlaps six others.
n, length, step = 40, 10, 3
spans = [(step * i, step * i + length - 1) for i in range(n)]
ind = sampling.get_ind_matrix(spans, list(range(step * n + length)))

def sample_uniqueness(drawn):
    return sampling.get_ind_mat_average_uniqueness([[row[c] for c in drawn] for row in ind])

rng, trials = random.Random(4), 200
standard = sum(sample_uniqueness([rng.randrange(n) for _ in range(n)]) for _ in range(trials)) / trials
sequential = sum(sample_uniqueness(sampling.seq_bootstrap(ind)) for _ in range(trials)) / trials
print(f"all {n} labels, each once: {sample_uniqueness(range(n)):.3f}")
print(f"standard bootstrap:       {standard:.3f}")
print(f"sequential is higher:     {sequential > standard}")
```

```text
all 40 labels, each once: 0.318
standard bootstrap:       0.301
sequential is higher:     True
```

The sequential figure is not printed because `seq_bootstrap` cannot be seeded (below); over
repeated runs of this script it is about 0.312. Read the three numbers together. Sequential
sampling recovers most of what a uniform bootstrap loses to repeated and adjacent draws, but
it cannot exceed what the label set allows: with every label overlapping six others, no
sample of 40 is much more than 30% unique. AFML's Monte Carlo (§4.5.4) reports a larger gap —
median uniqueness of 0.6 against 0.7 — on random label sets with far less structural overlap.
If average uniqueness is low, the first remedy is upstream: a wider
[CUSUM threshold](/modules/filters/#choosing-h) or a shorter vertical barrier.

## From Rust

```rust
use openquant::sampling::{
    bootstrap_loop_run, get_av_uniqueness_from_triple_barrier, get_ind_matrix, seq_bootstrap,
};

let spans = vec![(0, 2), (2, 3), (4, 5)];
let bars: Vec<usize> = (0..6).collect();
let ind = get_ind_matrix(&spans, &bars)?;
assert_eq!(ind[2], vec![1, 1, 0]);

let uniqueness = get_av_uniqueness_from_triple_barrier(&spans, bars.len())?;
assert!((uniqueness[0] - 5.0 / 6.0).abs() < 1e-12);

// Second-draw probabilities after label 1, as in AFML §4.5.3.
let concurrency: Vec<f64> = ind.iter().map(|row| f64::from(row[1])).collect();
let u = bootstrap_loop_run(&ind, &concurrency)?;
let total: f64 = u.iter().sum();
assert!((u[0] / total - 5.0 / 14.0).abs() < 1e-12);
assert!((u[2] / total - 6.0 / 14.0).abs() < 1e-12);

// One label index per draw; `None` draws as many as there are labels.
let drawn = seq_bootstrap(&ind, None, None)?;
assert_eq!(drawn.len(), 3);
assert!(drawn.iter().all(|&label| label < 3));
```

A span whose start is after its end, a ragged matrix, a concurrency vector of the wrong
length and a warm-up index beyond the last label are each an `InputError`.

## What to watch for

- **`seq_bootstrap` is not reproducible.** It draws from the thread-local generator and takes
  no seed. `warmup_samples` forces the first draws (it is consumed from the *end* of the
  list), which is how the tests pin it; it is not a seed. Tracked in
  [#90](https://github.com/Open-Quant/openquant/issues/90).
- **The matrix is dense.** `get_ind_matrix` allocates bars × labels bytes and
  `seq_bootstrap` rescans all of it for every draw, so a full-length sample costs on the
  order of bars × labels² operations. Ten thousand labels over a hundred thousand bars is a
  gigabyte and will not finish. Bootstrap within blocks, or draw `sample_length` well below
  the label count.
- **Spans are inclusive bar positions.** Convert event and barrier-touch timestamps to
  positions in the *same* bar series before calling. An `end` beyond the last bar is
  not an error: both `num_concurrent_events` and `get_ind_matrix` truncate the span silently.
- **`num_concurrent_events` ignores its third argument.** It is kept for signature
  compatibility with Snippet 4.1; pass an empty list.
- **From Python, the matrix is a list of lists of 0/1 ints**, one inner list per bar. Each
  cell is an eight-byte pointer where Rust stores one byte, so the density warning above bites
  sooner from Python.
- **Uniqueness is not a weight yet.** A sample weight also reflects how much happened during
  the label; see [`sample-weights`](/modules/sample-weights/).

## Related modules

- [`labeling`](/modules/labeling/) — produces the event and barrier-touch times that become
  spans.
- [`sample-weights`](/modules/sample-weights/) — weights from the same concurrency counts.
- [`sb-bagging`](/modules/sb-bagging/) — a bagging ensemble built on `seq_bootstrap`.
- [`cross-validation`](/modules/cross-validation/) — purging and embargo, which deal with
  the same overlap across the train/test boundary.
