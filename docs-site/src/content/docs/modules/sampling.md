---
title: "sampling"
description: "Label concurrency, average uniqueness and the sequential bootstrap, for training sets whose labels overlap in time."
status: authored
last_authored: '2026-11-07'
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
  - "seq_bootstrap_with_rng"
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
column per label. `bar_index` contains the bar positions to report, not merely the number of
rows: its values may be sparse or reordered, and the matrix follows their supplied order.
`get_ind_mat_label_uniqueness` returns $u_{t,i}$ **transposed**, with one row per label and
one column per input matrix row (`result[label][bar_row]`);
`get_ind_mat_average_uniqueness` the mean of $\bar u_i$ over labels that cover at least
one reported bar as a single number, and
`num_concurrent_events(n_bars, spans, _)` just $c_t$. The scalar mean excludes all-zero
label columns and returns 0 when none are covered. In contrast,
`get_av_uniqueness_from_triple_barrier` returns one value per input label, including 0 for
a label covering no bar. Check coverage separately when selecting sparse rows.

```python
from openquant import sampling

# Three bar rows, two label columns.
ind = sampling.get_ind_matrix([(0, 1), (1, 2)], [0, 1, 2])
assert ind == [[1, 0], [1, 1], [0, 1]]
# The uniqueness result is label-first: two label rows, three bar columns.
assert sampling.get_ind_mat_label_uniqueness(ind) == [
    [1.0, 0.5, 0.0],
    [0.0, 0.5, 1.0],
]
# Each row tests the actual listed position, here 4, then 2, then 8.
assert sampling.get_ind_matrix([(2, 4)], [4, 2, 8]) == [[1], [1], [0]]
assert sampling.get_ind_mat_average_uniqueness([[1, 0]]) == 1.0
assert sampling.get_ind_mat_average_uniqueness([[0, 0]]) == 0.0
```

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

# 40 labels of 10 bars each, a new one every 3 bars; interior labels overlap six others.
n, length, step = 40, 10, 3
spans = [(step * i, step * i + length - 1) for i in range(n)]
ind = sampling.get_ind_matrix(spans, list(range(step * n + length)))

def sample_uniqueness(drawn):
    return sampling.get_ind_mat_average_uniqueness([[row[c] for c in drawn] for row in ind])

rng, trials = random.Random(4), 200
standard = sum(sample_uniqueness([rng.randrange(n) for _ in range(n)]) for _ in range(trials)) / trials
sequential = sum(sample_uniqueness(sampling.seq_bootstrap(ind, random_state=s)) for s in range(trials)) / trials
print(f"all {n} labels, each once: {sample_uniqueness(range(n)):.3f}")
print(f"standard bootstrap:       {standard:.3f}")
print(f"sequential bootstrap:     {sequential:.3f}")
```

```text
all 40 labels, each once: 0.318
standard bootstrap:       0.301
sequential bootstrap:     0.312
```

Read the three numbers together. Sequential sampling recovers most of what a uniform
bootstrap loses to repeated and adjacent draws. Here every draw covers exactly 10 bars,
so a 40-draw sample has 400 label-bar memberships. Summed across active labels, uniqueness
is 1 at each covered bar. The union can cover at most 127 bars, making the sample
average at most $127/400 = 0.3175$, attained by taking every label once. This bound is
specific to these equal-length spans; it is not a universal bootstrap ceiling. AFML's Monte Carlo (§4.5.4) reports a larger gap —
median uniqueness of 0.6 against 0.7 — on random label sets with far less structural overlap.
If average uniqueness is low, the first remedy is upstream: a wider
[CUSUM threshold](/modules/filters/#choosing-h) or a shorter vertical barrier.

## Bootstrap only the training fold

`seq_bootstrap` returns positions in the indicator matrix's **label columns**. After
selecting a training fold, those are local positions, not original dataset row numbers.
Build the matrix from the training spans, map draws back through the training indices,
and apply the same repeated index sequence to features, labels and fitting weights:

```python
import numpy as np
from openquant import cross_validation as cv, sampling

spans = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12)]
starts, ends = np.array(spans).T
train, test = cv.purged_kfold_splits(starts, ends, 3, 0.0)[0]
train_spans = [spans[i] for i in train]
# Keep the original bar-position clock; each column now names one training label.
indicator = sampling.get_ind_matrix(train_spans, list(range(13)))
local_draws = np.array(sampling.seq_bootstrap(
    indicator, sample_length=5, random_state=7,
))
drawn_rows = train[local_draws]

# Synthetic row identifiers make alignment visible; no model is fitted here.
features = np.column_stack([np.arange(6), np.arange(6) + 10])
labels = np.array([0, 1, 0, 1, 0, 1])
train_weights = np.ones(len(train))  # illustrative unit fitting weights
sampled_x = features[drawn_rows]
sampled_y = labels[drawn_rows]
sampled_weights = train_weights[local_draws]
assert sampled_x[:, 0].tolist() == drawn_rows.tolist()
assert len(sampled_y) == len(sampled_weights) == 5
assert set(drawn_rows).issubset(set(train))
assert not set(drawn_rows).intersection(test)
print("training rows:", train.tolist(), "test rows:", test.tolist())
print("local draws:", local_draws.tolist())
print("original rows:", drawn_rows.tolist())
```

```text
training rows: [3, 4, 5] test rows: [0, 1]
local draws: [0, 1, 0, 2, 0]
original rows: [3, 4, 3, 5, 3]
```

Row 2 was purged because its closed span touches the first test fold. This particular
seed repeats row 3; repeated draws are valid and must remain repeated in the fitting
arrays. Do not turn the result into a set. Warm-up indices, when used, must also refer
to the local matrix columns.

The sequential draw probabilities depend on overlap and previously selected columns,
not the fitting weights supplied to a later learner. Return-attribution or decay weights
are a separate choice: [recompute them inside the training fold](/modules/sample-weights/)
and align them to `train` before indexing by `local_draws`. A repeated row contributes
its fitting weight on every draw; multiplying that row by its draw count as well would
count its multiplicity twice.

Purged k-fold can train on rows later than the test fold, as this example does. It is
not a walk-forward forecast. Choose a chronological training scheme when that is the
research question, and keep preprocessing and model selection inside its training
boundary. Sequential bootstrap changes sampling; it does not replace a valid split.

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

The matrix builders reject a span whose start is after its end. In contrast,
`num_concurrent_events` silently skips such a span, so zero concurrency is not proof that
the input spans were valid. Matrix consumers reject ragged rows; `bootstrap_loop_run`
rejects a concurrency vector of the wrong length, and a nonempty sequential-bootstrap
request rejects an out-of-range warm-up index. These errors are `InputError` in Rust and
`ValueError` in Python. Pass a binary 0/1 matrix and finite, non-negative concurrency;
shape checks do not validate all numerical contents.

## What to watch for

- **`seq_bootstrap` is reproducible only with a seed.** Without `random_state` it draws from
  the thread-local generator, so repeated calls differ; from Rust, `seq_bootstrap_with_rng`
  takes any generator. `warmup_samples` forces the first draws and is consumed from the
  *end* of the list; every later draw is uniqueness-weighted.
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
