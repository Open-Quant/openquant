---
title: "sampling"
description: "Indicator matrix and sequential bootstrap tooling."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "sampling"
api_surface: "both"
afml_chapters:
  - 4
risk_notes:
  - "Indicator matrix quality drives bootstrap quality."
  - "Use average uniqueness as a diagnostics KPI."
rust_api:
  - "get_ind_matrix"
  - "get_ind_mat_average_uniqueness"
  - "get_ind_mat_label_uniqueness"
  - "bootstrap_loop_run"
  - "seq_bootstrap"
  - "get_av_uniqueness_from_triple_barrier"
  - "num_concurrent_events"
sidebar:
  badge: Module
---

## Concept Overview

Standard bootstrap assumes IID observations: draw N samples with replacement uniformly. But AFML labels overlap in time — event A might span bars 1-5 while event B spans bars 3-8. Drawing both A and B into the same bootstrap sample introduces information leakage between train/test, because they share bars 3-5.

The **sequential bootstrap** (AFML Chapter 4) fixes this by making draws overlap-aware. It builds an **indicator matrix** that maps which bars each label spans. At each draw step, it computes the average uniqueness of each remaining label *given what's already been drawn*, then samples proportionally to uniqueness. Labels that would create heavy overlap with already-drawn samples have low uniqueness and are unlikely to be selected.

The result is a bootstrap sample where the drawn labels are as independent as possible given the underlying overlap structure. This is critical for bagging classifiers trained on financial labels, where naive bootstrap produces ensembles with highly correlated base learners.

**Average uniqueness** is the key diagnostic: it tells you what fraction of each label's information is non-redundant. Low average uniqueness (< 0.5) means heavy overlap and sequential bootstrap becomes essential.

## When to Use

Use sequential bootstrap whenever you're bagging or bootstrapping with overlapping labels. It replaces standard `np.random.choice` in any ensemble or bootstrap workflow.

**Prerequisites**: An indicator matrix from event start/end times, and optionally the concurrent event count per bar.

**Alternatives**: Standard IID bootstrap (fast but leakage-prone), or sample weighting (correct expected value but doesn't reduce sample correlation).

## Mathematical Foundations

### Average Uniqueness

$$u_i=\frac{1}{|T_i|}\sum_{t\in T_i}\frac{1}{c_t}$$

### Sequential Draw Prob

$$P(i)\propto E[u_i \mid \mathcal{S}]$$

## Key Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `ind_matrix` | `Vec<Vec<i32>>` | Indicator matrix: rows=bars, cols=labels. Entry is 1 if bar i is active during label j | — |
| `n_samples` | `Option<usize>` | Number of bootstrap draws; defaults to number of labels | None (= n_labels) |

## Usage Examples

### Python

#### Sequential bootstrap with overlap-aware sampling

```python
from openquant._core import sampling

# Indicator matrix: rows=bars, cols=labels
# 1 means bar i is active for label j
ind_matrix = [
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [0, 1, 0],
    [1, 0, 0],
]

# Average uniqueness per label (diagnostic)
avg_u = sampling.get_ind_mat_average_uniqueness(ind_matrix)
# e.g., [0.72, 0.58, 0.44] — label 0 is most unique

# Sequential bootstrap: draw n samples favoring unique labels
drawn_indices = sampling.seq_bootstrap(ind_matrix, n_samples=3)
# Returns label indices selected with overlap-aware probabilities
```

### Rust

#### Run sequential bootstrap

```rust
use openquant::sampling::seq_bootstrap;

let ind = vec![vec![1,0,1], vec![0,1,1], vec![1,1,0]];
let idx = seq_bootstrap(&ind, Some(3), None);
```

## Common Pitfalls

- Building the indicator matrix with wrong event boundaries — off-by-one errors silently break uniqueness calculations.
- Using sequential bootstrap with very short labels that don't overlap — it degenerates to standard bootstrap and just adds overhead.
- Forgetting to pass sequential bootstrap indices to the bagging estimator — the sampling module produces indices, your estimator must use them.

## API Reference

### Python API

- `sampling.get_ind_matrix`
- `sampling.get_ind_mat_average_uniqueness`
- `sampling.get_ind_mat_label_uniqueness`
- `sampling.bootstrap_loop_run`
- `sampling.seq_bootstrap`
- `sampling.get_av_uniqueness_from_triple_barrier`
- `sampling.num_concurrent_events`

### Rust API

- `get_ind_matrix`
- `get_ind_mat_average_uniqueness`
- `get_ind_mat_label_uniqueness`
- `bootstrap_loop_run`
- `seq_bootstrap`
- `get_av_uniqueness_from_triple_barrier`
- `num_concurrent_events`

## Implementation Notes

- Indicator matrix quality drives bootstrap quality.
- Use average uniqueness as a diagnostics KPI.

## Related Modules

- [`sample-weights`](/modules/sample-weights/)
- [`sb-bagging`](/modules/sb-bagging/)
- [`labeling`](/modules/labeling/)
