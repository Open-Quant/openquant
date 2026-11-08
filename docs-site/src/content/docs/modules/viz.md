---
title: "viz"
description: "Visualization payload builders for feature importance, drawdown, regime, frontier, and cluster charts."
status: authored
last_authored: '2026-11-07'
audience:
  - quant-dev
  - platform-engineering
module: "viz"
api_surface: "python-only"
python_api:
  - "viz.prepare_feature_importance_payload"
  - "viz.prepare_feature_importance_comparison_payload"
  - "viz.prepare_drawdown_payload"
  - "viz.prepare_regime_payload"
  - "viz.prepare_frontier_payload"
  - "viz.prepare_cluster_payload"
citation:
  - "OpenQuant implementation: python/openquant/viz.py."
  - "OpenQuant regression tests: python/tests/test_adapters_viz.py."
sidebar:
  badge: Module
---

`viz` builds plain dictionaries for a renderer you supply. It does not draw figures,
fit models, calculate feature importance or optimize portfolios. The
[implementation](https://github.com/Open-Quant/openquant/blob/main/python/openquant/viz.py)
defines the payloads; [tests](https://github.com/Open-Quant/openquant/blob/main/python/tests/test_adapters_viz.py)
exercise every builder. [Feature diagnostics](/modules/feature-diagnostics/) uses
these builders in its reports; [pipeline](/modules/pipeline/) returns raw sequences
and frames that you can pass to them yourself.

## Payload contracts

| Builder | `chart` | Data and behavior |
| --- | --- | --- |
| `prepare_feature_importance_payload` | `bar` | `x` names, `y` scores, optional `error_y`; descending importance, then `top_n` |
| `prepare_feature_importance_comparison_payload` | `grouped_bar` | `left` and `right`, each with `name`, `labels`, `values`; input order retained |
| `prepare_drawdown_payload` | `line` | Timestamp strings in `x`, original `equity`, computed `drawdown` |
| `prepare_regime_payload` | `line+step` | `x`, `score`, `threshold`, `regime`; +1 at or above threshold, otherwise −1 |
| `prepare_frontier_payload` | `scatter` | Volatility in `x`, return in `y`; optional Sharpe `color` |
| `prepare_cluster_payload` | `tree` | `node_id`, `parent_id`; optional `height` |

All builders check corresponding sequence lengths and raise `ValueError` on mismatch.
They do not check that supplied statistics have economic meaning. Comparison series
are neither sorted nor aligned by label: their labels may deliberately differ.
Cluster parents are not checked against node IDs, and cycles are not rejected.

## Sort scores while retaining their error bars

```python
from openquant.viz import prepare_feature_importance_payload

payload = prepare_feature_importance_payload(
    feature_names=["spread", "momentum", "vol"],
    importance=[0.20, 0.45, 0.35],
    std=[0.02, 0.05, 0.03],
    top_n=2,
)
print("chart:", payload["chart"])
print("features:", payload["x"])
print("importance:", payload["y"])
print("error bars:", payload["error_y"])
```

```text
chart: bar
features: ['momentum', 'vol']
importance: [0.45, 0.35]
error bars: [0.05, 0.03]
```

The numbers here are illustrative inputs, not fitted importance estimates. Error
bars are passed through from `std`; the helper does not infer confidence intervals.
Use a positive `top_n` for an ordinary top-feature view and explicitly label the
importance method and uncertainty convention in your renderer.

## Compare reports by feature name

The comparison builder preserves each side's order and does not align labels. A renderer
that pairs the first left value with the first right value can therefore compare different
features. Read the nested `left` and `right` payloads and join by feature name when the
reports use a common feature namespace:

```python
import polars as pl
from openquant.viz import prepare_feature_importance_comparison_payload

# Illustrative scores from two reports using the same importance definition and units.
payload = prepare_feature_importance_comparison_payload(
    left_labels=["spread", "momentum"], left_values=[0.2, 0.8],
    right_labels=["momentum", "volume"], right_values=[0.6, 0.4],
    left_name="baseline", right_name="candidate",
)
assert payload["chart"] == "grouped_bar"
assert payload["left"]["name"] == "baseline"
assert payload["right"]["labels"] == ["momentum", "volume"]


def side_frame(side, column):
    # Require one measurement per feature before joining; duplicate labels are ambiguous.
    assert len(set(side["labels"])) == len(side["labels"])
    return pl.DataFrame({"feature": side["labels"], column: side["values"]})


comparison = side_frame(payload["left"], "baseline").join(
    side_frame(payload["right"], "candidate"),
    on="feature", how="full", coalesce=True,
).sort("feature")
assert comparison.to_dicts() == [
    {"feature": "momentum", "baseline": 0.8, "candidate": 0.6},
    {"feature": "spread", "baseline": 0.2, "candidate": None},
    {"feature": "volume", "baseline": None, "candidate": 0.4},
]
for row in comparison.iter_rows(named=True):
    print(row["feature"], row["baseline"], row["candidate"])
```

```text
momentum 0.8 0.6
spread 0.2 None
volume None 0.4
```

The joined table is also a text alternative to a grouped chart. Its nulls mean **not
supplied in that report**, not zero importance; show a missing marker rather than an
invented zero-height bar. The helper does not normalize units or make different importance
methods comparable. Confirm the method, score definition, evaluation sample and feature
meaning before interpreting differences. If labels intentionally describe different objects
(such as raw features versus principal components), display separate labeled series instead
of forcing a feature-by-feature join.

## Drawdown is relative to the running high

For positive equity, drawdown is `equity / running_max(equity) - 1`: zero at a new
high and negative below it. This is a fractional drawdown, not a currency loss or
a positive loss magnitude. The builder preserves row order, so sort and align the
inputs before calling it.

```python
from openquant.viz import prepare_drawdown_payload, prepare_regime_payload

payload = prepare_drawdown_payload(
    timestamps=["2024-01-02 16:00:00", "2024-01-03 16:00:00", "2024-01-04 16:00:00"],
    equity_curve=[100.0, 99.0, 101.0],
)
print("equity:", payload["equity"])
print("drawdown:", [round(value, 4) for value in payload["drawdown"]])
print("first timestamp:", payload["x"][0])
regime = prepare_regime_payload(["t1", "t2", "t3"], [0.3, -0.2, 0.0])
print("regimes:", regime["regime"])
```

```text
equity: [100.0, 99.0, 101.0]
drawdown: [0.0, -0.01, 0.0]
first timestamp: 2024-01-02 16:00:00
regimes: [1, -1, 1]
```

<figure>
<img class="dark:sl-hidden" src="/figures/workflow-viz-drawdown-light.svg" alt="Equity falls from 100 to 99 and rises to 101. Drawdown is zero, minus one percent, then zero at the new high." />
<img class="light:sl-hidden" src="/figures/workflow-viz-drawdown-dark.svg" alt="Equity falls from 100 to 99 and rises to 101. Drawdown is zero, minus one percent, then zero at the new high." />
<figcaption>The example's three synthetic points: drawdown resets at a new high. Generated from the definition by <a href="https://github.com/Open-Quant/openquant/blob/main/docs-site/scripts/figures/workflow_viz_figures.py">workflow_viz_figures.py</a>; the rendering is separate from the payload builder.</figcaption>
</figure>

The drawdown builder parses timestamps with the shared data parser and stringifies
the results; an invalid or null parsed entry becomes the string `"None"`, including
a string series in which every value is invalid. The regime builder instead
passes timestamp labels through unchanged. These dictionaries are convenient chart
inputs, not a universal interchange schema: frontier labels are fixed to volatility
and return, and your plotting code must choose units, axes and formatting.

For frame construction and input-shape validation, see [adapters](/modules/adapters/).


## Validate and label the rendered result

The payload builders do not reject nonfinite statistics. For example, a NaN score compares
false to the regime threshold and becomes regime −1, which can hide missing data as a
meaningful state. Validate or mask missing values before constructing the payload:

```python
import math
from openquant.viz import prepare_drawdown_payload, prepare_regime_payload

assert prepare_drawdown_payload(["invalid"], [100.0])["x"] == ["None"]
score = [0.2, float("nan")]
assert prepare_regime_payload(["t1", "t2"], score)["regime"] == [1, -1]
assert not all(math.isfinite(value) for value in score)  # reject or mask before rendering
```

When exporting to strict JSON, reject or explicitly encode NaN and infinity; a plain Python
dictionary does not guarantee a portable JSON payload. Do not silently replace missing
returns or risks with zero merely to make a chart render.

A payload can be valid for Python plotting but fail a strict JSON consumer. Export with
`allow_nan=False` so missing/nonfinite numeric values require an explicit decision. This
example retains an absent comparison value as JSON `null` and rejects a nonfinite score:

```python
import json
from openquant.viz import prepare_regime_payload

comparison_row = {"feature": "spread", "baseline": 0.2, "candidate": None}
encoded = json.dumps(comparison_row, allow_nan=False)
assert json.loads(encoded)["candidate"] is None

payload = prepare_regime_payload(["t1", "t2"], [0.2, float("nan")])
try:
    json.dumps(payload, allow_nan=False)
except ValueError:
    print("nonfinite payload rejected before export")
else:
    raise AssertionError("expected strict JSON to reject NaN")
```

```text
nonfinite payload rejected before export
```

Choose a missing-data policy before constructing a chart, and apply it consistently to
values, labels and derived states. Strict serialization rejects invalid numbers; it cannot
repair the regime already assigned to a missing score or validate the economic meaning of
finite values. Python's default JSON encoder permits nonstandard NaN/Infinity tokens, so a
successful default `json.dumps(payload)` call alone is not a portability check.

The renderer also owns accessibility and interpretation: provide an informative title,
axis units and sampling period, text labels or markers in addition to color, and a table or
text summary of the plotted values. Name the uncertainty convention for `error_y` and show
missing observations as missing. A frontier-shaped scatter plot does not establish that its
points are efficient portfolios, and a regime step plot does not validate a predictive model.
