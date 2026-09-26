"""Reference events for the CUSUM and z-score filters, independent of this library.

    uv run --with pandas python tests/fixtures/filters/generate.py

Writes events.json next to this file. Imports neither openquant nor mlfinlab.

CUSUM: AFML snippet 2.4 (the symmetric CUSUM filter, section 2.5.2.1) applied to log prices,
so the increments are log returns. The dynamic case uses the threshold of the bar being
tested, h_t = close_t * 1e-5.

Z-score: an event is every bar whose close is at least `threshold` rolling standard deviations
above its rolling mean: close_t >= mean_{w}(close)_t + threshold * std_{w'}(close)_t, with
pandas' rolling mean and rolling sample standard deviation (ddof=1), windows ending at and
including t. This filter is not in AFML; the rule is stated here in full and implemented
directly.

Both "_timestamps" and "_index" keys hold the event timestamps (ISO format); the Rust test maps
its index output back to timestamps before comparing, so the two must agree.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
data = pd.read_csv(HERE.parent / "shared" / "dollar_bar_sample.csv", index_col="date_time", parse_dates=True)
close = data["close"]


def cusum_events(close, h):
    """AFML snippet 2.4 on log prices. `h` is a scalar or a Series aligned with `close`."""
    events, s_pos, s_neg = [], 0.0, 0.0
    diff = np.log(close).diff()
    for t in diff.index[1:]:
        h_t = h.loc[t] if isinstance(h, pd.Series) else h
        s_pos, s_neg = max(0.0, s_pos + diff.loc[t]), min(0.0, s_neg + diff.loc[t])
        if s_neg < -h_t:
            s_neg = 0.0
            events.append(t)
        elif s_pos > h_t:
            s_pos = 0.0
            events.append(t)
    return events


def z_score_events(close, mean_window, std_window, threshold):
    bound = close.rolling(mean_window).mean() + threshold * close.rolling(std_window).std()
    return list(close.index[close >= bound])


def iso(events):
    return [t.isoformat(timespec="microseconds") for t in events]


result = {
    "source": "tests/fixtures/filters/generate.py: AFML snippet 2.4 (CUSUM) and a rolling z-score "
              "rule, pandas %s, on tests/fixtures/shared/dollar_bar_sample.csv" % pd.__version__,
    "meta": {"rows": len(data), "columns": list(data.columns)},
    "cusum": {},
}
for thresh in [0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.04]:
    events = iso(cusum_events(close, thresh))
    result["cusum"][f"{thresh}_timestamps"] = events
    result["cusum"][f"{thresh}_index"] = events

dynamic = iso(cusum_events(close, close * 1e-5))
result["cusum_dynamic"] = {"timestamps": dynamic, "index": dynamic}

z = iso(z_score_events(close, 100, 100, 2))
result["z_score"] = {"timestamps": z, "index": z}

(HERE / "events.json").write_text(json.dumps(result, indent=2) + "\n")
print({k: len(v) for k, v in result["cusum"].items() if k.endswith("_timestamps")},
      "dynamic", len(dynamic), "z_score", len(z))
