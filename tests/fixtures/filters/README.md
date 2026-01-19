# Filters fixtures

- `dollar_bar_sample.csv`: copied from `mlfinlab/tests/test_data/dollar_bar_sample.csv`.
- `events.json`: precomputed outputs for `cusum_filter` and `z_score_filter` on `dollar_bar_sample.csv`.

Generation command (Python 3.8 env `.venv_x86`):
```bash
arch -x86_64 .venv_x86/bin/python - <<'PY'
import json
import pandas as pd
from mlfinlab.filters.filters import cusum_filter, z_score_filter

path = 'mlfinlab/tests/test_data/dollar_bar_sample.csv'
data = pd.read_csv(path, index_col='date_time')
data.index = pd.to_datetime(data.index)

thresholds = [0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.04]
result = {"source_csv": path, "meta": {"rows": len(data), "columns": list(data.columns)}, "cusum": {}}

def serialize(e):
    return e.isoformat() if hasattr(e, 'isoformat') else int(e)

for thresh in thresholds:
    for ts in (True, False):
        events = cusum_filter(data['close'], threshold=thresh, time_stamps=ts)
        key = f"{thresh}_{'timestamps' if ts else 'index'}"
        result["cusum"][key] = [serialize(e) for e in events]

result["cusum_dynamic"] = {}
dyn = data['close'] * 1e-5
for ts in (True, False):
    events = cusum_filter(data['close'], threshold=dyn, time_stamps=ts)
    key = f"{'timestamps' if ts else 'index'}"
    result["cusum_dynamic"][key] = [serialize(e) for e in events]

result["z_score"] = {}
for ts in (True, False):
    events = z_score_filter(data['close'], 100, 100, 2, time_stamps=ts)
    key = f"{'timestamps' if ts else 'index'}"
    result["z_score"][key] = [serialize(e) for e in events]

with open('openquant-rs/tests/fixtures/filters/events.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
PY
```

Use these fixtures to mirror Python assertions in Rust tests:
- CUSUM: counts and sequences for thresholds (0.005–0.04) and dynamic threshold (`close * 1e-5`).
- Z-score: full event sequences for window/lag 100, threshold 2.
