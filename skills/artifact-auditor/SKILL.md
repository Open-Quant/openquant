---
name: artifact-auditor
description: Audit experiment artifact folders for completeness, schema consistency, and promotion traceability across single and grid runs.
---

# Artifact Auditor

Use this skill when checking saved experiment artifacts before another quant reviews the run. Structural checks support review; they do not establish publishability, independent provenance, profitability or absence of leakage.

## What this skill does

1. Verifies required files exist for each run.
2. Checks expected columns in `metrics.parquet` and `leaderboard.parquet`.
3. Checks decision-file presence and manifest/leaderboard linkage for grid members.
4. Reports missing files/fields with actionable remediation.

## Required files (single run)

- `run_manifest.json`
- `metrics.parquet`
- `events.parquet`
- `signals.parquet`
- `weights.parquet`
- `backtest.parquet`
- `decision.md`
- `equity_curve.svg` and `drawdown.svg`

## Required files (grid run)

- top-level `leaderboard.parquet`
- top-level `run_manifest.json`
- sub-run directories each containing the single-run file set

## Structural audit command

Run from the repository root in the prepared development environment. Replace the path with the exact single-run or grid-run directory printed by `experiments/run_pipeline.py`, not the enclosing artifacts root. This read-only check exits nonzero on missing files, empty run selection, unreadable JSON/Parquet, missing core columns or inconsistent grid membership. It stops at the first failure; restore or regenerate that run in a new output root and rerun the audit.

```bash
uv run --no-sync --python .venv/bin/python - /path/to/printed-run-directory <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import polars as pl

root = Path(sys.argv[1]).resolve()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def manifest(folder):
    value = json.loads((folder / "run_manifest.json").read_text())
    require(isinstance(value, dict), f"{folder}: manifest must be an object")
    for key in ("config_digest", "git_sha", "python", "dataset"):
        require(key in value, f"{folder}: missing manifest field {key}")
    return value

def table(folder, name, columns):
    frame = pl.read_parquet(folder / f"{name}.parquet")
    missing = set(columns.split()) - set(frame.columns)
    require(not missing, f"{folder}/{name}: missing columns {sorted(missing)}")
    return frame

def single(folder):
    info = manifest(folder)
    for key in ("run_name", "config_path", "config"):
        require(key in info, f"{folder}: missing manifest field {key}")
    metrics = table(folder, "metrics", "net_sharpe realized_sharpe inputs_aligned")
    require(metrics.height == 1, f"{folder}: expected one metrics row")
    # Zero event rows can be valid; structural presence is still required.
    table(folder, "events", "start_ts end_ts prob side")
    table(folder, "signals", "ts signal")
    table(folder, "weights", "asset weight")
    table(folder, "backtest", "ts equity returns position")
    for name in ("decision.md", "equity_curve.svg", "drawdown.svg"):
        require((folder / name).is_file() and (folder / name).stat().st_size > 0,
                f"{folder}: missing or empty {name}")
    return info

try:
    require(root.is_dir(), f"{root}: select an existing single-run or grid-run directory")
    top = manifest(root)
    if top.get("mode") == "grid":
        board = table(root, "leaderboard", "run_name run_index config_digest net_sharpe promote_candidate")
        count = top.get("run_count")
        require(isinstance(count, int) and count > 0 and board.height == count,
                f"{root}: nonempty leaderboard/run_count mismatch")
        require(sorted(board["run_index"].to_list()) == list(range(count)),
                f"{root}: invalid or duplicate run indices")
        require(Counter(board["run_name"].to_list()) == Counter(top.get("run_names", [])),
                f"{root}: manifest/leaderboard run names differ")
        expected = set()
        for row in board.iter_rows(named=True):
            child = (root / f"{row['run_name']}-{row['config_digest']}").resolve()
            require(child.parent == root and child not in expected,
                    f"{root}: unsafe or duplicate child mapping {child}")
            expected.add(child)
            info = single(child)
            require(info["run_name"] == row["run_name"], f"{child}: run name mismatch")
            require(info["dataset"] == top["dataset"], f"{child}: dataset metadata mismatch")
        actual = {p.resolve() for p in root.iterdir() if p.is_dir()}
        require(actual == expected, f"{root}: unexpected or missing child directories")
    else:
        single(root)
        count = 1
    print(f"Structural artifact audit passed: {count} run(s)")
except Exception as exc:
    raise SystemExit(f"Artifact audit failed: {exc}") from exc
PY
```

The check allows zero event rows and checks only the listed core columns, not full dtypes, finite values, timestamp alignment, numerical recomputation, chart contents or decision flags. Grid directory names use each leaderboard entry's effective-config digest; the child manifest digest hashes a different wrapper containing `base_config_path` and `grid_entry`, so those digests must not be compared directly. Manifest equality does not verify the dataset hash against original inputs or identify the loaded native binary. Consult the [artifact readback example](../../docs-site/src/content/docs/examples/catalog.md) and inspect configuration, data lineage and decision evidence separately before drawing research conclusions.
