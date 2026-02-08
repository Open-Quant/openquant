#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description="Fail if benchmark regressions exceed threshold")
    parser.add_argument("--baseline", default="benchmarks/baseline_benchmarks.json")
    parser.add_argument("--latest", default="benchmarks/latest_benchmarks.json")
    parser.add_argument("--max-regression-pct", type=float, default=25.0)
    parser.add_argument("--min-baseline-ms", type=float, default=0.05)
    args = parser.parse_args()

    baseline = load(Path(args.baseline))
    latest = load(Path(args.latest))

    base = baseline.get("benchmarks", {})
    curr = latest.get("benchmarks", {})

    regressions = []
    checked = 0
    for name, b in base.items():
        if name not in curr:
            continue
        b_ms = b["mean_ms"]
        c_ms = curr[name]["mean_ms"]
        if b_ms < args.min_baseline_ms:
            continue
        checked += 1
        delta_pct = ((c_ms - b_ms) / b_ms) * 100.0
        if delta_pct > args.max_regression_pct:
            regressions.append((name, b_ms, c_ms, delta_pct))

    print(f"checked {checked} benchmarks against baseline")
    if regressions:
        print("regressions above threshold:")
        for name, b_ms, c_ms, d in regressions:
            print(f"- {name}: baseline={b_ms:.3f}ms latest={c_ms:.3f}ms delta={d:.1f}%")
        return 1

    print("no benchmark regressions above threshold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
