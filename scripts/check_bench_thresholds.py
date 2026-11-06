#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def finite_number(value, label, *, positive=False):
    """Reject unusable measurements before comparisons can silently ignore them."""
    requirement = "positive" if positive else "non-negative"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite {requirement} number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (value <= 0 if positive else value < 0):
        raise ValueError(f"{label} must be a finite {requirement} number")
    return value


def benchmark_timings(payload, label):
    if not isinstance(payload, dict) or not isinstance(payload.get("benchmarks"), dict):
        raise ValueError(f"{label} must contain a benchmarks object")
    benchmarks = payload["benchmarks"]
    for name, measurement in benchmarks.items():
        if not isinstance(measurement, dict) or "mean_ms" not in measurement:
            raise ValueError(f"{label} benchmark {name!r} must contain mean_ms")
        finite_number(measurement["mean_ms"], f"{label} benchmark {name!r} mean_ms", positive=True)
    return benchmarks


def main():
    parser = argparse.ArgumentParser(description="Fail if benchmark regressions exceed threshold")
    parser.add_argument("--baseline", default="benchmarks/baseline_benchmarks.json")
    parser.add_argument("--latest", default="benchmarks/latest_benchmarks.json")
    parser.add_argument("--max-regression-pct", type=float, default=25.0)
    parser.add_argument("--min-baseline-ms", type=float, default=0.05)
    parser.add_argument(
        "--overrides",
        default=None,
        help=(
            "Optional JSON file mapping benchmark name -> max regression percent override "
            "(e.g. {\"synthetic_ticker_pipeline_end_to_end\": 50.0})"
        ),
    )
    args = parser.parse_args()

    try:
        finite_number(args.max_regression_pct, "--max-regression-pct")
        finite_number(args.min_baseline_ms, "--min-baseline-ms")
        base = benchmark_timings(load(Path(args.baseline)), "baseline")
        curr = benchmark_timings(load(Path(args.latest)), "latest")
        overrides = load(Path(args.overrides)) if args.overrides else {}
        if not isinstance(overrides, dict):
            raise ValueError("overrides must be an object mapping benchmark names to thresholds")
        for name, value in overrides.items():
            finite_number(value, f"override for {name!r}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    regressions = []
    missing = []
    rows = []
    checked = 0
    for name, b in base.items():
        if name not in curr:
            missing.append(name)
            continue
        b_ms = b["mean_ms"]
        c_ms = curr[name]["mean_ms"]
        if b_ms < args.min_baseline_ms:
            continue
        checked += 1
        delta_pct = ((c_ms - b_ms) / b_ms) * 100.0
        threshold_pct = float(overrides.get(name, args.max_regression_pct))
        rows.append((name, b_ms, c_ms, delta_pct, threshold_pct))
        if delta_pct > threshold_pct:
            regressions.append((name, b_ms, c_ms, delta_pct, threshold_pct))

    print(f"checked {checked} benchmarks against baseline")
    for name, b_ms, c_ms, d, t in rows:
        print(f"  {name}: baseline={b_ms:.3f}ms latest={c_ms:.3f}ms delta={d:+.1f}% threshold={t:.1f}%")
    for name in sorted(set(curr) - set(base)):
        print(f"  {name}: new benchmark, no baseline to compare against")

    failed = False
    if missing:
        # A benchmark that stops producing results must not read as "no regression".
        print("benchmarks in the baseline but missing from the latest run:")
        for name in missing:
            print(f"- {name}")
        failed = True
    if checked == 0:
        print("no benchmarks were compared; refusing to report success")
        failed = True
    if regressions:
        print("regressions above threshold:")
        for name, b_ms, c_ms, d, t in regressions:
            print(
                f"- {name}: baseline={b_ms:.3f}ms latest={c_ms:.3f}ms "
                f"delta={d:.1f}% threshold={t:.1f}%"
            )
        failed = True

    if failed:
        return 1
    print("no benchmark regressions above threshold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
