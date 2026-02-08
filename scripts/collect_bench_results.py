#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def find_estimates(criterion_dir: Path):
    for path in criterion_dir.rglob("new/estimates.json"):
        yield path


def bench_name(criterion_dir: Path, estimates_path: Path) -> str:
    rel = estimates_path.relative_to(criterion_dir)
    # strip trailing new/estimates.json
    parts = rel.parts[:-2]
    return "/".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Collect criterion benchmark means into a single JSON file")
    parser.add_argument("--criterion-dir", default="target/criterion")
    parser.add_argument("--out", default="benchmarks/latest_benchmarks.json")
    parser.add_argument("--commit", default="")
    parser.add_argument("--allow-list", default="", help="JSON file with benchmark names to include")
    args = parser.parse_args()

    criterion_dir = Path(args.criterion_dir)
    out_path = Path(args.out)

    allow = None
    if args.allow_list:
        allow = set(json.loads(Path(args.allow_list).read_text()))

    benchmarks = {}
    for est_path in find_estimates(criterion_dir):
        try:
            data = json.loads(est_path.read_text())
            mean = data["mean"]["point_estimate"]
            lower = data["mean"]["confidence_interval"]["lower_bound"]
            upper = data["mean"]["confidence_interval"]["upper_bound"]
        except Exception:
            continue
        name = bench_name(criterion_dir, est_path)
        if allow is not None and name not in allow:
            continue
        benchmarks[name] = {
            "mean_ns": mean,
            "mean_ms": mean / 1_000_000.0,
            "ci_lower_ns": lower,
            "ci_upper_ns": upper,
        }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": args.commit,
        "count": len(benchmarks),
        "benchmarks": dict(sorted(benchmarks.items())),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path} with {len(benchmarks)} benchmarks")


if __name__ == "__main__":
    main()
