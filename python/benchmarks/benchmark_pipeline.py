from __future__ import annotations

import argparse
import statistics
import time

import openquant


def run_benchmark(iterations: int, bars: int, seed: int) -> dict[str, float]:
    # Warm-up to stabilize JIT/cache effects.
    warmup_ds = openquant.research.make_synthetic_futures_dataset(n_bars=bars, seed=seed)
    openquant.research.run_flywheel_iteration(warmup_ds)

    timings: list[float] = []
    for i in range(iterations):
        ds = openquant.research.make_synthetic_futures_dataset(n_bars=bars, seed=seed + i)
        t0 = time.perf_counter()
        openquant.research.run_flywheel_iteration(ds)
        timings.append(time.perf_counter() - t0)

    mean_s = statistics.mean(timings)
    p95_s = sorted(timings)[max(int(len(timings) * 0.95) - 1, 0)]
    return {
        "iterations": float(iterations),
        "bars_per_run": float(bars),
        "mean_ms": mean_s * 1000.0,
        "p95_ms": p95_s * 1000.0,
        "runs_per_sec": 1.0 / mean_s if mean_s > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OpenQuant Python pipeline runtime.")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--bars", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = run_benchmark(iterations=args.iterations, bars=args.bars, seed=args.seed)
    print(
        "pipeline_bench "
        f"iterations={int(out['iterations'])} "
        f"bars={int(out['bars_per_run'])} "
        f"mean_ms={out['mean_ms']:.3f} "
        f"p95_ms={out['p95_ms']:.3f} "
        f"runs_per_sec={out['runs_per_sec']:.2f}"
    )


if __name__ == "__main__":
    main()
