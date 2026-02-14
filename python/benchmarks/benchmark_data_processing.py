from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import polars as pl

import openquant


@dataclass(frozen=True)
class BenchStats:
    name: str
    rows: int
    bytes_estimate: int
    iterations: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    std_ms: float
    rows_per_sec: float
    mb_per_sec: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rows": self.rows,
            "bytes_estimate": self.bytes_estimate,
            "iterations": self.iterations,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "std_ms": self.std_ms,
            "rows_per_sec": self.rows_per_sec,
            "mb_per_sec": self.mb_per_sec,
        }


def _estimate_df_bytes(df: pl.DataFrame) -> int:
    if df.height == 0:
        return 0
    # Keep this cheap and deterministic for repeated benchmark runs.
    return int(df.estimated_size())


def _measure(
    name: str,
    rows: int,
    bytes_estimate: int,
    iterations: int,
    fn: Callable[[], Any],
) -> BenchStats:
    timings: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - t0)
    timings.sort()
    mean_s = statistics.mean(timings)
    p50_s = timings[len(timings) // 2]
    p95_s = timings[max(int(len(timings) * 0.95) - 1, 0)]
    std_s = statistics.pstdev(timings) if len(timings) > 1 else 0.0
    return BenchStats(
        name=name,
        rows=rows,
        bytes_estimate=bytes_estimate,
        iterations=iterations,
        mean_ms=mean_s * 1000.0,
        p50_ms=p50_s * 1000.0,
        p95_ms=p95_s * 1000.0,
        std_ms=std_s * 1000.0,
        rows_per_sec=(rows / mean_s) if mean_s > 0 else 0.0,
        mb_per_sec=((bytes_estimate / 1_000_000.0) / mean_s) if mean_s > 0 else 0.0,
    )


def _format_rate(v: float) -> str:
    if v >= 1_000_000_000:
        return f"{v / 1_000_000_000:.2f}B/s"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M/s"
    if v >= 1_000:
        return f"{v / 1_000:.2f}K/s"
    return f"{v:.2f}/s"


def _format_table(stats: list[BenchStats]) -> str:
    header = (
        "name".ljust(22)
        + "rows".rjust(12)
        + "mean_ms".rjust(12)
        + "p95_ms".rjust(12)
        + "rows/s".rjust(14)
        + "MB/s".rjust(12)
    )
    lines = [header, "-" * len(header)]
    for s in stats:
        lines.append(
            s.name.ljust(22)
            + f"{s.rows:,}".rjust(12)
            + f"{s.mean_ms:,.2f}".rjust(12)
            + f"{s.p95_ms:,.2f}".rjust(12)
            + _format_rate(s.rows_per_sec).rjust(14)
            + f"{s.mb_per_sec:,.1f}".rjust(12)
        )
    return "\n".join(lines)


def _compare_against_baseline(
    current: list[BenchStats],
    baseline_path: Path,
) -> list[dict[str, Any]]:
    if not baseline_path.exists():
        return []
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_map = {
        item["name"]: item
        for item in baseline.get("functions", [])
        if isinstance(item, dict) and "name" in item
    }
    rows: list[dict[str, Any]] = []
    for cur in current:
        prev = base_map.get(cur.name)
        if not prev:
            continue
        prev_mean = float(prev.get("mean_ms", 0.0))
        if prev_mean <= 0:
            continue
        regression_pct = ((cur.mean_ms - prev_mean) / prev_mean) * 100.0
        rows.append(
            {
                "name": cur.name,
                "baseline_mean_ms": prev_mean,
                "current_mean_ms": cur.mean_ms,
                "regression_pct": regression_pct,
            }
        )
    return rows


def make_dataset(rows_per_symbol: int, symbols: list[str]) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    ts_base = pl.datetime_range(
        start=pl.datetime(2020, 1, 1),
        end=pl.datetime(2020, 1, 1) + pl.duration(minutes=rows_per_symbol - 1),
        interval="1m",
        eager=True,
    )
    for symbol in symbols:
        frames.append(
            pl.DataFrame(
                {
                    "ts": ts_base,
                    "symbol": [symbol] * rows_per_symbol,
                    "open": pl.arange(0, rows_per_symbol, eager=True).cast(pl.Float64) + 100.0,
                    "high": pl.arange(0, rows_per_symbol, eager=True).cast(pl.Float64) + 100.5,
                    "low": pl.arange(0, rows_per_symbol, eager=True).cast(pl.Float64) + 99.5,
                    "close": pl.arange(0, rows_per_symbol, eager=True).cast(pl.Float64) + 100.2,
                    "volume": pl.repeat(1000.0, rows_per_symbol, eager=True),
                }
            )
        )
    return pl.concat(frames, rechunk=True)


def run_benchmarks(rows_per_symbol: int, symbols: int, iterations: int) -> dict[str, Any]:
    symbol_names = [f"SYM{i}" for i in range(symbols)]
    base = make_dataset(rows_per_symbol, symbol_names)
    total_rows = base.height
    base_bytes = _estimate_df_bytes(base)

    # Warm-up to stabilize lazy-plan compile and allocation effects.
    clean_warm = openquant.data.clean_ohlcv(base)
    _ = openquant.data.data_quality_report(base)
    _ = openquant.data.align_calendar(clean_warm, interval="1m")

    clean = openquant.data.clean_ohlcv(base)
    clean_rows = clean.height
    clean_bytes = _estimate_df_bytes(clean)

    function_stats: list[BenchStats] = [
        _measure(
            name="clean_ohlcv",
            rows=total_rows,
            bytes_estimate=base_bytes,
            iterations=iterations,
            fn=lambda: openquant.data.clean_ohlcv(base),
        ),
        _measure(
            name="data_quality_report",
            rows=total_rows,
            bytes_estimate=base_bytes,
            iterations=iterations,
            fn=lambda: openquant.data.data_quality_report(base),
        ),
        _measure(
            name="align_calendar",
            rows=clean_rows,
            bytes_estimate=clean_bytes,
            iterations=iterations,
            fn=lambda: openquant.data.align_calendar(clean, interval="1m"),
        ),
    ]

    io_stats: list[BenchStats] = []
    with TemporaryDirectory(prefix="openquant_bench_") as tmp:
        tmp_dir = Path(tmp)
        csv_path = tmp_dir / "bench_ohlcv.csv"
        pq_path = tmp_dir / "bench_ohlcv.parquet"
        base.write_csv(csv_path)
        base.write_parquet(pq_path)

        # Warm-up file paths.
        _ = openquant.data.load_ohlcv(csv_path)
        _ = openquant.data.load_ohlcv(pq_path)

        io_stats.append(
            _measure(
                name="load_ohlcv_csv",
                rows=total_rows,
                bytes_estimate=csv_path.stat().st_size,
                iterations=iterations,
                fn=lambda: openquant.data.load_ohlcv(csv_path),
            )
        )
        io_stats.append(
            _measure(
                name="load_ohlcv_parquet",
                rows=total_rows,
                bytes_estimate=pq_path.stat().st_size,
                iterations=iterations,
                fn=lambda: openquant.data.load_ohlcv(pq_path),
            )
        )

    all_stats = function_stats + io_stats
    return {
        "dataset": {
            "rows_per_symbol": rows_per_symbol,
            "symbols": symbols,
            "total_rows": total_rows,
            "estimated_bytes": base_bytes,
        },
        "iterations": iterations,
        "functions": [s.as_dict() for s in all_stats],
        "tables": {
            "core": _format_table(function_stats),
            "io": _format_table(io_stats),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark openquant.data with per-function throughput and latency metrics."
    )
    parser.add_argument("--rows-per-symbol", type=int, default=200_000)
    parser.add_argument("--symbols", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    args = parser.parse_args()

    result = run_benchmarks(
        rows_per_symbol=args.rows_per_symbol,
        symbols=args.symbols,
        iterations=args.iterations,
    )

    print("Core Data Functions")
    print(result["tables"]["core"])
    print()
    print("I/O + End-to-End Load")
    print(result["tables"]["io"])
    print()

    if args.baseline is not None:
        diff_rows = _compare_against_baseline(
            [BenchStats(**row) for row in result["functions"]],
            args.baseline,
        )
        result["baseline_comparison"] = diff_rows
        if diff_rows:
            print("Baseline Comparison (mean_ms regression; negative is faster)")
            for row in diff_rows:
                print(
                    f"{row['name']}: baseline={row['baseline_mean_ms']:.2f}ms "
                    f"current={row['current_mean_ms']:.2f}ms "
                    f"regression_pct={row['regression_pct']:+.2f}%"
                )
            print()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote benchmark metrics to {args.out}")


if __name__ == "__main__":
    main()
