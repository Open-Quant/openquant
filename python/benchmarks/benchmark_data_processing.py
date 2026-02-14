from __future__ import annotations

import argparse
import json
import time

import polars as pl

import openquant


def make_dataset(rows_per_symbol: int, symbols: list[str]) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for symbol in symbols:
        frames.append(
            pl.DataFrame(
                {
                    "ts": pl.datetime_range(
                        start=pl.datetime(2020, 1, 1),
                        end=pl.datetime(2020, 1, 1) + pl.duration(minutes=rows_per_symbol - 1),
                        interval="1m",
                        eager=True,
                    ),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark openquant.data throughput.")
    parser.add_argument("--rows-per-symbol", type=int, default=200_000)
    parser.add_argument("--symbols", type=int, default=4)
    args = parser.parse_args()

    symbol_names = [f"SYM{i}" for i in range(args.symbols)]
    base = make_dataset(args.rows_per_symbol, symbol_names)
    total_rows = base.height

    # Warm-up to stabilize lazy-plan compile and allocation effects.
    _ = openquant.data.clean_ohlcv(base)

    t0 = time.perf_counter()
    clean = openquant.data.clean_ohlcv(base)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    quality = openquant.data.data_quality_report(base)
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    aligned = openquant.data.align_calendar(clean, interval="1m")
    t5 = time.perf_counter()

    print(
        json.dumps(
            {
                "rows": total_rows,
                "clean_rows": clean.height,
                "aligned_rows": aligned.height,
                "clean_seconds": t1 - t0,
                "clean_rows_per_sec": total_rows / max(t1 - t0, 1e-9),
                "quality_seconds": t3 - t2,
                "quality_rows_per_sec": total_rows / max(t3 - t2, 1e-9),
                "align_seconds": t5 - t4,
                "align_rows_per_sec": clean.height / max(t5 - t4, 1e-9),
                "quality_report": {
                    "row_count": quality["row_count"],
                    "symbol_count": quality["symbol_count"],
                    "duplicate_key_count": quality["duplicate_key_count"],
                    "gap_interval_count": quality["gap_interval_count"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
