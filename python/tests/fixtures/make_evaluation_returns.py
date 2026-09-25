"""Regenerate `evaluation_returns.csv`, the returns series `test_evaluation.py` checks.

    python python/tests/fixtures/make_evaluation_returns.py

It uses only the standard library's `random.Random`, whose `gauss` and `random` streams
are stable across CPython versions for a given seed, and writes each return rounded to
six decimals. The rounded values in the CSV — not this script — are what the hand
computations in the test are based on, so the file is committed and the script is only a
record of where it came from.

The series is 120 daily returns from a strategy with a positive drift, a 1% daily
volatility and an occasional 2.5% gap down: a per-period Sharpe ratio around 0.1, with
negative skew and fat tails, so that every term of the PSR, DSR and MinTRL formulas is
exercised.
"""

from __future__ import annotations

import random
from pathlib import Path

SEED = 27
N_OBS = 120
OUT = Path(__file__).with_name("evaluation_returns.csv")


def main() -> None:
    rng = random.Random(SEED)
    rows = []
    for _ in range(N_OBS):
        gap = 0.025 if rng.random() < 0.04 else 0.0
        rows.append(round(rng.gauss(0.0015, 0.01) - gap, 6))
    OUT.write_text("return\n" + "".join(f"{r:.6f}\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} returns to {OUT}")


if __name__ == "__main__":
    main()
