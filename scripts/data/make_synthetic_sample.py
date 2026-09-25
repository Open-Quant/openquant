#!/usr/bin/env python3
"""Generate the SYNTHETIC daily OHLCV sample shipped with `openquant.data`.

This is not market data. Every number is drawn from a seeded random process by
this script, so the file has no third-party terms and may be redistributed under
the repository's MIT license. It exists so CI, the docs and the runbooks can run
offline until the owner picks a real redistributable sample (see DATA_SOURCES.md).

The symbols (`SYN_A` ... `SYN_E`) are deliberately not real tickers.

Model: one market factor plus an idiosyncratic term per symbol, compounded as a
log-normal walk over Monday-Friday business days (no exchange holidays).
Open gaps from the previous close; high/low widen the open-close range; volume is
log-normal. `adj_close` equals `close` (there are no corporate actions).

Only the standard library is used, and every value is rounded before writing, so
the output is byte-identical across platforms and Python versions:

    python3 scripts/data/make_synthetic_sample.py          # rewrite the file
    python3 scripts/data/make_synthetic_sample.py --check  # exit 1 if it differs
"""

from __future__ import annotations

import argparse
import io
import math
import random
import sys
from datetime import date, timedelta
from pathlib import Path

SEED = 20260919
START = date(2022, 1, 3)
END = date(2023, 12, 29)
# symbol: (initial price, market beta, idiosyncratic daily vol, mean volume)
SYMBOLS = {
    "SYN_A": (100.0, 1.00, 0.008, 2_500_000),
    "SYN_B": (45.0, 1.30, 0.014, 6_000_000),
    "SYN_C": (210.0, 0.70, 0.006, 900_000),
    "SYN_D": (32.0, 0.20, 0.011, 3_200_000),
    "SYN_E": (78.0, -0.30, 0.012, 1_400_000),
}
MARKET_VOL = 0.010
MARKET_DRIFT = 0.0002

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "python" / "openquant" / "_sample_data" / "synthetic_daily_ohlcv.csv"
HEADER = "date,symbol,open,high,low,close,volume,adj_close\n"


def business_days(start: date, end: date) -> list[date]:
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def generate() -> str:
    rng = random.Random(SEED)
    days = business_days(START, END)
    market = [rng.gauss(MARKET_DRIFT, MARKET_VOL) for _ in days]

    rows: list[tuple[str, str, float, float, float, float, int]] = []
    for symbol, (p0, beta, idio, mean_vol) in SYMBOLS.items():
        prev_close = p0
        for i, d in enumerate(days):
            gap = rng.gauss(0.0, idio * 0.25)
            ret = beta * market[i] + rng.gauss(0.0, idio)
            open_ = prev_close * math.exp(gap)
            close = prev_close * math.exp(gap + ret)
            spread = abs(rng.gauss(0.0, (idio + MARKET_VOL) * 0.6))
            high = max(open_, close) * math.exp(spread * rng.random())
            low = min(open_, close) * math.exp(-spread * rng.random())
            volume = int(mean_vol * math.exp(rng.gauss(0.0, 0.35) - 0.35**2 / 2))
            o, h, l, c = (round(x, 4) for x in (open_, high, low, close))
            # Rounding can only narrow the range by < 1e-4; keep low <= o,c <= high.
            h = max(h, o, c)
            l = min(l, o, c)
            rows.append((d.isoformat(), symbol, o, h, l, c, volume))
            prev_close = c

    buf = io.StringIO()
    buf.write(HEADER)
    for d, s, o, h, l, c, v in rows:
        buf.write(f"{d},{s},{o:.4f},{h:.4f},{l:.4f},{c:.4f},{v},{c:.4f}\n")
    return buf.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the committed file is stale")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    text = generate()
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else None
        if current != text:
            print(f"{args.out} is stale: rerun scripts/data/make_synthetic_sample.py", file=sys.stderr)
            return 1
        print(f"{args.out} is current")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(text):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
