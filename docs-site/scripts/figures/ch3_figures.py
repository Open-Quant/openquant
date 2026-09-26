"""Figures for the labeling and bet-sizing pages. Run from the repo root:

    .venv/bin/python docs-site/scripts/figures/ch3_figures.py

Same seeded series and parameters as the page's Python example; the two events drawn are taken
from that run.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

from _svg import MONO, THEMES, Chart
from openquant import bet_sizing, filters, labeling, volatility

PANELS = [("2024-01-12 12:00:00", "profit barrier first"), ("2024-01-09 02:00:00", "time runs out")]
WIDTH = 1.5  # barrier width in targets
HORIZON = 48  # hours


def series():
    rng = random.Random(3)
    start, price, stamps, close = datetime(2024, 1, 1), 100.0, [], []
    for i in range(24 * 90):
        price *= math.exp(rng.gauss(0, 0.004))
        stamps.append((start + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"))
        close.append(price)
    return stamps, close


def main():
    stamps, close = series()
    vol = [(t, v) for t, v in volatility.get_daily_vol(stamps, close, 100) if not math.isnan(v)]
    events = filters.cusum_filter_timestamps(close, stamps, 0.02)
    vertical = labeling.add_vertical_barrier(events, stamps, num_days=2)
    found = labeling.get_events(
        stamps, close, events, (WIDTH, WIDTH), [t for t, _ in vol], [v for _, v in vol], 0.005,
        vertical_barrier_times=vertical,
    )
    bins = {b[0]: b for b in labeling.get_bins(found, stamps, close)}
    by_start = {f[0]: f for f in found}
    index = {t: i for i, t in enumerate(stamps)}

    for name, theme in THEMES.items():
        ch = Chart(760, 300, theme, "Two triple-barrier events: one ends at the profit barrier, one at the time limit")
        for k, (when, title) in enumerate(PANELS):
            left = 56 + k * 372
            x = ch.scale(0, HORIZON, left, left + 300)
            y = ch.scale(-2.2, 2.2, 262, 40)
            event, (_, ret, target, label, _) = by_start[when], bins[when]
            i0, i1 = index[when], index[event[1]]
            path = [(close[i] / close[i0] - 1) / target for i in range(i0, i1 + 1)]

            ch.label(left, 28, title, "muted")
            ch.rule(left, y(0), left + 300, y(0))
            ch.rule(left, y(WIDTH), left + 300, y(WIDTH), "text", 1.4)
            ch.rule(left, y(-WIDTH), left + 300, y(-WIDTH), "text", 1.4)
            ch.rule(x(HORIZON), y(-WIDTH), x(HORIZON), y(WIDTH), "text", 1.4)
            ch.text(left - 8, y(WIDTH) + 4, "+1.5", size=11, anchor="end", family=MONO)
            ch.text(left - 8, y(-WIDTH) + 4, "−1.5", size=11, anchor="end", family=MONO)
            ch.text(left - 8, y(0) + 4, "0", size=11, anchor="end", family=MONO)
            ch.line([(x(h), y(v)) for h, v in enumerate(path)], "text", 1.3)
            ch.dot(x(len(path) - 1), y(path[-1]), 4)
            # Above a barrier touch; otherwise in the clear band under the path.
            ch.text(x(len(path) - 1) - 8, y(path[-1]) - 10 if path[-1] >= 1 else y(-1.0),
                    f"label {label:+d} · {ret / target:+.2f} targets", "accent", 12, "end", weight=600)
            for h in (0, 24, 48):
                ch.text(x(h), 284, f"{h}h", size=11, anchor="middle", family=MONO)
        print(ch.save("ch3-triple-barrier", name))


def bet_size_figure():
    probs = [0.5 + i * 0.0025 for i in range(200)]  # stop short of p = 1, where z is infinite
    sizes = bet_sizing.get_signal(probs, 2)
    w = bet_sizing.get_w_sigmoid(10.0, 0.95)
    gaps = [i * 0.25 - 20 for i in range(161)]
    dynamic = [bet_sizing.bet_size_sigmoid(w, g) for g in gaps]

    for name, theme in THEMES.items():
        ch = Chart(760, 300, theme, "Bet size as a function of predicted probability, and of forecast divergence")
        for k, (title, xs, ys, xlo, xhi, ylo, ticks, fmt) in enumerate([
            ("size from a probability · 2 classes", probs, sizes, 0.5, 1.0, 0.0, (0.5, 0.75, 1.0), "{:.2f}"),
            ("size from forecast − market", gaps, dynamic, -20.0, 20.0, -1.0, (-20, -10, 0, 10, 20), "{:+.0f}"),
        ]):
            left = 56 + k * 372
            x = ch.scale(xlo, xhi, left, left + 300)
            y = ch.scale(ylo, 1.0, 262, 40)
            ch.label(left, 28, title)
            ch.rule(left, y(ylo), left + 300, y(ylo))
            ch.rule(left, y(1.0), left + 300, y(1.0))
            if ylo < 0:
                ch.rule(left, y(0), left + 300, y(0))
            for v in ((0.0, 0.5, 1.0) if ylo == 0 else (-1.0, 0.0, 1.0)):
                ch.text(left - 8, y(v) + 4, "0" if v == 0 else f"{v:+.1f}" if ylo < 0 else f"{v:.1f}", size=11, anchor="end", family=MONO)
            ch.line([(x(a), y(b)) for a, b in zip(xs, ys)], "accent", 1.8)
            for t in ticks:
                ch.text(x(t), 284, "0" if t == 0 else fmt.format(t), size=11, anchor="middle", family=MONO)
        # The calibration point of bet_size_dynamic: a divergence of 10 is a 0.95 bet.
        x = ch.scale(-20.0, 20.0, 56 + 372, 56 + 372 + 300)
        y = ch.scale(-1.0, 1.0, 262, 40)
        ch.dot(x(10), y(0.95), 3.6, "text")
        ch.text(x(10) + 6, y(0.95) + 18, "(10, 0.95)", "muted", 11, "start", MONO)
        print(ch.save("ch10-bet-size", name))


if __name__ == "__main__":
    main()
    bet_size_figure()
