"""Figures for the Chapter 4 module pages (sampling, sample_weights).

Run from the repo root with the built extension on the path:

    .venv/bin/python docs-site/scripts/figures/ch4_figures.py
"""

from __future__ import annotations

from datetime import datetime, timedelta

from _svg import MONO, THEMES, Chart
from openquant import sample_weights, sampling

# AFML §4.5.3's example: three labels over six bars.
SPANS = [(0, 2), (2, 3), (4, 5)]
N_BARS = 6


def concurrency_figure():
    ind = [list(row) for row in sampling.get_ind_matrix(SPANS, list(range(N_BARS)))]
    concurrency = [sum(row) for row in ind]
    uniqueness = sampling.get_av_uniqueness_from_triple_barrier(SPANS, N_BARS)

    for name, theme in THEMES.items():
        ch = Chart(760, 250, theme, "Three labels over six bars, the number of labels alive at each bar, and each label's average uniqueness")
        left, right = 120, 600
        cell = (right - left) / N_BARS
        x = lambda bar: round(left + bar * cell, 1)  # noqa: E731 - left edge of a bar's cell
        ch.label(left, 28, "label spans")
        ch.label(right + 24, 28, "avg. uniqueness")
        base = 52 + 3 * 34 + 12
        for bar in range(1, N_BARS):  # grid first, so the spans sit on top of it
            ch.rule(x(bar), 44, x(bar), base + 32)
        for k, ((start, end), u) in enumerate(zip(SPANS, uniqueness)):
            y = 52 + k * 34
            ch.text(left - 12, y + 15, f"label {k}", size=12, anchor="end")
            ch.band(x(start) + 3, x(end + 1) - 3, y, y + 22, "accent", 1.0)
            ch.text(right + 24, y + 15, f"{u:.3f}", "text", 12, family=MONO)
        ch.rule(left, base, right, base)
        ch.text(left - 12, base + 24, "concurrent", size=12, anchor="end")
        for bar, c in enumerate(concurrency):
            ch.text(x(bar) + cell / 2, base + 24, str(c), "accent" if c > 1 else "text", 13, "middle", MONO, 600 if c > 1 else 400)
            ch.text(x(bar) + cell / 2, base + 50, f"bar {bar}", size=11, anchor="middle", family=MONO)
        print(ch.save("ch4-concurrency", name))


def time_decay_figure():
    # Twenty back-to-back labels that never overlap, so cumulative uniqueness is just the count
    # and the curves show the decay function alone.
    start = datetime(2024, 1, 2, 9, 30)
    stamps = [(start + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(40)]
    close = [100.0 + 0.1 * i for i in range(40)]
    events = [(stamps[2 * i], stamps[2 * i + 1], 1.0) for i in range(20)]
    curves = {c: [w for _, w in sample_weights.get_weights_by_time_decay(events, stamps, close, c)]
              for c in (1.0, 0.5, 0.0, -0.5)}

    for name, theme in THEMES.items():
        ch = Chart(760, 300, theme, "Time-decay weights from the oldest label to the newest, for four values of the decay parameter")
        left, right = 56, 600
        x = ch.scale(0, 19, left, right)
        y = ch.scale(0.0, 1.0, 256, 40)
        ch.label(left, 28, "weight by label age")
        for v in (0.0, 0.5, 1.0):
            ch.rule(left, y(v), right, y(v))
            ch.text(left - 8, y(v) + 4, "0" if v == 0 else f"{v:.1f}", size=11, anchor="end", family=MONO)
        # (index to label at, dx, dy, anchor): above-left of a rising line, below a flat one.
        spots = {1.0: (10, 0, 16, "middle"), 0.5: (5, -6, -6, "end"), 0.0: (5, -6, -6, "end"), -0.5: (12, 8, 14, "start")}
        for c, ws in curves.items():
            ch.line([(x(i), y(w)) for i, w in enumerate(ws)], "accent" if c != 1.0 else "text", 1.8)
            i, dx, dy, anchor = spots[c]
            ch.text(x(i) + dx, y(ws[i]) + dy, f"c = {c:g}", "muted", 11, anchor, MONO)
        ch.text(left, 280, "oldest", size=11, family=MONO)
        ch.text(right, 280, "newest", size=11, anchor="end", family=MONO)
        print(ch.save("ch4-time-decay", name))


if __name__ == "__main__":
    concurrency_figure()
    time_decay_figure()
