"""Figures for the strategy_risk and codependence module pages.

    .venv/bin/python docs-site/scripts/figures/ch15_mlam3_figures.py
"""

from __future__ import annotations

import math
import random

from _svg import MONO, THEMES, Chart
from openquant import codependence as cd
from openquant import strategy_risk as sr


def dependence_figure():
    # The example on the codependence page, verbatim, thinned to 250 points for drawing.
    rng = random.Random(6)
    n = 1000
    x = [rng.gauss(0, 1) for _ in range(n)]
    cases = {
        "linear": [0.8 * v + 0.6 * rng.gauss(0, 1) for v in x],
        "negative": [-0.8 * v + 0.6 * rng.gauss(0, 1) for v in x],
        "parabola": [v * v + 0.3 * rng.gauss(0, 1) for v in x],
        "independent": [rng.gauss(0, 1) for _ in x],
    }

    def pearson(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        cov = sum((p - ma) * (q - mb) for p, q in zip(a, b))
        return cov / math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))

    for name, theme in THEMES.items():
        ch = Chart(760, 250, theme, "Scatter plots of four relationships with their correlation, distance correlation and normalised mutual information")
        for k, (title, y) in enumerate(cases.items()):
            left, size = 24 + k * 184, 160
            sx = ch.scale(-3.2, 3.2, left, left + size)
            lo, hi = (-1.5, 9.0) if title == "parabola" else (-3.6, 3.6)
            sy = ch.scale(lo, hi, 44 + size, 44)
            ch.label(left, 28, title)
            ch.band(left, left + size, 44, 44 + size, "rule", 0.25)
            for a, b in list(zip(x, y))[::4]:
                if -3.2 < a < 3.2 and lo < b < hi:
                    ch.dot(sx(a), sy(b), 1.6, "accent")
            stats = (("corr", f"{pearson(x, y):+.2f}"), ("dcor", f"{cd.distance_correlation(x, y):.2f}"),
                     ("nmi", f"{cd.get_mutual_info(x, y, None, True):.2f}"))
            for j, (label, value) in enumerate(stats):
                ch.text(left + j * 58, 228, label, size=10.5, family=MONO)
                ch.text(left + j * 58, 242, value, "text", 12, family=MONO, weight=600)
        print(ch.save("mlam3-dependence", name))


def precision_figure():
    freqs = [10 ** (1 + i * 2.6 / 80) for i in range(81)]  # 10 to about 4,000 bets a year
    curves = {
        "symmetric": [sr.implied_precision_symmetric(2.0, n) for n in freqs],
        "+1% / −2%": [sr.implied_precision_asymmetric(2.0, n, 0.01, -0.02) for n in freqs],
    }
    for name, theme in THEMES.items():
        ch = Chart(760, 300, theme, "Precision needed for an annualised Sharpe ratio of 2 against the number of bets per year, for symmetric and asymmetric payouts")
        left, right = 56, 600
        x = lambda n: round(left + (math.log10(n) - 1) / 2.6 * (right - left), 1)  # noqa: E731
        y = ch.scale(0.5, 0.9, 256, 40)
        ch.label(left, 28, "precision needed for a sharpe ratio of 2")
        for v in (0.5, 0.6, 0.7, 0.8, 0.9):
            ch.rule(left, y(v), right, y(v))
            ch.text(left - 8, y(v) + 4, f"{v:.1f}", size=11, anchor="end", family=MONO)
        # Break-even precision for the asymmetric payout: 2 / 3.
        ch.rule(left, y(2 / 3), right, y(2 / 3), "muted", 1.0)
        ch.text(right + 10, y(2 / 3) + 4, "break-even 0.667", "muted", 11, "start", MONO)
        for (label, ps), color in zip(curves.items(), ("text", "accent")):
            ch.line([(x(n), y(p)) for n, p in zip(freqs, ps) if p <= 0.9], color, 1.8)
            ch.text(right + 10, y(ps[-1]) + (-14 if label != "symmetric" else 4), label, "muted", 11, "start", MONO)
        for n, tick in ((12, "monthly"), (52, "weekly"), (260, "daily"), (2600, "10 a day")):
            ch.text(x(n), 280, tick, size=11, anchor="middle", family=MONO)
        print(ch.save("ch15-precision", name))


if __name__ == "__main__":
    dependence_figure()
    precision_figure()
