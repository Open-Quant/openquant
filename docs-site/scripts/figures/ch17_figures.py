"""Figure for the structural_breaks module page.

    .venv/bin/python docs-site/scripts/figures/ch17_figures.py
"""

from __future__ import annotations

import math
import random

from _svg import MONO, THEMES, Chart
from openquant import structural_breaks as sb


def sadf_figure():
    # The example on the structural_breaks page, verbatim.
    rng = random.Random(14)
    y = [math.log(100.0)]
    for t in range(1, 300):
        step = rng.gauss(0, 0.01)
        if 180 <= t < 240:
            step += 0.001 * 1.06 ** (t - 180)
        elif t == 240:
            step -= 2 / 3 * (y[-1] - y[179])
        y.append(y[-1] + step)
    sadf = sb.get_sadf(y, "linear", True, 30, 1)
    offset = len(y) - len(sadf)

    for name, theme in THEMES.items():
        ch = Chart(760, 360, theme, "A simulated log price with a run-up and crash, and its SADF statistic")
        left, right = 56, 704
        x = ch.scale(0, 299, left, right)
        ch.band(x(180), x(240), 40, 320, "rule", 0.35)

        yp = ch.scale(min(y), max(y), 150, 44)
        ch.label(left, 28, "log price")
        ch.line([(x(t), yp(v)) for t, v in enumerate(y)], "text", 1.3)

        ys = ch.scale(-3.0, 4.0, 320, 190)
        ch.label(left, 180, "sadf")
        for v in (-2.0, 0.0, 2.0, 4.0):
            ch.rule(left, ys(v), right, ys(v))
            ch.text(left - 8, ys(v) + 4, "0" if v == 0 else f"{v:+.0f}", size=11, anchor="end", family=MONO)
        ch.rule(left, ys(1.5), right, ys(1.5), "muted", 1.0)
        ch.text(right + 6, ys(1.5) + 4, "1.5", "muted", 11, "start", MONO)
        ch.line([(x(i + offset), ys(max(v, -3.0))) for i, v in enumerate(sadf)], "accent", 1.6)
        for t in (0, 60, 120, 180, 240, 299):
            ch.text(x(t), 342, str(t), size=11, anchor="middle", family=MONO)
        print(ch.save("ch17-sadf", name))


if __name__ == "__main__":
    sadf_figure()
