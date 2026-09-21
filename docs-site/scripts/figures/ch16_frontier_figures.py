"""Figure for the cla module page: the efficient frontier and its turning points.

    .venv/bin/python docs-site/scripts/figures/ch16_frontier_figures.py
"""

from __future__ import annotations

from _svg import MONO, THEMES, Chart
from openquant import cla

NAMES = ["bonds", "equity", "small cap", "gold"]
MU = [0.03, 0.07, 0.09, 0.04]
VOL = [0.05, 0.16, 0.22, 0.15]
RHO = [[1.0, 0.1, 0.1, 0.1], [0.1, 1.0, 0.8, 0.0], [0.1, 0.8, 1.0, 0.0], [0.1, 0.0, 0.0, 1.0]]
COV = [[RHO[i][j] * VOL[i] * VOL[j] for j in range(4)] for i in range(4)]


def stats(w):
    ret = sum(a * b for a, b in zip(w, MU))
    risk = sum(w[i] * COV[i][j] * w[j] for i in range(4) for j in range(4)) ** 0.5
    return risk, ret


def frontier_figure():
    frontier = cla.allocate_cla(expected_returns=MU, covariance_matrix=COV, solution="efficient_frontier")
    curve = list(zip(frontier["efficient_frontier_sigma"], frontier["efficient_frontier_means"]))
    turning = [stats(w) for w in cla.allocate_cla(expected_returns=MU, covariance_matrix=COV)["weights"]]
    best = stats(cla.allocate_cla(expected_returns=MU, covariance_matrix=COV, solution="max_sharpe")["weights"][0])

    for name, theme in THEMES.items():
        ch = Chart(760, 330, theme, "The long-only efficient frontier of four assets, with the turning points of the critical line algorithm")
        left, right = 64, 620
        x = ch.scale(0.0, 0.24, left, right)
        y = ch.scale(0.02, 0.10, 286, 40)
        ch.label(left, 28, "expected return against volatility")
        for v in (0.02, 0.04, 0.06, 0.08, 0.10):
            ch.rule(left, y(v), right, y(v))
            ch.text(left - 8, y(v) + 4, f"{v:.0%}", size=11, anchor="end", family=MONO)
        for v in (0.0, 0.06, 0.12, 0.18, 0.24):
            ch.text(x(v), 308, f"{v:.0%}", size=11, anchor="middle", family=MONO)
        ch.line([(x(s), y(m)) for s, m in curve], "accent", 2.0)
        for s, m in turning:
            ch.dot(x(s), y(m), 4.2, "accent")
        ch.dot(x(best[0]), y(best[1]), 4.2, "text")
        ch.text(x(best[0]) + 10, y(best[1]) + 14, "max Sharpe", "muted", 11, "start", MONO)
        for asset, s, m in zip(NAMES, VOL, MU):
            ch.dot(x(s), y(m), 3.0, "muted")
            ch.text(x(s) + 8, y(m) + 4, asset, "muted", 11, "start", MONO)
        print(ch.save("ch16-frontier", name))


if __name__ == "__main__":
    frontier_figure()
