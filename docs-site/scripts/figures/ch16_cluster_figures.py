"""Figure for the onc module page: a correlation matrix before and after clustering.

    .venv/bin/python docs-site/scripts/figures/ch16_cluster_figures.py
"""

from __future__ import annotations

import random

from _svg import MONO, THEMES, Chart
from openquant import onc


def onc_figure():
    # The example on the onc page, verbatim.
    rng = random.Random(7)
    group_of = [0] * 5 + [1] * 4 + [2] * 3
    rng.shuffle(group_of)
    factors = [[rng.gauss(0, 1) for _ in range(600)] for _ in range(3)]
    series = [[0.8 * f + 0.6 * rng.gauss(0, 1) for f in factors[g]] for g in group_of]

    def corr(a, b):
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        return cov / (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5

    matrix = [[corr(a, b) for b in series] for a in series]
    result = onc.get_onc_clusters(matrix, 5)
    ordered = [i for _, members in sorted(result["clusters"].items()) for i in members]

    panels = (("as given", list(range(12))), ("ordered by onc cluster", ordered))
    for name, theme in THEMES.items():
        ch = Chart(760, 330, theme, "A twelve by twelve correlation matrix as given and reordered by ONC cluster")
        for k, (title, order) in enumerate(panels):
            left, top, cell = 56 + k * 360, 48, 22
            ch.label(left, 28, title)
            for r, i in enumerate(order):
                ch.text(left - 8, top + r * cell + 15, str(i), size=10, anchor="end", family=MONO)
                ch.text(left + r * cell + cell / 2 - 1, top + 12 * cell + 14, str(i), size=10, anchor="middle", family=MONO)
                for c, j in enumerate(order):
                    x, y = left + c * cell, top + r * cell
                    ch.band(x, x + cell - 2, y, y + cell - 2, "rule", 0.3)
                    ch.band(x, x + cell - 2, y, y + cell - 2, "accent", round(max(matrix[i][j], 0.0), 3))
        print(ch.save("mlam4-onc", name))


if __name__ == "__main__":
    onc_figure()
