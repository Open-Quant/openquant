"""Figures for the Chapter 11-13 module pages (backtesting_engine, synthetic_backtesting).

    .venv/bin/python docs-site/scripts/figures/ch10_12_figures.py
"""

from __future__ import annotations

import random
from itertools import combinations

from _svg import MONO, THEMES, Chart
from openquant import synthetic_bt as sb


def cpcv_figure():
    """AFML figure 12.1: which groups each of the 15 splits tests, and the path each cell joins."""
    n_groups, k = 6, 2
    splits = list(combinations(range(n_groups), k))
    seen = [0] * n_groups
    for name, theme in THEMES.items():
        ch = Chart(760, 330, theme, "The fifteen splits of combinatorial purged cross-validation with six groups and two test groups, and the five paths they form")
        left, top, cw, rh = 150, 56, 34, 38
        ch.label(24, 28, "cpcv(6, 2) · 15 splits · each test cell is numbered with its path")
        for g in range(n_groups):
            ch.text(24, top + g * rh + 24, f"group {g}", size=12, family=MONO)
        seen = [0] * n_groups
        for s, groups in enumerate(splits):
            x = left + s * cw
            ch.text(x + cw / 2 - 2, top - 8, str(s), size=10.5, anchor="middle", family=MONO)
            for g in range(n_groups):
                y = top + g * rh
                if g in groups:
                    ch.band(x, x + cw - 4, y + 4, y + rh - 4, "accent", 1.0)
                    ch.text(x + cw / 2 - 2, y + 25, str(seen[g]), "ground", 12, "middle", MONO, 600)
                    seen[g] += 1
                else:
                    ch.band(x, x + cw - 4, y + 4, y + rh - 4, "rule", 0.45)
        ch.text(left, top + n_groups * rh + 24, "split →   filled = tested in that split, pale = available for training (before purging)", size=11, family=MONO)
        print(ch.save("ch12-cpcv", name))


def history(phi, equilibrium=100.0, sigma=1.0, n=1500, seed=3):
    rng, p, out = random.Random(seed), equilibrium, []
    for _ in range(n):
        p = (1 - phi) * equilibrium + phi * p + sigma * rng.gauss(0, 1)
        out.append(p)
    return out


def surface_figure():
    grid = [0.5, 1.0, 2.0, 4.0, 8.0]
    panels = []
    for label, phi in (("mean-reverting · φ = 0.90", 0.90), ("near random walk · φ = 0.995", 0.995)):
        res = sb.run_synthetic_otr_workflow(
            history(phi), initial_price=97.0, n_paths=4000, horizon=60, seed=11,
            profit_taking_grid=grid, stop_loss_grid=grid, max_holding_steps=59, annualization_factor=1.0)
        cells = {(p["profit_taking"], p["stop_loss"]): p["sharpe"] for p in res["response_surface"]}
        panels.append((label, cells))
    top_sharpe = max(v for _, cells in panels for v in cells.values())

    for name, theme in THEMES.items():
        ch = Chart(760, 330, theme, "Sharpe ratio of each profit-taking and stop-loss pair on simulated paths, for a mean-reverting process and a near random walk")
        for k, (label, cells) in enumerate(panels):
            left, top, cell = 84 + k * 350, 56, 46
            ch.label(left, 28, label)
            for r, sl in enumerate(reversed(grid)):
                ch.text(left - 10, top + r * cell + 28, f"{sl:g}", size=11, anchor="end", family=MONO)
                for c, pt in enumerate(grid):
                    v = cells[(pt, sl)]
                    x, y = left + c * cell, top + r * cell
                    ch.band(x, x + cell - 3, y, y + cell - 3, "rule", 0.35)
                    ch.band(x, x + cell - 3, y, y + cell - 3, "accent", round(max(v, 0.0) / top_sharpe, 3))
                    strong = v / top_sharpe > 0.55
                    ch.text(x + (cell - 3) / 2, y + 27, f"{v:.1f}" if abs(v) >= 0.05 else "0", "ground" if strong else "text", 11, "middle", MONO)
            for c, pt in enumerate(grid):
                ch.text(left + c * cell + (cell - 3) / 2, top + 5 * cell + 14, f"{pt:g}", size=11, anchor="middle", family=MONO)
            ch.text(left + 2.5 * cell, top + 5 * cell + 34, "profit-taking width", size=11, anchor="middle")
            if k == 0:
                ch.text(left - 44, top - 10, "stop", size=11)
        print(ch.save("ch13-otr-surface", name))


if __name__ == "__main__":
    cpcv_figure()
    surface_figure()
