"""Figures for the Chapter 5-7 module pages (fracdiff, ensemble_methods, cross_validation).

    .venv/bin/python docs-site/scripts/figures/ch5_7_figures.py
"""

from __future__ import annotations

from _svg import MONO, THEMES, Chart
from openquant import ensemble, fracdiff


def weights_figure():
    lags = 8
    curves = {d: list(reversed(fracdiff.get_weights(d, lags + 1))) for d in (0.2, 0.5, 0.8, 1.0)}
    for name, theme in THEMES.items():
        ch = Chart(760, 300, theme, "Fractional differencing weights at lags 1 to 8 for d = 0.2, 0.5, 0.8 and 1")
        left, right = 120, 600
        # Lag 0 is left out: its weight is 1 for every d, and it would flatten everything else.
        x = ch.scale(1, lags, left, right)
        y = ch.scale(-1.0, 0.0, 256, 40)
        ch.label(56, 28, "weight on the value k bars ago · the weight at k = 0 is always 1")
        for v in (-1.0, -0.5, 0.0):
            ch.rule(left, y(v), right, y(v))
            ch.text(right + 10, y(v) + 4, "0" if v == 0 else f"{v:+.1f}", size=11, family=MONO)
        for d, ws in curves.items():
            color = "text" if d == 1.0 else "accent"
            ch.line([(x(k), y(ws[k])) for k in range(1, lags + 1)], color, 1.6)
            for k in range(1, lags + 1):
                ch.dot(x(k), y(ws[k]), 2.6, color)
            ch.text(left - 10, y(ws[1]) + 4, f"d = {d:g}", "muted", 11, "end", MONO)
        for k in range(1, lags + 1):
            ch.text(x(k), 280, f"k = {k}", size=11, anchor="middle", family=MONO)
        print(ch.save("ch5-fracdiff-weights", name))


def purge_figure():
    n, test_lo, test_hi, span, embargo = 40, 16, 23, 4, 6
    for name, theme in THEMES.items():
        ch = Chart(760, 230, theme, "One fold of purged k-fold: test labels, purged labels on both sides, and the embargo")
        left, right = 56, 704
        x = ch.scale(0, n, left, right)
        row = 96
        ch.label(left, 28, "fold 3 of 5 · 40 labels of 4 bars · embargo 0.15")
        kinds = {"train": ("text", 0.8), "purged": ("muted", 0.5), "embargo": ("rule", 1.0), "test": ("accent", 1.0)}
        for i in range(n):
            if test_lo <= i <= test_hi:
                kind = "test"
            elif i + span - 1 >= test_lo and i <= test_hi + span - 1:
                kind = "purged"  # the label's span touches the window the test labels cover
            elif test_lo - embargo <= i <= test_hi + embargo:
                kind = "embargo"  # counted from the fold's edges, as PurgedKFold does
            else:
                kind = "train"
            color, opacity = kinds[kind]
            ch.band(x(i) + 1.5, x(i + 1) - 1.5, row, row + 30, color, opacity)
        for k, (kind, (color, opacity)) in enumerate(kinds.items()):
            lx = left + k * 150
            ch.band(lx, lx + 14, 168, 182, color, opacity)
            ch.text(lx + 22, 180, kind, "text", 12)
        ch.text(x(test_lo), row - 10, "test fold", "accent", 12, weight=600)
        ch.text(left, 214, "time →", size=11, family=MONO)
        print(ch.save("ch7-purged-fold", name))


def bagging_variance_figure():
    ns = list(range(1, 51))
    curves = {rho: [ensemble.bagging_ensemble_variance(1.0, rho, n) for n in ns] for rho in (0.0, 0.3, 0.6, 0.9)}
    for name, theme in THEMES.items():
        ch = Chart(760, 300, theme, "Variance of a bagged ensemble relative to one estimator, against the number of estimators, for four correlations")
        left, right = 56, 600
        x = ch.scale(1, 50, left, right)
        y = ch.scale(0.0, 1.0, 256, 40)
        ch.label(left, 28, "ensemble variance ÷ single-estimator variance")
        for v in (0.0, 0.5, 1.0):
            ch.rule(left, y(v), right, y(v))
            ch.text(left - 8, y(v) + 4, "0" if v == 0 else f"{v:.1f}", size=11, anchor="end", family=MONO)
        for rho, vs in curves.items():
            ch.line([(x(n), y(v)) for n, v in zip(ns, vs)], "accent", 1.8)
            ch.text(right + 10, y(vs[-1]) + 4, f"ρ = {rho:g}", "muted", 11, "start", MONO)
        for n in (1, 10, 20, 30, 40, 50):
            ch.text(x(n), 280, str(n), size=11, anchor="middle", family=MONO)
        ch.text(right, 296, "estimators", size=11, anchor="end", family=MONO)
        print(ch.save("ch6-bagging-variance", name))


if __name__ == "__main__":
    weights_figure()
    purge_figure()
    bagging_variance_figure()
