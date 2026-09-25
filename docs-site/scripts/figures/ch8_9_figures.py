"""Figures for the Chapter 8-9 module pages (feature_diagnostics, hyperparameter_tuning).

    .venv/bin/python docs-site/scripts/figures/ch8_9_figures.py
"""

from __future__ import annotations

import math
import random

from _svg import MONO, THEMES, Chart
from openquant import feature_diagnostics as fd

# hyperparameter_tuning has no Python binding. These are the grid-search means asserted by
# crates/openquant/tests/docs_ch8_9_examples.rs; if that test changes, change these.
K = [0.5, 2.0, 4.0, 8.0, 32.0]
NEG_LOG_LOSS = [-0.5941, -0.3933, -0.2745, -0.2243, -0.4711]
ACCURACY = [0.8950] * 5


def importance_figure():
    # The example on the feature_diagnostics page, verbatim.
    rng = random.Random(8)
    n = 1200
    a = [rng.gauss(0, 1) for _ in range(n)]
    b = [rng.gauss(0, 1) for _ in range(n)]
    names = ["a", "a_copy", "b", "noise_1", "noise_2"]
    X = [[a[i], a[i] + rng.gauss(0, 0.2), b[i], rng.gauss(0, 1), rng.gauss(0, 1)] for i in range(n)]
    y = [1.0 if a[i] + 0.6 * b[i] + rng.gauss(0, 0.7) > 0 else 0.0 for i in range(n)]
    ends = [min(i + 5, n - 1) for i in range(n)]
    common = dict(feature_names=names, event_end_indices=ends, n_splits=5, pct_embargo=0.01)
    mean = lambda report: {r["feature"]: r["mean"] for r in report["records"]}  # noqa: E731
    mdi, mda, sfi = mean(fd.mdi_importance(X, y, feature_names=names)), mean(fd.mda_importance(X, y, **common)), mean(fd.sfi_importance(X, y, **common))
    coin = -math.log(2)
    panels = [
        ("MDI · coefficient share", [mdi[f] for f in names]),
        ("MDA · loss of score", [mda[f] for f in names]),
        ("SFI · gain over a coin flip", [sfi[f] - coin for f in names]),
    ]

    for name, theme in THEMES.items():
        ch = Chart(760, 250, theme, "Importance of five features under three methods")
        for k, (title, values) in enumerate(panels):
            left = 84 + k * 232
            width = 130
            top = max(values)
            ch.label(left - 60, 28, title)
            zero = left + (width * 0.25 if min(values) < -1e-3 else 0)
            x = lambda v: round(zero + v / top * (left + width - zero), 1)  # noqa: E731
            ch.rule(zero, 44, zero, 44 + 5 * 36)
            for row, (feature, v) in enumerate(zip(names, values)):
                yy = 52 + row * 36
                if k == 0:
                    ch.text(left - 60, yy + 14, feature, size=12, family=MONO)
                lo, hi = sorted((zero, x(v)))
                ch.band(lo, max(hi, lo + 1), yy, yy + 20, "accent", 1.0)
                ch.text((hi if v >= 0 else zero) + 6, yy + 14, f"{round(v, 3) + 0.0:+.3f}" if k == 2 else f"{round(v, 3) + 0.0:.3f}", size=11, family=MONO)
        print(ch.save("ch8-importance", name))


def scoring_figure():
    for name, theme in THEMES.items():
        ch = Chart(760, 300, theme, "Cross-validated score against sharpness k under accuracy and negative log loss")
        left, right = 72, 600
        x = lambda k: round(left + (math.log(k) - math.log(0.5)) / (math.log(32) - math.log(0.5)) * (right - left), 1)  # noqa: E731
        y = ch.scale(-0.7, 1.0, 256, 40)
        ch.label(left, 28, "mean score over five purged folds")
        for v in (-0.5, 0.0, 0.5, 1.0):
            ch.rule(left, y(v), right, y(v))
            ch.text(left - 8, y(v) + 4, "0" if v == 0 else f"{v:+.1f}", size=11, anchor="end", family=MONO)
        for series, color, label in ((ACCURACY, "text", "accuracy"), (NEG_LOG_LOSS, "accent", "neg log loss")):
            ch.line([(x(k), y(v)) for k, v in zip(K, series)], color, 1.8)
            for k, v in zip(K, series):
                ch.dot(x(k), y(v), 3.2, color)
            ch.text(right + 10, y(series[-1]) + 4, label, "muted", 11, "start", MONO)
        best = NEG_LOG_LOSS.index(max(NEG_LOG_LOSS))
        ch.text(x(K[best]), y(NEG_LOG_LOSS[best]) - 10, "best", "accent", 11, "middle", MONO, 600)
        for k in K:
            ch.text(x(k), 280, f"k = {k:g}", size=11, anchor="middle", family=MONO)
        print(ch.save("ch9-scoring", name))


if __name__ == "__main__":
    importance_figure()
    scoring_figure()
