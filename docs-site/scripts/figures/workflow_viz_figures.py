"""Reproduce the viz page's illustrative equity/drawdown figure.

    python3 docs-site/scripts/figures/workflow_viz_figures.py

Inputs match the executable example on modules/viz.md. Drawdown is computed here
from its definition, independently of openquant.viz: equity / running high - 1.
This is a synthetic three-point illustration, not backtest performance.
"""
from _svg import MONO, THEMES, Chart


def drawdown_figure():
    equity = [100.0, 99.0, 101.0]
    drawdown = [value / max(equity[: i + 1]) - 1.0 for i, value in enumerate(equity)]
    for name, theme in THEMES.items():
        chart = Chart(760, 290, theme, "Illustrative equity and fractional drawdown across three dates")
        for panel, title, values, ticks, bounds in (
            (0, "Equity", equity, [99, 100, 101], (98.8, 101.2)),
            (1, "Drawdown (%)", [100 * v for v in drawdown], [-1, -0.5, 0], (-1.1, 0.1)),
        ):
            left = 65 + panel * 380
            right = left + 255
            x = chart.scale(0, 2, left, right)
            y = chart.scale(*bounds, 228, 55)
            chart.label(left, 28, title)
            for tick in ticks:
                chart.rule(left, y(tick), right, y(tick))
                chart.text(left - 10, y(tick) + 4, f"{tick:g}", anchor="end", family=MONO)
            chart.line([(x(i), y(v)) for i, v in enumerate(values)], "accent", 2)
            for i, value in enumerate(values):
                chart.dot(x(i), y(value))
                chart.text(x(i), 257, f"Jan {i + 2}", anchor="middle", family=MONO)
        print(chart.save("workflow-viz-drawdown", name))


if __name__ == "__main__":
    drawdown_figure()
