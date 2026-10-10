"""Plot the four library-scored candidates in the dynamic-allocation page example.

Run with an interpreter containing openquant:
    python docs-site/scripts/figures/ch21_allocation_figures.py
"""

from _svg import MONO, THEMES, Chart
from openquant import dynamic_allocation as da


def allocation_figure():
    means = [[0.10, -0.05]]
    covariances = [[[0.04, 0.0], [0.0, 0.04]]]
    costs = [[0.01, 0.01]]
    weights = da.all_weights(1, 2)
    scores = [
        da.trajectory_sharpe_ratio([weight], means, covariances, costs, [0.0, 0.0])
        for weight in weights
    ]
    best = da.dynamic_optimal_portfolio(means, covariances, costs, k=1)
    labels = ["Short asset 0", "Long asset 0", "Short asset 1", "Long asset 1"]
    for name, theme in THEMES.items():
        chart = Chart(760, 330, theme, "Net Sharpe of four synthetic portfolio candidates")
        x = chart.scale(-0.6, 0.6, 220, 690)
        chart.label(26, 30, "One horizon / two assets / K = 1")
        chart.text(26, 52, "Synthetic forecasts; every entry costs 0.01")
        for tick in (-0.6, -0.3, 0.0, 0.3, 0.6):
            chart.rule(x(tick), 72, x(tick), 266)
            chart.text(x(tick), 287, f"{tick:.1f}", anchor="middle", family=MONO)
        chart.rule(x(0), 72, x(0), 266, "muted")
        for index, (label, weight, score) in enumerate(zip(labels, weights, scores)):
            y = 95 + index * 48
            color = "accent" if [weight] == best["weights"] else "muted"
            chart.text(150, y + 4, label, anchor="end")
            chart.band(min(x(0), x(score)), max(x(0), x(score)), y - 12, y + 12, color, 0.8)
            chart.text(
                x(score) + (8 if score > 0 else -8),
                y + 4,
                f"{score:+.2f}",
                color,
                anchor="start" if score > 0 else "end",
                family=MONO,
            )
        chart.text(430, 314, "Forecast Sharpe after transaction costs", anchor="middle")
        print(chart.save("ch21-allocation", name))


if __name__ == "__main__":
    allocation_figure()
