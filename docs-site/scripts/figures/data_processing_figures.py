"""Reproduce: python3 docs-site/scripts/figures/data_processing_figures.py.

Synthetic AAA closes reproduce the module example in
crates/openquant/src/data_processing.rs (clean_ohlcv_rows / align_calendar_rows).
Behavior is covered by crates/openquant/tests/data_processing.rs:
dedupe_keeps_input_order_and_alignment_output_is_sorted,
gap_count_follows_the_bar_frequency, and align_calendar_reports_off_grid_bars.

The standard-library calculation below independently derives keep-last rows,
the modal spacing (smallest on a tie), weekday-gap intervals, and missing slots.
No market data, network, interpolation, or openquant runtime is required.
"""

from collections import Counter
from datetime import date, timedelta

from _svg import MONO, THEMES, Chart

RAW = [(date(2024, 1, day), close) for day, close in [(2, 10), (2, 10.5), (3, 11), (5, 12)]]
CLEAN = sorted(dict(RAW).items())  # Last input occurrence wins for each AAA/date key.
REMOVED = len(RAW) - len(CLEAN)
SPACINGS = Counter(right[0] - left[0] for left, right in zip(CLEAN, CLEAN[1:]))
INTERVAL = min(SPACINGS, key=lambda spacing: (-SPACINGS[spacing], spacing))
GAPS = sum(
    any(
        (left[0] + timedelta(days=day)).weekday() < 5 for day in range(1, (right[0] - left[0]).days)
    )
    for left, right in zip(CLEAN, CLEAN[1:])
)
GRID = [
    CLEAN[0][0] + INTERVAL * step for step in range((CLEAN[-1][0] - CLEAN[0][0]) // INTERVAL + 1)
]
ALIGNED = [(day, dict(CLEAN).get(day)) for day in GRID]

assert REMOVED == 1 and GAPS == 1 and INTERVAL == timedelta(days=1)
assert [close for _, close in CLEAN] == [10.5, 11, 12]
assert [close for _, close in ALIGNED] == [10.5, 11, None, 12]


def table(chart, x, rows, *, missing_column=False, raw=False):
    """Tabular values avoid drawing a price path across the missing observation."""
    chart.label(x, 128, "UTC date")
    chart.label(x + 134, 128, "Close")
    if missing_column:
        chart.label(x + 202, 128, "Missing?")
    for index, (day, close) in enumerate(rows):
        y = 163 + index * 35
        missing = close is None
        if missing or (raw and index < 2):
            chart.band(
                x - 8,
                x + (280 if missing_column else 190),
                y - 22,
                y + 10,
                color="accent",
                opacity=0.10,
            )
        chart.text(x, y, day.strftime("Jan %d"), color="text", size=15, family=MONO)
        chart.text(
            x + 134,
            y,
            "null" if missing else f"{close:g}",
            color="accent" if missing else "text",
            size=15,
            family=MONO,
        )
        if missing_column:
            chart.text(
                x + 202,
                y,
                str(missing).lower(),
                color="accent" if missing else "muted",
                size=15,
                family=MONO,
            )
        if raw and index == 0:
            chart.rule(x + 132, y - 5, x + 156, y - 5, color="accent", width=1.5)


for theme_name, theme in THEMES.items():
    c = Chart(
        840, 395, theme, "Synthetic AAA bars: keep the last duplicate and mark January 4 as missing"
    )
    c.text(28, 32, "Prepare the data, preserve the gaps", color="text", size=21, weight=600)
    c.text(28, 57, "Synthetic AAA closes · January 2024 · all timestamps at 00:00 UTC", size=13)

    for x, title, subtitle in [
        (28, "1  Raw rows", "Repeated (symbol, timestamp)"),
        (283, "2  Keep last", "Sorted unique keys"),
        (535, "3  Daily grid", "86,400-second spacing"),
    ]:
        c.text(x, 89, title, color="text", size=17, weight=600)
        c.text(x, 108, subtitle, size=12)
    for x in [245, 497]:
        c.line([(x - 10, 192), (x + 12, 192)], color="accent", width=1.8)
        c.line([(x + 5, 186), (x + 12, 192), (x + 5, 198)], color="accent", width=1.8)

    table(c, 28, RAW, raw=True)
    table(c, 283, CLEAN)
    table(c, 535, ALIGNED, missing_column=True)
    c.text(28, 296, "10.5 is the later Jan 2 row.", size=12)
    c.text(283, 296, f"{REMOVED} duplicate removed · {GAPS} weekday-gap interval", size=11)
    c.text(535, 296, "Missing? = is_missing_bar", size=12)
    c.rule(28, 315, 812, 315)
    c.text(
        28,
        341,
        "Jan 4 is absent: OHLCV fields stay null (Rust: None), never zero.",
        color="text",
        size=15,
    )
    c.text(28, 367, "The missing-bar flag is true. Alignment does not interpolate prices.", size=14)
    c.save("data-processing", theme_name)
