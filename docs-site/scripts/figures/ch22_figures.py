"""Figure for the streaming_hpc module page: VPIN, its CDF, HHI and the alert around a synthetic crash.

    .venv/bin/python docs-site/scripts/figures/ch22_figures.py
"""

from __future__ import annotations

from _svg import MONO, THEMES, Chart
from openquant import streaming_hpc

VPIN_CDF_THRESHOLD, HHI_THRESHOLD = 0.99, 0.5
FIRST, LAST = 650, 800


def early_warning_figure():
    events = streaming_hpc.generate_synthetic_flash_crash_stream(
        events=1000, crash_start_fraction=0.7, calm_venues=4, shock_venue=0
    )
    snaps = streaming_hpc.run_streaming_pipeline(
        events,
        bucket_volume=1000.0,
        support_buckets=10,
        lookback_events=50,
        cdf_lookback=100,
        vpin_cdf_threshold=VPIN_CDF_THRESHOLD,
        hhi_threshold=HHI_THRESHOLD,
    )["snapshots"]
    window = range(FIRST, LAST + 1)
    alerts = [i for i in window if snaps[i][5]]
    first_alert, last_alert = alerts[0], alerts[-1]

    for name, theme in THEMES.items():
        ch = Chart(760, 330, theme, "VPIN, the CDF of VPIN and HHI around a synthetic flash crash, with the thresholds and the alert region")
        left, right = 64, 640
        x = ch.scale(FIRST, LAST, left, right)
        y = ch.scale(0.0, 1.0, 286, 40)
        ch.label(left, 28, "vpin, cdf(vpin) and hhi by event")
        ch.band(x(first_alert), x(last_alert + 1), y(1.0), y(0.0), "rule", 0.5)
        ch.text(x(last_alert + 1) + 6, y(0.08), "alert", "muted", 11, "start", MONO)
        for v in (0.0, 0.25, 0.5, 0.75, 1.0):
            ch.rule(left, y(v), right, y(v))
            ch.text(left - 8, y(v) + 4, f"{v:.2f}", size=11, anchor="end", family=MONO)
        for e in range(FIRST, LAST + 1, 25):
            ch.text(x(e), 308, str(e), size=11, anchor="middle", family=MONO)
        ch.rule(x(700), y(1.0), x(700), y(0.0), "muted")
        ch.text(x(700) - 6, y(0.93), "crash", "muted", 11, "end", MONO)
        ch.rule(left, y(VPIN_CDF_THRESHOLD), right, y(VPIN_CDF_THRESHOLD), "accent", 0.8)
        ch.rule(left, y(HHI_THRESHOLD), right, y(HHI_THRESHOLD), "text", 0.8)
        ch.line([(x(i), y(snaps[i][2])) for i in window], "muted", 1.2)
        ch.line([(x(i), y(snaps[i][6])) for i in window], "accent", 2.0)
        ch.line([(x(i), y(snaps[i][3])) for i in window], "text", 2.0)
        ch.text(right + 8, y(snaps[LAST][2]) + 4, "VPIN", "muted", 11, "start", MONO)
        ch.text(right + 8, y(snaps[LAST][6]) + 4, "CDF(VPIN)", "accent", 11, "start", MONO)
        ch.text(right + 8, y(1.0) - 6, "HHI", "text", 11, "start", MONO)
        ch.text(right + 8, y(VPIN_CDF_THRESHOLD) + 10, "0.99", "accent", 11, "start", MONO)
        ch.text(right + 8, y(HHI_THRESHOLD) + 4, "0.5", "text", 11, "start", MONO)
        print(ch.save("ch22-early-warning", name))


if __name__ == "__main__":
    early_warning_figure()
