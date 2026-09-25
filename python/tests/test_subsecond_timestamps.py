"""Sub-second timestamps cross the Python bindings intact (issue #87).

Timestamps cross the PyO3 boundary as "%Y-%m-%d %H:%M:%S" strings with an optional
fractional second. Returned strings are written as str(datetime) writes them: no fraction on
a whole second (so whole-second output is unchanged), six digits, or nine below a microsecond.
"""

from __future__ import annotations

import datetime as dt

import openquant
import polars as pl
import pytest
from openquant import _core

# Ticks around one second at microsecond and nanosecond resolution, in the form
# str(datetime) and str(pandas.Timestamp) produce.
TICKS = [
    "2024-01-02 09:30:01",
    "2024-01-02 09:30:01.120000",
    "2024-01-02 09:30:01.250500",
    "2024-01-02 09:30:01.500000",
    "2024-01-02 09:30:01.760917",
    "2024-01-02 09:30:01.999999",
    "2024-01-02 09:30:02.000000001",
    "2024-01-02 09:30:02.123456789",
]
CLOSE = [100.0, 100.6, 101.3, 100.2, 99.1, 99.6, 101.1, 102.0]


def _tick_frame(ts: list) -> pl.DataFrame:
    close = CLOSE[: len(ts)]
    return pl.DataFrame(
        {
            "ts": ts,
            "symbol": ["X"] * len(ts),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [10.0] * len(ts),
        }
    )


def _us_datetimes() -> list[dt.datetime]:
    base = dt.datetime(2024, 1, 2, 9, 30, 1)
    return [
        base + dt.timedelta(microseconds=u)
        for u in (0, 120_000, 250_500, 500_000, 760_917, 999_999)
    ]


# --- openquant.bars (the issue's reproduction) -------------------------------------------


@pytest.mark.parametrize(
    "builder, kwargs",
    [
        (openquant.bars.build_tick_bars, {"ticks_per_bar": 2}),
        (openquant.bars.build_volume_bars, {"volume_per_bar": 20.0}),
        (openquant.bars.build_dollar_bars, {"dollar_value_per_bar": 1_900.0}),
    ],
)
def test_bars_from_microsecond_datetimes_keep_the_fraction(builder, kwargs):
    # Six ticks inside one second, two per bar.
    stamps = _us_datetimes()
    out = builder(_tick_frame(stamps), **kwargs)
    assert out.height == 3
    assert out["start_ts"].to_list() == stamps[0::2]
    assert out["ts"].to_list() == stamps[1::2]


def test_bars_from_mixed_precision_strings_drop_no_rows():
    # A string column mixing whole and fractional seconds: polars' format inference alone
    # would null the rows that do not match the first value, and clean_ohlcv drops nulls.
    stamps = _us_datetimes()
    as_str = [str(s) for s in stamps]
    assert as_str[0] == "2024-01-02 09:30:01" and as_str[1] == "2024-01-02 09:30:01.120000"
    out = openquant.bars.build_tick_bars(_tick_frame(as_str), ticks_per_bar=1)
    assert out["ts"].to_list() == stamps


def test_core_tick_bars_accept_and_return_sub_second_strings():
    rows = _core.bars.build_tick_bars(TICKS, CLOSE, [1.0] * len(TICKS), 1)
    assert [r[1] for r in rows] == TICKS
    assert len({r[1] for r in rows}) == len(TICKS)


def test_whole_second_bar_output_is_unchanged():
    ts = [f"2024-01-02 09:3{m}:0{s}" for m in range(3) for s in range(3)]
    rows = _core.bars.build_tick_bars(ts, [100.0] * len(ts), [1.0] * len(ts), 3)
    assert [(r[0], r[1]) for r in rows] == [(ts[i], ts[i + 2]) for i in (0, 3, 6)]


# --- filters, labeling, bet sizing, sample weights ---------------------------------------


def test_cusum_filter_timestamps_round_trip():
    out = _core.filters.cusum_filter_timestamps(CLOSE, TICKS, 0.005)
    assert out
    assert set(out) <= set(TICKS)
    assert any("." in t for t in out)


def test_get_events_and_get_bins_round_trip():
    events = _core.labeling.get_events(
        TICKS, CLOSE, TICKS[:5], (1.0, 1.0), TICKS, [0.005] * len(TICKS), 0.0, 1, None, None
    )
    assert [e[0] for e in events] == TICKS[:5]
    assert all(e[1] in TICKS for e in events)
    bins = _core.labeling.get_bins(events, TICKS, CLOSE)
    assert [b[0] for b in bins] == TICKS[:5]
    assert _core.labeling.drop_labels(bins, 0.0) == bins


def test_vertical_barriers_accept_sub_second_times():
    events = _core.labeling.triple_barrier_events(
        TICKS,
        CLOSE,
        TICKS[:2],
        TICKS,
        [0.5] * len(TICKS),
        1.0,
        1.0,
        0.0,
        [(TICKS[0], TICKS[3]), (TICKS[1], TICKS[4])],
        None,
    )
    assert [(e[0], e[1]) for e in events] == [(TICKS[0], TICKS[3]), (TICKS[1], TICKS[4])]


def test_bet_sizing_round_trip():
    sides = _core.bet_sizing.get_concurrent_sides(TICKS[:3], TICKS[2:5], [1.0, -1.0, 1.0])
    assert [s[0] for s in sides] == TICKS[:3]
    avg = _core.bet_sizing.avg_active_signals(TICKS[:3], [0.5, 0.2, 0.1], TICKS[2:5])
    assert [a[0] for a in avg] == TICKS[:5]


def test_sample_weights_round_trip():
    events = [(TICKS[0], TICKS[3], 1.0), (TICKS[2], TICKS[6], 1.0)]
    out = _core.sample_weights.get_weights_by_time_decay(events, TICKS, CLOSE, 0.5)
    assert [t for t, _ in out] == [TICKS[0], TICKS[2]]


def test_data_quality_report_keeps_the_fraction():
    stamps = _us_datetimes()
    report = openquant.data.data_quality_report(_tick_frame(stamps))
    assert report["ts_min"] == "2024-01-02 09:30:01"
    assert report["ts_max"] == "2024-01-02 09:30:01.999999"


# --- format contract -----------------------------------------------------------------------


def test_output_is_written_as_str_of_the_datetime():
    # Any fraction is accepted on input; output is written as str(datetime) writes it, so
    # strings built with str() come back unchanged and can be used as join keys.
    ts = ["2024-01-02 09:30:01.5", "2024-01-02 09:30:01.6"]
    out = _core.filters.cusum_filter_timestamps([1.0, 2.0], ts, 0.1)
    assert out == [str(dt.datetime(2024, 1, 2, 9, 30, 1, 600_000))]


@pytest.mark.parametrize(
    "bad", ["2024-01-02 09:30", "2024-01-02T09:30:01", "2024-01-02 09:30:01.x"]
)
def test_malformed_timestamps_still_raise(bad):
    with pytest.raises(ValueError, match="invalid datetime"):
        _core.filters.cusum_filter_timestamps([1.0, 2.0], ["2024-01-02 09:30:00", bad], 0.1)
