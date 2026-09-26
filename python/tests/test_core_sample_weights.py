import math

import pytest
from _core_fixtures import load_csv_columns, load_timestamps
from openquant import filters, labeling, sample_weights, volatility

DATES = [f"2000-01-{day:02d} 00:00:00" for day in range(1, 11)]
PRICES = [100.0, 101.0, 102.0, 102.0, 102.0, 108.0, 108.0, 108.0, 108.0, 108.0]
# Two events that never overlap, so every bar has concurrency 1.
EVENTS = [
    ("2000-01-01 00:00:00", "2000-01-03 00:00:00", 0.01),
    ("2000-01-04 00:00:00", "2000-01-06 00:00:00", 0.01),
]


def _setup_events():
    # Same pipeline as crates/openquant/tests/sample_weights.rs::setup_events
    path = "filters/dollar_bar_sample.csv"
    (close,) = load_csv_columns(path, ["close"])
    timestamps = load_timestamps(path)

    daily_vol = volatility.get_daily_vol(timestamps, close, 100)
    cusum_events = filters.cusum_filter_timestamps(close, timestamps, 0.02)
    vertical_barriers = labeling.add_vertical_barrier(cusum_events, timestamps, close, 2, 0, 0, 0)
    events = labeling.get_events(
        timestamps,
        close,
        cusum_events,
        (4.0, 4.0),
        [ts for ts, _ in daily_vol],
        [v for _, v in daily_vol],
        0.005,
        3,
        vertical_barriers,
        [(ts, 1.0) for ts in timestamps],
    )
    simple = [(ts, t1, trgt) for ts, t1, trgt, _, _, _ in events if t1 is not None]
    return simple, timestamps, close


def test_return_attribution_for_non_overlapping_events():
    # AFML 4.10 with concurrency 1: weight_i = |sum of log returns over event i|,
    # rescaled so the weights sum to the number of events.
    raw = [math.log(102.0 / 100.0), math.log(108.0 / 102.0)]
    expected = [r * len(raw) / sum(raw) for r in raw]

    out = sample_weights.get_weights_by_return(EVENTS, DATES, PRICES)

    assert [ts for ts, _ in out] == [EVENTS[0][0], EVENTS[1][0]]
    assert [w for _, w in out] == pytest.approx(expected, abs=1e-12)


def test_time_decay_for_non_overlapping_events():
    # AFML 4.11: cumulative uniqueness is [1, 2]; with decay c = 0.5 the line through
    # (2, 1) has slope (1 - c) / 2 = 0.25, so the weights are [0.75, 1.0].
    out = sample_weights.get_weights_by_time_decay(EVENTS, DATES, PRICES, 0.5)
    assert [w for _, w in out] == pytest.approx([0.75, 1.0], abs=1e-12)


def test_return_attribution_on_fixture():
    # Mirrors crates/openquant/tests/sample_weights.rs::test_ret_attribution. The Rust test
    # compares against the mlfinlab values with a tolerance of 1e5, i.e. not at all, so the
    # only portable value here is the normalisation.
    events, timestamps, close = _setup_events()
    out = sample_weights.get_weights_by_return(events, timestamps, close)

    assert len(out) == len(events)
    assert sum(w for _, w in out) == pytest.approx(len(events), abs=1e-9)
    assert all(w > 0.0 for _, w in out)


def test_time_decay_weights_on_fixture():
    # Mirrors crates/openquant/tests/sample_weights.rs::test_time_decay_weights
    events, timestamps, close = _setup_events()

    def decay(value):
        return [
            w for _, w in sample_weights.get_weights_by_time_decay(events, timestamps, close, value)
        ]

    standard = decay(0.5)
    no_decay = decay(1.0)
    neg_decay = decay(-0.5)
    converge = decay(0.0)

    for weights in (standard, no_decay, neg_decay, converge):
        assert len(weights) == len(events)

    assert standard[-1] == 1.0
    assert all(abs(w - 1.0) < 1e-12 for w in no_decay)
    assert sum(1 for w in neg_decay if w == 0.0) == 3


@pytest.mark.parametrize("bad", [1.5, -1.0, float("nan")])
def test_time_decay_rejects_decay_outside_afml_domain(bad):
    # AFML's domain is (-1, 1]; 1.5 used to be accepted and weighted old labels more (#186).
    with pytest.raises(ValueError, match=r"decay must be in \(-1, 1\]"):
        sample_weights.get_weights_by_time_decay(EVENTS, DATES, PRICES, bad)


def test_time_decay_keeps_one_weight_per_event_when_starts_coincide():
    # Mirrors crates/openquant/tests/sample_weights.rs, issue #91. Average uniqueness is
    # [4/9, 11/24, 11/18]; the tie at 09:30 is broken by input order, so the cumulative
    # uniqueness is [32/72, 65/72, 109/72] and decay 0.5 gives 0.5 + 36/109 * x.
    ts = [f"2024-01-02 09:3{i}:00" for i in range(6)]
    events = [(ts[0], ts[2], 1.0), (ts[0], ts[3], 1.0), (ts[2], ts[4], 1.0)]
    out = sample_weights.get_weights_by_time_decay(
        events, ts, [100, 101, 102, 101, 100, 103.0], 0.5
    )

    assert [t for t, _ in out] == [ts[0], ts[0], ts[2]]
    expected = [0.5 + 36 / 109 * x for x in (32 / 72, 65 / 72)] + [1.0]
    assert [w for _, w in out] == pytest.approx(expected, abs=1e-12)


def test_sample_weights_reject_invalid_events():
    # Mirrors crates/openquant/tests/sample_weights.rs::test_value_error_raise
    events = [EVENTS[0], (EVENTS[1][1], EVENTS[1][0], EVENTS[1][2])]

    with pytest.raises(ValueError, match="event 1 ends before it starts"):
        sample_weights.get_weights_by_return(events, DATES, PRICES)
    with pytest.raises(ValueError, match="event 1 ends before it starts"):
        sample_weights.get_weights_by_time_decay(events, DATES, PRICES, 0.5)
    # 1970-01-01 is an ordinary bar, not a missing value.
    epoch = ["1970-01-01 00:00:00", "1970-01-02 00:00:00"]
    out = sample_weights.get_weights_by_time_decay([(*epoch, 1.0)], epoch, [100.0, 101.0], 0.5)
    assert out == [(epoch[0], 1.0)]
    with pytest.raises(ValueError, match="length mismatch"):
        sample_weights.get_weights_by_return(EVENTS, DATES, [1.0])
