"""Large signed barrier components retain timestamp semantics without native panics."""

import openquant
import pytest

STAMPS = ["2024-01-01 00:00:00", "2024-01-02 00:00:00"]


@pytest.mark.parametrize(
    ("offsets", "expected"),
    [
        ({"num_days": 2**63 - 1}, []),
        ({"num_days": -(2**63)}, [(STAMPS[1], STAMPS[0])]),
        (
            {"num_days": 100_000_000_000_000, "num_hours": -2_400_000_000_000_000},
            [(STAMPS[1], STAMPS[1])],
        ),
    ],
)
def test_extreme_barrier_components_use_the_combined_signed_horizon(offsets, expected):
    assert openquant.labeling.add_vertical_barrier([STAMPS[1]], STAMPS, **offsets) == expected


def test_nan_minimum_return_does_not_select_any_target():
    assert (
        openquant.labeling.triple_barrier_events(
            STAMPS,
            [100.0, 101.0],
            [STAMPS[0]],
            [STAMPS[0]],
            [0.01],
            min_ret=float("nan"),
        )
        == []
    )


@pytest.mark.parametrize(
    ("target", "pt", "sl", "side", "touch"),
    [
        (1e308, 2.0, 0.0, None, STAMPS[1]),
        (1.0, 4.0, 0.0, 1e-308, None),
        (1e308, 0.0, 2.0, -1.0, STAMPS[1]),
    ],
)
def test_horizontal_barriers_compare_unrepresentable_intermediates(target, pt, sl, side, touch):
    events = openquant.labeling.triple_barrier_events(
        STAMPS,
        [1e-308, 3.0],
        [STAMPS[0]],
        [STAMPS[0]],
        [target],
        pt=pt,
        sl=sl,
        side_prediction=None if side is None else [(STAMPS[0], side)],
    )
    assert len(events) == 1
    assert events[0][1] == touch
