import importlib.util
from pathlib import Path

import numpy as np
import pytest

s = importlib.util.spec_from_file_location(
    "delayed_label", Path(__file__).resolve().parents[2] / "experiments/delayed_label_lab.py"
)
assert s is not None and s.loader is not None
lab = importlib.util.module_from_spec(s)
s.loader.exec_module(lab)


@pytest.mark.parametrize("h,n", [(1, 1), (1, 4), (4, 4), (4, 16), (16, 16), (16, 64), (4, 2)])
@pytest.mark.parametrize("lag", [0, 2])
def test_independent_overlap_closed_form(h, n, lag):
    d = lab.design(h, n, lag)
    v = (n * h + 2 * sum((n - k) * (h - k) for k in range(1, min(n - 1, h - 1) + 1))) / n**2
    c = sum(min(n, r) for r in range(1, h)) / n
    safe = d["methods"]["mature_only"]
    naive = d["methods"]["complete_archive"]
    assert safe["prediction_variance"] == pytest.approx(v)
    assert naive["prediction_variance"] == pytest.approx(v)
    assert safe["target_covariance"] == 0
    assert naive["target_covariance"] == pytest.approx(c)
    assert safe["expected_mse"] == pytest.approx(h + v)
    assert naive["expected_mse"] == pytest.approx(h + v - 2 * c)


def test_integer_clock_selection_and_unavailability():
    assert lab.eligible_training([-3, -2, -1, 0], [0, 1, 2, 0], 0, 1) == [-3]
    assert lab.eligible_training([-3, -2, -1], [0, 1, 2], 0, 2) is None
    assert lab.eligible_training([-1, -3, -2], [2, 0, 1], 0, 1) == [-3]


def test_delay_contract_is_not_always_future_content():
    naive = lab.design(1, 4, 2)["methods"]["complete_archive"]
    assert naive["unmatured_count"] == 2
    assert naive["future_content_label_count"] == 0


@pytest.mark.parametrize(
    "decisions,available", [([-1, -1], [0, 0]), ([-1], []), ([-1], [-2]), ([-1.0], [0])]
)
def test_bad_clocks(decisions, available):
    with pytest.raises((ValueError, TypeError)):
        lab.eligible_training(decisions, available, 0, 1)


def test_fixed_cutoff_shift_and_replay():
    assert (
        lab.design(4, 16, 2)["methods"]["mature_only"]["innovation_weights"]
        == lab.design(4, 16, 2, cutoff=100)["methods"]["mature_only"]["innovation_weights"]
    )
    r = lab.run_lab(horizons=[4], training_multipliers=[1], lags=[0], repetitions=5)
    assert r["records"] == lab.run_lab(**r["config"])["records"]


def test_unsigned_grid_multiplication_cannot_wrap_into_scope():
    with pytest.raises(ValueError):
        lab.run_lab(
            horizons=[np.uint64(4)],
            training_multipliers=[np.uint64(2**62 + 1)],
            lags=[0],
            repetitions=2,
        )
