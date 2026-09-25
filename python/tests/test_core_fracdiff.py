import math

import pytest
from _core_fixtures import load_csv_columns
from openquant import fracdiff


def _load_close():
    (close,) = load_csv_columns("backtest_statistics/dollar_bar_sample.csv", ["close"])
    return close


def _reference_ffd_weights(diff_amt, thresh):
    # AFML snippet 5.3: w_k = -w_{k-1} * (d - k + 1) / k, stop once |w_k| < thresh.
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (diff_amt - k + 1) / k
        if abs(w) < thresh:
            break
        weights.append(w)
        k += 1
    return weights  # most recent observation first


def test_get_weights_values():
    # Mirrors crates/openquant/tests/fracdiff.rs::test_get_weights
    weights = fracdiff.get_weights(0.9, 100)
    assert len(weights) == 100
    assert weights[-1] == 1.0
    # AFML 5.1 recursion, oldest weight first.
    assert weights[-4:] == pytest.approx([-0.0165, -0.045, -0.9, 1.0], abs=1e-12)


def test_get_weights_ffd_values():
    # Mirrors crates/openquant/tests/fracdiff.rs::test_get_weights_ffd
    weights = fracdiff.get_weights_ffd(0.9, 1e-3, 100)
    assert len(weights) == 12
    assert weights[-1] == 1.0
    assert weights[::-1] == pytest.approx(_reference_ffd_weights(0.9, 1e-3), abs=1e-12)


def test_frac_diff_ffd_matches_reference_convolution():
    # Mirrors crates/openquant/tests/fracdiff.rs::test_frac_diff_ffd (same fixture and
    # length/NaN checks) and adds the values the Rust test leaves unasserted.
    close = _load_close()
    for i in range(1, 10):
        out = fracdiff.frac_diff_ffd(close, i / 10.0, 1e-5)
        assert len(out) == len(close)
        assert math.isnan(out[0])

    diff_amt, thresh = 0.5, 1e-3
    weights = _reference_ffd_weights(diff_amt, thresh)
    width = len(weights)
    out = fracdiff.frac_diff_ffd(close, diff_amt, thresh)
    assert all(math.isnan(v) for v in out[: width - 1])
    for idx in range(width - 1, len(close)):
        expected = sum(w * close[idx - k] for k, w in enumerate(weights))
        assert out[idx] == pytest.approx(expected, abs=1e-8)


def test_frac_diff_lengths_and_leading_nan():
    # Mirrors crates/openquant/tests/fracdiff.rs::test_frac_diff
    close = _load_close()
    for i in range(1, 10):
        out = fracdiff.frac_diff(close, i / 10.0, 0.01)
        assert len(out) == len(close)
        assert math.isnan(out[0])


def test_fracdiff_rejects_invalid_arguments():
    with pytest.raises(OverflowError):
        fracdiff.get_weights(0.5, -1)
    with pytest.raises(TypeError):
        fracdiff.frac_diff_ffd(["a", "b"], 0.5, 1e-5)
