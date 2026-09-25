import math

import pytest

from _core_fixtures import load_csv_columns, load_json, nanmean

from openquant import volatility

# Expected values from tests/fixtures/volatility/generate_range.py (Parkinson 1980,
# Garman & Klass 1980, Yang & Zhang 2000, computed in pandas independently of this library).
RANGE_REFERENCE = load_json("volatility/range_reference.json")


def _assert_matches(actual, expected):
    assert sum(math.isnan(v) for v in actual) == expected["n_nan"]
    assert abs(nanmean(actual) - expected["mean_excluding_nan"]) < 1e-12
    for sample in expected["samples"]:
        value = actual[sample["position"]]
        if sample["value"] is None:
            assert math.isnan(value)
        else:
            assert abs(value - sample["value"]) < 1e-12


def _load_ohlc():
    return load_csv_columns(
        "backtest_statistics/dollar_bar_sample.csv", ["open", "high", "low", "close"]
    )


def test_range_estimators_match_reference():
    # Mirrors crates/openquant/tests/volatility_features.rs::
    # test_volatility_estimators_match_reference
    open_, high, low, close = _load_ohlc()
    gm_vol = volatility.get_garman_class_vol(open_, high, low, close, 20)
    yz_vol = volatility.get_yang_zhang_vol(open_, high, low, close, 20)
    park_vol = volatility.get_parkinson_vol(high, low, 20)

    assert len(gm_vol) == len(close)
    assert len(yz_vol) == len(close)
    assert len(park_vol) == len(close)

    _assert_matches(gm_vol, RANGE_REFERENCE["garman_klass"])
    # `yang_zhang` is the library's form of the estimator, which departs from the paper; see
    # generate_range.py and test_yang_zhang_matches_paper below.
    _assert_matches(yz_vol, RANGE_REFERENCE["yang_zhang"])
    _assert_matches(park_vol, RANGE_REFERENCE["parkinson"])


@pytest.mark.xfail(
    strict=True,
    reason="FINDING: get_yang_zhang_vol uses ln(C_t/O_{t-1}) for the close term instead of "
    "Yang & Zhang's open-to-close ln(C_t/O_t), and undemeaned moments; see "
    "tests/fixtures/volatility/generate_range.py",
)
def test_yang_zhang_matches_paper():
    # Mirrors crates/openquant/tests/volatility_features.rs::test_yang_zhang_matches_paper
    open_, high, low, close = _load_ohlc()
    yz_vol = volatility.get_yang_zhang_vol(open_, high, low, close, 20)
    _assert_matches(yz_vol, RANGE_REFERENCE["yang_zhang_paper"])


def test_daily_vol_is_zero_for_constant_daily_return():
    # No Rust value test exists for get_daily_vol (it is only used as an input in
    # crates/openquant/tests/sample_weights.rs). A series compounding at exactly 1% per
    # day has identical one-day returns, so any EWM standard deviation of them is zero.
    timestamps = [f"2024-01-{day:02d} 00:00:00" for day in range(1, 11)]
    prices = [100.0 * 1.01**i for i in range(10)]

    out = volatility.get_daily_vol(timestamps, prices, 5)

    # Snippet 3.1 compares each bar with the last bar *strictly* more than a day older
    # (`searchsorted(t - 1 day) - 1`), so with bars exactly one day apart the first estimate is
    # on the third bar, against the first. pandas gives the same 8 rows: NaN for the first
    # (one observation has no sample variance), then exactly 0.
    assert [ts for ts, _ in out] == timestamps[2:]
    assert math.isnan(out[0][1])
    assert [v for _, v in out[1:]] == pytest.approx([0.0] * 7, abs=1e-15)


def test_daily_vol_rejects_length_mismatch_and_bad_timestamps():
    with pytest.raises(ValueError, match="length mismatch"):
        volatility.get_daily_vol(["2024-01-01 00:00:00"], [1.0, 2.0], 5)
    with pytest.raises(ValueError, match="invalid datetime"):
        volatility.get_daily_vol(["2024-01-01"], [1.0], 5)


@pytest.mark.parametrize(
    "call",
    [
        lambda: volatility.get_parkinson_vol([1.0, 2.0, 3.0], [1.0], 2),
        lambda: volatility.get_garman_class_vol([1.0, 2.0, 3.0], [1.0], [1.0], [1.0], 2),
        lambda: volatility.get_yang_zhang_vol([1.0, 2.0, 3.0], [1.0], [1.0], [1.0], 2),
    ],
    ids=["parkinson", "garman_klass", "yang_zhang"],
)
def test_range_estimators_length_mismatch_raises_value_error(call):
    with pytest.raises(ValueError):
        call()
