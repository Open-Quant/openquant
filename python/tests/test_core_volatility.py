import math
from datetime import datetime, timedelta

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
    return load_csv_columns("shared/dollar_bar_sample.csv", ["open", "high", "low", "close"])


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
    _assert_matches(yz_vol, RANGE_REFERENCE["yang_zhang_paper"])
    _assert_matches(park_vol, RANGE_REFERENCE["parkinson"])


def test_yang_zhang_matches_paper():
    # Mirrors crates/openquant/tests/volatility_features.rs::test_yang_zhang_matches_paper
    # (issue #165): open-to-close ln(C_t/O_t) and demeaned moments, as in Yang & Zhang (2000).
    open_, high, low, close = _load_ohlc()
    yz_vol = volatility.get_yang_zhang_vol(open_, high, low, close, 20)
    _assert_matches(yz_vol, RANGE_REFERENCE["yang_zhang_paper"])


def test_yang_zhang_is_zero_for_pure_drift():
    # Mirrors volatility_features.rs::test_yang_zhang_is_zero_for_pure_drift: constant
    # overnight and open-to-close returns have zero variance once demeaned, and with the high
    # at the close and the low at the open the Rogers-Satchell term is zero.
    open_, close = [100.0], [101.0]
    for i in range(1, 40):
        open_.append(close[i - 1] * 1.002)
        close.append(open_[i] * 1.01)
    yz_vol = volatility.get_yang_zhang_vol(open_, close, open_, close, 10)
    assert all(math.isnan(v) for v in yz_vol[:10])
    assert max(abs(v) for v in yz_vol[10:]) < 1e-12


def test_daily_vol_is_zero_for_constant_daily_return():
    # A series compounding at exactly 1% per day has identical eligible two-day
    # returns, so any EWM standard deviation of them is zero.
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
    ("prices", "expected"),
    [
        (
            [100.0, 110.0, 120.0, math.nan, 150.0, 140.0, 170.0, 180.0],
            [
                math.nan,
                math.nan,
                0.03535533905932741,
                0.03535533905932741,
                0.07528605062021397,
                0.09357397530988633,
            ],
        ),
        (
            [math.nan, math.nan, 100.0, 110.0, 120.0, 130.0, 140.0],
            [math.nan, math.nan, math.nan, 0.012856486930664428, 0.015745916432444287],
        ),
        (
            [100.0, 110.0, 120.0, 125.0, math.nan, 150.0, 160.0, 170.0],
            [
                math.nan,
                0.044997704257325646,
                0.044997704257325646,
                0.037440353071835054,
                0.037440353071835054,
                0.0423121471969789,
            ],
        ),
        ([math.nan] * 6, [math.nan] * 4),
        (
            [100.0, 110.0, 120.0, math.inf, 150.0, 140.0, 170.0, 180.0],
            [
                math.nan,
                math.nan,
                0.03535533905932741,
                0.8361845163924918,
                0.6709403362030547,
                0.5297712044961865,
            ],
        ),
    ],
    ids=["interior_nan", "leading_nan", "hold_then_recover", "all_nan", "infinite_return"],
)
def test_daily_vol_matches_pandas_with_missing_returns(prices, expected):
    # Independent pandas EWM golden values shared with crates/openquant/tests/daily_vol.rs.
    timestamps = [f"2024-01-{i + 1:02d} 00:00:00" for i in range(len(prices))]
    out = volatility.get_daily_vol(timestamps, prices, 3)
    assert [timestamp for timestamp, _ in out] == timestamps[2:]
    assert [value for _, value in out] == pytest.approx(expected, nan_ok=True, rel=0, abs=1e-12)


def test_daily_vol_does_not_underflow_to_zero_across_a_long_missing_gap():
    prices = [100.0, 101.0, 102.0, 103.0] + [math.nan] * 530
    start = datetime(2024, 1, 1)
    timestamps = [
        (start + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(len(prices))
    ]
    out = volatility.get_daily_vol(timestamps, prices, 3)
    # Two-observation sample standard deviation, carried forward by pandas.
    expected = math.sqrt((102.0 / 100.0 - 103.0 / 101.0) ** 2 / 2)
    assert [timestamp for timestamp, _ in out] == timestamps[2:]
    assert math.isnan(out[0][1])
    assert [value for _, value in out[1:]] == pytest.approx(
        [expected] * (len(out) - 1), rel=0, abs=1e-12
    )


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
