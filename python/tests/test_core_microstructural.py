import math

import pytest
from _core_fixtures import finite_max, finite_mean, load_csv_columns, load_json
from openquant import microstructural

# Written by tests/fixtures/microstructural_features/generate.py (AFML ch. 19 in pandas).
REFERENCE = load_json("microstructural_features/reference.json")
# Same arithmetic as the reference, so only rounding separates the two.
REL_TOL = 1e-9


def _load_dollar_bars():
    return load_csv_columns(
        "shared/dollar_bar_sample.csv",
        ["close", "high", "low", "cum_dollar", "cum_vol"],
    )


def _assert_matches(series, key):
    expected = REFERENCE[key]
    first = expected["first_finite"]
    assert all(math.isnan(v) for v in series[:first]), key
    assert math.isfinite(series[first]), key
    assert finite_max(series) == pytest.approx(expected["max"], rel=REL_TOL)
    assert finite_mean(series) == pytest.approx(expected["mean"], rel=REL_TOL)
    assert series[expected["position"]] == pytest.approx(expected["at_position"], rel=REL_TOL)


def test_first_generation_features():
    # Mirrors crates/openquant/tests/microstructural_features.rs::test_first_generation_features
    close, high, low, cum_dollar, _ = _load_dollar_bars()
    roll = microstructural.get_roll_measure(close, 20)
    roll_impact = microstructural.get_roll_impact(close, cum_dollar, 20)
    corwin_schultz = microstructural.get_corwin_schultz_estimator(high, low, 20)
    bekker = microstructural.get_bekker_parkinson_vol(high, low, 20)

    for series in (roll, roll_impact, corwin_schultz, bekker):
        assert len(series) == len(close)

    _assert_matches(roll, "roll_measure")
    _assert_matches(roll_impact, "roll_impact")
    _assert_matches(corwin_schultz, "corwin_schultz")
    _assert_matches(bekker, "becker_parkinson")


def test_second_generation_bar_based_lambdas():
    # Mirrors crates/openquant/tests/microstructural_features.rs::
    # test_second_generation_intra_bar
    close, _, _, cum_dollar, cum_vol = _load_dollar_bars()
    kyle = microstructural.get_bar_based_kyle_lambda(close, cum_vol, 20)
    amihud = microstructural.get_bar_based_amihud_lambda(close, cum_dollar, 20)
    hasbrouck = microstructural.get_bar_based_hasbrouck_lambda(close, cum_dollar, 20)

    _assert_matches(kyle, "kyle_lambda")
    _assert_matches(amihud, "amihud_lambda")
    _assert_matches(hasbrouck, "hasbrouck_lambda")


def test_third_generation_vpin():
    # Mirrors crates/openquant/tests/microstructural_features.rs::test_third_generation_features
    close, _, _, _, cum_vol = _load_dollar_bars()
    buy_volume = microstructural.get_bvc_buy_volume(close, cum_vol, 20)
    vpin_1 = microstructural.get_vpin(cum_vol, buy_volume, 1)
    vpin_20 = microstructural.get_vpin(cum_vol, buy_volume, 20)

    _assert_matches(vpin_1, "vpin_1")
    _assert_matches(vpin_20, "vpin_20")


def test_tick_rule_encoding():
    # Mirrors crates/openquant/tests/microstructural_features.rs::test_tick_rule_encoding
    assert microstructural.encode_tick_rule_array([-1, 1, 0, 0]) == "bacc"
    with pytest.raises(ValueError, match="Unknown value for tick rule"):
        microstructural.encode_tick_rule_array([-1, 1, 0, 20000000])


def test_entropy_calculations():
    # Mirrors crates/openquant/tests/microstructural_features.rs::test_entropy_calculations
    message = "11100001"
    message_array = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    plug_in = microstructural.get_plug_in_entropy(message, 1)
    encoding = microstructural.quantile_mapping(message_array, 2)
    encoded = microstructural.encode_array(message_array, encoding)

    # Worked by hand from AFML ch. 18 for "11100001":
    # Shannon: four 1s and four 0s, so -2 * (1/2) log2(1/2) = 1.
    assert microstructural.get_shannon_entropy(message) == pytest.approx(1.0, abs=1e-12)
    # Lempel-Ziv (snippet 18.2): library 1, 11, 0, 00, 01 -> 5 words / 8 chars.
    assert microstructural.get_lempel_ziv_entropy(message) == pytest.approx(5 / 8, abs=1e-12)
    # Plug-in, word length 1 (snippet 18.1): the snippet's pmf reads msg[i-1] for
    # i in 1..len, i.e. "1110000" (the last character is not counted):
    # p(1) = 3/7, p(0) = 4/7.
    p1, p0 = 3 / 7, 4 / 7
    assert plug_in == pytest.approx(-(p1 * math.log2(p1) + p0 * math.log2(p0)), abs=1e-12)
    # Kontoyiannis, expanding window (snippets 18.3-18.4): points i = 1..4 with
    # match length + 1 of 2, 2, 1, 4, so h = mean(log2(i+1) / L_i).
    konto = (math.log2(2) / 2 + math.log2(3) / 2 + math.log2(4) / 1 + math.log2(5) / 4) / 4
    assert microstructural.get_konto_entropy(message, 0) == pytest.approx(konto, abs=1e-12)
    assert abs(plug_in - microstructural.get_plug_in_entropy(encoded, 1)) < 1e-9


def test_trade_based_scalars():
    # No Rust integration test covers these; values are worked by hand.
    assert microstructural.vwap([10.0, 20.0], [1.0, 1.0]) == pytest.approx(15.0)
    assert microstructural.get_avg_tick_size([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    # Kyle lambda = OLS slope through the origin of price_diff on signed volume:
    # sum(x*y)/sum(x*x) with x = [10, -20, 30], y = [0.1, -0.1, 0.2] -> 9 / 1400.
    kyle = microstructural.get_trades_based_kyle_lambda(
        [0.1, -0.1, 0.2], [10.0, 20.0, 30.0], [1.0, -1.0, 1.0]
    )
    assert kyle == pytest.approx(9.0 / 1400.0, abs=1e-12)


@pytest.mark.parametrize(
    ("volumes", "returns", "expected"),
    [
        ([2.0**600, 2.0**601], [1.0, -2.0], 2.0**-600),
        ([2.0**-600, 2.0**-599], [1.0, -2.0], 2.0**600),
        (
            [4.0208288850127243e-256] * 2,
            [7.228216483043474e52] * 2,
            float.fromhex("0x1.fffffffffffffp1023"),
        ),
        (
            [4.0208288850127243e-256] * 2,
            [math.nextafter(7.228216483043474e52, math.inf)] * 2,
            math.inf,
        ),
    ],
)
def test_trade_amihud_extreme_finite_inputs(volumes, returns, expected):
    # Exercise list conversion and the native extension at both ends of binary64.
    assert microstructural.get_trades_based_amihud_lambda(returns, volumes) == expected


def test_sigma_mapping_rejects_oversized_alphabet():
    with pytest.raises(ValueError, match="exceeds ASCII table"):
        microstructural.sigma_mapping([float(i) for i in range(1000)], 0.001)


@pytest.mark.parametrize(
    "call",
    [
        lambda: microstructural.quantile_mapping([], 2),
        lambda: microstructural.get_corwin_schultz_estimator([1.0, 2.0, 3.0], [1.0], 2),
        lambda: microstructural.get_vpin([1.0, 2.0, 3.0], [1.0], 2),
        lambda: microstructural.get_bar_based_kyle_lambda([1.0, 2.0, 3.0], [1.0], 2),
        lambda: microstructural.get_plug_in_entropy("11", 5),
    ],
    ids=[
        "quantile_mapping_empty",
        "corwin_schultz_length_mismatch",
        "vpin_length_mismatch",
        "kyle_lambda_length_mismatch",
        "plug_in_entropy_word_longer_than_message",
    ],
)
def test_invalid_input_raises_value_error(call):
    with pytest.raises(ValueError):
        call()


def test_encoding_edge_cases_raise_instead_of_dropping_values():
    # #185 item 16.
    codebook = microstructural.quantile_mapping([1.0, 2.0, 3.0, 4.0, 5.0], 2)
    with pytest.raises(ValueError, match="NaN"):
        microstructural.encode_array([1.0, float("nan")], codebook)
    with pytest.raises(ValueError, match="codebook"):
        microstructural.encode_array([1.0], [])
    with pytest.raises(ValueError, match="step"):
        microstructural.sigma_mapping([0.0, 1.0], float("nan"))
    with pytest.raises(ValueError, match="empty"):
        microstructural.sigma_mapping([], 0.5)
    with pytest.raises(ValueError, match="window"):
        microstructural.get_bvc_buy_volume([10.0, 11.0, 10.0], [100.0] * 3, 0)


@pytest.mark.parametrize("exponent", [0, -38, -40, -44, -60])
def test_bvc_positive_deviation_is_invariant_to_price_units(exponent):
    # Exact binary rescaling: no loss of price differences or variance underflow.
    scale = math.ldexp(1.0, exponent)
    close = [p * scale for p in [10.0, 11.0, 10.0, 11.0]]
    actual = microstructural.get_bvc_buy_volume(close, [100.0] * 4, 2)
    # Changes [-scale, +scale] have sample deviation sqrt(2) * scale.
    # Independent standard-normal CDF via Python's erfc, not Rust/statrs.
    expected = [50.0 * math.erfc(0.5), 50.0 * math.erfc(-0.5)]
    assert all(math.isnan(value) for value in actual[:2])
    assert actual[2:] == pytest.approx(expected, rel=0, abs=1e-8)


def test_bvc_preserves_zero_deviation_and_ordinary_outputs_bitwise():
    scale = math.ldexp(1.0, -44)
    # Captured from the pre-fix extension; lock down the untouched numeric paths.
    # Constant nonzero changes must retain Phi(dp / 1e-12), not a sign/half split.
    cases = [
        ([10.0] * 4, 2, ["0x1.9000000000000p+5"] * 2),
        ([p * scale for p in [10.0, 11.0, 12.0, 13.0]], 2, ["0x1.a221ccad359cdp+5"] * 2),
        ([p * scale for p in [13.0, 12.0, 11.0, 10.0]], 2, ["0x1.7dde3352ca633p+5"] * 2),
        (
            [10.0, 12.0, 11.0, 14.0, 14.0, 12.0, 15.0],
            3,
            [
                "0x1.72177b8e9e3aep+6",
                "0x1.9000000000000p+5",
                "0x1.556be02a9f9b6p+4",
                "0x1.615a9cd2e4d0ep+6",
            ],
        ),
    ]
    for close, window, expected in cases:
        actual = microstructural.get_bvc_buy_volume(close, [100.0] * len(close), window)
        assert all(math.isnan(value) for value in actual[:window])
        assert [value.hex() for value in actual[window:]] == expected


def test_bvc_preserves_missing_nonfinite_and_window_behavior():
    for window in [0, 1]:
        with pytest.raises(ValueError):
            microstructural.get_bvc_buy_volume([1.0, 2.0], [100.0] * 2, window)
    with pytest.raises(ValueError):
        microstructural.get_bvc_buy_volume([1.0, 2.0], [100.0], 2)
    assert microstructural.get_bvc_buy_volume([], [], 2) == []
    short = microstructural.get_bvc_buy_volume([1.0, 2.0], [100.0] * 2, 2)
    assert len(short) == 2 and all(math.isnan(value) for value in short)
    for missing in [math.nan, math.inf, -math.inf]:
        close = [1.0, missing, 2.0, 3.0, 4.0, 3.0]
        actual = microstructural.get_bvc_buy_volume(close, [100.0] * len(close), 2)
        assert all(math.isnan(value) for value in actual[:4])
        assert actual[4] == 100.0  # Recovery after a complete clean window.
        assert actual[5] == pytest.approx(50.0 * math.erfc(0.5), rel=0, abs=1e-8)
    # Finite changes with an overflowing variance still have sigma=+inf and z=0.
    actual = microstructural.get_bvc_buy_volume([1e200, 2e200, 1e200, 2e200], [100.0] * 4, 2)
    assert actual[2:] == [50.0, 50.0]
