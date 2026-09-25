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
        "microstructural_features/dollar_bar_sample.csv",
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
