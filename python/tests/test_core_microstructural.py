import pytest

from _core_fixtures import finite_max, finite_mean, load_csv_columns

from openquant import microstructural


def _load_dollar_bars():
    return load_csv_columns(
        "microstructural_features/dollar_bar_sample.csv",
        ["close", "high", "low", "cum_dollar", "cum_vol"],
    )


def test_first_generation_features():
    # Mirrors crates/openquant/tests/microstructural_features.rs::test_first_generation_features
    close, high, low, cum_dollar, _ = _load_dollar_bars()
    roll = microstructural.get_roll_measure(close, 20)
    roll_impact = microstructural.get_roll_impact(close, cum_dollar, 20)
    corwin_schultz = microstructural.get_corwin_schultz_estimator(high, low, 20)
    bekker = microstructural.get_bekker_parkinson_vol(high, low, 20)

    for series in (roll, roll_impact, corwin_schultz, bekker):
        assert len(series) == len(close)

    assert abs(finite_max(roll) - 7.1584) < 1e-3
    assert abs(finite_mean(roll) - 2.341) < 1e-3
    assert abs(roll[25] - 1.176) < 1e-3

    assert abs(finite_max(roll_impact) - 1.022e-7) < 1e-8
    assert abs(finite_mean(roll_impact) - 3.3445e-8) < 1e-8
    assert abs(roll_impact[25] - 1.6807e-8) < 1e-6

    assert abs(finite_max(corwin_schultz) - 0.01652) < 1e-4
    assert abs(finite_mean(corwin_schultz) - 0.00151602) < 1e-4
    assert abs(corwin_schultz[25] - 0.00139617) < 1e-4

    assert abs(finite_max(bekker) - 0.018773) < 1e-4
    assert abs(finite_mean(bekker) - 0.001456) < 1e-4
    assert abs(bekker[25] - 0.000517) < 1e-4


def test_second_generation_bar_based_lambdas():
    # Mirrors crates/openquant/tests/microstructural_features.rs::
    # test_second_generation_intra_bar
    close, _, _, cum_dollar, cum_vol = _load_dollar_bars()
    kyle = microstructural.get_bar_based_kyle_lambda(close, cum_vol, 20)
    amihud = microstructural.get_bar_based_amihud_lambda(close, cum_dollar, 20)
    hasbrouck = microstructural.get_bar_based_hasbrouck_lambda(close, cum_dollar, 20)

    assert abs(finite_max(kyle) - 0.000163423) < 1e-6
    assert abs(finite_mean(kyle) - 7.02e-5) < 1e-5
    assert abs(kyle[25] - 7.76e-5) < 1e-5

    assert abs(finite_max(amihud) - 4.057838e-11) < 1e-13
    assert abs(finite_mean(amihud) - 1.7213e-11) < 1e-12
    assert abs(amihud[25] - 1.8439e-11) < 1e-12

    assert abs(finite_max(hasbrouck) - 3.39527e-7) < 1e-9
    assert abs(finite_mean(hasbrouck) - 1.44037e-7) < 1e-8
    assert abs(hasbrouck[25] - 1.5433e-7) < 1e-8


def test_third_generation_vpin():
    # Mirrors crates/openquant/tests/microstructural_features.rs::test_third_generation_features
    close, _, _, _, cum_vol = _load_dollar_bars()
    buy_volume = microstructural.get_bvc_buy_volume(close, cum_vol, 20)
    vpin_1 = microstructural.get_vpin(cum_vol, buy_volume, 1)
    vpin_20 = microstructural.get_vpin(cum_vol, buy_volume, 20)

    assert abs(finite_max(vpin_1) - 0.999) < 1e-3
    assert abs(finite_mean(vpin_1) - 0.501) < 1e-3
    assert abs(vpin_1[25] - 0.554) < 1e-3

    assert abs(finite_max(vpin_20) - 0.6811) < 1e-3
    assert abs(finite_mean(vpin_20) - 0.500) < 1e-3
    assert abs(vpin_20[45] - 0.4638) < 1e-3


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

    assert abs(microstructural.get_shannon_entropy(message) - 1.0) < 1e-3
    assert abs(microstructural.get_lempel_ziv_entropy(message) - 0.625) < 1e-3
    assert abs(plug_in - 0.985) < 1e-3
    assert abs(microstructural.get_konto_entropy(message, 0) - 0.9682) < 1e-3
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FINDING: these microstructural bindings index out of bounds on short or "
        "mismatched input and surface pyo3 PanicException instead of ValueError"
    ),
)
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
