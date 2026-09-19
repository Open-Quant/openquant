import math

import pytest

from _core_fixtures import load_csv_columns

from openquant import structural_breaks


def _log_prices():
    (close,) = load_csv_columns("structural_breaks/dollar_bar_sample.csv", ["close"])
    return [math.log(v) for v in close]


def _mean(values):
    return sum(values) / len(values)


def test_chow_type_stat():
    # Mirrors crates/openquant/tests/structural_breaks.rs::test_chow_test
    min_length = 10
    log_prices = _log_prices()
    stats = structural_breaks.get_chow_type_stat(log_prices, min_length)

    assert len(stats) == len(log_prices) - min_length * 2
    assert abs(max(stats) - 0.179) < 0.001
    assert abs(_mean(stats) + 0.653) < 0.001
    assert abs(stats[3] + 0.6649) < 0.001


def test_chu_stinchcombe_white_statistics():
    # Mirrors crates/openquant/tests/structural_breaks.rs::test_chu_stinchcombe_white_test
    log_prices = _log_prices()
    one_critical, one_stat = structural_breaks.get_chu_stinchcombe_white_statistics(
        log_prices, "one_sided"
    )
    two_critical, two_stat = structural_breaks.get_chu_stinchcombe_white_statistics(
        log_prices, "two_sided"
    )

    assert len(one_critical) == len(log_prices) - 2
    assert len(two_critical) == len(log_prices) - 2

    assert abs(max(one_critical) - 3.265) < 0.001
    assert abs(_mean(one_critical) - 2.7809) < 0.001
    assert abs(one_critical[20] - 2.4466) < 0.001

    assert abs(max(one_stat) - 3729.001) < 0.001
    assert abs(_mean(one_stat) - 836.509) < 0.001
    assert abs(one_stat[20] - 380.137) < 0.001

    assert abs(max(two_critical) - 3.235) < 0.001
    assert abs(_mean(two_critical) - 2.769) < 0.001
    assert abs(two_critical[20] - 2.715) < 0.001

    assert abs(max(two_stat) - 5518.519) < 0.001
    assert abs(_mean(two_stat) - 1264.582) < 0.001
    assert abs(two_stat[20] - 921.2979) < 0.001


@pytest.mark.parametrize(
    "model, expected",
    [
        ("sm_power", -4.281),
        ("linear", -0.717),
        ("quadratic", -1.065),
        ("sm_poly_1", 0.8268),
        ("sm_poly_2", 0.822),
        ("sm_exp", -5.821),
    ],
)
def test_sadf_pointwise_reference_values(model, expected):
    # Mirrors the `<model>[29]` assertions of
    # crates/openquant/tests/structural_breaks.rs::test_sadf_test. SADF at bar t only looks
    # at bars <= t, so element 29 is identical on a 60-bar prefix of the fixture. The full
    # series is not run here (the Rust test is #[ignore]d as long-running; one model takes
    # minutes through a debug build), so the whole-series means are not ported.
    min_length, lags = 20, 5
    log_prices = _log_prices()[:60]
    out = structural_breaks.get_sadf(log_prices, model, True, min_length, lags)

    assert len(out) == len(log_prices) - min_length - lags - 1
    assert abs(out[29] - expected) < 0.001


def test_sadf_constant_series_is_negative_infinity():
    # Mirrors the `ones` case of crates/openquant/tests/structural_breaks.rs::test_sadf_test
    # on a shorter series.
    min_length, lags = 20, 5
    series = [1.0] * 40
    out = structural_breaks.get_sadf(series, "sm_power", True, min_length, lags)

    assert len(out) == len(series) - min_length - lags - 1
    assert all(v == -math.inf for v in out)


def test_structural_breaks_reject_invalid_inputs():
    # Mirrors the error cases of test_chu_stinchcombe_white_test and test_sadf_test
    log_prices = _log_prices()[:60]
    with pytest.raises(ValueError, match="InvalidTestType"):
        structural_breaks.get_chu_stinchcombe_white_statistics(log_prices, "rubbish text")
    with pytest.raises(ValueError, match="InvalidModel"):
        structural_breaks.get_sadf(log_prices, "rubbish_string", True, 20, 5)
    with pytest.raises(ValueError, match="InputTooShort"):
        structural_breaks.get_sadf([1.0, 2.0, 3.0], "linear", True, 20, 5)
