import math

import pytest
from _core_fixtures import load_csv_columns, load_json
from openquant import structural_breaks

# AFML chapter 17 recomputed in numpy by tests/fixtures/structural_breaks/generate.py.
REFERENCE = load_json("structural_breaks/reference.json")


def _log_prices():
    (close,) = load_csv_columns("shared/dollar_bar_sample.csv", ["close"])
    return [math.log(v) for v in close]


def _mean(values):
    return sum(values) / len(values)


def _close(got, want, rel=1e-8):
    return abs(got - want) <= rel * max(abs(want), 1.0)


def test_chow_type_stat():
    # Mirrors crates/openquant/tests/structural_breaks.rs::test_chow_test
    min_length = 10
    log_prices = _log_prices()
    stats = structural_breaks.get_chow_type_stat(log_prices, min_length)

    want = REFERENCE["chow"]
    assert len(stats) == len(log_prices) - min_length * 2 == want["len"]
    assert _close(max(stats), want["max"])
    assert _close(_mean(stats), want["mean"])
    assert _close(stats[3], want["at_3"])


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

    # AFML 17.3.2 values (#104 fixed sigma_t^2 for sigma_t, #173 the divisor of sigma_t^2).
    _assert_csw(one_critical, two_critical, "critical_value")
    _assert_csw(one_stat, two_stat, "stat")


def _assert_csw(one, two, field):
    for name, values in [("one_sided", one), ("two_sided", two)]:
        want = REFERENCE["chu_stinchcombe_white"][name][field]
        assert _close(max(values), want["max"], rel=1e-10), (name, field, "max")
        assert _close(_mean(values), want["mean"], rel=1e-10), (name, field, "mean")
        assert _close(values[20], want["at_20"], rel=1e-10), (name, field, "[20]")


def test_chu_stinchcombe_white_statistic_matches_afml():
    # #147's FINDING, fixed by #173: sigma_t^2 is the mean of the squared differences up to
    # bar t (AFML 17.3.2); it used to divide their sum by one fewer than their number.
    log_prices = _log_prices()
    _, one_stat = structural_breaks.get_chu_stinchcombe_white_statistics(log_prices, "one_sided")
    _, two_stat = structural_breaks.get_chu_stinchcombe_white_statistics(log_prices, "two_sided")
    _assert_csw(one_stat, two_stat, "stat")


def test_chu_stinchcombe_white_by_hand():
    # Mirrors structural_breaks.rs::test_chu_stinchcombe_white_by_hand: y = 0, 1, 3 gives
    # sigma^2 = (1 + 4) / 2 and S = 3 / sqrt 5 (the old divisor gave 3 / sqrt 10).
    critical, stat = structural_breaks.get_chu_stinchcombe_white_statistics(
        [0.0, 1.0, 3.0], "one_sided"
    )
    assert _close(stat[0], 3 / math.sqrt(5), rel=1e-14)
    assert _close(critical[0], math.sqrt(4.6 + math.log(2)), rel=1e-14)


@pytest.mark.parametrize(
    "model", ["linear", "quadratic", "sm_power", "sm_poly_1", "sm_poly_2", "sm_exp"]
)
def test_sadf_pointwise_reference_values(model):
    # Mirrors crates/openquant/tests/structural_breaks.rs::sadf_*_match_afml_on_prefix and
    # sadf_quadratic_and_martingale_models_match_afml (fixed in #166). SADF at bar t only
    # looks at bars <= t, so every value on a 60-bar prefix of the fixture equals the
    # full-series one (the generator checks this). The full series is not run here (the Rust
    # test_sadf_test is #[ignore]d as long-running; one model takes minutes through a debug
    # build), so the whole-series means are not ported.
    min_length, lags = 20, 5
    prefix = REFERENCE["sadf_prefix"]
    log_prices = _log_prices()[: prefix["n_bars"]]
    out = structural_breaks.get_sadf(log_prices, model, True, min_length, lags)

    want = prefix["models"][model]
    assert len(out) == len(log_prices) - min_length - lags - 1 == want["len"]
    # 1e-7: the normal-equations inverse (snippet 17.4, as in the library) is off from a QR
    # solve by up to ~2e-9 relative in these statistics.
    assert _close(out[29], want["at_29"], rel=1e-7)
    for i, (got, ref) in enumerate(zip(out, want["values"], strict=True)):
        assert _close(got, ref, rel=1e-7), (model, i, got, ref)


@pytest.mark.parametrize("model", ["sm_poly_2", "sm_exp", "sm_power"])
def test_sadf_martingale_statistic_ignores_the_trend_sign(model):
    # AFML 17.4.3 takes |beta| / se (#166): the reciprocal series negates log y and so beta,
    # and must give the same, non-negative statistic.
    (close,) = load_csv_columns("shared/dollar_bar_sample.csv", ["close"])
    prices = list(close[:60])
    up = structural_breaks.get_sadf(prices, model, True, 20, 1)
    down = structural_breaks.get_sadf([1.0 / p for p in prices], model, True, 20, 1)
    assert all(math.isfinite(v) and v >= 0.0 for v in up)
    assert all(_close(a, b) for a, b in zip(up, down, strict=True))


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
    with pytest.raises(ValueError, match="unknown test type"):
        structural_breaks.get_chu_stinchcombe_white_statistics(log_prices, "rubbish text")
    with pytest.raises(ValueError, match="unknown model"):
        structural_breaks.get_sadf(log_prices, "rubbish_string", True, 20, 5)
    with pytest.raises(ValueError, match="too short"):
        structural_breaks.get_sadf([1.0, 2.0, 3.0], "linear", True, 20, 5)
