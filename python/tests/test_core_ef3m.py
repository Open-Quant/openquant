import math

import pytest
from openquant import ef3m


def test_centered_moment_result():
    # Mirrors crates/openquant/tests/ef3m.rs::test_centered_moment_result
    raw = [0.701756, 2.591815, 0.450519, 24.689030, -57.756735]
    expected = 0.0
    for j in range(6):
        add_on = 1.0 if j == 5 else raw[5 - j - 1]
        expected += (-1.0) ** j * add_on * raw[0] ** j * math.comb(5, j)

    assert abs(ef3m.centered_moment(raw, 5) - expected) < 1e-7


def test_raw_moment_result():
    # Mirrors crates/openquant/tests/ef3m.rs::test_raw_moment_result
    centered = [0.0, 2.11, -4.373999999999999, 30.803699999999996, -153.58572]
    raw = ef3m.raw_moment(centered, 0.7)
    expected = [0.7, 2.6, 0.4, 25.0, -59.8]

    assert len(raw) == len(expected)
    for got, want in zip(raw, expected):
        assert abs(got - want) < 1e-7


def test_most_likely_parameters_result():
    # Mirrors crates/openquant/tests/ef3m.rs::test_most_likely_parameters_result (keys), and
    # pins the values: sigma/p columns are constant and mu_1/mu_2 are spread evenly around
    # 1.05/2.05, so the density modes must sit there.
    rows = [
        (1.0, 2.0, 0.5, 0.6, 0.3, 0.01),
        (1.1, 2.1, 0.5, 0.6, 0.3, 0.01),
        (1.05, 2.05, 0.5, 0.6, 0.3, 0.01),
    ]
    out = ef3m.most_likely_parameters(rows, 1000)

    assert sorted(out) == ["mu_1", "mu_2", "p_1", "sigma_1", "sigma_2"]
    assert out["mu_1"] == pytest.approx(1.05, abs=1e-3)
    assert out["mu_2"] == pytest.approx(2.05, abs=1e-3)
    assert out["sigma_1"] == pytest.approx(0.5, abs=1e-3)
    assert out["sigma_2"] == pytest.approx(0.6, abs=1e-3)
    assert out["p_1"] == pytest.approx(0.3, abs=1e-3)


def test_fit_m2n_single_loop_shape():
    # Mirrors crates/openquant/tests/ef3m.rs::test_single_fit_loop_and_mp_fit_types
    moments = [0.7, 2.6, 0.4, 25.0, -59.8]
    out = ef3m.fit_m2n(moments, epsilon=1e-2, factor=5.0, n_runs=3, variant=2, max_iter=1000)

    assert len(out) <= 1
    for mu_1, mu_2, sigma_1, sigma_2, p_1, error in out:
        assert 0.0 <= p_1 <= 1.0
        assert sigma_1 > 0.0 and sigma_2 > 0.0
        assert error >= 0.0


def test_ef3m_rejects_wrongly_typed_input():
    with pytest.raises(TypeError):
        ef3m.centered_moment(["a"], 1)
    with pytest.raises(OverflowError):
        ef3m.centered_moment([0.7, 2.6, 0.4, 25.0, -59.8], -1)


def test_centered_moment_too_few_moments_raises_value_error():
    with pytest.raises(ValueError):
        ef3m.centered_moment([0.7], 5)
