"""The i.i.d. null should not gain drift or silently drop imaginary inputs."""

import math
import warnings
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from openquant import cross_validation as cv


@pytest.mark.parametrize("magnitude", [1e16, 1e100, 1e300])
def test_bootstrap_centers_small_residual_during_large_cancellation(magnitude):
    values = np.array([[magnitude, 0.01], [1.0, 0.02], [-magnitude, 0.03]])
    exact = [Fraction(float(x)) for x in values[:, 0]]
    mean = sum(exact) / len(exact)
    expected = np.array([float(x - mean) for x in exact])
    ordinary = values[:, 1] - values[:, 1].mean()
    rng = np.random.default_rng(81)
    indices = rng.integers(0, len(values), len(values))
    with np.errstate(all="raise"):
        drawn = cv.bootstrap_returns(values)(np.random.default_rng(81))
    np.testing.assert_array_equal(drawn[:, 0], expected[indices])
    np.testing.assert_array_equal(drawn[:, 1], ordinary[indices])


def test_bootstrap_centers_before_rounding_a_subnormal_mean():
    values = np.array([0.0, math.ulp(0.0)])
    with np.errstate(all="raise"):
        drawn = cv.bootstrap_returns(values)(np.random.default_rng(81))
    # Both deviations are exactly half the least subnormal, so each rounds to zero.
    np.testing.assert_array_equal(drawn, np.zeros(2))


@pytest.mark.parametrize(
    "values",
    [
        np.array([0.25 + 0.5j, 0.75 - 0.25j]),
        np.array([np.complex128(0.25 + 0.5j), 0.75], dtype=object),
    ],
)
@pytest.mark.parametrize("method", ["bootstrap", "p_value"])
def test_complex_null_inputs_are_never_cast_to_reals(values, method):
    if method == "bootstrap":
        with pytest.raises(ValueError, match="returns must be real"):
            cv.bootstrap_returns(values)
    else:
        with pytest.raises(ValueError, match="null scores must be real"):
            cv.null_p_value(0.5, values)


@pytest.mark.parametrize("method", ["observed", "evaluated"])
def test_complex_observed_or_evaluated_scalar_is_rejected(method):
    score = np.complex128(1.5 + 0.5j)
    if method == "observed":
        with pytest.raises(ValueError, match="observed score must be real"):
            cv.null_p_value(score, [1.0, 2.0])
    else:
        with pytest.raises(ValueError, match="null score must be real"):
            cv.null_score_distribution(lambda data, splits: score, [None], lambda rng: None)


def test_seeded_cancelling_panels_match_exact_binary64_reference():
    rng = np.random.default_rng(171)
    for seed in range(100):
        # Place residuals between equal-and-opposite leading values. Random
        # exponents exercise representable magnitudes without overflow.
        leading = np.ldexp(rng.uniform(0.5, 1.0, 3), rng.integers(-900, 900, 3))
        residue = np.ldexp(leading, -rng.integers(52, 78, 3))
        panel = np.array([leading, residue, -leading])
        exact = [[Fraction(float(x)) for x in column] for column in panel.T]
        expected = np.asarray([[float(x - sum(c) / len(c)) for x in c] for c in exact]).T
        indices = np.random.default_rng(seed).integers(0, len(panel), len(panel))
        np.testing.assert_array_equal(
            cv.bootstrap_returns(panel)(np.random.default_rng(seed)), expected[indices]
        )


@pytest.mark.parametrize(
    "observations",
    [[0.1, 0.1, 0.1], [1e16, 1e16, 1e16 + 2.0], [-1e16, -1e16, -1e16 - 2.0]],
)
def test_large_common_offset_does_not_destroy_sub_ulp_deviations(observations):
    exact = list(map(Fraction, observations))
    mean = sum(exact) / len(exact)
    indices = np.random.default_rng(81).integers(0, len(exact), len(exact))
    expected = np.array([float(value - mean) for value in exact])
    np.testing.assert_array_equal(
        cv.bootstrap_returns(observations)(np.random.default_rng(81)), expected[indices]
    )


@pytest.mark.parametrize(
    "values",
    [
        [0.0, math.ulp(0.0), math.ulp(0.0)],
        [1.0, 1e300, -1e300, math.ulp(0.0), -1.0],
        [np.finfo(float).max, np.finfo(float).max, -np.finfo(float).max],
    ],
)
def test_fallback_exact_dyadic_centering_matches_fraction_bits(values):
    exact = list(map(Fraction, map(float, values)))
    mean = sum(exact) / len(exact)
    try:
        wanted = np.array([float(value - mean) for value in exact])
    except OverflowError:
        with pytest.raises(ValueError, match="representable as finite floats"):
            cv.bootstrap_returns(values)
        return
    indices = np.random.default_rng(23).integers(0, len(values), len(values))
    drawn = cv.bootstrap_returns(values)(np.random.default_rng(23))
    np.testing.assert_array_equal(drawn.view(np.uint64), wanted[indices].view(np.uint64))


def test_missing_masked_null_draw_never_uses_hidden_payload():
    expected = Fraction(1, 3)  # neither of the two visible draws meets the threshold
    low_payload = np.ma.array([1.0, -100.0, 2.0], mask=[False, True, False])
    high_payload = np.ma.array([1.0, 100.0, 2.0], mask=[False, True, False])
    assert cv.null_p_value(3.0, low_payload) == float(expected)
    assert cv.null_p_value(3.0, high_payload) == float(expected)
    with pytest.raises(ValueError, match="no finite scores"):
        cv.null_p_value(3.0, np.ma.array([1.0, 2.0], mask=True))


def test_bootstrap_rejects_masked_returns_instead_of_sampling_hidden_payload():
    sample = np.ma.array([1.0, 2.0, 500.0], mask=[False, False, True])
    with pytest.raises(ValueError, match="returns.*masked"):
        cv.bootstrap_returns(sample)


def test_masked_callback_scores_are_recorded_as_nan_without_reading_payload():
    scores = iter([np.ma.array(999.0, mask=True), np.ma.masked, 2.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = cv.null_score_distribution(
            lambda data, splits: next(scores), [None], lambda rng: None, n_null=3
        )
    np.testing.assert_array_equal(result, [np.nan, np.nan, 2.0])


def _exact_score(value):
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    return Fraction(*value.as_integer_ratio())


@pytest.mark.parametrize(
    ("observed", "draws"),
    [
        (2**53 + 1, np.array([2**53, 2**53 + 2], dtype=np.int64)),
        (np.uint64(2**64 - 2), np.array([2**64 - 3, 2**64 - 1], dtype=np.uint64)),
        (np.uint64(2**63), np.array([-1, 2**63 - 1], dtype=np.int64)),
        (2**53 + 1, np.array([2**53, 2**53 + 2], dtype=np.float64)),
        (np.float64(2**53), np.array([2**53 - 1, 2**53], dtype=np.uint64)),
        (2**53 + 1, [np.uint64(2**53), np.int64(2**53 + 2), 2.0]),
        (np.nextafter(1.0, 2.0), np.array([1.0, 2.0], dtype=np.float32)),
        (-0.25, np.array([-1, 0, 1], dtype=np.int64)),
        (math.ulp(0.0), np.array([-0.0, 0.0, np.nextafter(np.float32(0), np.float32(1))])),
        (Decimal("1.0000000000000001"), np.array([1.0, np.nextafter(1.0, 2.0)])),
        (Fraction(2**53 + 1, 2**53), np.array([1.0, np.nextafter(1.0, 2.0)])),
        (2**1024, np.array([-1.0, 0.0, np.finfo(float).max])),
        (-(2**1024), np.array([-np.finfo(float).max, -1.0])),
    ],
)
def test_null_tail_count_matches_exact_rational_score_ordering(observed, draws):
    scores = [_exact_score(value) for value in draws]
    count = sum(score >= _exact_score(observed) for score in scores)
    assert cv.null_p_value(observed, draws) == float(Fraction(count + 1, len(scores) + 1))


@pytest.mark.skipif(
    np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant,
    reason="platform has no wider longdouble mantissa",
)
def test_null_ordering_preserves_extended_precision_inside_binary64_range():
    low = np.longdouble(1)
    high = np.nextafter(low, np.longdouble(2))
    assert cv.null_p_value(high, np.array([low, high], dtype=np.longdouble)) == 2 / 3


@pytest.mark.skipif(
    np.finfo(np.longdouble).maxexp <= np.finfo(np.float64).maxexp,
    reason="platform has no wider longdouble exponent range",
)
def test_null_ordering_preserves_finite_extended_range_and_ignores_infinity():
    high = np.longdouble(2) ** 1500
    with np.errstate(all="raise"):
        assert cv.null_p_value(high, np.array([0, high, np.inf], dtype=np.longdouble)) == 2 / 3


@pytest.mark.skipif(
    np.finfo(np.longdouble).maxexp <= 15001,
    reason="platform has no longdouble range at exponent 15000",
)
def test_exact_integer_threshold_near_longdouble_top_needs_no_decimal_string():
    integer = 2**15000
    high = np.ldexp(np.longdouble(1), 15000)
    draws = np.array([0, high, np.nextafter(high, np.longdouble(np.inf))])
    assert cv.null_p_value(integer + 1, draws) == 1 / 2


def test_null_mixed_inputs_skip_missing_without_rounding_visible_values():
    draws = np.ma.array(
        [np.uint64(2**64 - 1), np.int64(1), np.uint64(2**64 - 2), np.nan],
        dtype=object,
        mask=[True, False, False, False],
    )
    assert cv.null_p_value(np.uint64(2**64 - 2), draws) == 2 / 3
    assert cv.null_p_value(np.uint64(2**64 - 1), draws) == 1 / 3


def test_null_object_inputs_keep_none_as_missing():
    assert cv.null_p_value(2.0, np.array([None, 1.0, 2.0], dtype=object)) == 2 / 3


@pytest.mark.parametrize("container", [list, tuple])
def test_null_float_sequence_matches_exact_array_around_integer_threshold(container):
    low, high = float(2**53), float(2**53 + 2)
    scores = container([low, high, np.nan, float("inf")])
    assert (
        cv.null_p_value(2**53 + 1, scores)
        == cv.null_p_value(2**53 + 1, np.asarray(scores))
        == 2 / 3
    )
