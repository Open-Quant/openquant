import math
import sys
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest
from openquant import feature_diagnostics as fd


def decimal_fraction(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


@pytest.mark.parametrize("magnitude", [709.0, 740.0, 745.0, 746.0, 760.0, 1392.0, 4096.0])
def test_subnormal_logistic_tail_matches_high_precision(magnitude):
    prob, tail = fd._fraction_logistic(Fraction(-magnitude))
    assert prob > 0 and tail > 0
    with localcontext() as ctx:
        ctx.prec = 100
        q = (-Decimal.from_float(magnitude)).exp()
        # log1p(q) differs from q by <q²/2, far below exp's relative float error.
        assert abs(decimal_fraction(tail) / q - 1) < Decimal("2e-14")
        assert abs(decimal_fraction(prob) / (q / (1 + q)) - 1) < Decimal("2e-14")


@pytest.mark.parametrize(
    "ridge,feature,max_iter", [(1e-25, 0.0, 1000), (1e-300, 0.0, 2000), (1e-3, 1e10, 1000)]
)
def test_underflow_recovery_reaches_stationary_logit(ridge, feature, max_iter):
    model = fd._fit_linear_probability_model(
        [[feature]], [1.0], [1e308], ridge=ridge, max_iter=max_iter
    )
    observed = model.intercept + feature * model.coeffs[0]
    with localcontext() as ctx:
        ctx.prec = 100
        penalty = (
            Decimal.from_float(ridge)
            / (1 + Decimal.from_float(feature) ** 2)
            / Decimal.from_float(1e308)
        )
        beta = Decimal(0)
        for _ in range(max_iter):
            q = (-beta).exp()
            p = q / (1 + q)
            beta -= (-p + penalty * beta) / (p * (1 - p) + penalty)
        assert observed == pytest.approx(float(beta), abs=2e-5)
        got = Decimal.from_float(observed)
        q = (-got).exp()
        assert abs((q / (1 + q)) / (penalty * got) - 1) < Decimal("2e-5")


def test_normal_logistic_tails_keep_existing_bits():
    for magnitude in [0.0, 0.5, 1.0, 32.0, 100.0, 700.0, -math.log(sys.float_info.min)]:
        exponential = math.exp(-magnitude)
        if exponential < sys.float_info.min:
            continue
        q = Fraction(exponential)
        for sign in [-1, 1]:
            z = Fraction(sign * magnitude)
            prob, tail = fd._fraction_logistic(z)
            assert prob == (1 / (1 + q) if z >= 0 else q / (1 + q))
            assert tail == Fraction(math.log1p(exponential))


def test_tail_recovery_bound_is_exact():
    for z in [Fraction(math.nextafter(4096.0, math.inf)), Fraction(4096) + Fraction(1, 2**50)]:
        prob, tail = fd._fraction_logistic(-z)
        assert prob == tail == 0


@pytest.mark.parametrize("direction", [-math.inf, 0.0, math.inf])
def test_tail_recovery_normal_subnormal_boundary(direction):
    boundary = -math.log(sys.float_info.min)
    z = boundary if direction == 0 else math.nextafter(boundary, direction)
    _, tail = fd._fraction_logistic(Fraction(z))
    with localcontext() as ctx:
        ctx.prec = 100
        expected = (-Decimal.from_float(z)).exp()
        assert abs(decimal_fraction(tail) / expected - 1) < Decimal("2e-14")


def test_underflow_recovery_preserves_private_iteration_cap():
    model = fd._fit_linear_probability_model([[0.0]], [1.0], [1e308], ridge=1e-300, max_iter=1000)
    with localcontext() as ctx:
        ctx.prec = 100
        penalty = Decimal.from_float(1e-300) / Decimal.from_float(1e308)
        beta = Decimal(0)
        for _ in range(1000):
            q = (-beta).exp()
            p = q / (1 + q)
            beta -= (-p + penalty * beta) / (p * (1 - p) + penalty)
        assert model.intercept == pytest.approx(float(beta), abs=2e-5)
        assert model.intercept < 1100  # The true stationary logit is about1392.7.
