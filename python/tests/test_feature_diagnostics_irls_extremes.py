import math
from decimal import Decimal, localcontext

import pytest
from openquant import feature_diagnostics as fd

X = [[-1.0]] * 4 + [[1.0]] * 4
Y = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]


def test_huge_weight_logistic_fit_matches_analytic_nonseparable_model():
    model = fd._fit_linear_probability_model(X, Y, [1e308] * 8)
    assert model.intercept == pytest.approx(0.0, abs=1e-10)
    assert model.coeffs == pytest.approx([math.log(3.0)], abs=1e-8)


@pytest.mark.parametrize(
    "design,labels,want",
    [
        ([[v[0], v[0]] for v in X], Y, [0.0, math.log(3.0) / 2, math.log(3.0) / 2]),
        ([[1.0]] * 8, [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [math.log(3.0) / 2] * 2),
        ([[0.0]] * 8, Y, [0.0, 0.0]),
        (X, [0.0, 1.0] * 4, [0.0, 0.0]),
    ],
)
def test_recovery_preserves_ridge_for_degenerate_designs(design, labels, want):
    model = fd._fit_linear_probability_model(design, labels, [1e308] * 8)
    assert [model.intercept, *model.coeffs] == pytest.approx(want, abs=1e-8)


def test_recovery_preserves_non_negligible_ridge():
    # Independent Decimal bisection for 8*(sigmoid(beta)-3/4)+beta=0.
    with localcontext() as ctx:
        ctx.prec = 80
        lo, hi = Decimal(0), Decimal(2)
        for _ in range(220):
            mid = (lo + hi) / 2
            gradient = 8 * (1 / (1 + (-mid).exp()) - Decimal("0.75")) + mid
            if gradient > 0:
                hi = mid
            else:
                lo = mid
        want = float((lo + hi) / 2)
    model = fd._fit_linear_probability_model(X, Y, [1e308] * 8, ridge=1e308)
    assert model.coeffs == pytest.approx([want], abs=1e-8)


def test_finite_initial_loss_recovers_material_tail_cancellation():
    # The original initial loss and first Newton system are finite. Decimal follows
    # 50 exact Newton steps; this tests the iteration cap, not convergence of separable data.
    with localcontext() as ctx:
        ctx.prec = 100
        beta = Decimal(0)
        ridge = Decimal.from_float(1e-3) / Decimal.from_float(1e308)
        for _ in range(50):
            q = (-beta).exp()
            p = q / (1 + q)
            beta -= (-p + ridge * beta) / (p * (1 - p) + ridge)
        want = float(beta)
    model = fd._fit_linear_probability_model([[0.0]], [1.0], [1e308])
    assert model.intercept == pytest.approx(want, abs=1e-8)
    assert model.coeffs == [0.0]


def test_recovery_checks_logit_change_with_large_finite_features():
    design = [[row[0] * 1e150] for row in X]
    model = fd._fit_linear_probability_model(design, Y, [1e150] * 8)
    assert fd._predict_proba(model, [[1e150]]) == pytest.approx([0.75], abs=1e-9)
    assert model.coeffs[0] * 1e150 == pytest.approx(math.log(3.0), abs=1e-8)


def test_recovery_keeps_correlated_feature_signal_with_tiny_ridge():
    design, labels = [], []
    for x, z, positive in [(1.0, 1.0, 3), (1.0, -1.0, 2), (-1.0, 1.0, 2), (-1.0, -1.0, 1)]:
        design.extend([[x, x + 1e-7 * z]] * 4)
        labels.extend([1.0] * positive + [0.0] * (4 - positive))
    model = fd._fit_linear_probability_model(design, labels, [1e308] * 16)
    # Four binomial cells give logits ln3,0,0,-ln3. Solve that two-by-two
    # linear system using the actual rounded input feature coordinates.
    a, b = 1.0 + 1e-7, 1.0 - 1e-7
    second = math.log(3.0) / (a - b)
    assert model.coeffs == pytest.approx([-b * second, second], rel=1e-7)
    assert fd._predict_proba(model, design[::4]) == pytest.approx([0.75, 0.5, 0.5, 0.25], abs=1e-8)


@pytest.mark.parametrize("label", [-1.0, 0.5, 1e308])
def test_exceptional_recovery_does_not_extend_the_binary_label_domain(label):
    # The public contract is binary labels. Leave out-of-domain legacy behavior
    # to the ordinary path instead of creating giant exact Newton steps.
    assert fd._recover_weighted_logistic([[1.0, 0.0]], [label], [1e308], 1e-3, 50, 1e-8) is None
