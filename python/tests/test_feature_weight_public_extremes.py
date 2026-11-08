from math import log

import pytest
from openquant import feature_diagnostics as diagnostics

X = [[-1.0]] * 4 + [[1.0]] * 4
Y = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]


def test_huge_finite_weights_keep_single_feature_bootstrap_importance():
    result = diagnostics.mdi_importance(X, Y, sample_weight=[1e308] * 8, n_estimators=2, seed=42)
    assert result["records"][0]["mean"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("method", "expected"),
    [("sfi_importance", -log(2.0)), ("mda_importance", 0.0)],
)
def test_huge_finite_weights_do_not_break_rank_deficient_training_folds(method, expected):
    # Constant-X folds make intercept and slope collinear, but positive ridge
    # still defines a unique fit. Minimum-norm training coefficients cancel
    # on the opposite-sign test X, giving held-out probability 0.5.
    result = getattr(diagnostics, method)(
        X, Y, sample_weight=[1e308] * 8, n_splits=2, allow_unpurged=True
    )
    assert result["records"][0]["mean"] == pytest.approx(expected, rel=1e-7, abs=1e-7)
