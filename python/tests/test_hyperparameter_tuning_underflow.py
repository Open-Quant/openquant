"""Tiny positive score weights must not quantize log loss or change selection."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from openquant import hyperparameter_tuning as ht


def test_subnormal_weighted_log_loss_matches_decimal_reference():
    weights = [float.fromhex("0x0.0000000000001p-1022"), float.fromhex("0x0.0000000000003p-1022")]
    with localcontext() as ctx:
        ctx.prec = 80
        expected = float(
            (
                Decimal.from_float(weights[0]) * Decimal("0.75").ln()
                + Decimal.from_float(weights[1]) * Decimal("0.5").ln()
            )
            / sum(Decimal.from_float(w) for w in weights)
        )
    actual = ht.classification_score([1, 0], [0.75, 0.5], weights)
    assert actual == pytest.approx(expected, rel=0.0, abs=1e-14)


def test_purged_search_tiny_weights_do_not_turn_different_candidates_into_ties():
    class ConstantProbability:
        def __init__(self, params):
            self.p = params["p"]

        def fit(self, X, y, sample_weight=None):
            return self

        def predict_proba(self, X):
            return np.full(len(X), self.p)

    t0 = np.datetime64("2020-01-01", "ns") + np.arange(4) * np.timedelta64(1, "D")
    result = ht.purged_search(
        ConstantProbability,
        [{"p": 0.9}, {"p": 0.75}],
        np.ones((4, 1)),
        np.ones(4),
        t0,
        t0,
        n_splits=2,
        pct_embargo=0.0,
        sample_weight=np.full(4, np.nextafter(0.0, 1.0)),
    )
    assert result["best_params"] == {"p": 0.9}
    assert result["best_score"] == pytest.approx(np.log(0.9), rel=0.0, abs=1e-14)
    assert result["trials"][1]["mean_score"] == pytest.approx(np.log(0.75), rel=0.0, abs=1e-14)
