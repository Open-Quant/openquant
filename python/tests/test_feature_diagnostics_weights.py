import math

import pytest
from openquant import feature_diagnostics as fd


@pytest.mark.parametrize("method", [fd.mda_importance, fd.sfi_importance])
@pytest.mark.parametrize(
    "weights",
    [
        [0.0] * 12,
        [-0.0] * 12,
        [math.nan] + [1.0] * 11,
        [math.inf] + [1.0] * 11,
        [-1.0] + [1.0] * 11,
    ],
)
def test_invalid_weights_are_rejected(method, weights):
    with pytest.raises(ValueError, match="sample_weight"):
        method(
            [[float(i % 2)] for i in range(12)],
            [float(i % 2) for i in range(12)],
            sample_weight=weights,
            event_end_indices=list(range(12)),
            n_splits=3,
            pct_embargo=0,
        )


@pytest.mark.parametrize("method", [fd.mda_importance, fd.sfi_importance])
@pytest.mark.parametrize(
    ("weights", "role"), [([0.0] * 4 + [1.0] * 8, "test"), ([1.0] * 4 + [0.0] * 8, "train")]
)
def test_fold_requires_positive_weight_mass(method, weights, role):
    with pytest.raises(ValueError, match=rf"fold 0.*{role}.*positive"):
        method(
            [[float(i % 2)] for i in range(12)],
            [float(i % 2) for i in range(12)],
            sample_weight=weights,
            event_end_indices=list(range(12)),
            n_splits=3,
            pct_embargo=0,
        )


@pytest.mark.parametrize("metric", ["neg_log_loss", "accuracy", "f1"])
def test_empty_weighted_score_is_not_a_score(metric):
    with pytest.raises(ValueError, match="positive"):
        fd._score([0.0, 1.0], [0.9, 0.1], metric, [0.0, -0.0])


def test_positive_mass_all_negative_f1_remains_zero():
    assert fd._score([0.0, 0.0], [0.1, 0.1], "f1", [0.0, 2.0]) == 0.0


def test_validation_does_not_sum_or_rescale_finite_weights():
    weights = [0.0, -0.0, 1e308, 1e308, math.ulp(0.0)]
    assert fd._sample_weight(weights, len(weights)) == weights


@pytest.mark.parametrize("method", [fd.mda_importance, fd.sfi_importance])
def test_mixed_zero_positive_folds_remain_valid(method):
    out = method(
        [[float(i % 2)] for i in range(12)],
        [float(i % 2) for i in range(12)],
        sample_weight=[0.0, 1.0] * 6,
        event_end_indices=list(range(12)),
        n_splits=3,
        pct_embargo=0,
    )
    assert all(math.isfinite(r["mean"]) for r in out["records"])


def test_mdi_zero_mass_bootstrap_is_not_a_trained_model():
    with pytest.raises(ValueError, match="bootstrap training.*positive"):
        fd.mdi_importance(
            [[float(i % 2)] for i in range(12)],
            [float(i % 2) for i in range(12)],
            sample_weight=[1.0] + [0.0] * 11,
            n_estimators=2,
            seed=1,
        )
