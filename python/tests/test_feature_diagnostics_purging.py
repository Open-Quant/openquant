"""Regression tests for purged-CV leakage in feature_diagnostics.

The default ``event_end_indices=None`` used to build degenerate one-row
label intervals ``(i, i)``, which can only overlap the test rows themselves.
Purging therefore removed zero *extra* rows and the returned ``cv`` block
still claimed ``method == "purged_kfold"`` -- silent leakage.
"""

import random

import pytest

import openquant
from openquant.feature_diagnostics import _build_intervals, _purged_kfold_splits


def _dataset(n: int = 120):
    rng = random.Random(7)
    x = []
    y = []
    for i in range(n):
        f0 = rng.gauss(0.0, 1.0)
        f1 = rng.gauss(0.0, 1.0)
        score = 0.8 * f0 + 0.3 * rng.gauss(0.0, 1.0)
        x.append([f0, f1])
        y.append(1.0 if score > 0 else 0.0)
    return x, y, ["f0", "f1"]


def test_purging_removes_rows_whose_labels_overlap_the_test_fold():
    """With real label spans, purging must drop train rows -- and the
    degenerate one-row intervals must be shown to drop none."""
    n = 100
    horizon = 10
    spans = [min(n - 1, i + horizon) for i in range(n)]

    purged = _purged_kfold_splits(
        _build_intervals(spans, n),
        n_splits=5,
        pct_embargo=0.0,
    )
    degenerate = _purged_kfold_splits(
        [(i, i) for i in range(n)],
        n_splits=5,
        pct_embargo=0.0,
    )

    # Fold 1 tests rows 20..39; rows 10..19 carry labels ending at 20..29.
    train_purged, test_idx = purged[1]
    train_degenerate, _ = degenerate[1]
    assert test_idx[0] == 20 and test_idx[-1] == 39

    leaked = [i for i in range(10, 20)]
    assert all(i in train_degenerate for i in leaked), "degenerate intervals purge nothing"
    assert not any(i in train_purged for i in leaked), "purging must drop overlapping rows"
    assert len(train_purged) < len(train_degenerate)


@pytest.mark.parametrize(
    "fn",
    ["mda_importance", "sfi_importance", "substitution_effect_report"],
)
def test_missing_label_spans_raises_instead_of_silently_not_purging(fn):
    x, y, names = _dataset()
    with pytest.raises(ValueError, match="event_end_indices"):
        getattr(openquant.feature_diagnostics, fn)(x, y, feature_names=names)


def test_explicit_opt_out_is_reported_as_unpurged():
    x, y, names = _dataset()
    out = openquant.feature_diagnostics.mda_importance(
        x, y, feature_names=names, allow_unpurged=True
    )
    assert out["cv"]["purged"] is False
    assert out["cv"]["method"] == "kfold_embargo_only"

    spans = [min(len(x) - 1, i + 5) for i in range(len(x))]
    purged_out = openquant.feature_diagnostics.mda_importance(
        x, y, feature_names=names, event_end_indices=spans
    )
    assert purged_out["cv"]["purged"] is True
    assert purged_out["cv"]["method"] == "purged_kfold"
