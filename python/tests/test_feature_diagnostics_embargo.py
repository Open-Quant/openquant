"""Feature-diagnostic embargoes must extend beyond test labels, not just test rows."""

import pytest
from openquant import feature_diagnostics as fd


def _two_row_horizon():
    return [(i, min(i + 2, 11)) for i in range(12)]


def test_embargo_counts_fresh_rows_after_the_test_label_horizon():
    # Fold 0 tests rows 0..3, whose labels run through row 5. Rows 6 and 7
    # therefore consume the two-row embargo, rather than rows already purged.
    assert fd._purged_kfold_splits(_two_row_horizon(), 3, 1 / 6) == [
        ([8, 9, 10, 11], [0, 1, 2, 3]),
        ([0, 1], [4, 5, 6, 7]),
        ([0, 1, 2, 3, 4, 5], [8, 9, 10, 11]),
    ]


def test_embargo_uses_latest_test_label_end_not_last_test_row_end():
    intervals = [(i, 8 if i == 1 else i) for i in range(12)]
    train, test = fd._purged_kfold_splits(intervals, 3, 1 / 6)[0]
    assert test == [0, 1, 2, 3]
    assert train == [11]  # Rows 9 and 10 follow the label ending at row 8.


def test_a_fully_embargoed_training_fold_is_rejected():
    intervals = [(i, 8 if i == 1 else i) for i in range(12)]
    with pytest.raises(ValueError, match="empty train/test fold"):
        fd._purged_kfold_splits(intervals, 3, 0.25)


def test_documented_pre_fold_embargo_is_preserved():
    intervals = [(i, i) for i in range(12)]
    train, test = fd._purged_kfold_splits(intervals, 3, 1 / 6)[1]
    assert test == [4, 5, 6, 7]
    assert train == [0, 1, 10, 11]  # Retains the Python API's extra pre-fold exclusion.


def test_zero_embargo_keeps_the_existing_purge_only_splits():
    assert fd._purged_kfold_splits(_two_row_horizon(), 3, 0.0) == [
        ([6, 7, 8, 9, 10, 11], [0, 1, 2, 3]),
        ([0, 1, 10, 11], [4, 5, 6, 7]),
        ([0, 1, 2, 3, 4, 5], [8, 9, 10, 11]),
    ]


@pytest.mark.parametrize("method", ["mda", "sfi", "substitution"])
def test_public_feature_scores_use_the_post_label_embargo(method):
    x = [[float(i)] for i in range(12)]
    y = [1.0] * 8 + [0.0] * 4
    kwargs = dict(
        event_end_indices=[end for _, end in _two_row_horizon()],
        n_splits=3,
        pct_embargo=1 / 6,
        scoring="accuracy",
    )
    if method == "mda":
        score = fd.mda_importance(x, y, **kwargs)["cv"]["mean_base_score"]
    elif method == "sfi":
        score = fd.sfi_importance(x, y, **kwargs)["records"][0]["mean"]
    else:
        score = fd.substitution_effect_report(x, y, orthogonalize=False, **kwargs)["baseline_mda"][
            "cv"
        ]["mean_base_score"]
    # The corrected training folds contain only labels 0, 1, and 1 respectively.
    # Their test accuracies are therefore 0, 1, and 0: an independent 1/3 mean.
    assert score == pytest.approx(1 / 3)
