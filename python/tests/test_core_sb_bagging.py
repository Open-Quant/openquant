import math

import numpy as np
import pytest
from openquant import sampling, sb_bagging
from openquant.cross_validation import purged_kfold_splits

N_SAMPLES = 240


def _synthetic_dataset():
    # Same data as crates/openquant/tests/sb_bagging.rs::synthetic_dataset
    x, y_clf, y_reg = [], [], []
    for i in range(N_SAMPLES):
        t = float(i)
        f0 = math.sin(t / 12.0)
        f1 = math.cos(t / 19.0)
        f2 = math.sin(t / 7.0) * 0.5
        signal = 1.4 * f0 + 0.8 * f1 - 0.3 * f2
        y_clf.append(1 if signal > 0.0 else 0)
        y_reg.append(2.0 if signal > 0.0 else 1.0)
        x.append(
            [
                signal,
                f0,
                f1,
                f2,
                math.sin(t / 5.0) * 0.2,
                math.cos(t / 17.0) * 0.2,
                (i % 11) / 11.0,
                ((i * 7) % 13) / 13.0,
            ]
        )
    # One label per row of x, each spanning the next 6 bars.
    t1 = [(start, min(start + 6, N_SAMPLES - 1)) for start in range(N_SAMPLES)]
    ind_mat = sampling.get_ind_matrix(t1, list(range(N_SAMPLES)))
    return x, y_clf, y_reg, ind_mat


def _accuracy(predictions, y):
    return sum(p == t for p, t in zip(predictions, y)) / len(y)


def _errors(predictions, y):
    mse = sum((p - t) ** 2 for p, t in zip(predictions, y)) / len(y)
    mae = sum(abs(p - t) for p, t in zip(predictions, y)) / len(y)
    return mse, mae


# The bindings fit and predict on the same rows, so unlike the Rust tests these are
# in-sample scores; the thresholds are the Rust ones.


def test_sb_classifier_single_feature():
    # Mirrors crates/openquant/tests/sb_bagging.rs::test_sb_classifier
    x, y, _, ind_mat = _synthetic_dataset()
    signal_only = [[row[0]] for row in x]

    out = sb_bagging.fit_predict_sb_classifier(
        signal_only, y, ind_mat, n_estimators=10, random_state=1
    )

    assert len(out["predictions"]) == N_SAMPLES
    assert _accuracy(out["predictions"], y) >= 0.55
    assert math.isfinite(out["oob_score"])
    again = sb_bagging.fit_predict_sb_classifier(
        signal_only, y, ind_mat, n_estimators=10, random_state=1
    )
    assert again["predictions"] == out["predictions"]


def test_sb_regressor_single_feature():
    # Mirrors crates/openquant/tests/sb_bagging.rs::test_sb_regressor
    x, _, y, ind_mat = _synthetic_dataset()
    signal_only = [[row[0]] for row in x]

    out = sb_bagging.fit_predict_sb_regressor(
        signal_only, y, ind_mat, n_estimators=10, random_state=1
    )

    mse, mae = _errors(out["predictions"], y)
    assert len(out["predictions"]) == N_SAMPLES
    assert mse < 0.4
    assert mae < 0.5
    assert math.isfinite(out["oob_score"])


def test_sb_bagging_rejects_invalid_inputs():
    # Mirrors crates/openquant/tests/sb_bagging.rs::test_value_error_raise
    x, y, _, ind_mat = _synthetic_dataset()
    with pytest.raises(ValueError, match="n_estimators must be positive"):
        sb_bagging.fit_predict_sb_classifier(x, y, ind_mat, n_estimators=0)
    with pytest.raises(ValueError, match="max_samples is out of range"):
        sb_bagging.fit_predict_sb_classifier(x, y, ind_mat, max_samples=2.0)
    with pytest.raises(ValueError, match="disagree on the number of samples"):
        sb_bagging.fit_predict_sb_classifier(x, y[:-1], ind_mat)
    with pytest.raises(ValueError, match="rectangular"):
        sb_bagging.fit_predict_sb_regressor([[1.0, 2.0], [1.0]], [0.0, 1.0], ind_mat)
    # ind_mat must have one label column per row of x.
    with pytest.raises(ValueError, match="disagree on the number of samples"):
        sb_bagging.fit_predict_sb_classifier(x[:-5], y[:-5], ind_mat)
    with pytest.raises(ValueError, match="disagree on the number of samples"):
        sb_bagging.fit_predict_sb_classifier(x, y, ind_mat, sample_weight=[1.0])
    with pytest.raises(ValueError, match="sample weights must be finite"):
        sb_bagging.fit_predict_sb_classifier(x, y, ind_mat, sample_weight=[-1.0] * N_SAMPLES)


def test_sb_bagging_uses_sample_weight():
    # Rows 0..30 follow y = x and rows 30..60 follow y = -x; zero weight on the second half
    # leaves the first half's line (y = x) at every row.
    n = 60
    x = [[(r % 30) - 14.5] for r in range(n)]
    y = [row[0] if r < 30 else -row[0] for r, row in enumerate(x)]
    ind_mat = sampling.get_ind_matrix(
        [(2 * i, 2 * i + 7) for i in range(n)], list(range(2 * n + 8))
    )
    weights = [1.0 if r < 30 else 0.0 for r in range(n)]

    weighted = sb_bagging.fit_predict_sb_regressor(x, y, ind_mat, sample_weight=weights)
    unweighted = sb_bagging.fit_predict_sb_regressor(x, y, ind_mat)
    assert weighted["predictions"] == pytest.approx([row[0] for row in x])
    assert weighted["predictions"] != pytest.approx(unweighted["predictions"])


def test_sb_classifier_all_features():
    # Mirrors crates/openquant/tests/sb_bagging.rs::test_sb_classifier
    x, y, _, ind_mat = _synthetic_dataset()
    out = sb_bagging.fit_predict_sb_classifier(x, y, ind_mat, n_estimators=10, random_state=1)
    assert _accuracy(out["predictions"], y) >= 0.55


# --- the model object: fit on some rows, predict on others (issue #187) -------------------

SPAN = 6


def _labels(rows):
    """Spans of the labels at `rows` (each spans the next SPAN bars) and their ind_mat."""
    spans = [(r, min(r + SPAN, N_SAMPLES - 1)) for r in rows]
    return spans, sampling.get_ind_matrix(spans, list(range(N_SAMPLES)))


def _proba_ok(proba, n_rows):
    arr = np.asarray(proba, dtype=float)
    assert arr.shape == (n_rows, 2)
    assert ((arr >= 0.0) & (arr <= 1.0)).all()
    np.testing.assert_allclose(arr.sum(axis=1), 1.0, atol=1e-12)
    return arr


def test_classifier_predicts_held_out_rows():
    # Mirrors crates/openquant/tests/sb_bagging.rs::test_sb_classifier, through Python.
    x, y, _, _ = _synthetic_dataset()
    split = int(N_SAMPLES * 0.6)
    _, ind_train = _labels(range(split))
    model = sb_bagging.SequentiallyBootstrappedBaggingClassifier(
        n_estimators=100, oob_score=True, random_state=1
    )
    assert model.fit(x[:split], y[:split], ind_train) is model
    assert model.n_features_in_ == 8 and model.classes_ == [0, 1]
    assert len(model.estimators_samples_) == 100
    assert math.isfinite(model.oob_score_)

    predictions = model.predict(x[split:])
    assert _accuracy(predictions, y[split:]) >= 0.55
    proba = _proba_ok(model.predict_proba(x[split:]), N_SAMPLES - split)
    # The probability is the share of 100 votes, and predict is its majority.
    np.testing.assert_allclose(proba[:, 1] * 100, np.round(proba[:, 1] * 100), atol=1e-9)
    assert predictions == [int(p >= 0.5) for p in proba[:, 1]]


def test_classifier_is_reproducible_with_a_seed():
    x, y, _, ind_mat = _synthetic_dataset()
    proba = [
        sb_bagging.SequentiallyBootstrappedBaggingClassifier(n_estimators=15, random_state=s)
        .fit(x, y, ind_mat)
        .predict_proba(x)
        for s in (4, 4, 5)
    ]
    assert proba[0] == proba[1]
    assert proba[0] != proba[2]


def test_classifier_matches_fit_predict_sb_classifier():
    x, y, _, ind_mat = _synthetic_dataset()
    old = sb_bagging.fit_predict_sb_classifier(x, y, ind_mat, n_estimators=12, random_state=9)
    model = sb_bagging.SequentiallyBootstrappedBaggingClassifier(
        n_estimators=12, random_state=9, oob_score=True
    ).fit(x, y, ind_mat)
    assert model.predict(x) == old["predictions"]
    assert model.oob_score_ == old["oob_score"]


def test_classifier_out_of_fold_with_purged_kfold():
    x, y, _, _ = _synthetic_dataset()
    x_arr, y_arr = np.asarray(x), np.asarray(y)
    spans, ind_mat = _labels(range(N_SAMPLES))
    t0 = [s for s, _ in spans]
    t1 = [e for _, e in spans]
    ind_arr = np.asarray(ind_mat)

    oof = np.full((N_SAMPLES, 2), np.nan)
    splits = purged_kfold_splits(t0, t1, n_splits=4, pct_embargo=0.01)
    for train, test in splits:
        # The training labels' columns of the indicator matrix: one per training row.
        model = sb_bagging.SequentiallyBootstrappedBaggingClassifier(
            n_estimators=25, random_state=11
        ).fit(x_arr[train].tolist(), y_arr[train].tolist(), ind_arr[:, train].tolist())
        oof[test] = model.predict_proba(x_arr[test].tolist())

    covered = np.concatenate([test for _, test in splits])
    assert sorted(covered.tolist()) == list(range(N_SAMPLES))  # every row scored exactly once
    _proba_ok(oof, N_SAMPLES)
    assert _accuracy((oof[:, 1] >= 0.5).astype(int).tolist(), y) >= 0.55


def test_classifier_validates_prediction_inputs():
    x, y, _, ind_mat = _synthetic_dataset()
    model = sb_bagging.SequentiallyBootstrappedBaggingClassifier(n_estimators=3)
    assert model.n_features_in_ is None and model.oob_score_ is None
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(x)
    with pytest.raises(ValueError, match="not fitted"):
        model.predict_proba(x)
    model.fit(x, y, ind_mat)
    with pytest.raises(ValueError, match="2 feature columns but the model was fitted on 8"):
        model.predict_proba([row[:2] for row in x])
    with pytest.raises(ValueError, match="disagree on the number of samples"):
        model.fit(x[:-5], y[:-5], ind_mat)
    # A failed fit leaves the model unfitted rather than half-fitted.
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(x)
    assert "n_estimators=3" in repr(model)
