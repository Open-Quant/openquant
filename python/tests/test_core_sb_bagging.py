import math

import pytest

from openquant import sampling, sb_bagging

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
    t1 = [(start, start + 6) for start in range(0, N_SAMPLES - 6, 3)]
    ind_mat = sampling.get_ind_matrix(t1, list(range(N_SAMPLES)))
    return x, y_clf, y_reg, ind_mat


def _accuracy(predictions, y):
    # Class predictions cross the boundary as `bytes` (Rust Vec<u8>), hence list().
    return sum(p == t for p, t in zip(list(predictions), y)) / len(y)


def _errors(predictions, y):
    mse = sum((p - t) ** 2 for p, t in zip(predictions, y)) / len(y)
    mae = sum(abs(p - t) for p, t in zip(predictions, y)) / len(y)
    return mse, mae


# The bindings fit and predict on the same rows, so unlike the Rust tests these are
# in-sample scores; the thresholds are the Rust ones. A single feature column is used for
# the passing tests because multi-column matrices are scrambled (see the xfails below).


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
    assert list(again["predictions"]) == list(out["predictions"])


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
    with pytest.raises(ValueError, match="InvalidEstimators"):
        sb_bagging.fit_predict_sb_classifier(x, y, ind_mat, n_estimators=0)
    with pytest.raises(ValueError, match="MaxSamplesOutOfRange"):
        sb_bagging.fit_predict_sb_classifier(x, y, ind_mat, max_samples=2.0)
    with pytest.raises(ValueError, match="DimensionMismatch"):
        sb_bagging.fit_predict_sb_classifier(x, y[:-1], ind_mat)
    with pytest.raises(ValueError, match="rectangular"):
        sb_bagging.fit_predict_sb_regressor([[1.0, 2.0], [1.0]], [0.0, 1.0], ind_mat)


def test_sb_classifier_all_features():
    # Mirrors crates/openquant/tests/sb_bagging.rs::test_sb_classifier
    x, y, _, ind_mat = _synthetic_dataset()
    out = sb_bagging.fit_predict_sb_classifier(x, y, ind_mat, n_estimators=10, random_state=1)
    assert _accuracy(out["predictions"], y) >= 0.55

