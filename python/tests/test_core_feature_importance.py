"""openquant.feature_importance: MDI, and MDA / SFI on purged folds that need label spans.

Fixtures and expected values are those of crates/openquant/tests/feature_importance.rs and
feature_importance_reference.rs. The Rust tests hand a classifier to Rust; here its
out-of-sample probabilities are computed in Python and passed in, or a Python port of the
classifier is passed to the estimator-driven functions.
"""

from math import log, sqrt

import numpy as np
import pytest

from openquant import _core
from openquant import feature_importance as fi


def point_spans(n):
    """Labels that start and end on their own bar: purging removes nothing, so the purged
    folds are the plain contiguous folds the Rust fixtures pass in."""
    t = np.arange(n)
    return t, t


# --- the reference fixture: `mda_data` and `SignOfFirstColumn` ------------------------------

X_REF = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
Y_REF = np.array([1.0, 0.0, 1.0, 1.0])
# Two folds: rows 0-1 and rows 2-3, as the Rust `mda_data` splits.
T0_REF, T1_REF = point_spans(4)


class SignOfFirstColumn:
    """P(y = 1) = 0.9 when the first column it is shown is positive, else 0.1. `fit` is a no-op."""

    def fit(self, X, y, sample_weight=None):
        return self

    def predict_proba(self, X):
        return np.where(np.asarray(X)[:, 0] > 0, 0.9, 0.1)


def sign_of_first(x):
    return np.where(x[:, 0] > 0, 0.9, 0.1)


# Out-of-sample probabilities of that classifier: unpermuted, then with each feature's two test
# values swapped (the only non-trivial permutation of a two-row fold). f1 is never read.
BASE = sign_of_first(X_REF)
PERMUTED = np.column_stack(
    [sign_of_first(X_REF[[1, 0, 3, 2]] * [1, 0] + X_REF * [0, 1]), BASE]
)
# SFI: the classifier shown f0 alone, then f1 alone.
SFI_PROBA = np.column_stack([sign_of_first(X_REF[:, [0]]), sign_of_first(X_REF[:, [1]])])


def test_mdi_means_hand_worked():
    # Rust: mdi_means_hand_worked, mdi_ignores_zero_importances_hand_worked
    mdi = fi.mean_decrease_impurity([[0.6, 0.4], [0.8, 0.2], [0.7, 0.3]])
    assert list(mdi) == ["f0", "f1"]
    assert mdi["f0"]["mean"] == pytest.approx(0.7, abs=1e-15)
    assert mdi["f1"]["mean"] == pytest.approx(0.3, abs=1e-15)
    zeros = fi.mean_decrease_impurity([[1.0, 0.0], [0.5, 0.5]], feature_names=["a", "b"])
    assert zeros["a"]["mean"] == pytest.approx(0.6, abs=1e-15)
    assert zeros["b"]["mean"] == pytest.approx(0.4, abs=1e-15)


def test_mdi_standard_error_is_symmetric_for_mirrored_columns():
    # Rust: mdi_standard_error_is_symmetric_for_mirrored_columns. Bracketed between the ddof=0
    # and ddof=1 closed forms, so it holds before and after PR #132 changes the ddof.
    mdi = fi.mean_decrease_impurity([[0.6, 0.4], [0.8, 0.2], [0.7, 0.3]])
    assert mdi["f0"]["std"] == pytest.approx(mdi["f1"]["std"], abs=1e-15)
    assert 0.047 < mdi["f0"]["std"] < 0.058


def test_mdi_orders_the_rust_forest_fixture():
    # Rust: test_feature_importance_mdi_mda_sfi (MDI part)
    per_tree = [[0.50, 0.35, 0.15], [0.60, 0.30, 0.10], [0.58, 0.32, 0.10], [0.52, 0.34, 0.14]]
    mdi = fi.mean_decrease_impurity(per_tree, feature_names=["f0", "f1", "f2"])
    assert sum(v["mean"] for v in mdi.values()) == pytest.approx(1.0, abs=1e-9)
    assert mdi["f0"]["mean"] > mdi["f1"]["mean"] > mdi["f2"]["mean"]


def test_mda_accuracy_hand_worked():
    # Rust: mda_accuracy_hand_worked
    mda = fi.mda_from_probabilities(
        Y_REF, T0_REF, T1_REF, BASE, PERMUTED, n_splits=2, scoring="accuracy"
    )
    assert mda["f0"]["mean"] == pytest.approx(0.5, abs=1e-15)
    assert mda["f1"] == {"mean": 0.0, "std": 0.0}


def test_mda_neg_log_loss_hand_worked():
    # Rust: mda_neg_log_loss_hand_worked
    mda = fi.mda_from_probabilities(
        Y_REF, T0_REF, T1_REF, BASE, PERMUTED, n_splits=2, scoring="neg_log_loss"
    )
    want = (1.0 - log(0.9) / log(0.1)) / 2.0
    assert mda["f0"]["mean"] == pytest.approx(want, abs=1e-14)
    assert mda["f1"]["mean"] == 0.0


def test_sfi_accuracy_hand_worked():
    # Rust: sfi_accuracy_hand_worked
    sfi = fi.sfi_from_probabilities(Y_REF, T0_REF, T1_REF, SFI_PROBA, n_splits=2, scoring="accuracy")
    assert sfi["f0"]["mean"] == pytest.approx(0.75, abs=1e-15)
    assert sfi["f1"]["mean"] == pytest.approx(0.5, abs=1e-15)
    assert sfi["f0"]["std"] == pytest.approx(0.25 / sqrt(2), abs=1e-15)
    assert sfi["f1"]["std"] == pytest.approx(0.5 / sqrt(2), abs=1e-15)


def test_estimator_driven_sfi_reproduces_the_hand_worked_values():
    sfi = fi.single_feature_importance(
        SignOfFirstColumn(), X_REF, Y_REF, T0_REF, T1_REF, n_splits=2, pct_embargo=0.0,
        scoring="accuracy",
    )
    assert sfi["f0"]["mean"] == pytest.approx(0.75, abs=1e-15)
    assert sfi["f1"]["mean"] == pytest.approx(0.5, abs=1e-15)


def test_estimator_driven_mda_leaves_an_unread_feature_at_zero():
    # However the two-row test folds are shuffled, a feature the model never reads scores 0.
    for seed in range(4):
        mda = fi.mean_decrease_accuracy(
            SignOfFirstColumn(), X_REF, Y_REF, T0_REF, T1_REF, n_splits=2, pct_embargo=0.0,
            scoring="accuracy", seed=seed,
        )
        assert mda["f1"] == {"mean": 0.0, "std": 0.0}
        assert mda["f0"]["mean"] in (0.0, 0.5)


class LinearProbClassifier:
    """Python port of the Rust `LinearProbClassifier` test fixture."""

    def fit(self, X, y, sample_weight=None):
        X, y = np.asarray(X), np.asarray(y)
        pos, neg = X[y > 0.5], X[y <= 0.5]
        pos_mean = pos.mean(axis=0) if len(pos) else np.zeros(X.shape[1])
        neg_mean = neg.mean(axis=0) if len(neg) else np.zeros(X.shape[1])
        self.w = pos_mean - neg_mean
        return self

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-(np.asarray(X) @ self.w)))


def make_dataset():
    """Rust `make_dataset`: 120 rows, f1 correlated with f0, f2 noise, y = f0 > 0."""
    i = np.arange(120)
    f0 = np.sin(i / 10.0)
    f1 = 0.7 * f0 + 0.3 * np.cos(i / 7.0)
    f2 = ((i * 37) % 17) / 17.0 - 0.5
    return np.column_stack([f0, f1, f2]), (f0 > 0).astype(float)


def test_mda_and_sfi_order_the_rust_fixture():
    # Rust: test_feature_importance_mdi_mda_sfi. Its four contiguous folds of 30 are the purged
    # folds of point labels.
    x, y = make_dataset()
    t0, t1 = point_spans(120)
    kw = {"n_splits": 4, "pct_embargo": 0.0, "feature_names": ["f0", "f1", "f2"]}
    for scoring in ("accuracy", "f1"):
        mda = fi.mean_decrease_accuracy(LinearProbClassifier(), x, y, t0, t1, scoring=scoring, **kw)
        assert mda["f0"]["mean"] > mda["f2"]["mean"]
        sfi = fi.single_feature_importance(LinearProbClassifier(), x, y, t0, t1, scoring=scoring, **kw)
        assert sfi["f0"]["mean"] >= sfi["f2"]["mean"]
    again = fi.mean_decrease_accuracy(LinearProbClassifier(), x, y, t0, t1, scoring="f1", **kw)
    assert again == fi.mean_decrease_accuracy(LinearProbClassifier(), x, y, t0, t1, scoring="f1", **kw)


def test_feature_importance_cannot_run_without_label_spans():
    x, y = make_dataset()
    with pytest.raises(TypeError):
        fi.mean_decrease_accuracy(LinearProbClassifier(), x, y)
    with pytest.raises(TypeError):
        fi.single_feature_importance(LinearProbClassifier(), x, y)
    with pytest.raises(TypeError):
        fi.mda_from_probabilities(Y_REF, base_proba=BASE, permuted_proba=PERMUTED, n_splits=2)
    with pytest.raises(TypeError):
        fi.sfi_from_probabilities(Y_REF, proba=SFI_PROBA, n_splits=2)
    t = np.arange(len(y))
    for t0, t1 in ((None, t), (t, None), (None, None)):
        with pytest.raises(ValueError, match="is required"):
            fi.mean_decrease_accuracy(LinearProbClassifier(), x, y, t0, t1)
        with pytest.raises(ValueError, match="is required"):
            fi.single_feature_importance(LinearProbClassifier(), x, y, t0, t1)
    with pytest.raises(ValueError, match="is required"):
        fi.mda_from_probabilities(Y_REF, None, None, BASE, PERMUTED, n_splits=2)
    # The compiled functions take the spans positionally too; there is no unpurged path.
    with pytest.raises(TypeError):
        _core.feature_importance.mda_from_probabilities(
            Y_REF.tolist(), BASE.tolist(), PERMUTED.T.tolist(), ["f0", "f1"],
            n_splits=2, pct_embargo=0.0, scoring="accuracy",
        )


def test_invalid_input_raises_value_error():
    kw = {"n_splits": 2, "scoring": "accuracy"}
    with pytest.raises(ValueError, match="scoring must be one of"):
        fi.mda_from_probabilities(Y_REF, T0_REF, T1_REF, BASE, PERMUTED, n_splits=2, scoring="auc")
    with pytest.raises(ValueError, match="base_proba has 3 values"):
        fi.mda_from_probabilities(Y_REF, T0_REF, T1_REF, BASE[:3], PERMUTED, **kw)
    with pytest.raises(ValueError, match="probabilities in"):
        fi.mda_from_probabilities(Y_REF, T0_REF, T1_REF, BASE + 1.0, PERMUTED, **kw)
    with pytest.raises(ValueError, match="y has 3 values"):
        fi.mda_from_probabilities(Y_REF[:3], T0_REF, T1_REF, BASE, PERMUTED, **kw)
    with pytest.raises(ValueError, match="feature_names has 3 names"):
        fi.sfi_from_probabilities(Y_REF, T0_REF, T1_REF, SFI_PROBA, feature_names=list("abc"), **kw)
    with pytest.raises(ValueError, match="sample_weight has 2 values"):
        fi.sfi_from_probabilities(Y_REF, T0_REF, T1_REF, SFI_PROBA, sample_weight=[1, 1], **kw)
    with pytest.raises(ValueError, match="ImportanceRowLength|importance row length mismatch"):
        _core.feature_importance.mean_decrease_impurity([[0.5, 0.5], [1.0]], ["a", "b"])
    with pytest.raises(ValueError, match="n_splits"):
        fi.mda_from_probabilities(Y_REF, T0_REF, T1_REF, BASE, PERMUTED, n_splits=5)


def test_sklearn_models():
    # scikit-learn is not a dependency of openquant; this runs where it is installed.
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    t0 = np.arange(n)
    t1 = t0 + 5

    forest = RandomForestClassifier(n_estimators=50, max_features=1, random_state=0).fit(x, y)
    mdi = fi.mean_decrease_impurity([t.feature_importances_ for t in forest.estimators_])
    assert max(mdi, key=lambda k: mdi[k]["mean"]) == "f0"

    mda = fi.mean_decrease_accuracy(LogisticRegression(), x, y, t0, t1, n_splits=5)
    assert max(mda, key=lambda k: mda[k]["mean"]) == "f0"
    sfi = fi.single_feature_importance(LogisticRegression(), x, y, t0, t1, n_splits=5)
    assert max(sfi, key=lambda k: sfi[k]["mean"]) == "f0"
