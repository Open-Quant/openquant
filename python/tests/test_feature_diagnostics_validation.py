"""Public validation regressions for issue #278.

The documented model is binary classification: labels outside {0, 1} are not
valid targets, and feature names must identify columns without aliasing them.
"""

import math

import polars as pl
import pytest
from openquant import feature_diagnostics as fd

X = [[float(i % 3), float((i * 7) % 5)] for i in range(12)]
Y = [float(i % 2) for i in range(12)]
METHODS = ["mdi_importance", "mda_importance", "sfi_importance", "substitution_effect_report"]


def _call(method, y, names=None):
    kwargs = {"feature_names": names}
    if method == "mdi_importance":
        kwargs["n_estimators"] = 2
    else:
        kwargs.update(event_end_indices=list(range(12)), n_splits=3, pct_embargo=0.0)
    if method == "substitution_effect_report":
        kwargs["orthogonalize"] = False
    return getattr(fd, method)(X, y, **kwargs)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "bad_label",
    [
        math.nan,
        math.inf,
        -math.inf,
        -1.0,
        2.0,
        0.5,
        math.nextafter(0.0, 1.0),
        math.nextafter(1.0, 0.0),
    ],
)
def test_importance_rejects_nonfinite_or_nonbinary_labels(method, bad_label):
    # Without validation, the solver consumes invalid targets or silently scores
    # fractional/out-of-domain labels as if they were binary observations.
    labels = Y.copy()
    labels[5] = bad_label
    with pytest.raises(ValueError, match=r"y.*finite.*(0.*1|binary)"):
        _call(method, labels)


@pytest.mark.parametrize("method", METHODS)
def test_importance_rejects_duplicate_feature_names(method):
    with pytest.raises(ValueError, match=r"feature_names.*unique"):
        _call(method, Y, ["signal", "signal"])


@pytest.mark.parametrize("as_frame", [False, True])
def test_screen_rejects_names_that_merge_distinct_columns(as_frame):
    # One column is constant and one varies. Aliasing their names merges their
    # rejection reasons and can wrongly reject the varying feature too.
    rows = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    features = (
        pl.DataFrame(rows, schema=["constant", "varying"], orient="row") if as_frame else rows
    )
    with pytest.raises(ValueError, match=r"feature_names.*unique"):
        fd.feature_screen_report(features, feature_names=["signal", "signal"])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("label", [0.0, 1.0])
def test_binary_single_class_labels_remain_valid(method, label):
    result = _call(method, [label] * len(Y), ["a", "b"])
    table = (
        result["baseline_mda"]["table"]
        if method == "substitution_effect_report"
        else result["table"]
    )
    assert set(table["feature"].to_list()) == {"a", "b"}


def test_screen_default_dataframe_names_keep_reasons_separate():
    result = fd.feature_screen_report(
        pl.DataFrame({"constant": [1.0] * 3, "varying": [0.0, 1.0, 2.0]})
    )
    rows = {row["feature"]: row for row in result["table"].to_dicts()}
    assert "constant_or_near_constant" in rows["constant"]["reasons"]
    assert "constant_or_near_constant" not in rows["varying"]["reasons"]
