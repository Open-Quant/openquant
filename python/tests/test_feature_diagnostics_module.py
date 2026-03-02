import random

import openquant


def _dataset(n: int = 220):
    rng = random.Random(123)
    x = []
    y = []
    event_end = []
    for i in range(n):
        f0 = rng.gauss(0.0, 1.0)
        f1 = 0.96 * f0 + 0.04 * rng.gauss(0.0, 1.0)
        f2 = rng.gauss(0.0, 1.0)
        score = 0.9 * f0 + 0.2 * f2 + 0.2 * rng.gauss(0.0, 1.0)
        x.append([f0, f1, f2])
        y.append(1.0 if score > 0 else 0.0)
        event_end.append(min(n - 1, i + (i % 5)))
    return x, y, ["f0", "f1", "f2"], event_end


def test_mda_default_cv_and_scoring_modes():
    x, y, names, event_end = _dataset()

    out = openquant.feature_diagnostics.mda_importance(
        x,
        y,
        feature_names=names,
        event_end_indices=event_end,
    )
    assert out["method"] == "mda"
    assert out["cv"]["method"] == "purged_kfold"
    assert out["cv"]["n_splits"] == 5
    assert out["cv"]["pct_embargo"] == 0.01
    assert out["cv"]["scoring"] == "neg_log_loss"
    assert out["table"].height == 3

    out_f1 = openquant.feature_diagnostics.mda_importance(
        x,
        y,
        feature_names=names,
        event_end_indices=event_end,
        scoring="f1",
    )
    assert out_f1["table"].height == 3


def test_substitution_dilution_and_orthogonalization():
    x, y, names, event_end = _dataset()

    report = openquant.feature_diagnostics.substitution_effect_report(
        x,
        y,
        feature_names=names,
        event_end_indices=event_end,
        corr_threshold=0.85,
        orthogonalize=True,
    )

    pairs = report["pairs"]
    assert pairs.height >= 1
    assert "dilution_ratio" in pairs.columns

    top = pairs.sort("dilution_ratio", descending=True).row(0, named=True)
    assert top["dilution_ratio"] > 0.75
    assert top["group_importance"] > top["single_sum"] * 0.5

    ortho = report["orthogonalized"]
    assert ortho["max_abs_corr_after"] < ortho["max_abs_corr_before"]
    assert ortho["mda"]["table"].height >= 1


def test_mdi_sfi_and_pca_api_shapes():
    x, y, names, event_end = _dataset()

    mdi = openquant.feature_diagnostics.mdi_importance(x, y, feature_names=names, n_estimators=12)
    sfi = openquant.feature_diagnostics.sfi_importance(
        x,
        y,
        feature_names=names,
        event_end_indices=event_end,
        scoring="accuracy",
    )
    pca = openquant.feature_diagnostics.orthogonalize_features_pca(x, variance_threshold=0.9)

    assert mdi["table"].height == 3
    assert sfi["table"].height == 3
    assert len(pca["explained_variance_ratio"]) >= 1
    assert pca["table"].height == len(x)
