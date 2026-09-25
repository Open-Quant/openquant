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


def _mda_of_ar1_feature(phi: float, seed: int = 42) -> float:
    """MDA of one informative AR(1) feature (unit variance) beside one noise feature."""
    rng = random.Random(11)
    n = 1500
    scale = (1.0 - phi * phi) ** 0.5
    f = rng.gauss(0.0, 1.0)
    x, y = [], []
    for _ in range(n):
        f = phi * f + scale * rng.gauss(0.0, 1.0)
        x.append([f, rng.gauss(0.0, 1.0)])
        y.append(1.0 if f + 0.5 * rng.gauss(0.0, 1.0) > 0 else 0.0)
    ends = [min(i + 5, n - 1) for i in range(n)]
    out = openquant.feature_diagnostics.mda_importance(
        x, y, feature_names=["signal", "noise"], event_end_indices=ends, seed=seed
    )
    return {r["feature"]: r["mean"] for r in out["records"]}["signal"]


def test_mda_of_persistent_feature_matches_iid_case():
    # Regression for #98: the column used to be rotated by fold_index + 1 rows, which barely
    # changes a persistent feature, so its importance collapsed. A shuffle (AFML Snippet 8.3)
    # breaks the feature-label link however persistent the feature is.
    iid = _mda_of_ar1_feature(0.0)
    assert iid > 0.1
    for phi in (0.5, 0.9, 0.95):
        persistent = _mda_of_ar1_feature(phi)
        assert abs(persistent - iid) < 0.15 * iid, (phi, persistent, iid)
    # At 0.99 a fold spans only a few swings of the feature, so a within-fold shuffle damages
    # it less; some understatement is inherent to Snippet 8.3. Not the shift's collapse, though.
    assert _mda_of_ar1_feature(0.99) > 0.4 * iid


def test_mda_is_reproducible_for_a_seed():
    x, y, names, event_end = _dataset()
    common = dict(feature_names=names, event_end_indices=event_end)
    fd = openquant.feature_diagnostics
    a = fd.mda_importance(x, y, seed=3, **common)
    b = fd.mda_importance(x, y, seed=3, **common)
    c = fd.mda_importance(x, y, seed=4, **common)
    assert a["records"] == b["records"]
    assert a["records"] != c["records"]
    assert a["cv"]["seed"] == 3
