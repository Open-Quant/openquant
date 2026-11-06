"""PCA must retain variance even when an all-ones start misses an eigenspace."""

import numpy as np
import pytest
from openquant.feature_diagnostics import orthogonalize_features_pca


def test_pca_retains_perfectly_anticorrelated_features():
    rows = [[1.0, -1.0], [-1.0, 1.0], [2.0, -2.0], [-2.0, 2.0]]
    out = orthogonalize_features_pca(rows)
    assert out["explained_variance_ratio"] == pytest.approx([1.0])
    assert len(out["components"]) == 1
    component = out["components"][0]
    assert component[0] == pytest.approx(-component[1])
    assert sum(value * value for value in component) == pytest.approx(1.0)
    # Standardized variables each have sample variance one. The single component
    # must therefore retain their total variance of two, regardless of its sign.
    assert np.var(out["transformed"], axis=0, ddof=1) == pytest.approx([2.0])


def test_pca_retains_repeated_eigenvalues():
    rows = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    out = orthogonalize_features_pca(rows, variance_threshold=1.0)
    assert out["explained_variance_ratio"] == pytest.approx([0.5, 0.5])
    components = np.asarray(out["components"])
    np.testing.assert_allclose(components @ components.T, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(np.cov(np.asarray(out["transformed"]).T), np.eye(2), atol=1e-12)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -float("inf"), 0.0, 1.1])
def test_pca_rejects_invalid_variance_threshold(threshold):
    with pytest.raises(ValueError, match="variance_threshold"):
        orthogonalize_features_pca([[1.0], [2.0]], variance_threshold=threshold)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_pca_rejects_nonfinite_features(value):
    with pytest.raises(ValueError, match="finite"):
        orthogonalize_features_pca([[1.0, value], [2.0, 3.0]])


def test_pca_rejects_truly_constant_features():
    with pytest.raises(ValueError, match="zero variance"):
        orthogonalize_features_pca([[2.0, -1.0], [2.0, -1.0]])
