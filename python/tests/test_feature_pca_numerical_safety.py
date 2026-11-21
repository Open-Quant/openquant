"""PCA must not lose representable variation while standardizing finite inputs."""

from math import nextafter

import numpy as np
import pytest
from openquant.feature_diagnostics import orthogonalize_features_pca


@pytest.mark.parametrize("scale", [1e200, 1e-200, 1e-160, 1e-161])
def test_pca_is_scale_invariant_when_variance_overflows_or_underflows(scale):
    # Independent analytic sample std of [-1, 0, 1] is one.
    out = orthogonalize_features_pca([[-scale], [0.0], [scale]])
    assert out["mean"] == [0.0]
    assert out["std"][0] / scale == pytest.approx(1.0)
    assert out["explained_variance_ratio"] == pytest.approx([1.0])
    assert np.abs(np.asarray(out["transformed"])[:, 0]) == pytest.approx([1.0, 0.0, 1.0])


def test_pca_centers_high_offset_adjacent_floats_without_rounding_away_mean():
    lo = 1e100
    hi = nextafter(lo, float("inf"))
    # Exact mean lies halfway between adjacent floats; +/-1/sqrt(2) scores.
    out = orthogonalize_features_pca([[lo], [hi]])
    scores = np.asarray(out["transformed"])[:, 0]
    assert scores.sum() == pytest.approx(0.0, abs=1e-15)
    assert np.abs(scores) == pytest.approx([2**-0.5, 2**-0.5])
    assert out["std"][0] / (hi - lo) == pytest.approx(2**-0.5)


def test_pca_handles_overflowing_mean_sum_and_constant_column():
    out = orthogonalize_features_pca([[1e308, 1.0], [1e308, 2.0], [1e308, 3.0]])
    assert out["mean"] == [1e308, 2.0]
    assert out["std"] == [1.0, 1.0]
    assert np.abs(np.asarray(out["transformed"])[:, 0]) == pytest.approx([1.0, 0.0, 1.0])


def test_pca_retains_variance_at_smallest_subnormal_scale():
    unit = float.fromhex("0x0.0000000000001p-1022")
    out = orthogonalize_features_pca([[-unit], [0.0], [unit]])
    assert out["std"] == [unit]
    assert np.abs(np.asarray(out["transformed"])[:, 0]) == pytest.approx([1.0, 0.0, 1.0])


def test_pca_unrepresentable_std_has_explicit_range_error():
    with pytest.raises(ValueError, match="standard deviation.*float64 range"):
        orthogonalize_features_pca([[-1.7e308], [1.7e308]])


def test_pca_extreme_standardization_matches_decimal_reference():
    from decimal import Decimal, localcontext
    from random import Random

    rng = Random(1711)
    with localcontext() as ctx:
        ctx.prec = 100
        for exponent in [-300, -160, -100, 0, 100, 200, 300]:
            rows = [[rng.uniform(-1, 1) * 10.0**exponent for _ in range(3)] for _ in range(9)]
            reference = np.zeros((9, 3))
            for col in range(3):
                values = [Decimal(row[col]) for row in rows]
                mean = sum(values) / len(values)
                sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)).sqrt()
                reference[:, col] = [float((value - mean) / sd) for value in values]
            out = orthogonalize_features_pca(rows, variance_threshold=1.0)
            scores = np.asarray(out["transformed"])
            # A full orthonormal rotation must preserve sample Gram geometry.
            np.testing.assert_allclose(scores @ scores.T, reference @ reference.T, atol=2e-13)
            np.testing.assert_allclose(scores.mean(axis=0), 0.0, atol=2e-15)
