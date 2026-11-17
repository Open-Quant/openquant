from __future__ import annotations

import math

import openquant
import pytest


@pytest.mark.parametrize("parameter", ["min_coverage", "max_corr"])
def test_feature_screen_rejects_nan_threshold(parameter):
    with pytest.raises(ValueError, match=parameter):
        openquant.feature_diagnostics.feature_screen_report(
            [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], **{parameter: math.nan}
        )


def test_feature_screen_accepts_complete_coverage_boundary():
    report = openquant.feature_diagnostics.feature_screen_report(
        [[1.0], [2.0], [3.0]], min_coverage=1.0
    )
    assert report["selected_features"] == ["f0"]
    assert report["records"][0]["coverage"] == 1.0


@pytest.mark.parametrize("scale", [1e155, 1e308])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_feature_screen_extreme_finite_correlated_features(scale, direction):
    # [-a, 0, a] has sample std=a and exact signed correlation +/-1.
    report = openquant.feature_diagnostics.feature_screen_report(
        [[-scale, -direction * scale], [0.0, 0.0], [scale, direction * scale]]
    )
    assert report["selected_features"] == ["f0"]
    assert report["rejected_features"] == ["f1"]
    for row in report["records"]:
        assert row["std"] == pytest.approx(scale)
        assert row["max_abs_corr"] == pytest.approx(1.0)


def test_feature_screen_large_constant_is_rejected():
    report = openquant.feature_diagnostics.feature_screen_report([[1e308], [1e308], [1e308]])
    assert report["selected_features"] == []
    assert report["records"][0]["std"] == 0.0
    assert "constant_or_near_constant" in report["rejection_reasons"]["f0"]


def test_feature_screen_large_offset_keeps_small_spread():
    # Three adjacent representable values have sample std equal to one ULP.
    center = 1e308
    step = math.ulp(center)
    report = openquant.feature_diagnostics.feature_screen_report(
        [[center - step, -center + step], [center, -center], [center + step, -center - step]]
    )
    for row in report["records"]:
        assert row["std"] == pytest.approx(step, rel=1e-14)
        assert row["max_abs_corr"] == pytest.approx(1.0, abs=1e-14)


def test_feature_screen_extreme_statistics_against_decimal_reference():
    from decimal import Decimal, localcontext
    from random import Random

    rng = Random(50115)
    with localcontext() as ctx:
        ctx.prec = 100
        for exponent in (155, 250, 307):
            for _ in range(12):
                xs = [rng.uniform(-1.0, 1.0) * 10.0**exponent for _ in range(9)]
                ys = [rng.uniform(-1.0, 1.0) * 10.0**exponent for _ in range(9)]
                a, b = [Decimal(x) for x in xs], [Decimal(y) for y in ys]
                am, bm = sum(a) / len(a), sum(b) / len(b)
                aa = sum((x - am) ** 2 for x in a)
                bb = sum((y - bm) ** 2 for y in b)
                corr = sum((x - am) * (y - bm) for x, y in zip(a, b)) / (aa * bb).sqrt()
                report = openquant.feature_diagnostics.feature_screen_report(list(zip(xs, ys)))
                rows = {row["feature"]: row for row in report["records"]}
                assert rows["f0"]["std"] == pytest.approx(float((aa / 8).sqrt()), rel=1e-14)
                assert rows["f1"]["std"] == pytest.approx(float((bb / 8).sqrt()), rel=1e-14)
                assert rows["f0"]["max_abs_corr"] == pytest.approx(abs(float(corr)), abs=1e-14)


def test_feature_screen_unrepresentable_std_has_clear_error():
    from sys import float_info

    with pytest.raises(ValueError, match="standard deviation exceeds float64 range"):
        openquant.feature_diagnostics.feature_screen_report([[-float_info.max], [float_info.max]])


def test_feature_screen_adjacent_floats_with_large_offset_are_anticorrelated():
    a = 1e12
    b = math.nextafter(a, math.inf)
    report = openquant.feature_diagnostics.feature_screen_report([[a, b], [b, a]])
    assert report["selected_features"] == ["f0"]
    for row in report["records"]:
        assert row["std"] == pytest.approx((b - a) / math.sqrt(2), rel=1e-14)
        assert row["max_abs_corr"] == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("target_corr,expected_count", [(0.949, 2), (0.951, 1), (-0.951, 1)])
def test_feature_screen_extreme_correlation_around_cutoff(target_corr, expected_count):
    scale = 1e200
    perpendicular_weight = math.sqrt((1 - target_corr**2) / 3)
    xs = [-1.0, 0.0, 1.0]
    orthogonal = [1.0, -2.0, 1.0]
    ys = [target_corr * x + perpendicular_weight * p for x, p in zip(xs, orthogonal)]
    report = openquant.feature_diagnostics.feature_screen_report(
        [[x * scale, y * scale] for x, y in zip(xs, ys)], max_corr=0.95
    )
    assert len(report["selected_features"]) == expected_count
    assert report["records"][0]["max_abs_corr"] == pytest.approx(abs(target_corr), abs=1e-14)
