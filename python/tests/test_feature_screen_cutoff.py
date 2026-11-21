import random
from fractions import Fraction
from math import nextafter

import pytest
from openquant.feature_diagnostics import feature_screen_report


def exceeds(left, right, threshold):
    # Independent exact centered Pearson predicate for finite binary64 inputs.
    a, b = list(map(Fraction, left)), list(map(Fraction, right))
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    a, b = [x - ma for x in a], [x - mb for x in b]
    aa, bb = sum(x * x for x in a), sum(x * x for x in b)
    return bool(
        aa and bb and sum(x * y for x, y in zip(a, b)) ** 2 > Fraction(threshold) ** 2 * aa * bb
    )


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize(
    "values",
    [
        [0.0, 1.0, 2.0],
        [1e150, 2e150, 3e150],
        [
            1e100,
            nextafter(1e100, float("inf")),
            nextafter(nextafter(1e100, float("inf")), float("inf")),
        ],
    ],
)
def test_identical_or_opposite_columns_exceed_largest_valid_cutoff(values, sign):
    report = feature_screen_report([[x, sign * x] for x in values], max_corr=nextafter(1.0, 0.0))
    assert len(report["rejected_features"]) == 1


@pytest.mark.parametrize("threshold", [nextafter(0.5, 0.0), 0.5, nextafter(0.5, 1.0)])
def test_strict_cutoff_around_exact_half_correlation(threshold):
    a, b = [-1.0, 0.0, 1.0], [-1.0, 1.0, 0.0]
    report = feature_screen_report(list(zip(a, b)), max_corr=threshold)
    assert bool(report["rejected_features"]) == exceeds(a, b, threshold)


def test_seeded_near_unit_decisions_match_exact_reference():
    rng = random.Random(1611)
    for _ in range(100):
        a = [rng.uniform(-100, 100) for _ in range(rng.randrange(3, 40))]
        b = [x + rng.uniform(-1e-6, 1e-6) for x in a]
        threshold = nextafter(1.0, 0.0)
        report = feature_screen_report(list(zip(a, b)), max_corr=threshold)
        assert bool(report["rejected_features"]) == exceeds(a, b, threshold)


def test_reported_correlation_is_bounded_for_identical_finite_columns():
    a = [
        0.5414124727934966,
        0.9391491627785106,
        0.38120423768821243,
        0.21659939713061338,
        0.4221165755827173,
        0.029040787574867943,
        0.22169166627303505,
    ]
    report = feature_screen_report(list(zip(a, a)))
    assert all(0.0 <= row["max_abs_corr"] <= 1.0 for row in report["records"])


@pytest.mark.parametrize("rows", [[[1.0, None], [2.0, None]], [[1.0, 3.0], [2.0, None]]])
def test_missing_pair_data_does_not_enter_exact_predicate(rows):
    report = feature_screen_report(rows, min_coverage=0.1, max_corr=nextafter(0.0, 1.0))
    assert all(row["max_abs_corr"] == 0.0 for row in report["records"])


def test_exact_zero_covariance_stays_below_smallest_positive_cutoff():
    report = feature_screen_report(
        [[-1.0, 1.0], [0.0, -2.0], [1.0, 1.0]], max_corr=nextafter(0.0, 1.0)
    )
    assert report["rejected_features"] == []
