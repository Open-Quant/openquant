"""Independent finite-binomial and information-boundary checks."""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[2] / "experiments" / "adaptive_sign_audit.py"
spec = importlib.util.spec_from_file_location("adaptive_sign_audit", MODULE)
assert spec is not None and spec.loader is not None
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


@pytest.mark.parametrize("positives,tail", [(0, 16), (1, 15), (2, 11), (3, 5), (4, 1)])
def test_four_sign_hand_tail(positives, tail):
    result = lab.sign_test([1] * positives + [-1] * (4 - positives), alpha=1 / 16)
    assert result["p_value"] == tail / 16
    assert result["tail_integer"] == str(tail)
    assert result["rejected"] is (positives == 4)


def test_ten_sign_tail_and_inclusive_threshold():
    assert lab.sign_test([1] * 8 + [-1] * 2)["p_value"] == 56 / 1024
    assert not lab.sign_test([1] * 8 + [-1] * 2)["rejected"]
    assert lab.sign_test([1] * 9 + [-1])["p_value"] == 11 / 1024
    assert lab.sign_test([1] * 9 + [-1], alpha=11 / 1024)["rejected"]


def test_zero_ties_dropped_and_all_ties_unavailable():
    a = lab.sign_test([1, 1, -1, -1])
    b = lab.sign_test([0, -0.0, 1, 1, -1, -1])
    assert a["p_value"] == b["p_value"]
    assert b["ties"] == 2
    allzero = lab.sign_test([0, 0, 0])
    assert allzero["p_value"] is allzero["rejected"] is None
    assert allzero["n_nonzero"] == 0


def test_exhaustive_fair_sign_patterns_are_superuniform():
    # No Monte Carlo tolerance: enumerate all 1024 equally probable iid null patterns.
    for alpha in (0.01, 0.05, 0.1, 0.5):
        count = sum(
            lab.sign_test(pattern, alpha=alpha)["rejected"]
            for pattern in itertools.product((-1, 1), repeat=10)
        )
        assert count / 1024 <= alpha
    assert (
        sum(lab.sign_test(pattern)["rejected"] for pattern in itertools.product((-1, 1), repeat=4))
        == 0
    )


def test_exact_integer_threshold_avoids_rounded_float_boundary():
    # 1/2**1075 rounds to float zero, but exact positive tail is retained in evidence.
    result = lab.sign_test(np.ones(1075), alpha=np.nextafter(0.0, 1.0))
    assert result["p_value"] == 0
    assert result["tail_integer"] == "1"
    assert result["denominator_power_of_two"] == 1075
    assert result["rejected"]


@pytest.mark.parametrize(
    "values", [[], [1, float("nan")], [1, float("inf")], [[1, 2]], ["1", "2"], [True, False]]
)
def test_invalid_returns_rejected(values):
    with pytest.raises(ValueError):
        lab.sign_test(values)


@pytest.mark.parametrize("alpha", [0, 1, -0.1, float("nan"), True, np.bool_(False)])
def test_invalid_alpha_rejected(alpha):
    with pytest.raises((TypeError, ValueError)):
        lab.sign_test([1, -1], alpha=alpha)


def test_audit_mutation_cannot_change_selected_weights_or_validation_tests():
    rng = np.random.default_rng(46)
    train, valid, audit = [rng.normal(size=(30, 5)) for _ in range(3)]
    a = lab.evaluate(train, valid, audit, rounds=4, alpha=0.05)
    b = lab.evaluate(train, valid, -audit, rounds=4, alpha=0.05)
    for key in ("selected_weights", "training_order", "frozen_candidate", "single_candidate"):
        assert a[key] == b[key]
    for key in ("adaptive_validation", "single_validation", "frozen_validation"):
        assert a["tests"][key] == b["tests"][key]
    assert a["tests"]["adaptive_audit"]["positive"] == b["tests"]["adaptive_audit"]["negative"]


def test_mixture_ignores_inactive_extreme_column():
    values = np.array([[1e-308, -1e308], [2e-308, 0], [3e-308, 1e308]])
    result = lab.sign_test(lab._mixture(values, np.array([1.0, 0.0])))
    assert result["positive"] == 3


def test_zero_rounds_matches_frozen_sign_control():
    report = lab.run_lab(
        repetitions=3, sample_size=20, n_candidates=4, rounds=0, families=["iid_null"]
    )
    for record in report["scenarios"][0]["records"]:
        assert record["tests"]["adaptive_audit"] == record["tests"]["frozen_audit"]
        assert record["tests"]["adaptive_validation"] == record["tests"]["frozen_validation"]


def test_paired_mcse_and_unavailable_outcomes():
    records = [
        {"tests": {"a": {"rejected": a}, "b": {"rejected": b}}}
        for a, b in [(True, False), (False, True), (True, True), (None, False)]
    ]
    result = lab._paired(records, "a", "b")
    assert result["available_pairs"] == 3 and result["unavailable_pairs"] == 1
    assert result["mean"] == 0
    assert result["monte_carlo_standard_error"] == pytest.approx(1 / math.sqrt(3))
    rate = lab._rate(records, "a")
    assert rate["available"] == 3 and rate["unavailable"] == 1
    assert rate["rejection_rate"] == 2 / 3
    assert 0 < rate["wilson_95_interval"][0] < 2 / 3 < rate["wilson_95_interval"][1] < 1


def test_wilson_endpoints_and_missing_values():
    assert lab._wilson(0, 0) is None
    assert lab._wilson(0, 10)[0] == 0
    assert lab._wilson(10, 10)[1] == 1


def test_known_family_roles_and_reordered_streams():
    options = dict(repetitions=3, sample_size=16, n_candidates=3, rounds=2)
    a = lab.run_lab(**options)
    b = lab.run_lab(**options, families=list(reversed(lab.base.FAMILIES)))
    assert a["scenarios"] == list(reversed(b["scenarios"]))
    roles = {s["family"]: s["interpretation"] for s in a["scenarios"]}
    assert roles["serial_null"] == "serial_dependence_violates_sign_test"
    assert roles["positive_drift"] == "positive_drift_power"
    assert roles["heavy_tailed_null"] == "iid_symmetric_null"
    json.dumps(a, allow_nan=False)


def test_cli_writes_replay_compatible_provenance(tmp_path):
    target = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(MODULE),
            "--out",
            str(target),
            "--repetitions",
            "2",
            "--sample-size",
            "12",
            "--n-candidates",
            "2",
            "--rounds",
            "1",
            "--families",
            "iid_null",
        ],
        check=True,
        capture_output=True,
    )
    report = json.loads(target.read_text())
    assert set(report["provenance"]["dependency_sha256"]) == {
        "adaptive_holdout_lab.py",
        "validation_lab.py",
    }
    assert len(report["provenance"]["source_sha256"]) == 64
    assert report["lab"] == "adaptive_sign_audit"


def test_attainable_size_preserves_discreteness():
    small = lab.attainable_boundary(4, 0.05)
    assert small["critical_positive_count"] is None
    assert small["exact_null_rejection_rate"] == 0
    ten = lab.attainable_boundary(10, 0.05)
    assert ten["critical_positive_count"] == 9
    assert ten["exact_null_rejection_rate"] == 11 / 1024
    inclusive = lab.attainable_boundary(4, 1 / 16)
    assert inclusive["critical_positive_count"] == 4
    assert inclusive["exact_null_rejection_rate"] == 1 / 16
    full = lab.attainable_boundary(252, 0.05)
    k = full["critical_positive_count"]
    assert lab.sign_test([1] * k + [-1] * (252 - k))["rejected"]
    assert not lab.sign_test([1] * (k - 1) + [-1] * (253 - k))["rejected"]


@pytest.mark.parametrize(
    "alpha", [0.05 + 1j, np.complex64(0.05 + 1j), np.complex128(0.05 + 1j), np.complex128(0.05)]
)
def test_complex_alpha_is_rejected_without_discarding_imaginary_part(alpha):
    with pytest.raises((TypeError, ValueError)):
        lab.sign_test([1, -1], alpha=alpha)


def test_ties_define_conditional_not_unconditional_positive_probability():
    # 40% positive overall, but 100% of nonzero observations are positive.
    result = lab.sign_test([0] * 60 + [1] * 40)
    assert result["n_nonzero"] == 40
    assert result["p_value"] == 2**-40
    assert result["rejected"]


def test_rounded_pvalue_equality_does_not_replace_exact_threshold():
    n, k = 100, 59
    tail = sum(math.comb(n, j) for j in range(k, n + 1))
    rounded = tail / (1 << n)
    numerator, denominator = rounded.as_integer_ratio()
    assert tail * denominator > numerator * (1 << n)  # float rounded downward
    result = lab.sign_test([1] * k + [-1] * (n - k), alpha=rounded)
    assert result["p_value"] == rounded
    assert not result["rejected"]  # comparing the two float presentations would be wrong
