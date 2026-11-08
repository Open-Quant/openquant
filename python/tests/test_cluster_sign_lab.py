"""Independent-unit algebra, exact-tail and simulation contracts."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[2] / "experiments" / "cluster_sign_lab.py"
spec = importlib.util.spec_from_file_location("cluster_sign_lab", MODULE)
assert spec is not None and spec.loader is not None
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def test_ar_sum_variance_matches_covariance_matrix_and_hand_values():
    assert lab.block_sum_variance(1, 0.9) == 1
    assert lab.block_sum_variance(2, 0.6) == pytest.approx(3.2)
    assert lab.block_sum_variance(3, 0.5) == pytest.approx(5.5)
    for m in (4, 16, 64):
        for phi in (0, 0.6, 0.9):
            cov = phi ** np.abs(np.arange(m)[:, None] - np.arange(m)[None, :])
            assert lab.block_sum_variance(m, phi) == pytest.approx(cov.sum())


def test_exact_power_hand_calculation_all_five_positive():
    # n=5 at .05 can reject only if all five positive; power=q**5.
    out = lab.independent_block_reference(5, 1, 0, 0.2, 0.05)
    q = 0.5 * math.erfc(-0.2 / math.sqrt(2))
    assert out["critical_positive_count"] == 5
    assert out["rejection_probability"] == pytest.approx(q**5)
    null = lab.independent_block_reference(16, 16, 0.9, 0, 0.05)
    assert (
        null["rejection_probability"]
        == lab.audit.attainable_boundary(16, 0.05)["exact_null_rejection_rate"]
    )


def test_four_blocks_cannot_reject_even_with_all_positive_sums():
    assert lab.independent_block_reference(4, 64, 0.9, 100, 0.05)["rejection_probability"] == 0
    assert lab.evaluate(np.ones((4, 16)), alpha=0.05)["tests"]["block_sums"]["rejected"] is False
    assert (
        lab.evaluate(np.ones((4, 16)), alpha=0.05)["tests"]["raw_observations"]["rejected"] is True
    )


def test_phi_zero_independent_and_continuous_use_identical_draws_given_same_seed():
    options = dict(n_blocks=8, block_size=16, phi=0, drift=0.1, seed=14)
    np.testing.assert_array_equal(
        lab.generate("independent_blocks", **options), lab.generate("continuous_path", **options)
    )


def test_one_observation_per_independent_block_matches_raw_sign_test():
    values = lab.generate(
        "independent_blocks", n_blocks=16, block_size=1, phi=0.9, drift=0.2, seed=3
    )
    result = lab.evaluate(values, alpha=0.05)
    assert result["tests"]["raw_observations"] == result["tests"]["block_sums"]


def test_ties_drop_differently_for_raw_and_sum_units_and_all_ties_unavailable():
    result = lab.evaluate([[1, -1], [2, -1], [-2, 1], [0, 0]], alpha=0.05)
    assert result["tests"]["block_sums"]["n_nonzero"] == 2
    assert result["tests"]["raw_observations"]["n_nonzero"] == 6
    assert lab.evaluate(np.zeros((3, 4)), alpha=0.05)["tests"]["block_sums"]["rejected"] is None


def test_cached_sign_results_cannot_be_mutated_by_caller():
    values = np.ones((5, 2))
    first = lab.evaluate(values, alpha=0.05)
    first["tests"]["block_sums"]["rejected"] = "modified"
    assert lab.evaluate(values, alpha=0.05)["tests"]["block_sums"]["rejected"] is True


def test_generators_have_declared_cross_block_covariance():
    # Many independent replications: adjacent sum correlation stays .5 for shared AR,
    # and is negligible for separately regenerated blocks.
    sums = {}
    for family in lab.FAMILIES:
        sums[family] = np.array(
            [
                lab.generate(family, n_blocks=2, block_size=4, phi=0.9, drift=0, seed=s).sum(axis=1)
                for s in range(2000)
            ]
        )
    assert abs(np.corrcoef(sums["independent_blocks"].T)[0, 1]) < 0.08
    assert np.corrcoef(sums["shared_factor_blocks"].T)[0, 1] == pytest.approx(0.5, abs=0.07)
    assert np.corrcoef(sums["continuous_path"].T)[0, 1] > 0.6


def test_report_replay_order_and_reference_validity():
    a = lab.run_lab(repetitions=8, n_blocks=16, block_size=4, phi=0.6)
    assert a == lab.run_lab(**a["config"])
    b = lab.run_lab(**{**a["config"], "families": list(reversed(lab.FAMILIES))})
    assert a["scenarios"] == b["scenarios"][::-1]
    assert [s["independent_block_reference_valid"] for s in a["scenarios"]] == [True, False, False]
    json.dumps(a, allow_nan=False)


def test_mcse_and_wilson_are_replication_level():
    report = lab.run_lab(repetitions=20, n_blocks=5, block_size=4, phi=0.9)
    for scenario in report["scenarios"]:
        for name, rate in scenario["rates"].items():
            flags = np.array([r["tests"][name]["rejected"] for r in scenario["records"]], dtype=int)
            assert rate["monte_carlo_standard_error"] == pytest.approx(
                flags.std(ddof=1) / np.sqrt(20)
            )
            assert rate["wilson_95_interval"][0] <= flags.mean() <= rate["wilson_95_interval"][1]


@pytest.mark.parametrize(
    "config",
    [
        dict(repetitions=1),
        dict(n_blocks=True),
        dict(block_size=1.1),
        dict(phi=1),
        dict(phi=-0.1),
        dict(phi=".6"),
        dict(drift=-0.1),
        dict(drift=float("inf")),
        dict(alpha=True),
        dict(families=[]),
        dict(families=["independent_blocks"] * 2),
    ],
)
def test_bad_config_fails(config):
    with pytest.raises((TypeError, ValueError)):
        lab.run_lab(**config)


def test_alpha_cannot_be_coerced_from_boolean_and_nonrepresentable_sums_fail():
    with pytest.raises(TypeError):
        lab.evaluate(np.ones((2, 3)), alpha=True)
    with pytest.raises(ValueError):
        lab.evaluate([[1e308, 1e308]], alpha=0.05)


def test_cli_serializes_all_source_hashes(tmp_path):
    out = tmp_path / "report.json"
    subprocess.run(
        [sys.executable, str(MODULE), "--repetitions", "2", "--out", str(out)],
        check=True,
        capture_output=True,
    )
    report = json.loads(out.read_text())
    assert set(report["provenance"]["dependency_sha256"]) == {
        "adaptive_sign_audit.py",
        "adaptive_holdout_lab.py",
        "validation_lab.py",
    }
    assert report["performance"]["wall_seconds"] > 0
    replay = subprocess.run(
        [
            sys.executable,
            str(MODULE.parent / "replay_offline_report.py"),
            str(out),
            "--lab",
            "cluster_sign",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(replay.stdout)["matched"] is True


def test_degenerate_independence_metadata_matches_generator():
    report = lab.run_lab(repetitions=2, n_blocks=1, block_size=1, phi=0.9)
    assert all(s["independent_block_reference_valid"] for s in report["scenarios"])
    assert all(s["raw_iid_sign_assumption_valid"] for s in report["scenarios"])
    report = lab.run_lab(repetitions=2, n_blocks=16, block_size=1, phi=0.9)
    assert report["scenarios"][0]["raw_iid_sign_assumption_valid"] is True
    assert report["scenarios"][1]["raw_iid_sign_assumption_valid"] is False
    report = lab.run_lab(repetitions=2, n_blocks=1, block_size=4, phi=0)
    assert all(s["raw_iid_sign_assumption_valid"] for s in report["scenarios"])


def test_complex_alpha_cannot_lose_imaginary_part():
    with pytest.raises(TypeError):
        lab.evaluate(np.ones((2, 3)), alpha=np.complex128(0.05 + 1j))
