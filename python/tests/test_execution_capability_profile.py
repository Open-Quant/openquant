"""Profile preflight is not economic/engine qualification."""

import copy
from typing import cast

import pytest

pytest.importorskip("backtrader")
from experiments.execution_capability_profile import (
    OBSERVED_SUBSET,
    PORTABLE_V2,
    assess,
    characterize,
)


@pytest.fixture(scope="module", params=[True, False])
def profile(request):
    return characterize(checksubmit=request.param)


def test_portable_contract_cannot_pass(profile):
    result = assess(profile, PORTABLE_V2)
    assert result["status"] == "blocked"
    assert result["external_engine_validated"] is False
    assert result["case_comparison_executed"] is False
    assert "volume_scope" in result["mismatches"]
    assert "arithmetic" in result["mismatches"]
    assert {"quote_model", "sides", "deadlines", "terminal_liquidation"} <= set(
        result["unqualified"]
    )


def test_subset_match_is_eligible_not_pass(profile):
    requirements = dict(OBSERVED_SUBSET, admission=profile["capabilities"]["admission"])
    result = assess(profile, requirements)
    assert result["status"] == "eligible_for_case_comparison"
    assert result["external_engine_validated"] is False
    assert result["case_comparison_executed"] is False
    assert result["omitted_dimensions"] == [
        "arithmetic",
        "deadlines",
        "quote_model",
        "sides",
        "terminal_liquidation",
    ]
    assert "passed" not in result


def test_terminal_residuals_separated_from_active_and_admission(profile):
    end = profile["evidence"]["outcomes"]
    assert end["execution_refusal"]["terminal_stage"] == "execution"
    assert end["execution_refusal"]["remaining"] == 3
    assert end["feed_end"]["state"] == "active_at_feed_end"
    assert end["feed_end"]["remaining"] == 3
    expected = "admission" if profile["checksubmit"] else "execution"
    assert end["cash_refusal"]["terminal_stage"] == expected
    assert end["completed"]["state"] == "completed"


def test_fresh_evidence_and_configuration_bound(profile):
    assert profile["engine_version"] == "1.9.78.123"
    assert profile["schema"] == "openquant_execution_capabilities_v1"
    assert len(profile["evidence"]["trace_sha256"]) == 6
    assert all(len(x) == 64 for x in profile["evidence"]["trace_sha256"].values())
    assert profile["evidence"]["observed_volume_excess"] == 2
    assert profile["evidence"]["fixed_fee_fill_count"] == 2


@pytest.mark.parametrize(
    "requirements",
    [
        {},
        {"bogus": "anything"},
        {"volume_scope": "unknown"},
        {"volume_scope": True},
        {"volume_scope": "shared", "arithmetic": None},
    ],
)
def test_invalid_requirement_fails_closed(profile, requirements):
    with pytest.raises(ValueError):
        assess(profile, requirements)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "future"),
        ("engine_version", "future"),
        ("external_engine_validated", True),
        ("checksubmit", 1),
    ],
)
def test_invalid_profile_fails_closed(profile, field, value):
    changed = copy.deepcopy(profile)
    changed[field] = value
    with pytest.raises(ValueError):
        assess(changed, OBSERVED_SUBSET)


def test_corrupted_profile_evidence_rejected(profile):
    changed = copy.deepcopy(profile)
    changed["evidence"]["observed_volume_excess"] = 0
    with pytest.raises(ValueError):
        assess(changed, OBSERVED_SUBSET)


def test_unknown_value_never_matches(profile):
    requirements = {"sides": "buy_and_sell"}
    result = assess(profile, requirements)
    assert result["status"] == "blocked"
    assert result["unqualified"] == ["sides"]


def test_bad_bool_not_config():
    with pytest.raises(ValueError):
        characterize(checksubmit=cast(bool, 1))


@pytest.mark.parametrize("field,value", [("synthetic_only", 1), ("external_engine_validated", 0)])
def test_integer_cannot_impersonate_boolean(profile, field, value):
    changed = copy.deepcopy(profile)
    changed[field] = value
    with pytest.raises(ValueError):
        assess(changed, OBSERVED_SUBSET)


def test_partial_profile_must_not_imply_full_contract(profile):
    result = assess(profile, {"fee_basis": "per_fill"})
    assert result["status"] == "eligible_for_case_comparison"
    assert len(result["omitted_dimensions"]) == 9
    assert result["external_engine_validated"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["traces"].pop("volume"),
        lambda p: p["traces"]["volume"]["config"].update(checksubmit="off"),
        lambda p: p["traces"]["completed"]["source_sha256"].update(broker="0" * 64),
        lambda p: p["traces"]["completed"]["fills"].pop(),
        lambda p: p["capabilities"].update(volume_scope="shared"),
        lambda p: p.update(extra="ignored?"),
    ],
)
def test_mutated_or_incomplete_report_fails_closed(profile, mutate):
    changed = copy.deepcopy(profile)
    mutate(changed)
    with pytest.raises(ValueError):
        assess(changed, OBSERVED_SUBSET)


def test_profile_roundtrips_and_repeats_exactly(profile):
    import json

    restored = json.loads(json.dumps(profile, allow_nan=False))
    assert assess(restored, PORTABLE_V2) == assess(profile, PORTABLE_V2)
    assert characterize(checksubmit=profile["checksubmit"]) == profile


def test_cli_exit_codes_are_not_qualification(tmp_path):
    import json
    import subprocess
    import sys

    for setting in ("on", "off"):
        for requirement, code in (("portable-v2", 2), ("observed-subset", 0)):
            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.execution_capability_profile",
                    "--checksubmit",
                    setting,
                    "--requirements",
                    requirement,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            assert run.returncode == code, run.stderr
            output = json.loads(run.stdout)
            assert output["preflight"]["external_engine_validated"] is False
            assert output["preflight"]["case_comparison_executed"] is False


def test_decimal_boundary_behavior_drift_cannot_requalify(profile):
    from experiments.backtrader_admission_probe import observe
    from experiments.execution_capability_profile import _derive

    changed = copy.deepcopy(profile["traces"])
    cfg = dict(changed["decimal_boundary"]["config"], cash=51.05000000001)
    trace = observe(**cfg)
    assert trace["bars"][-1]["orders"][0]["status"] == "Completed"
    # Tiny balance differences are inside the old bookkeeping tolerance. They
    # must not erase the boundary behavior required by this new profile.
    trace["config"]["cash"] = 51.05
    changed["decimal_boundary"] = trace
    with pytest.raises(ValueError, match="fixed-control behavior changed"):
        _derive(changed, profile["checksubmit"])


def test_named_screen_is_bound_to_reviewed_fixture(profile, tmp_path):
    from experiments.execution_capability_profile import PORTABLE_FIXTURE, assess_portable

    result = assess_portable(profile)
    assert (
        result["requirements_source"]["sha256"]
        == "2f34d8b986711e164dcf8f97a87b9e0b4ce24b2375810796842fe3027287a440"
    )
    candidate = tmp_path / "fixture.json"
    candidate.write_bytes(PORTABLE_FIXTURE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="unreviewed portable fixture"):
        assess_portable(profile, fixture=candidate)


def test_named_screen_metadata_not_claiming_execution(profile):
    from experiments.execution_capability_profile import assess_portable

    result = assess_portable(profile)
    assert (
        result["requirements_source"]["mapping"]
        == "manual_convention_screen_not_full_specification"
    )
    assert result["case_comparison_executed"] is False


def test_order_summary_keeps_terminal_active_and_delivery_distinct():
    from experiments.backtrader_admission_probe import observe
    from experiments.execution_capability_profile import summarize_orders

    trace = observe(
        cash=25,
        prices=[10, 10, 8, 1],
        volumes=[0, 2, 3, 10],
        fixed=1,
        rate=0.001,
        orders=[
            dict(id="large", bar=0, shares=5),
            dict(id="small", bar=1, shares=1),
            dict(id="last", bar=3, shares=1),
        ],
    )
    states = {x["id"]: x for x in summarize_orders(trace)}
    assert states["large"]["state"] == "terminal_residual"
    assert states["large"]["terminal_stage"] == "admission"
    assert states["large"]["last_fill_bar"] is None
    assert states["large"]["terminal_event_bar"] == 1
    assert states["small"]["state"] == "completed"
    assert states["small"]["last_fill_bar"] == 2
    assert states["small"]["remaining"] == 0
    assert states["last"]["state"] == "active_at_feed_end"
    assert states["last"]["last_delivery_bar"] is None
    assert states["last"]["undelivered_notifications"] == 1
    assert states["last"]["terminal_event_bar"] is None
    assert states["last"]["remaining"] == 1


def test_summary_retains_last_fill_before_terminal_refusal():
    from experiments.backtrader_admission_probe import observe
    from experiments.execution_capability_profile import summarize_orders

    trace = observe(
        cash=25,
        prices=[10, 10, 8, 1],
        volumes=[0, 2, 3, 10],
        fixed=1,
        rate=0.001,
        orders=[dict(id="large", bar=0, shares=5)],
        checksubmit=False,
    )
    (state,) = summarize_orders(trace)
    assert state["last_fill_bar"] == 1
    assert state["terminal_event_bar"] == 2
    assert state["last_delivery_bar"] == 2
    assert state["terminal_stage"] == "execution"
    assert state["remaining"] == 3


def test_summary_rejects_corrupt_trace(profile):
    from experiments.execution_capability_profile import summarize_orders

    trace = copy.deepcopy(profile["traces"]["completed"])
    trace["fills"].pop()
    with pytest.raises(ValueError):
        summarize_orders(trace)
