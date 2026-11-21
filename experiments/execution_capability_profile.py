"""Versioned synthetic capability preflight, never an engine-acceptance verdict.

The profile is evidence-bearing, not authenticated. A match only permits a
separate exact-path comparison within the documented synthetic subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.backtrader_admission_probe import VERSION, audit_trace, observe

SCHEMA = "openquant_execution_capabilities_v1"
PORTABLE_FIXTURE = Path(__file__).with_name("fixtures") / "execution_cost_acceptance_v2.json"
PORTABLE_FIXTURE_SHA256 = "2f34d8b986711e164dcf8f97a87b9e0b4ce24b2375810796842fe3027287a440"
VALUES = {
    "admission": {"whole_order_precheck", "no_submission_precheck"},
    "execution_refusal": {"terminal_margin", "raise_overdraw_error", "retry_residual"},
    "feed_end": {"active_residual", "cancel_residual"},
    "volume_scope": {"per_order", "shared"},
    "fee_basis": {"per_fill", "per_order"},
    "arithmetic": {"binary64", "exact_rational"},
    "quote_model": {"bid_ask_mark"},
    "sides": {"buy_and_sell"},
    "deadlines": {"inclusive"},
    "terminal_liquidation": {"explicit_bucket"},
}
# A screen for required portable-v2 conventions, not its full contract.
# Feed-end order liveness is not exposed by the portable fixture output.
PORTABLE_V2 = {
    "admission": "no_submission_precheck",
    "execution_refusal": "raise_overdraw_error",
    "volume_scope": "shared",
    "fee_basis": "per_fill",
    "arithmetic": "exact_rational",
    "quote_model": "bid_ask_mark",
    "sides": "buy_and_sell",
    "deadlines": "inclusive",
    "terminal_liquidation": "explicit_bucket",
}
OBSERVED_SUBSET = {
    "admission": "whole_order_precheck",
    "execution_refusal": "terminal_margin",
    "feed_end": "active_residual",
    "volume_scope": "per_order",
    "fee_basis": "per_fill",
}


def _configs(checksubmit: bool) -> dict[str, dict[str, Any]]:
    if type(checksubmit) is not bool:
        raise ValueError("checksubmit must be bool")
    base = dict(
        cash=100,
        prices=[10, 10, 8, 1],
        volumes=[0, 2, 3, 10],
        orders=[dict(id="large", bar=0, shares=5)],
        fixed=1,
        rate=0.001,
        checksubmit=checksubmit,
    )
    return {
        "cash_refusal": dict(base, cash=25 if checksubmit else 1),
        "execution_refusal": dict(base, cash=51.5, prices=[10, 10, 10, 1]),
        "feed_end": dict(base, prices=[10, 10], volumes=[0, 2]),
        "completed": dict(base),
        "volume": dict(
            base,
            volumes=[0, 2, 0, 0],
            orders=[dict(id="a", bar=0, shares=2), dict(id="b", bar=0, shares=2)],
        ),
        "decimal_boundary": dict(base, cash=51.05, prices=[10, 10, 10, 10], volumes=[0, 5, 5, 5]),
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _outcome(trace: dict[str, Any]) -> dict[str, Any]:
    last = trace["bars"][-1]["orders"][0]
    stage = None
    if last["status"] == "Margin":
        stage = next(
            e["phase"]
            for e in reversed(trace["broker_events"])
            if e["id"] == last["id"] and e["status"] == "Margin"
        )
    return dict(
        state="completed"
        if last["status"] == "Completed"
        else "active_at_feed_end"
        if last["alive"]
        else "terminal_residual",
        terminal_stage=stage,
        remaining=last["remaining"],
    )


def summarize_orders(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Audit then summarize every order without converting refusal to retry.

    Bars are zero-based observer coordinates, not exchange timestamps. Feed
    exhaustion is neither cancellation nor a promise of future execution.
    """
    audit_trace(trace)
    result = []
    for order in trace["bars"][-1]["orders"]:
        name = order["id"]
        events = [e for e in trace["broker_events"] if e["id"] == name]
        fills = [f for f in trace["fills"] if f["id"] == name]
        deliveries = [n for n in trace["notifications"] if n["id"] == name]
        terminal = next(
            (e for e in reversed(events) if e["status"] in ("Margin", "Completed")), None
        )
        result.append(
            {
                "id": name,
                "state": "completed"
                if order["status"] == "Completed"
                else "active_at_feed_end"
                if order["alive"]
                else "terminal_residual",
                "terminal_stage": terminal["phase"]
                if terminal and order["status"] == "Margin"
                else None,
                "remaining": order["remaining"],
                "filled": order["filled"],
                "first_fill_bar": fills[0]["bar"] if fills else None,
                "last_fill_bar": fills[-1]["bar"] if fills else None,
                "terminal_event_bar": terminal["bar"] if terminal else None,
                "last_delivery_bar": deliveries[-1]["delivered_bar"] if deliveries else None,
                "undelivered_notifications": len(events) - len(deliveries),
            }
        )
    return result


def _derive(traces: dict[str, Any], checksubmit: bool) -> dict[str, Any]:
    configs = _configs(checksubmit)
    if type(traces) is not dict or set(traces) != set(configs):
        raise ValueError("all six fixed profile controls required")
    audits = {}
    for name, cfg in configs.items():
        trace = traces[name]
        if trace.get("config") != cfg:
            raise ValueError(f"profile configuration drift: {name}")
        audits[name] = audit_trace(trace)  # Never trust a stored audit result.
    hashes = traces["completed"]["source_sha256"]
    if any(t["source_sha256"] != hashes for t in traces.values()):
        raise ValueError("mixed source digests")
    outcomes = {
        name: _outcome(traces[name])
        for name in (
            "cash_refusal",
            "execution_refusal",
            "feed_end",
            "completed",
            "decimal_boundary",
        )
    }
    volume_excess = sum(f["size"] for f in traces["volume"]["fills"] if f["bar"] == 1) - 2
    fill_count = len(traces["completed"]["fills"])
    expectations = {
        "cash_refusal": dict(
            state="terminal_residual",
            terminal_stage="admission" if checksubmit else "execution",
            remaining=5,
        ),
        "execution_refusal": dict(
            state="terminal_residual", terminal_stage="execution", remaining=3
        ),
        "feed_end": dict(state="active_at_feed_end", terminal_stage=None, remaining=3),
        "completed": dict(state="completed", terminal_stage=None, remaining=0),
        "decimal_boundary": dict(
            state="terminal_residual",
            terminal_stage="admission" if checksubmit else "execution",
            remaining=5,
        ),
    }
    if (
        any(outcomes[k] != v for k, v in expectations.items())
        or volume_excess != 2
        or fill_count != 2
    ):
        raise ValueError("fixed-control behavior changed; do not silently revise profile")
    capabilities = dict(
        OBSERVED_SUBSET,
        admission="whole_order_precheck" if checksubmit else "no_submission_precheck",
        arithmetic="binary64",
    )
    capabilities.update({k: "unqualified" for k in VALUES if k not in capabilities})
    return {
        "schema": SCHEMA,
        "engine_version": VERSION,
        "checksubmit": checksubmit,
        "synthetic_only": True,
        "external_engine_validated": False,
        "capabilities": capabilities,
        "scope": "one_asset_buy_only_collapsed_ohlc_fixedsize_no_cheat",
        "evidence": {
            "trace_sha256": {k: _digest(v) for k, v in traces.items()},
            "source_sha256": hashes,
            "outcomes": outcomes,
            "order_outcomes": {name: summarize_orders(trace) for name, trace in traces.items()},
            "observed_volume_excess": volume_excess,
            "fixed_fee_fill_count": fill_count,
        },
        "traces": traces,
    }


def characterize(*, checksubmit: bool = True) -> dict[str, Any]:
    """Rerun six fixed controls on the pinned optional package. No network/data."""
    return _derive(
        {name: observe(**cfg) for name, cfg in _configs(checksubmit).items()}, checksubmit
    )


def assess(profile: dict[str, Any], requirements: dict[str, str]) -> dict[str, Any]:
    """Reject invalid reports; list mismatches/unknowns without claiming parity.

    Internal consistency/digests are not authenticity: a malicious producer can
    fabricate matching reports. Only controlled local runs are in scope.
    """
    if type(requirements) is not dict or not requirements:
        raise ValueError("nonempty explicit requirements required")
    for key, value in requirements.items():
        if key not in VALUES or type(value) is not str or value not in VALUES[key]:
            raise ValueError(f"unknown requirement or value: {key}")
    try:
        expected = _derive(profile["traces"], profile["checksubmit"])
        if _digest(profile) != _digest(expected):
            raise ValueError("profile metadata/evidence does not match traces")
    except (KeyError, TypeError, AttributeError, StopIteration) as exc:
        raise ValueError("invalid profile") from exc
    capabilities = profile["capabilities"]
    unknown = sorted(k for k in requirements if capabilities[k] == "unqualified")
    mismatches = {
        k: {"required": requirements[k], "observed": capabilities[k]}
        for k in sorted(requirements)
        if k not in unknown and capabilities[k] != requirements[k]
    }
    return {
        "schema": "openquant_execution_preflight_v1",
        "status": "blocked" if unknown or mismatches else "eligible_for_case_comparison",
        "external_engine_validated": False,
        "case_comparison_executed": False,
        "requirements": dict(requirements),
        "mismatches": mismatches,
        "unqualified": unknown,
        "omitted_dimensions": sorted(set(VALUES) - set(requirements)),
        "profile_sha256": _digest(profile),
    }


def assess_portable(profile: dict[str, Any], *, fixture: Path = PORTABLE_FIXTURE) -> dict[str, Any]:
    """Screen the exact reviewed fixture; a changed file needs a new review."""
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    if digest != PORTABLE_FIXTURE_SHA256:
        raise ValueError("unreviewed portable fixture; revise mapping explicitly")
    return dict(
        assess(profile, PORTABLE_V2),
        requirements_source={
            "path": "experiments/fixtures/execution_cost_acceptance_v2.json",
            "sha256": digest,
            "mapping": "manual_convention_screen_not_full_specification",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checksubmit", choices=("on", "off"), default="on")
    parser.add_argument(
        "--requirements", choices=("portable-v2", "observed-subset"), default="portable-v2"
    )
    args = parser.parse_args()
    profile = characterize(checksubmit=args.checksubmit == "on")
    requirements = (
        PORTABLE_V2
        if args.requirements == "portable-v2"
        else dict(OBSERVED_SUBSET, admission=profile["capabilities"]["admission"])
    )
    verdict = (
        assess_portable(profile)
        if args.requirements == "portable-v2"
        else assess(profile, requirements)
    )
    print(json.dumps(dict(profile=profile, preflight=verdict), indent=2, allow_nan=False))
    return 2 if verdict["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
