"""Fail-closed, exact-path synthetic execution conformance gate.

Success requires both separately implemented interpreters to match the complete
hand-calculated fixture, not merely each other. No external-engine claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction as Q
from pathlib import Path
from typing import Any

from experiments.execution_cost_conformance import CONTRACT, _observe
from experiments.execution_cost_reference import observe

FIXTURE = Path(__file__).with_name("fixtures") / "execution_cost_acceptance_v2.json"
PATHS = {
    "cash",
    "inventory",
    "equity",
    "commission",
    "fill_shares",
    "fill_prices",
    "fill_commissions",
}
FLAGS = {"terminal_completed", "decisions_completed"}


def differences(expected: dict[str, Any], observed: dict[str, Any]) -> list[dict[str, Any]]:
    """Explain field/row mismatches, retaining equivalent rational spellings."""
    result: list[dict[str, Any]] = []
    for key in sorted(expected.keys() | observed.keys()):
        if key not in expected or key not in observed:
            result.append({"field": key, "kind": "missing_or_extra_field"})
            continue
        wanted, actual = expected[key], observed[key]
        if isinstance(wanted, list) and isinstance(actual, list):
            for index in range(max(len(wanted), len(actual))):
                a = wanted[index] if index < len(wanted) else None
                b = actual[index] if index < len(actual) else None
                if a is None or b is None or Q(a) != Q(b):
                    result.append({"field": key, "index": index, "expected": a, "observed": b})
        elif type(actual) is not bool or type(wanted) is not bool or actual != wanted:
            result.append({"field": key, "expected": wanted, "observed": actual})
    return result


def _keys(record: Any, keys: set[str], name: str) -> None:
    if not isinstance(record, dict) or set(record) != keys:
        raise ValueError(f"{name} fields must be {sorted(keys)}")


def _number(value: Any, *, positive: bool = False, signed: bool = False) -> Q:
    if not isinstance(value, str):
        raise ValueError("fixture numbers must be rational strings")
    try:
        number = Q(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid rational string") from exc
    if not signed and (number < 0 or (positive and not number)):
        raise ValueError("invalid numeric sign")
    return number


def _time(value: Any) -> None:
    if type(value) is not int:
        raise ValueError("timestamp must be integer seconds")


def _validate(s: dict[str, Any], case: dict[str, Any]) -> None:
    _keys(s, {"initial", "decision", "buckets", "terminal_bucket", "participation"}, "scenario")
    _keys(s["initial"], {"at", "cash", "inventory", "mark"}, "initial")
    _time(s["initial"]["at"])
    for key in ("cash", "inventory", "mark"):
        _number(s["initial"][key], positive=key == "mark")
    d = s["decision"]
    _keys(
        d,
        {"id", "information_at", "decision_at", "earliest_fill_at", "deadline", "shares"},
        "decision",
    )
    if not isinstance(d["id"], str) or not d["id"].strip():
        raise ValueError("decision id must be nonempty")
    times = [d[key] for key in ("information_at", "decision_at", "earliest_fill_at", "deadline")]
    for t in times:
        _time(t)
    if times != sorted(times) or times[0] < s["initial"]["at"]:
        raise ValueError("invalid decision times")
    if not _number(d["shares"], signed=True):
        raise ValueError("shares must be nonzero")
    if not isinstance(s["buckets"], list) or not s["buckets"]:
        raise ValueError("nonempty bucket list required")
    previous = s["initial"]["at"]
    for b in [*s["buckets"], s["terminal_bucket"]]:
        _keys(b, {"at", "mark", "bid", "ask"}, "bucket")
        _time(b["at"])
        if b["at"] <= previous:
            raise ValueError("buckets must increase after initial time")
        previous = b["at"]
        for key in ("mark", "bid", "ask"):
            _number(b[key], positive=True)
        if Q(b["bid"]) > Q(b["ask"]):
            raise ValueError("crossed quote")
    if not 0 <= _number(s["participation"]) <= 1:
        raise ValueError("participation outside [0,1]")
    _number(case["fixed_fee"])
    _number(case["fee_rate"])
    if not isinstance(case["volumes"], list) or len(case["volumes"]) != len(s["buckets"]):
        raise ValueError("volume count must equal bucket count")
    for v in case["volumes"]:
        _number(v)
    if case["terminal_volume"] is not None:
        _number(case["terminal_volume"])
    if "expected_error" in case:
        if case["expected_error"] not in (
            "cash would become negative",
            "inventory would become negative",
        ):
            raise ValueError("unsupported expected error")
    else:
        expected = case["expected"]
        _keys(expected, PATHS | FLAGS, "expected")
        rows = len(s["buckets"]) + (case["terminal_volume"] is not None)
        for key in FLAGS:
            if type(expected[key]) is not bool:
                raise ValueError("completion flags must be bool")
        for key in PATHS:
            if not isinstance(expected[key], list):
                raise ValueError("paths must be lists")
            for value in expected[key]:
                _number(value, signed=True)
            if key in {"cash", "inventory", "equity", "commission"} and len(expected[key]) != rows:
                raise ValueError("missing or extra expected row")
        if (
            len({len(expected[key]) for key in ("fill_shares", "fill_prices", "fill_commissions")})
            != 1
        ):
            raise ValueError("fill path lengths differ")


def evaluate(document: dict[str, Any]) -> dict[str, Any]:
    """Return a failed/unsupported verdict rather than silently skip bad cases."""
    base: dict[str, Any] = {
        "schema_version": 1,
        "synthetic_only": True,
        "external_engine_validated": False,
        "implementations": ["openquant_experimental", "rational_reference"],
        "passed": False,
        "cases": [],
    }
    if type(document.get("schema_version")) is not int or document["schema_version"] not in (1, 2):
        return dict(base, status="unsupported", error="unsupported schema_version")
    if document.get("contract") != CONTRACT:
        return dict(base, status="unsupported", error="unsupported contract")
    try:
        fields = {"schema_version", "contract", "cases"}
        if document["schema_version"] == 1:
            fields.add("scenario")
        _keys(document, fields, "document")
        cases = document["cases"]
        if not isinstance(cases, list) or not cases:
            raise ValueError("nonempty cases required")
        ids = []
        for case in cases:
            keys = {"id", "fixed_fee", "fee_rate", "terminal_volume", "volumes"}
            keys.add("expected_error" if "expected_error" in case else "expected")
            if document["schema_version"] == 2:
                keys.add("scenario")
            _keys(case, keys, "case")
            if not isinstance(case["id"], str) or not case["id"].strip() or case["id"] in ids:
                raise ValueError("unique nonempty case IDs required")
            ids.append(case["id"])
            s = case["scenario"] if document["schema_version"] == 2 else document["scenario"]
            _validate(s, case)
    except (KeyError, TypeError, ValueError) as exc:
        return dict(base, status="invalid_fixture", error=str(exc))
    for case in cases:
        normalized = dict(document, scenario=case.get("scenario", document.get("scenario")))
        outcomes = {}
        for name, adapter in (
            ("openquant_experimental", _observe),
            ("rational_reference", observe),
        ):
            try:
                observed = adapter(normalized, case)
                if "expected_error" in case:
                    outcomes[name] = {"passed": False, "error": "expected rejection not raised"}
                else:
                    diff = differences(case["expected"], observed)
                    outcomes[name] = {"passed": not diff, "differences": diff, "observed": observed}
            except ValueError as exc:
                reason = case.get("expected_error")
                accepted = reason is not None and (
                    str(exc) == reason or str(exc).startswith(reason + " for ")
                )
                outcomes[name] = {"passed": accepted, "error": str(exc)}
        base["cases"].append(
            {
                "id": case["id"],
                "passed": all(x["passed"] for x in outcomes.values()),
                "outcomes": outcomes,
            }
        )
    base["passed"] = all(case["passed"] for case in base["cases"])
    base["status"] = "passed" if base["passed"] else "failed"
    return base


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def run(path: Path = FIXTURE) -> dict[str, Any]:
    raw = path.read_bytes()
    document = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(document, dict):
        return {"passed": False, "status": "invalid_fixture", "error": "object required"}
    source_dir = Path(__file__).parent
    names = (
        "execution_cost_acceptance.py",
        "execution_cost_reference.py",
        "execution_cost_conformance.py",
        "fill_ledger.py",
    )
    hashes = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest() for name in names}
    return dict(
        evaluate(document),
        fixture_sha256=hashlib.sha256(raw).hexdigest(),
        implementation_source_sha256=hashes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    args = parser.parse_args()
    try:
        report = run(args.fixture)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        report = {"passed": False, "status": "invalid_fixture", "error": str(exc)}
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
