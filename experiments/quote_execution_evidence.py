"""As-of observations and exact marked attribution for the synthetic quote tape.

These are analytical reports, not broker receipts or execution qualification.
A cutoff never cancels an order, imputes a mark, or liquidates a position.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction as Q
from pathlib import Path
from typing import Any

from experiments import quote_visibility_contract as quote


def _replay_prefix(
    document: dict[str, Any], at: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Whole-fixture schema validation is deliberate. Only the selected prefix
    # can affect financial state or trigger an execution funding error.
    quote._validate(document)
    buckets = [bucket for bucket in document["buckets"] if bucket["at"] <= at]
    if not buckets:
        return [], []
    observed = quote.replay(dict(document, buckets=buckets))
    return observed["rows"], observed["fills"]


def _mark_at(document: dict[str, Any], at: int) -> dict[str, Any] | None:
    mark = quote._visible_mark(document["marks"], at)
    if mark is None:
        return None
    return {
        "price": str(Q(mark["price"])),
        "observed_at": mark["observed_at"],
        "available_at": mark["available_at"],
    }


def _snapshot_at(
    document: dict[str, Any], at: int, rows: list[dict[str, Any]], fills: list[dict[str, Any]]
) -> dict[str, Any]:
    row = rows[-1] if rows else None
    order = document["order"]
    decision_at, deadline = order["decision_at"], order["deadline"]
    cash = Q(row["cash"] if row else document["cash"])
    inventory = Q(row["inventory"] if row else document["inventory"])
    remaining = row["remaining"] if row else order["shares"]
    last_fill = fills[-1] if fills else None
    if row and row["at"] >= decision_at:
        status, evidence_at = row["status"], row["at"]
    elif at >= decision_at:
        status, evidence_at = "active", decision_at
    else:
        status, evidence_at = "pending", row["at"] if row else None

    terminal_observation = None
    if status == "completed":
        assert last_fill is not None
        evidence_at = last_fill["at"]
        terminal_observation = {"kind": "completed_from_fill", "at": evidence_at}
    elif status == "expired":
        evidence_at = next(x["at"] for x in rows if x["status"] == "expired")
        terminal_observation = {"kind": "expiry_observed_in_replay", "at": evidence_at}

    mark = _mark_at(document, at)
    equity = (
        str(cash + inventory * Q(mark["price"])) if mark else str(cash) if not inventory else None
    )

    return {
        "schema": "openquant_quote_execution_snapshot_v1",
        "synthetic_only": True,
        "external_engine_validated": False,
        "as_of": at,
        "deadline": deadline,
        "status": status,
        "status_evidence_at": evidence_at,
        "last_replay_bucket_at": row["at"] if row else None,
        "past_deadline_unobserved": at > deadline and terminal_observation is None,
        "terminal_observation": terminal_observation,
        "filled_shares": sum(x["shares"] for x in fills),
        "remaining_shares": remaining,
        "last_fill": last_fill,
        "commission_paid": str(sum((Q(x["commission"]) for x in fills), Q(0))),
        "cash": str(cash),
        "inventory": str(inventory),
        "mark": mark,
        "equity": equity,
    }


def snapshot(document: dict[str, Any], as_of: int) -> dict[str, Any]:
    """Report last evidenced model status and independent mark at the cutoff.

    Active/partial is not a claim about executability after the deadline.
    Terminal observation is from this replay only, never a broker message.
    """
    at = quote._time(as_of)
    rows, fills = _replay_prefix(document, at)
    return _snapshot_at(document, at, rows, fills)


def attribute(document: dict[str, Any], start_at: int, end_at: int) -> dict[str, Any]:
    """Reconcile marked-equity change using only fills in (start_at, end_at].

    Mark both endpoints independently as of their own clocks. This values
    actually held inventory, not the opportunity cost of an unfilled residual.
    """
    start_at, end_at = quote._time(start_at), quote._time(end_at)
    if start_at >= end_at:
        raise ValueError("start_at must precede end_at")
    rows, fills = _replay_prefix(document, end_at)
    start = _snapshot_at(
        document,
        start_at,
        [x for x in rows if x["at"] <= start_at],
        [x for x in fills if x["at"] <= start_at],
    )
    end = _snapshot_at(document, end_at, rows, fills)
    interval = [x for x in fills if x["at"] > start_at]
    fees = sum((Q(x["commission"]) for x in interval), Q(0))
    cash_change = Q(end["cash"]) - Q(start["cash"])
    missing = [name for name, state in (("start", start), ("end", end)) if state["equity"] is None]
    change = None
    terms = None
    if missing:
        status = "unavailable_endpoint_mark"
    else:
        change = Q(end["equity"]) - Q(start["equity"])
        if end["mark"] is None:
            # With one fixed-sided order, two unmarked flat endpoints imply
            # no fills in this interval; cash suffices, no price is synthesized.
            status = "flat_cash_only"
        else:
            status = "reconciled_marked"
            opening = Q(start["inventory"])
            end_mark = Q(end["mark"]["price"])
            opening_move = opening * (end_mark - Q(start["mark"]["price"])) if opening else Q(0)
            fills_to_mark = sum(
                (Q(x["shares"]) * (end_mark - Q(x["price"])) for x in interval), Q(0)
            )
            if opening_move + fills_to_mark - fees != change:
                raise ArithmeticError("synthetic equity attribution does not reconcile")
            terms = {
                "opening_inventory_mark_move": str(opening_move),
                "fills_to_end_mark": str(fills_to_mark),
                "commission_debit": str(-fees),
            }
    return {
        "schema": "openquant_quote_execution_attribution_v1",
        "synthetic_only": True,
        "external_engine_validated": False,
        "status": status,
        "start": start,
        "end": end,
        "unavailable_endpoints": missing,
        "fill_count": len(interval),
        "interval_commissions": str(fees),
        "cash_change": str(cash_change),
        "equity_change": str(change) if change is not None else None,
        "terms": terms,
    }


def decision_benchmark(document: dict[str, Any], as_of: int) -> dict[str, Any]:
    """Compare actual fills to a fee-free, instantaneous paper order.

    The independently observed mark at the decision is not executable. This
    is incremental marked value in fixture currency, excluding opening
    inventory, not a return or a realizable strategy counterfactual.
    """
    as_of = quote._time(as_of)
    rows, fills = _replay_prefix(document, as_of)
    order = document["order"]
    decision_at = order["decision_at"]
    if as_of < decision_at:
        raise ValueError("cutoff precedes the declared decision")

    state = _snapshot_at(document, as_of, rows, fills)
    decision_mark = _mark_at(document, decision_at)
    end_mark = Q(state["mark"]["price"]) if state["mark"] is not None else None
    requested = order["shares"] * (1 if order["side"] == "buy" else -1)
    filled = state["filled_shares"]
    residual = requested - filled

    actual = None
    if filled == 0:
        actual = Q(0)
    elif end_mark is not None:
        actual = Q(state["cash"]) - Q(document["cash"]) + filled * end_mark

    paper = shortfall = terms = None
    status = "unavailable_decision_mark"
    if decision_mark is not None:
        # Visible marks persist, so a known decision mark guarantees a cutoff mark.
        assert end_mark is not None and actual is not None
        benchmark = Q(decision_mark["price"])
        paper = requested * (end_mark - benchmark)
        shortfall = paper - actual
        price_gap = sum((Q(x["shares"]) * (Q(x["price"]) - benchmark) for x in fills), Q(0))
        opportunity = residual * (end_mark - benchmark)
        fees = Q(state["commission_paid"])
        if price_gap + opportunity + fees != shortfall:
            raise ArithmeticError("synthetic paper benchmark does not reconcile")
        terms = {
            "fill_price_gap": str(price_gap),
            "residual_opportunity": str(opportunity),
            "commission_paid": str(fees),
        }
        status = "valued_paper_benchmark"
    return {
        "schema": "openquant_quote_decision_mark_benchmark_v1",
        "synthetic_only": True,
        "external_engine_validated": False,
        "paper_funding_and_execution_assessed": False,
        "status": status,
        "decision_mark": decision_mark,
        "signed_requested_shares": requested,
        "signed_residual_shares": residual,
        "actual_incremental_marked_value": str(actual) if actual is not None else None,
        "paper_incremental_marked_value": str(paper) if paper is not None else None,
        "shortfall": str(shortfall) if shortfall is not None else None,
        "terms": terms,
        "snapshot": state,
    }


def main() -> None:
    """Report a local fixture as deterministic JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    as_of = commands.add_parser("snapshot")
    as_of.add_argument("at", type=int)
    window = commands.add_parser("attribute")
    window.add_argument("start", type=int)
    window.add_argument("end", type=int)
    paper = commands.add_parser("benchmark")
    paper.add_argument("at", type=int)
    for command in (as_of, window, paper):
        command.add_argument("--fixture", type=Path, default=quote.DEMO)
    args = parser.parse_args()
    document = quote.load_fixture(args.fixture)
    if args.command == "snapshot":
        result = snapshot(document, args.at)
    elif args.command == "attribute":
        result = attribute(document, args.start, args.end)
    else:
        result = decision_benchmark(document, args.at)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
