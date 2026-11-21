"""Bounded synthetic admission/lifecycle observations, never engine qualification.

Buy-only, one asset, collapsed OHLC, no leverage/interest/slippage/cheat mode.
The wrappers record primitive snapshots and forward to the pinned implementation.
Submission shadow state is not a reservation of actual account cash.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

VERSION = "1.9.78.123"
SCHEMA = "backtrader_admission_observation_v1"


def _validate(cash, prices, volumes, orders, fixed, rate, checksubmit):
    if type(checksubmit) is not bool:
        raise ValueError("checksubmit must be bool")
    if not isinstance(prices, (list, tuple)) or not isinstance(volumes, (list, tuple)):
        raise ValueError("prices and volumes must be sequences")
    if len(prices) != len(volumes) or not 2 <= len(prices) <= 28:
        raise ValueError("2 to 28 aligned bars required")
    for value in [cash, fixed, rate, *prices]:
        if type(value) not in (int, float) or not 0 <= value <= 1e12 or not math.isfinite(value):
            raise ValueError("cash, fees and prices must be finite numbers in [0, 1e12]")
    if cash <= 0 or cash / 100.0 == 0:
        raise ValueError("positive cash with representable initial fund shares required")
    if any(p <= 0 for p in prices) or rate > 1:
        raise ValueError("positive prices and fee rate in [0, 1] required")
    if any(type(v) is not int or not 0 <= v <= 1_000_000 for v in volumes):
        raise ValueError("volumes must be integers in [0, 1000000]")
    if not isinstance(orders, (list, tuple)) or not 1 <= len(orders) <= 8:
        raise ValueError("1 to 8 orders required")
    ids = set()
    for order in orders:
        if not isinstance(order, dict) or set(order) != {"id", "bar", "shares"}:
            raise ValueError("order fields must be id, bar, shares")
        name, bar, shares = order["id"], order["bar"], order["shares"]
        if not isinstance(name, str) or not name.strip() or name in ids:
            raise ValueError("unique nonempty order IDs required")
        if type(bar) is not int or not 0 <= bar < len(prices):
            raise ValueError("submission bar out of range")
        if type(shares) is not int or not 1 <= shares <= 1_000_000:
            raise ValueError("shares must be integers in [1, 1000000]")
        ids.add(name)


def observe(*, cash, prices, volumes, orders, fixed=0, rate=0, checksubmit=True) -> dict[str, Any]:
    """Capture events and end-of-bar states for one bounded, offline run.

    Bars and decision bars are zero-based. A Market order cannot fill on its
    submission bar. The filler limits each order separately, not total bar flow.
    End-of-data does not cancel an outstanding order. No live accounts are used.
    """
    _validate(cash, prices, volumes, orders, fixed, rate, checksubmit)
    bt = importlib.import_module("backtrader")
    if bt.__version__ != VERSION:
        raise ValueError("unsupported backtrader version")
    config = dict(
        cash=cash,
        prices=list(prices),
        volumes=list(volumes),
        orders=[dict(x) for x in orders],
        fixed=fixed,
        rate=rate,
        checksubmit=checksubmit,
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "external_engine_validated": False,
        "config": config,
        "engine_version": VERSION,
        "execution_attempts": [],
        "broker_events": [],
        "notifications": [],
        "fills": [],
        "bars": [],
    }

    def snapshot(order):
        return dict(
            id=order.info["probe_id"],
            status=order.getstatusname(),
            filled=order.executed.size,
            remaining=order.executed.remsize,
            commission=order.executed.comm,
            alive=order.alive(),
        )

    # Bases come from the optional, version-checked dynamic module.
    class PerFill(bt.CommInfoBase):  # type: ignore[name-defined]
        params = (("stocklike", True), ("commtype", bt.CommInfoBase.COMM_PERC), ("percabs", True))

        def _getcommission(self, size, price, pseudoexec):
            return fixed + abs(size) * price * rate if size else 0.0

    class ObservedBroker(bt.brokers.BackBroker):  # type: ignore[name-defined]
        phase = "submission"

        def check_submitted(self):
            prior, self.phase = self.phase, "admission"
            try:
                return super().check_submitted()
            finally:
                self.phase = prior

        def notify(self, order):
            result["broker_events"].append(
                dict(
                    bar=len(order.data) - 1,
                    phase=self.phase,
                    **snapshot(order),
                    live_cash=self.cash,
                    live_inventory=self.positions[order.data].size,
                )
            )
            return super().notify(order)

        def _execute(self, order, ago=None, price=None, cash=None, position=None, dtcoc=None):
            pseudo = ago is None
            before = dict(
                bar=len(order.data) - 1,
                id=order.info["probe_id"],
                phase="admission" if pseudo else "execution",
                cash_before=cash if pseudo else self.cash,
                inventory_before=position.size if pseudo else self.positions[order.data].size,
                remaining_before=order.executed.remsize,
            )
            count = len(order.executed.exbits)
            prior, self.phase = self.phase, before["phase"]
            try:
                returned = super()._execute(
                    order, ago=ago, price=price, cash=cash, position=position, dtcoc=dtcoc
                )
            finally:
                self.phase = prior
            before.update(
                cash_after=returned if pseudo else self.cash,
                inventory_after=position.size if pseudo else self.positions[order.data].size,
                remaining_after=order.executed.remsize,
            )
            result["execution_attempts"].append(before)
            if not pseudo:
                # Read bits immediately, not shared histories from delayed notifications.
                for bit in list(order.executed.exbits)[count:]:
                    result["fills"].append(
                        dict(
                            bar=len(order.data) - 1,
                            id=order.info["probe_id"],
                            size=bit.size,
                            price=bit.price,
                            commission=bit.comm,
                        )
                    )
            return returned

    class Strategy(bt.Strategy):  # type: ignore[name-defined]
        def __init__(self):
            self.probe_orders = {}

        def next(self):
            bar = len(self) - 1
            for order in orders:
                if order["bar"] == bar:
                    self.probe_orders[order["id"]] = self.buy(
                        size=order["shares"], probe_id=order["id"]
                    )
            result["bars"].append(
                dict(
                    bar=bar,
                    cash=self.broker.getcash(),
                    inventory=self.position.size,
                    orders=[snapshot(x) for x in self.probe_orders.values()],
                )
            )

        def notify_order(self, order):
            # Delivery time is separate from emission time and from fill time.
            result["notifications"].append(dict(delivered_bar=len(self) - 1, **snapshot(order)))

    with tempfile.TemporaryDirectory() as tmp:
        feed = Path(tmp) / "synthetic.csv"
        rows = ["date,open,high,low,close,volume,openinterest"]
        rows += [
            f"2026-01-{i:02},{p},{p},{p},{p},{v},0"
            for i, (p, v) in enumerate(zip(prices, volumes, strict=True), 1)
        ]
        feed.write_text("\n".join(rows) + "\n")
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.adddata(bt.feeds.GenericCSVData(dataname=str(feed), dtformat="%Y-%m-%d"))
        cerebro.addstrategy(Strategy)
        broker = ObservedBroker(
            checksubmit=checksubmit,
            coo=False,
            coc=False,
            shortcash=True,
            slip_perc=0,
            slip_fixed=0,
            fundstartval=100.0,
        )
        broker.setcash(cash)
        broker.addcommissioninfo(PerFill())
        broker.set_filler(bt.fillers.FixedSize())
        cerebro.setbroker(broker)
        cerebro.run()
    result["source_sha256"] = {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in {
            "broker": bt.brokers.bbroker,
            "order": bt.order,
            "fillers": bt.fillers,
            "commission": bt.comminfo,
            "position": bt.position,
        }.items()
    }
    # Reject nonfinite output, rather than serializing an apparently usable trace.
    json.dumps(result, allow_nan=False)
    return result


def audit_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Check trace bookkeeping, not economic realism or complete engine parity.

    Decimal-string rational debits are compared to the engine's binary64 state
    with relative tolerance 1e-12 and absolute tolerance 1e-10. This is a bounded
    diagnostic, not a money rounding policy, authentication or proof of accuracy.
    Shadow cash is intentionally not checked as though it were real cash.
    """

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def numeric(value):
        require(
            type(value) in (int, float) and -1e30 <= value <= 1e30 and math.isfinite(value),
            "nonfinite or out-of-range trace number",
        )
        return Fraction(str(value))

    def fields(value, expected):
        require(isinstance(value, dict) and set(value) == set(expected), "invalid trace fields")

    def equal(actual, expected):
        numeric(actual)
        return math.isclose(actual, float(expected), rel_tol=1e-12, abs_tol=1e-10)

    fields(
        trace,
        (
            "schema",
            "external_engine_validated",
            "config",
            "engine_version",
            "execution_attempts",
            "broker_events",
            "notifications",
            "fills",
            "bars",
            "source_sha256",
        ),
    )
    require(trace.get("schema") == SCHEMA, "unsupported trace schema")
    require(trace.get("external_engine_validated") is False, "qualification is not permitted")
    require(trace.get("engine_version") == VERSION, "unsupported trace engine")
    cfg = trace["config"]
    fields(cfg, ("cash", "prices", "volumes", "orders", "fixed", "rate", "checksubmit"))
    _validate(**cfg)
    orders = {x["id"]: x for x in cfg["orders"]}
    for key in ("fills", "bars", "broker_events", "notifications", "execution_attempts"):
        require(isinstance(trace[key], list), "invalid trace array")
    hashes = trace["source_sha256"]
    fields(hashes, ("broker", "order", "fillers", "commission", "position"))
    require(
        all(
            isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)
            for x in hashes.values()
        ),
        "invalid source digest",
    )
    require(len(trace["bars"]) == len(cfg["prices"]), "bar count mismatch")

    state_fields = ("id", "status", "filled", "remaining", "commission", "alive")

    def state_shape(state, extra=()):
        fields(state, (*state_fields, *extra))
        require(isinstance(state["id"], str) and state["id"] in orders, "unknown order")
        filled, remaining = numeric(state["filled"]), numeric(state["remaining"])
        require(
            filled >= 0
            and remaining >= 0
            and filled.denominator == 1
            and remaining.denominator == 1
            and filled + remaining == orders[state["id"]]["shares"],
            "invalid order quantities",
        )
        require(
            numeric(state["commission"]) >= 0 and type(state["alive"]) is bool,
            "invalid order state",
        )

    events = trace["broker_events"]
    event_states: dict[int, dict[str, dict[str, Any]]] = {}
    latest: dict[str, dict[str, Any]] = {}
    previous_bar = -1
    for event in events:
        state_shape(event, ("bar", "phase", "live_cash", "live_inventory"))
        bar = event["bar"]
        name = event["id"]
        require(
            type(bar) is int
            and orders[name]["bar"] <= bar < len(cfg["prices"])
            and bar >= previous_bar,
            "invalid event time",
        )
        previous_bar = bar
        phase, status = event["phase"], event["status"]
        require(
            (
                phase == "submission"
                and bar == orders[name]["bar"]
                and status == ("Submitted" if cfg["checksubmit"] else "Accepted")
            )
            or (
                phase == "admission"
                and cfg["checksubmit"]
                and bar == orders[name]["bar"] + 1
                and status in ("Accepted", "Margin")
            )
            or (
                phase == "execution"
                and bar > orders[name]["bar"]
                and status in ("Partial", "Completed", "Margin")
            ),
            "invalid event phase",
        )
        require(
            numeric(event["live_cash"]) >= 0 and numeric(event["live_inventory"]) >= 0,
            "invalid live event state",
        )
        latest[name] = {key: event[key] for key in state_fields}
        event_states[bar] = dict(latest)
    by_bar: dict[int, list[dict[str, Any]]] = {}
    seen = set()
    previous_bar = -1
    for fill in trace["fills"]:
        fields(fill, ("bar", "id", "size", "price", "commission"))
        bar, name = fill["bar"], fill["id"]
        require(type(bar) is int and 0 <= bar < len(cfg["prices"]), "invalid fill bar")
        require(
            isinstance(name, str) and name in orders and orders[name]["bar"] < bar,
            "fill before next eligible bar",
        )
        require(bar >= previous_bar and (bar, name) not in seen, "duplicate or unordered fill")
        previous_bar = bar
        seen.add((bar, name))
        size = numeric(fill["size"])
        require(
            size > 0 and size.denominator == 1 and size <= cfg["volumes"][bar],
            "invalid per-order fill capacity",
        )
        require(fill["price"] == cfg["prices"][bar], "fill price mismatch")
        fee = Fraction(str(cfg["fixed"])) + size * numeric(fill["price"]) * Fraction(
            str(cfg["rate"])
        )
        require(equal(fill["commission"], fee), "fill commission mismatch")
        by_bar.setdefault(bar, []).append(fill)
    event_cash, event_inventory = numeric(cfg["cash"]), Fraction(0)
    event_fill_keys = set()
    event_sizes = {name: Fraction(0) for name in orders}
    event_fees = {name: Fraction(0) for name in orders}
    prior_status: dict[str, str] = {}
    for event in events:
        name, phase, status = event["id"], event["phase"], event["status"]
        before = prior_status.get(name)
        require(
            (phase == "submission" and before is None)
            or (phase == "admission" and before == "Submitted")
            or (phase == "execution" and before in ("Accepted", "Partial")),
            "invalid event transition",
        )
        if phase == "execution" and status in ("Partial", "Completed"):
            event_key = (event["bar"], name)
            matches = [x for x in by_bar.get(event["bar"], []) if x["id"] == name]
            require(
                len(matches) == 1 and event_key not in event_fill_keys, "event missing unique fill"
            )
            fill = matches[0]
            size, fee = numeric(fill["size"]), numeric(fill["commission"])
            event_cash -= size * numeric(fill["price"]) + fee
            event_inventory += size
            event_sizes[name] += size
            event_fees[name] += fee
            event_fill_keys.add(event_key)
        require(
            equal(event["live_cash"], event_cash)
            and numeric(event["live_inventory"]) == event_inventory,
            "live event accounting mismatch",
        )
        require(
            numeric(event["filled"]) == event_sizes[name]
            and equal(event["commission"], event_fees[name]),
            "event execution mismatch",
        )
        prior_status[name] = status
    require(event_fill_keys == seen, "fill missing execution event")
    cash, inventory = Fraction(str(cfg["cash"])), Fraction(0)
    sizes = {name: Fraction(0) for name in orders}
    fees = {name: Fraction(0) for name in orders}
    terminal: dict[str, str] = {}
    shared_excess = []
    states: dict[str, dict[str, Any]] = {}
    latest_events: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(trace["bars"]):
        fields(row, ("bar", "cash", "inventory", "orders"))
        require(type(row["bar"]) is int and row["bar"] == index, "unordered bar snapshots")
        require(isinstance(row["orders"], list), "invalid order snapshots")
        for state in row["orders"]:
            state_shape(state)
        fills = by_bar.get(index, [])
        if sum(x["size"] for x in fills) > cfg["volumes"][index]:
            shared_excess.append(index)
        for fill in fills:
            name, size = fill["id"], numeric(fill["size"])
            require(name not in terminal, "fill after terminal state")
            fee = Fraction(str(cfg["fixed"])) + size * numeric(fill["price"]) * Fraction(
                str(cfg["rate"])
            )
            cash -= size * numeric(fill["price"]) + fee
            inventory += size
            sizes[name] += size
            fees[name] += fee
        require(numeric(row["cash"]) >= 0 and equal(row["cash"], cash), "live cash mismatch")
        require(numeric(row["inventory"]) == inventory, "live inventory mismatch")
        states = {x["id"]: x for x in row["orders"]}
        require(len(states) == len(row["orders"]), "duplicate order snapshot")
        require(
            set(states) == {name for name, order in orders.items() if order["bar"] <= index},
            "missing or extra order snapshot",
        )
        latest_events = event_states.get(index, latest_events)
        require(states == latest_events, "event and bar snapshots disagree")
        for name, state in states.items():
            remaining = orders[name]["shares"] - sizes[name]
            require(
                remaining >= 0
                and numeric(state["filled"]) == sizes[name]
                and numeric(state["remaining"]) == remaining,
                "order quantity mismatch",
            )
            require(equal(state["commission"], fees[name]), "order commission mismatch")
            status, alive = state["status"], state["alive"]
            require(type(alive) is bool, "invalid alive flag")
            if name in terminal:
                require(status == terminal[name], "terminal state changed")
            if status == "Completed":
                require(remaining == 0 and not alive, "invalid completed order")
                terminal[name] = status
            elif status == "Margin":
                require(remaining > 0 and not alive, "invalid margin residual")
                terminal[name] = status
            elif status == "Partial":
                require(0 < sizes[name] < orders[name]["shares"] and alive, "invalid partial state")
            else:
                require(
                    status in ("Submitted", "Accepted") and sizes[name] == 0 and alive,
                    "unsupported order state",
                )
    # Every real _execute call must conserve live cash, inventory and residuals.
    # Pseudo records are shape/count checked only: their state is not account state.
    attempts = trace["execution_attempts"]
    admission_ids = []
    real_keys = set()
    previous_bar = -1
    actual_cash, actual_inventory = cfg["cash"], 0
    filled_so_far = {name: 0 for name in orders}
    for attempt in attempts:
        fields(
            attempt,
            (
                "bar",
                "id",
                "phase",
                "cash_before",
                "cash_after",
                "inventory_before",
                "inventory_after",
                "remaining_before",
                "remaining_after",
            ),
        )
        name, bar = attempt["id"], attempt["bar"]
        require(isinstance(name, str) and name in orders, "unknown attempt order")
        require(
            type(bar) is int
            and orders[name]["bar"] < bar < len(cfg["prices"])
            and bar >= previous_bar,
            "invalid attempt time",
        )
        previous_bar = bar
        for key in (
            "cash_before",
            "cash_after",
            "inventory_before",
            "inventory_after",
            "remaining_before",
            "remaining_after",
        ):
            numeric(attempt[key])
        if attempt["phase"] == "admission":
            require(
                cfg["checksubmit"] and bar == orders[name]["bar"] + 1,
                "unexpected admission attempt",
            )
            admission_ids.append(name)
            require(
                attempt["remaining_before"] == attempt["remaining_after"] == orders[name]["shares"],
                "admission changed actual residual",
            )
            continue
        require(
            attempt["phase"] == "execution" and (bar, name) not in real_keys,
            "invalid execution attempt",
        )
        real_keys.add((bar, name))
        fills = [x for x in by_bar.get(bar, []) if x["id"] == name]
        size = sum(x["size"] for x in fills)
        debit = sum(
            (numeric(x["size"]) * numeric(x["price"]) + numeric(x["commission"]) for x in fills),
            Fraction(0),
        )
        require(
            equal(attempt["cash_before"], actual_cash)
            and attempt["inventory_before"] == actual_inventory,
            "execution start state mismatch",
        )
        require(
            equal(attempt["cash_after"], numeric(actual_cash) - debit)
            and attempt["inventory_after"] == actual_inventory + size,
            "execution conservation mismatch",
        )
        require(
            attempt["remaining_before"] == orders[name]["shares"] - filled_so_far[name]
            and attempt["remaining_after"] == attempt["remaining_before"] - size,
            "execution residual mismatch",
        )
        filled_so_far[name] += size
        actual_cash, actual_inventory = attempt["cash_after"], attempt["inventory_after"]
    expected_admissions = [
        name
        for name, order in orders.items()
        if cfg["checksubmit"] and order["bar"] < len(cfg["prices"]) - 1
    ]
    require(
        len(admission_ids) == len(set(admission_ids))
        and set(admission_ids) == set(expected_admissions),
        "missing or duplicate admission",
    )
    require(seen <= real_keys, "fill missing execution attempt")
    expected_real_keys = set()
    admitted = {
        (event["bar"], event["id"])
        for event in events
        if event["phase"] == "admission" and event["status"] == "Accepted"
    }
    for bar in range(1, len(trace["bars"])):
        for state in trace["bars"][bar - 1]["orders"]:
            attempt_key = (bar, state["id"])
            if state["alive"] and (state["status"] != "Submitted" or attempt_key in admitted):
                expected_real_keys.add(attempt_key)
    require(real_keys == expected_real_keys, "missing or out-of-lifecycle execution attempt")
    require(
        all(
            (event["bar"], event["id"]) in real_keys
            for event in events
            if event["phase"] == "execution"
        ),
        "event missing execution attempt",
    )

    notifications = trace["notifications"]
    for notification in notifications:
        state_shape(notification, ("delivered_bar",))
    require(len(notifications) <= len(events), "extra notifications")
    for event, notification in zip(events, notifications):
        require(
            all(event[x] == notification[x] for x in state_fields), "notification snapshot mismatch"
        )
        delivered = notification["delivered_bar"]
        require(
            type(delivered) is int
            and delivered == max(event["bar"], orders[event["id"]]["bar"] + 1)
            and delivered < len(cfg["prices"]),
            "notification delivery time mismatch",
        )
    require(
        all(
            x["bar"] == len(cfg["prices"]) - 1 and x["phase"] == "submission"
            for x in events[len(notifications) :]
        ),
        "missing delivered notification",
    )
    return {
        "consistent": True,
        "external_engine_validated": False,
        "numerical_tolerance": {"relative": 1e-12, "absolute": 1e-10},
        "completed_ids": [name for name in orders if states[name]["status"] == "Completed"],
        "terminal_residual_ids": [name for name in orders if states[name]["status"] == "Margin"],
        "active_residual_ids": [name for name in orders if states[name]["alive"]],
        "shared_volume_excess_bars": shared_excess,
        "undelivered_notifications": len(events) - len(notifications),
    }


def run():
    """Fixed controls selected before observations; all results remain diagnostic."""
    common = dict(cash=25, prices=[10, 10, 8, 1], volumes=[0, 2, 3, 10], fixed=1, rate=0.001)
    big = {"id": "large", "bar": 0, "shares": 5}
    small = {"id": "small", "bar": 0, "shares": 1}
    scenarios = {
        "whole_order_refused": dict(common, orders=[big]),
        "partial_then_terminal_margin": dict(common, orders=[big], checksubmit=False),
        "same_cycle_large_first": dict(common, orders=[big, small]),
        "same_cycle_small_first": dict(common, orders=[small, big]),
        "separate_cycle_small": dict(common, orders=[big, dict(small, bar=1)]),
        "truncated_partial": dict(
            common, prices=[10, 10], volumes=[0, 2], orders=[big], checksubmit=False
        ),
        "per_order_not_shared_volume": dict(
            common, cash=100, orders=[small, dict(big, shares=2)], volumes=[0, 2, 0, 0]
        ),
        "one_fee_budget_not_repeated_fees": dict(
            common, cash=51.5, prices=[10, 10, 10, 1], orders=[big]
        ),
        "repeated_fees_budgeted": dict(common, cash=52.1, prices=[10, 10, 10, 1], orders=[big]),
        "exact_decimal_cash_boundary": dict(common, cash=51.05, orders=[big]),
    }
    cases = {name: observe(**args) for name, args in scenarios.items()}
    return {
        "schema": "backtrader_admission_study_v1",
        "external_engine_validated": False,
        "cases": cases,
        "audits": {name: audit_trace(trace) for name, trace in cases.items()},
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
