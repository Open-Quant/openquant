"""Independent hand fixtures for the experimental cash/share ledger."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone
from fractions import Fraction as F
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def ledger_module():
    path = ROOT / "experiments/fill_ledger.py"
    assert path.exists(), "the offline fill ledger has not been implemented"
    spec = importlib.util.spec_from_file_location("_fill_ledger_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ts(n):
    return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=n)


def initial(m, **kwargs):
    return m.InitialState(at=ts(0), cash=100, inventory=0, mark=10, **kwargs)


def decision(m, name="buy", shares=2, at=0, earliest=0, deadline=2, **kwargs):
    return m.Decision(name, ts(at), ts(at), ts(earliest), ts(deadline), shares, **kwargs)


def bucket(m, at=1, mark=11, bid="10.5", ask=10, volume=10, **kwargs):
    # Independent fixtures may have a mark outside the spread. Executable bid <= ask
    # remains a required market constraint, so normal fixtures use bid 9.5.
    return m.MarketBucket(ts(at), mark, bid, ask, volume, **kwargs)


def test_entry_and_terminal_cash_are_not_marked_wealth():
    m = ledger_module()
    report = m.run_ledger(
        initial(m),
        [decision(m)],
        [bucket(m, bid="9.5")],
        fixed_fee=1,
        terminal_policy="liquidate_at_declared_bid",
        terminal_bucket=bucket(m, at=2, ask=11),
    )
    entry, exit_ = report.rows
    assert (entry.cash, entry.inventory, entry.equity) == (79, 2, 101)
    assert (exit_.cash, exit_.inventory, exit_.equity) == (99, 0, 99)
    assert entry.market_pnl == 0 and entry.execution_pnl == 2 and entry.commission == 1
    assert exit_.execution_pnl == -1 and exit_.commission == 1
    assert report.terminal.completed and report.decisions_completed
    assert [f.shares for f in report.fills] == [2, -2]
    assert m.run_ledger(initial(m), [], [bucket(m, bid="9.5")]).terminal.cash == 100


def test_fifo_shares_capacity_across_orders_and_retains_zero_fill_attempts():
    m = ledger_module()
    report = m.run_ledger(
        initial(m),
        [decision(m, "b", 6), decision(m, "a", 6)],
        [bucket(m, bid="9.5")],
        participation="0.5",
    )
    assert [(f.decision_id, f.shares) for f in report.fills] == [("a", 5)]
    assert [r.remaining for r in report.orders] == [1, 6]
    assert [a.reason for a in report.attempts] == ["partial_volume", "no_volume"]
    assert report.rows[0].executed_volume == 5
    assert not report.decisions_completed


def test_terminal_capacity_retains_unsold_stock_and_marked_cash_distinction():
    m = ledger_module()
    report = m.run_ledger(
        initial(m),
        [decision(m)],
        [bucket(m, bid="9.5")],
        terminal_policy="liquidate_at_declared_bid",
        terminal_bucket=bucket(m, at=2, ask=11, volume=1),
    )
    assert report.terminal.inventory == 1
    assert report.terminal.cash == F("90.5")
    assert report.terminal.marked_equity == F("101.5")
    assert not report.terminal.completed


def test_fee_not_charged_on_zero_fill_and_deadline_cancels_remainder():
    m = ledger_module()
    report = m.run_ledger(
        initial(m),
        [decision(m, deadline=1)],
        [bucket(m, bid="9.5", volume=0), bucket(m, at=2, bid="9.5")],
        fixed_fee=5,
    )
    assert not report.fills
    assert report.terminal.cash == 100
    assert report.orders[0].status == "cancelled_at_deadline"
    assert report.orders[0].remaining == 2
    assert len(report.attempts) == 1


def test_next_bucket_causality_and_future_quote_mutation():
    m = ledger_module()
    d = decision(m, at=1, earliest=1, deadline=3)
    quotes = [bucket(m, bid="9.5"), bucket(m, at=2, bid="9.5"), bucket(m, at=3, bid="9.5")]
    a = m.run_ledger(initial(m), [d], quotes)
    b = m.run_ledger(initial(m), [d], quotes[:2] + [replace(quotes[2], mark=999)])
    assert a.fills == b.fills
    assert a.rows[:2] == b.rows[:2]
    assert a.fills[0].at == ts(2)
    assert a.rows[0].inventory == 0 and a.rows[0].equity == 100


def test_opposing_orders_consume_gross_capacity_not_net_volume():
    m = ledger_module()
    report = m.run_ledger(
        replace(initial(m), inventory=3),
        [decision(m, "a", 2), decision(m, "b", -3)],
        [bucket(m, bid="9.5", volume=4)],
    )
    assert [f.shares for f in report.fills] == [2, -2]
    assert report.rows[0].executed_volume == 4
    assert report.orders[1].remaining == -1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "NaN", None])
def test_invalid_amounts_fail_with_field_name(bad):
    m = ledger_module()
    with pytest.raises(ValueError, match="volume"):
        bucket(m, bid="9.5", volume=bad)


@pytest.mark.parametrize("field,value", [("volume", -1), ("mark", 0), ("bid", 0), ("ask", -1)])
def test_bad_market_fields_fail(field, value):
    m = ledger_module()
    with pytest.raises(ValueError, match=field):
        replace(bucket(m, bid="9.5"), **{field: value})


def test_timestamp_order_units_identifiers_and_duplicate_ids_fail():
    m = ledger_module()
    for bad in [1, datetime(2026, 1, 1), datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=1)))]:
        with pytest.raises(ValueError, match="UTC"):
            replace(initial(m), at=bad)
    with pytest.raises(ValueError, match="timestamp"):
        decision(m, at=2, earliest=1)
    with pytest.raises(ValueError, match="duplicate"):
        m.run_ledger(initial(m), [decision(m), decision(m)], [])
    with pytest.raises(ValueError, match="instrument"):
        m.run_ledger(initial(m), [], [bucket(m, bid="9.5", instrument="OTHER")])
    with pytest.raises(ValueError, match="currency"):
        m.run_ledger(initial(m), [decision(m, currency="EUR")], [])
    with pytest.raises(ValueError, match="increasing"):
        m.run_ledger(initial(m), [], [bucket(m, bid="9.5"), bucket(m, bid="9.5")])


def test_cash_borrowing_and_short_sales_are_errors_not_clipped_orders():
    m = ledger_module()
    with pytest.raises(ValueError, match="cash"):
        m.run_ledger(initial(m), [decision(m, shares=10)], [bucket(m, bid="9.5")], fixed_fee=1)
    with pytest.raises(ValueError, match="inventory"):
        m.run_ledger(initial(m), [decision(m, shares=-1)], [bucket(m, bid="9.5")])
    with pytest.raises(ValueError, match="cash"):
        replace(initial(m), cash=-1)
    with pytest.raises(ValueError, match="inventory"):
        replace(initial(m), inventory=-1)


@pytest.mark.parametrize(
    "kw",
    [{"fixed_fee": -1}, {"fee_rate": float("nan")}, {"participation": 2}, {"participation": -1}],
)
def test_invalid_execution_costs_and_participation_fail(kw):
    m = ledger_module()
    with pytest.raises(ValueError, match=next(iter(kw))):
        m.run_ledger(initial(m), [], [], **kw)


def test_terminal_quote_required_strictly_new_and_no_silent_quote_consumption():
    m = ledger_module()
    with pytest.raises(ValueError, match="terminal_bucket"):
        m.run_ledger(initial(m), [], [], terminal_policy="liquidate_at_declared_bid")
    with pytest.raises(ValueError, match="terminal_bucket"):
        m.run_ledger(initial(m), [], [], terminal_bucket=bucket(m, bid="9.5"))
    with pytest.raises(ValueError, match="after"):
        m.run_ledger(
            initial(m),
            [],
            [bucket(m, bid="9.5")],
            terminal_policy="liquidate_at_declared_bid",
            terminal_bucket=bucket(m, bid="9.5"),
        )


@pytest.mark.parametrize("scale", ["1", "1e-200", "1e200"])
def test_exact_decimal_accounting_avoids_binary64_underflow_and_overflow(scale):
    m = ledger_module()
    s = F(scale)
    state = m.InitialState(ts(0), 100 * s * s, 0, 10 * s)
    d = decision(m, shares=2 * s)
    quote = m.MarketBucket(ts(1), 11 * s, 9 * s, 10 * s, 10 * s)
    report = m.run_ledger(state, [d], [quote], fixed_fee=s * s)
    assert report.terminal.cash == 79 * s * s
    assert report.terminal.marked_equity == 101 * s * s


def test_records_are_immutable_and_decimal_float_policy_is_explicit():
    m = ledger_module()
    state = replace(initial(m), cash=0.1)
    assert state.cash == F(1, 10)
    with pytest.raises(FrozenInstanceError):
        state.cash = 2
