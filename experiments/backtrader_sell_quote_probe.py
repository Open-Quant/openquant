"""Fixed offline sell/expiry and price-basis controls; no portable parity claim."""

from __future__ import annotations

import hashlib
import importlib
import json
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any

from experiments.backtrader_path_probe import VERSION

# No vendor data. The external mark is metadata, not a broker input.
FIXED_SLIPPAGE = 1
EXIT_VALID_FOR_DAYS = 1

ROWS = (
    ("2026-01-01", 10, 10, 10, 10, 0, 10),
    ("2026-01-02", 10, 10, 9, 9, 2, 12),
    ("2026-01-03", 9, 10, 8, 10, 1, 12),
    ("2026-01-04", 11, 11, 11, 11, 2, 13),
)


def observe(
    *, slip_match: bool, slip_out: bool, exit_market: bool = False, exit_no_expiry: bool = False
) -> dict[str, Any]:
    """Observe the same four OHLC bars under one fixed broker configuration."""
    bt = importlib.import_module("backtrader")
    if bt.__version__ != VERSION:
        raise ValueError("unexpected Backtrader version")
    bars: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []

    def path() -> str:
        return (
            "\n".join(
                ["date,open,high,low,close,volume,openinterest"]
                + [",".join(map(str, (*row[:-1], 0))) for row in ROWS]
            )
            + "\n"
        )

    class Strategy(bt.Strategy):  # type: ignore[name-defined]
        def __init__(self) -> None:
            self.entry: Any = None
            self.exit_order: Any = None

        def next(self) -> None:
            bar = len(self) - 1
            if bar == 0:
                self.entry = self.buy(size=2)
            elif bar == 1 and self.position.size == 2:
                self.exit_order = self.sell(
                    size=2,
                    exectype=bt.Order.Market if exit_market else bt.Order.Limit,
                    price=None if exit_market else 9,
                    valid=(
                        None
                        if exit_no_expiry
                        else self.data.datetime.datetime() + timedelta(days=EXIT_VALID_FOR_DAYS)
                    ),
                )
            cash = self.broker.getcash()
            inventory = self.position.size
            bars.append(
                {
                    "cash": cash,
                    "inventory": inventory,
                    "engine_value": self.broker.getvalue(),
                    "reference_mark_value": cash + inventory * ROWS[bar][-1],
                    "exit_status": self.exit_order.getstatusname() if self.exit_order else None,
                    "exit_remaining": self.exit_order.executed.remsize if self.exit_order else None,
                    "exit_active": self.exit_order.alive() if self.exit_order else None,
                }
            )

        def stop(self) -> None:
            dates = {row[0]: index for index, row in enumerate(ROWS)}
            for name, order in (("entry", self.entry), ("exit", self.exit_order)):
                if order is not None:
                    for bit in order.executed.exbits:
                        fills.append(
                            {
                                "order": name,
                                "bar": dates[bt.num2date(bit.dt).date().isoformat()],
                                "size": bit.size,
                                "price": bit.price,
                            }
                        )

    with tempfile.TemporaryDirectory() as tmp:
        feed = Path(tmp) / "synthetic.csv"
        feed.write_text(path(), encoding="utf-8")
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.adddata(bt.feeds.GenericCSVData(dataname=str(feed), dtformat="%Y-%m-%d"))
        cerebro.addstrategy(Strategy)
        cerebro.broker.setcash(100.0)
        cerebro.broker.set_filler(bt.fillers.FixedSize())
        cerebro.broker.set_slippage_fixed(
            FIXED_SLIPPAGE, slip_open=True, slip_match=slip_match, slip_out=slip_out
        )
        cerebro.run()
    modules = ("backtrader.brokers.bbroker", "backtrader.order", "backtrader.fillers")
    source_sha256 = {}
    for name in modules:
        source_path = importlib.import_module(name).__file__
        if not source_path:
            raise ValueError(f"source provenance is unavailable for {name}")
        source_sha256[name] = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "external_engine_validated": False,
        "backtrader_version": VERSION,
        "source_sha256": source_sha256,
        "settings": {
            "fixed_slippage": FIXED_SLIPPAGE,
            "exit_valid_for_days": None if exit_no_expiry else EXIT_VALID_FOR_DAYS,
            "valuation_bases": ["broker_close", "external_mark"],
            "slip_match": slip_match,
            "slip_out": slip_out,
            "exit_type": "market" if exit_market else "limit",
            "exit_limit": None if exit_market else 9,
        },
        "bars": bars,
        "fills": fills,
    }


def main() -> None:
    """Emit the complete controls as JSON."""
    print(
        json.dumps(
            {
                name: observe(
                    slip_match=match, slip_out=outside, exit_market=market, exit_no_expiry=no_expiry
                )
                for name, match, outside, market, no_expiry in (
                    ("clamp", True, False, False, False),
                    ("allow_outside", True, True, False, False),
                    ("defer", False, False, False, False),
                    ("market_deadline_control", True, False, True, False),
                    ("untimed_limit_control", True, False, False, True),
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
