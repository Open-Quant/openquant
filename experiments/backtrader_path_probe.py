"""Offline synthetic broker-path probe, not a full portable-fixture adapter."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import tempfile
from pathlib import Path

VERSION = "1.9.78.123"


def observe(*, cash, volumes, prices, fixed, rate):
    bt = importlib.import_module("backtrader")
    if bt.__version__ != VERSION:
        raise ValueError("unsupported backtrader version")
    if len(volumes) != len(prices) or not 1 <= len(volumes) <= 28:
        raise ValueError("1 to 28 aligned volumes and prices required")
    for value in [cash, fixed, rate, *volumes, *prices]:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("finite numeric inputs required")
    if (
        cash < 0
        or fixed < 0
        or rate < 0
        or any(v < 0 for v in volumes)
        or any(p <= 0 for p in prices)
    ):
        raise ValueError("invalid sign for cash, costs, volume or price")

    class PerFill(bt.CommInfoBase):
        params = (("stocklike", True), ("commtype", bt.CommInfoBase.COMM_PERC), ("percabs", True))

        def _getcommission(self, size, price, pseudoexec):
            return fixed + abs(size) * price * rate if size else 0.0

    result = {"cash": [], "inventory": [], "fills": [], "statuses": []}

    class Strategy(bt.Strategy):
        def __init__(self):
            self.seen = 0

        def next(self):
            if len(self) == 1:
                self.buy(size=5)
            else:
                result["cash"].append(self.broker.getcash())
                result["inventory"].append(self.position.size)

        def notify_order(self, order):
            result["statuses"].append(order.getstatusname())
            # Notifications share execution-bit history: consume only new bits.
            bits = list(order.executed.exbits)
            for bit in bits[self.seen :]:
                result["fills"].append(
                    {"size": bit.size, "price": bit.price, "commission": bit.comm}
                )
            self.seen = len(bits)

    with tempfile.TemporaryDirectory() as tmp:
        feed = Path(tmp) / "synthetic.csv"
        rows = ["date,open,high,low,close,volume,openinterest", "2026-01-01,10,10,10,10,0,0"]
        for i, (price, volume) in enumerate(zip(prices, volumes), 2):
            rows.append(f"2026-01-{i:02},{price},{price},{price},{price},{volume},0")
        feed.write_text("\n".join(rows) + "\n")
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.adddata(bt.feeds.GenericCSVData(dataname=str(feed), dtformat="%Y-%m-%d"))
        cerebro.addstrategy(Strategy)
        cerebro.broker.setcash(cash)
        cerebro.broker.addcommissioninfo(PerFill())
        cerebro.broker.set_filler(bt.fillers.FixedSize())
        cerebro.run()
    return result


def run():
    bt = importlib.import_module("backtrader")
    cases = {
        name: observe(cash=cash, volumes=volumes, prices=prices, fixed=1, rate=0.001)
        for name, cash, volumes, prices in [
            ("partial_paid", 100, (2, 3), (10, 8)),
            ("zero_then_partial", 100, (0, 2, 3), (10, 10, 8)),
            ("submission_margin", 25, (2, 3), (10, 8)),
        ]
    }
    return {
        "scope": "synthetic_external_broker_path_subset",
        "external_engine_validated": False,
        "version": bt.__version__,
        "broker_source_sha256": hashlib.sha256(
            Path(bt.brokers.bbroker.__file__).read_bytes()
        ).hexdigest(),
        "cases": cases,
        "limits": [
            "No full v2 adapter: bid/ask marking, terminal liquidation, short rejection and deadline still unqualified.",
            "Default whole-order cash precheck differs from per-fill cash admission. Do not change the fixture to hide this mismatch.",
            "No private data, account, network feed or investment-performance evidence.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
