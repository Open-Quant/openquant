"""Optional external *commission primitive* probe, never full engine validation.

Install only experiments/requirements-cost-probe.txt into a private environment.
No strategy execution, network feeds, account connection, or broker is used.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

VERSION = "1.9.78.123"
SOURCE = "https://www.backtrader.com/docu/commission-schemes/commission-schemes/"


def assess(explicit: str, default: str, expected: str) -> dict[str, bool]:
    """Compare binary64 external results with an exact-rational reference."""
    exact = float(Fraction(expected))
    good = math.isfinite(float(explicit)) and math.isclose(
        float(explicit), exact, rel_tol=1e-13, abs_tol=1e-15
    )
    mismatch = not math.isclose(float(default), exact, rel_tol=1e-13, abs_tol=1e-15)
    return {"explicit_units_passed": good, "default_scale_mismatch_detected": mismatch}


def run() -> dict[str, Any]:
    try:
        bt = importlib.import_module("backtrader")
    except ImportError:
        return {"status": "unavailable", "passed": False, "required_version": VERSION}
    if bt.__version__ != VERSION:
        return {"status": "unsupported", "passed": False, "version": bt.__version__}
    cases = []
    for size in (0, 2, -5, 0.5):
        for price in (8, 10, 123.45):
            for rate in ("0", "1/1000", "3/1000"):
                exact = abs(Fraction(str(size))) * Fraction(str(price)) * Fraction(rate)
                kwargs = dict(
                    commission=float(Fraction(rate)),
                    commtype=bt.CommInfoBase.COMM_PERC,
                    stocklike=True,
                )
                explicit = bt.CommInfoBase(**kwargs, percabs=True).getcommission(size, price)
                default = bt.CommInfoBase(**kwargs).getcommission(size, price)
                cases.append(
                    dict(
                        size=str(size),
                        price=str(price),
                        rate=rate,
                        expected=str(exact),
                        explicit=repr(explicit),
                        default=repr(default),
                        **assess(repr(explicit), repr(default), str(exact)),
                    )
                )
    # COMM_FIXED is per unit. It cannot substitute for USD per nonzero fill:
    fixed = bt.CommInfoBase(
        commission=1, commtype=bt.CommInfoBase.COMM_FIXED, stocklike=True
    ).getcommission(5, 10)
    nonzero = [c for c in cases if Fraction(c["expected"]) != 0]
    passed = (
        all(c["explicit_units_passed"] for c in cases)
        and all(c["default_scale_mismatch_detected"] for c in nonzero)
        and fixed == 5
    )
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "scope": "external_commission_primitive_only",
        "external_engine_validated": False,
        "version": bt.__version__,
        "comminfo_source_sha256": hashlib.sha256(
            Path(bt.comminfo.__file__).read_bytes()
        ).hexdigest(),
        "source": SOURCE,
        "cases": cases,
        "fixed_fee_probe": {
            "size": 5,
            "commission_parameter": 1,
            "observed": fixed,
            "fixture_per_fill_fee": 1,
            "mapping_status": "unsupported_without_custom_scheme",
        },
        "limits": [
            "No order scheduling, fill allocation, cash, inventory or terminal path tested.",
            "Default-unit mismatch is a configuration hazard, not a newly discovered library defect.",
            "Fraction-to-binary64 comparison uses rel_tol=1e-13 and abs_tol=1e-15.",
        ],
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 2)
