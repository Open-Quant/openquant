import math
from fractions import Fraction

import pytest

pytest.importorskip("backtrader", reason="optional pinned external-engine probe")
from experiments.backtrader_path_probe import observe


def test_partial_fills_charge_per_fill_not_per_order():
    result = observe(cash=100, volumes=(2, 3), prices=(10, 8), fixed=1, rate=0.001)
    assert result["cash"] == [78.98, 53.956]
    assert result["inventory"] == [2, 5]
    assert len(result["fills"]) == 2
    assert result["fills"][0]["size"] == 2
    assert result["fills"][1]["size"] == 3
    assert math.isclose(sum(x["commission"] for x in result["fills"]), 2.044)


def test_zero_capacity_does_not_charge_fee():
    result = observe(cash=100, volumes=(0, 2, 3), prices=(10, 10, 8), fixed=1, rate=0.001)
    assert result["cash"] == [100, 78.98, 53.956]
    assert result["inventory"] == [0, 2, 5]
    assert len(result["fills"]) == 2


def test_submission_precheck_differs_from_partial_fill_contract():
    result = observe(cash=25, volumes=(2, 3), prices=(10, 8), fixed=1, rate=0.001)
    assert result["cash"] == [25, 25]
    assert result["inventory"] == [0, 0]
    assert "Margin" in result["statuses"]
    assert not result["fills"]
    # First partial fill would be affordable under per-fill admission.
    assert Fraction(25) - 2 * 10 - 1 - Fraction(20, 1000) > 0


def test_fifty_exact_rational_partial_paths():
    import random

    rng = random.Random(1611)
    for _ in range(50):
        volumes = [rng.randint(0, 3) for _ in range(5)]
        prices = [rng.randint(2, 12) for _ in range(5)]
        fixed, rate = Fraction(rng.randint(0, 2)), Fraction(rng.randint(0, 5), 1000)
        result = observe(
            cash=1000, volumes=volumes, prices=prices, fixed=float(fixed), rate=float(rate)
        )
        cash, inventory, remaining = Fraction(1000), 0, 5
        for i, (v, p) in enumerate(zip(volumes, prices)):
            n = min(v, remaining)
            cash -= n * p + (fixed + n * p * rate if n else 0)
            inventory += n
            remaining -= n
            assert math.isclose(result["cash"][i], float(cash), rel_tol=1e-13, abs_tol=1e-12)
            assert result["inventory"][i] == inventory


def test_invalid_input_is_rejected():
    import pytest

    for field, value in [
        ("cash", float("nan")),
        ("rate", -1),
        ("fixed", float("inf")),
        ("volumes", [-1]),
        ("prices", [0]),
    ]:
        args = dict(cash=100, volumes=[2], prices=[10], fixed=1, rate=0.001)
        args[field] = value
        with pytest.raises(ValueError):
            observe(**args)
