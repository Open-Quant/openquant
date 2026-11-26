"""Exact comparison of two declared synthetic execution tapes; never a receipt adapter."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction as Q
from typing import TypedDict


@dataclass(frozen=True)
class Execution:
    order: str
    ordinal: int
    at: int
    shares: Q
    price: Q
    fee: Q

    def __post_init__(self) -> None:
        if not isinstance(self.order, str) or not self.order.strip():
            raise ValueError("order must be nonempty")
        if type(self.ordinal) is not int or self.ordinal < 1:
            raise ValueError("execution ordinal must be a positive integer")
        if type(self.at) is not int or self.at < 0:
            raise ValueError("execution time must be a nonnegative integer")
        if any(not isinstance(value, Q) for value in (self.shares, self.price, self.fee)):
            raise ValueError("shares, price and fee must be exact fractions")
        if not self.shares or self.price <= 0 or self.fee < 0:
            raise ValueError(
                "execution requires nonzero shares, positive price and nonnegative fee"
            )

    @property
    def key(self) -> tuple[str, int]:
        return self.order, self.ordinal


@dataclass(frozen=True)
class Receipt:
    sequence: int
    identity: str
    execution: Execution

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("source sequence must be a positive integer")
        if not isinstance(self.identity, str) or not self.identity.strip():
            raise ValueError("receipt identity must be nonempty")


Balance = tuple[str, str, str]
Key = tuple[str, int]


class Comparison(TypedDict):
    status: str
    account_equal: bool | None
    expected_account: Balance
    observed_account: Balance
    order_mismatches: list[str]
    order_account_differences: dict[str, tuple[Balance | None, Balance | None]]
    missing_executions: list[Key]
    unexpected_executions: list[Key]
    changed_executions: list[Key]
    through: int
    verified_prefix_through: int
    redeliveries: int
    buffered_receipts: int
    terminal_authority: bool
    external_engine_validated: bool


def _totals(executions: Iterable[Execution]) -> tuple[Balance, dict[str, Balance]]:
    # The three axes are signed executed shares, signed cash change and paid fees.
    total = [Q(0), Q(0), Q(0)]
    by_order: dict[str, list[Q]] = defaultdict(lambda: [Q(0), Q(0), Q(0)])
    for row in executions:
        order = by_order[row.order]
        for i, value in enumerate((row.shares, -row.shares * row.price - row.fee, row.fee)):
            total[i] += value
            order[i] += value

    def encode(values: list[Q]) -> Balance:
        return str(values[0]), str(values[1]), str(values[2])

    return encode(total), {order: encode(values) for order, values in by_order.items()}


def _indexed(executions: Iterable[Execution]) -> dict[Key, Execution]:
    rows: dict[Key, Execution] = {}
    for execution in executions:
        if execution.key in rows:
            raise ValueError("duplicate order-local execution ordinal")
        rows[execution.key] = execution
    return rows


def compare(
    expected: Iterable[Execution], receipts: Iterable[Receipt], *, through: int
) -> Comparison:
    """Assess supplied synthetic fills through a declared source sequence.

    Expected fills describe that same cutoff. Sequence is local to the declared
    source and starts at one. This function cannot establish source authenticity
    or whether the caller suppressed an unannounced tail.
    """
    if type(through) is not int or through < 0:
        raise ValueError("through must be a nonnegative integer source sequence")
    ref = _indexed(expected)
    identities: dict[str, Receipt] = {}
    sequences: dict[int, Receipt] = {}
    redeliveries = 0
    for receipt in receipts:
        if receipt.sequence > through:
            continue
        prior_id = identities.get(receipt.identity)
        prior_seq = sequences.get(receipt.sequence)
        if prior_id == receipt and prior_seq == receipt:
            redeliveries += 1
            continue
        if prior_id is not None or prior_seq is not None:
            raise ValueError("conflicting receipt identity or source sequence")
        identities[receipt.identity] = sequences[receipt.sequence] = receipt

    prefix: list[Execution] = []
    for sequence in sorted(sequences):
        if sequence != len(prefix) + 1:
            break
        prefix.append(sequences[sequence].execution)
    _indexed(receipt.execution for receipt in sequences.values())
    actual = _indexed(prefix)
    complete = len(prefix) == through
    expected_total, expected_orders = _totals(ref.values())
    actual_total, actual_orders = _totals(actual.values())
    missing = sorted(ref.keys() - actual.keys()) if complete else []
    unexpected = sorted(actual.keys() - ref.keys())
    changed = sorted(key for key in actual.keys() & ref.keys() if actual[key] != ref[key])
    order_differences = {
        order: (expected_orders.get(order), actual_orders.get(order))
        for order in sorted(expected_orders.keys() | actual_orders.keys())
        if complete and expected_orders.get(order) != actual_orders.get(order)
    }
    if unexpected or changed or missing:
        status = "mismatch"
    elif not complete:
        status = "incomplete_declared_prefix"
    else:
        status = "matched_declared_synthetic_tape"
    return {
        "status": status,
        "account_equal": expected_total == actual_total if complete else None,
        "expected_account": expected_total,
        "observed_account": actual_total,
        "order_mismatches": list(order_differences),
        "order_account_differences": order_differences,
        "missing_executions": missing,
        "unexpected_executions": unexpected,
        "changed_executions": changed,
        "through": through,
        "verified_prefix_through": len(prefix),
        "redeliveries": redeliveries,
        "buffered_receipts": len(sequences) - len(prefix),
        "terminal_authority": False,
        "external_engine_validated": False,
    }
