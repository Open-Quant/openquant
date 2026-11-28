# Order-keyed receipt conformance, offline research

The [quote evidence study](QUOTE_VISIBILITY_CONTRACT.md) distinguishes model
expiry from a terminal receipt. The [receipt journal](RECEIPT_REPLAY_STUDY.md)
requires a trusted, gap-free source and an explicit terminal completeness
assertion. This comparator stays in between: it compares **two declared synthetic
execution tapes** without sending any event to the journal. It cannot release
cash, declare cancellation, or qualify a broker.

## A weaker accounting check can pass incorrectly

Two fills, +2 and −2 shares at 10 with fee 1 each, produce zero net shares and
cash change −2. Swapping which order gets which fill leaves both account totals
unchanged and reverses the order-level balances. Even matching order balances
is insufficient: executing two one-share fills at 9 and 11 has the same totals
as at 10 and 10, but both fill prices differ. A shifted fill timestamp can also
leave the entire accounting projection unchanged. These are falsification
controls, not estimates of how often real reconciliation fails.

## Minimal runnable example

Run from the checkout root, with Python 3.11 or newer and no extra dependencies:

```python
from fractions import Fraction as Q

from experiments.receipt_order_conformance import Execution, Receipt, compare

buy = Execution("buy", 1, 4, Q(2), Q(10), Q(1))
sell = Execution("sell", 1, 4, Q(-2), Q(10), Q(1))
result = compare([buy, sell], [Receipt(1, "e1", buy), Receipt(2, "e2", sell)], through=2)
assert result["expected_account"] == ("0", "-2", "2")  # shares, cash change, fees
assert result["status"] == "matched_declared_synthetic_tape"
assert not result["terminal_authority"]

gap = compare([buy, sell], [Receipt(2, "e2", sell)], through=2)
assert gap["status"] == "incomplete_declared_prefix"
assert gap["buffered_receipts"] == 1
assert gap["account_equal"] is None
```

Reference fills must describe the same declared cutoff and use the same order
IDs and per-order execution ordinals. The observed sequence is a separate,
single-source, **fill-only** counter starting at one. Its identity is not
inferred from economic amounts, ordinal or timestamp. An exact redelivery counts
once; reusing an identity, sequence or execution key inconsistently fails the
comparison. Events beyond the declared cutoff do not enter validation or totals.
Terminal and correction events, currency conversions, and real source
normalization are deliberately not modeled. Neither side authenticates a
source or proves that a caller declared the true tail.

Only the contiguous observed prefix contributes signed shares, cash and fee
totals. A known fill contradiction can still produce a mismatch with a gap;
missing executions and account/order equality are not decided until the declared
prefix is complete. Economic totals are exact rational strings: for signed fill
size (q), price (p) and fee (f), cash moves by (-qp-f). Negative shares
represent a sale, not permission to short. This is a single-instrument,
single-currency comparison; balances don't include opening cash, marks or
nonexecution flows.

## Sources and next gate

- [Paleologo's public Wiley introduction](https://catalogimages.wiley.com/images/db/pdf/9781394265459.excerpt.pdf)
  distinguishes pretrade analysis from posttrade attribution. This is motivation,
  not a formula citation or claim of full-book access.
- [FIX Order State Changes](https://www.fixtrading.org/wp-content/uploads/download-manager-files/Order-State-Changes.pdf)
  distinguishes the purpose of an execution report from current order status.
  This is not a FIX adapter and uses no claims about FIX delivery guarantees.
- [Trading Volume Alpha](https://www.nber.org/papers/w33037) studies forecasting
  volume when trading has implementation costs. Only the author abstract was
  consulted; this synthetic replay has no volume model or alpha result. First
  establish per-order conservation before comparing shared-capacity policies.

Next: define independent order manifests and a reconciliation witness that
binds the declared cutoff to the input fixture, then test an intentionally
truncated tail and explicit cancel/replace or correction events. Keep the
existing journal unconnected until an adapter's source identity, stream
completeness, rollback rules and reservation authority are independently
specified and tested. No broker parity, performance, live execution or
investment-return conclusion.
