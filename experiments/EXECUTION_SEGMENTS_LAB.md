# Cash handoff between synthetic execution segments

This follow-up to [partial-fill accounting](FILL_LEDGER_LAB.md) implements a small
guard for transferring a completed cash balance. It is not a CPCV execution
adapter, a policy optimizer, a statistical test or a performance result.

## Runnable example

```bash
python -m experiments.run_offline_lab run \
  --lab execution_segments \
  --config experiments/configs/offline-smoke/execution_segments.json \
  --out /tmp/execution-segments-new-report.json
python -m experiments.replay_offline_report /tmp/execution-segments-new-report.json \
  --lab execution_segments
```

The full default grid uses boundary volumes 0, 1, 2 and 4. It retains all four
scenarios, including the two rejected handoffs. The smoke grid has three volumes.
Every first-segment ledger, rejection reason and accepted second-segment ledger
is included in the report. Amounts are exact rational strings, not floats.

## Why resetting a return series is not an execution model

In the fixed hand example, start with cash 100, buy two shares at 10, pay 1 and
mark at 11. Equity is 101. At the boundary, sell at bid 10.5 and pay another 1.
Cash is 99. A subsequent cash-only segment stays at 99 even if the share mark
rises to 12. Two tempting shortcuts answer different questions:

1. Chaining the first segment's marked equity to a cash-only segment without
   boundary execution produces 101. It omits the spread-to-mark difference and
   the terminal commission.
2. Carrying the first segment's shares forward, although the next segment was
   specified to start in cash, produces equity 103. This implements a different
   inventory policy, not the independent cash-only segment.

If terminal capacity is one share, cash is 88.5, residual inventory is one share
and marked equity is 99.5. Treating the last number as cash invents 11 units of
spendable currency. A valid handoff must fail rather than erase that share.
These are different declared accounting policies; the numbers are not evidence
that one strategy outperforms another.

## Guard contract

`experiments.execution_segments_lab.cash_handoff(result, at=..., mark=...)`
returns a new `InitialState` only if:

- Every decision finished, with no remaining signed obligation.
- The terminal policy was explicit liquidation and its completion is true.
- Terminal inventory is exactly zero.
- The last ledger row is terminal and its cash, inventory and equity reconcile
  exactly to the terminal summary.
- The next initial timestamp is aware UTC and no earlier than the terminal row.

It preserves cash, instrument and currency. It never replenishes the original
budget, sells stock implicitly or copies marked equity into cash. A new mark is
required for the new segment, but zero stock means that mark cannot change the
cash balance. Equal boundary timestamps are allowed as an explicit handoff;
the next segment still cannot fill at its decision timestamp.

This is intentionally narrower than a general reset policy. Cancelled/incomplete
decisions, stock transfer, new deposits, financing and non-cash boundaries require
separate contracts. The input must be genuine, unmodified `run_ledger` output.
The helper checks terminal consistency, not the complete fill/commission history
of a manually constructed or replaced `LedgerResult`. A caller can construct
mutually consistent false summaries; this is not an authenticity verifier.

## Before adapting actual CPCV paths

Implementation-ready next requirements:

1. Identify the chronological segments and prove each segment's signal was
   available before its first fill. Preserve split/model provenance separately
   from market quote provenance.
2. Choose a common capital/unit contract: sequential inherited cash or separately
   funded evaluations. Never silently rescale a fixed-share order tape to fit cash.
3. Declare each boundary's transition: retain inventory with explicit economic
   meaning, or execute a separate liquidation with its own timestamp and liquidity.
4. Keep rejected and incomplete boundaries in the result; do not condition a study
   on the subset that could liquidate. Report completion and economic obligations.
5. Validate the adapter against independent cash/inventory ledgers, with split
   permutations, repeated segments, missing quotes, partial exits, overlapping
   time intervals and heterogeneous initial capital as adversarial fixtures.

No adapter in this change composes arbitrary `run_cpcv` output. The existing API
consumes return arrays; that does not certify these state transitions or the
statistical validity of a stitched execution path. The chronology concerns are
consistent with the accessible author-linked backtesting talk cited in
[the fill-ledger guide](FILL_LEDGER_LAB.md#sources-and-attribution); the numerical
examples and guard are original, not a claimed book implementation.
