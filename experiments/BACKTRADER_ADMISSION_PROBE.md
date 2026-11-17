# Admission, shadow state and residual orders (November 17, 2026)

This is a bounded offline diagnostic, not a trading adapter or a qualified
backtester. It follows the [November 16 path probe](BACKTRADER_PATH_PROBE.md)
without changing the portable execution-cost fixtures. All reports retain
`external_engine_validated=false`.

## Run and inspect

Install the optional pinned `experiments/requirements-cost-probe.txt` into an
isolated environment, then run from the repository root:

```sh
python -m experiments.backtrader_admission_probe > admission-study.json
python -m pytest -q python/tests/test_backtrader_admission_probe.py
```

The study contains ten fixed controls, their observations and bookkeeping audits.
`observe(...)` also accepts small synthetic configurations: one asset, buy-only
integer shares, 2–28 aligned bars, 1–8 explicitly named orders, collapsed OHLC,
integer per-order volume limits and a fee of `fixed + abs(size) * price * rate`.
No interest, leverage, slippage, cheat-on-open/close, accounts or remote feeds are
used. Cash and other monetary inputs are bounded above by 1e12. Cash must be
positive and must not underflow the initial fund-share calculation `cash / 100`.
The zero-cash engine path divides by zero; rejecting it is this probe's input
contract, not a broker repair. Optional imports leave production dependencies
and package initialization unchanged.

## Reproduced controls

Unless stated otherwise, cash starts at 25; order sizes are large=5 and small=1;
prices are `[10, 10, 8, 1]`; volumes are `[0, 2, 3, 10]`; fees are 1 per fill plus
.001 of notional. Orders are submitted after bar 0. Bars below are zero-based.

| Control | Observed result |
|---|---|
| Default whole-order check, large only | Rejected before any fill. Shadow cash is −26.05, but actual cash remains 25 and inventory remains zero. |
| Large then small in the same submission pass | Both become `Margin`. The small order inherits shadow cash −26.05 and cloned position 5, then leaves shadow cash −37.06 and cloned position 6. Neither is an actual fill. |
| Small then large | Small completes, large is rejected; actual cash 13.99 and inventory 1. |
| Large first, small submitted a bar later | Small succeeds in a fresh pass and fills at bar 2's price 8, leaving cash 15.992 and inventory 1. This is a different fill-time control, not a same-price performance comparison. |
| Disable submission precheck | Large fills two at 10, leaving cash 3.98. Its next three-share attempt is unaffordable; `Margin` leaves residual 3 terminal and inactive. A later price of 1 does not revive it. |
| Stop the same run after its first partial fill | Residual 3 is `Partial` and active. Feed exhaustion is not cancellation. |
| Cash 51.5, constant price 10 until the last bar | Whole-order one-fee estimate admits; two actual fills require two fixed fees. After two shares, cash is 30.48 and the three-share attempt becomes terminal `Margin`. |
| Cash 52.1, same price path | Both fills complete, leaving approximately 0.05. |
| Cash exactly 51.05 | Exact decimal whole-order cost equals cash, but engine subtraction produces about −2.89e−15 and rejects admission. The discrepancy is retained, not hidden with an epsilon or changed fixture. |
| Two orders totaling three shares, bar volume two | Both complete, consuming three shares in one bar. `FixedSize` is an order-local cap, not a shared liquidity pool. A separate regression uses two plus two and observes four. |

An independent diagnostic also uses constant-price, zero-fee reverse
and separate-cycle controls. Those isolate submission order/cycle effects from
fees and changing prices. This describes the pinned Backtrader implementation
and this configuration; it makes no claim about real broker admission policies.

## Trace contract and audit boundary

The observer forwards to Backtrader 1.9.78.123 and records detached primitive
values. It does not copy or replace the execution implementation.

- `execution_attempts` separates `admission` pseudo calls from real `execution`.
  Pseudo cash and position are shadow state, not reservations or holdings.
- `broker_events` records emission bar, phase, order quantities, status and live
  account state. `notifications` separately records callback delivery bar.
- `fills` captures newly appended execution bits immediately after real execution.
  Delayed notifications may share an execution-bit history containing later
  fills; summing that history at each callback double-counts them. A `Margin`
  event can also replace `executed.dt`, so it is not a last-fill timestamp.
- `bars` records end-of-bar live cash, inventory and all submitted order states.
  Terminal unexecuted residual and active residual are different outcomes.
- `source_sha256` records broker, position, order, filler and commission source
  digests. These are provenance observations, not a trusted signature or a
  digest allowlist.

`audit_trace` checks field shapes, finite numeric types, order conservation,
actual fill debits, event transitions and live accounting, event/bar agreement,
execution-attempt completeness for eligible live orders, and callback delivery
timing in this bounded setup. It rejects missing zero-volume attempts, fabricated
post-terminal attempts, missing execution notifications, malformed provenance
and contradictory cash or residuals. Only a submission on the last bar may have
an undelivered callback here.

**It does not validate pseudo cash/position arithmetic**, authenticate a trace,
predict correct admission decisions, enforce market-wide liquidity, or prove
complete engine parity. Changing a finite pseudo cash observation alone can pass
this bookkeeping audit. That field remains evidence for source inspection and
the independent controls, not an audited financial account. A different engine
or configuration requires a new contract, not relaxed expectations.

Decimal-string rational fill debits are compared with binary64 values using
relative tolerance 1e-12 and absolute tolerance 1e-10. This can permit roughly a
one-dollar discrepancy at a cash scale of 1e12; it is **not universal cent-level
accuracy** or a money-rounding rule. Exact order quantities remain exact. The
decimal admission-boundary mismatch is intentionally tested separately.

## Evidence and next decision

The regression suite includes 200 prespecified seeded six-bar paths, the fixed
controls, invalid-input rejection and corrupted-trace checks. Independent review
compared 259 successful wrapped and uninstrumented broker runs exactly on bar
states, notifications and unique final fills. An additional zero-cash run failed
in the bare engine and now receives a clear adapter-boundary rejection. Review
found several audit omissions; targeted negative tests now cover those repairs.
These are synthetic correctness checks, not investment results or speed claims.

Next useful integration is an explicit, versioned admission/residual capability
profile for the portable acceptance gate. It must distinguish whole-order
precheck, per-fill rejection, terminal residual, feed-end active residual and
order-local versus shared volume. Do not translate `Margin` to retryable pending
or change fixtures to obtain agreement. Bid/ask/mark separation, sells/shorts,
deadlines, terminal liquidation and actual receipt-journal ingestion remain
unqualified. No production package, native source, docs site or workflow changed.

## Primary sources

- [Official broker configuration and cash-check documentation](https://www.backtrader.com/docu/broker/)
- [Official broker source, submission-check loop](https://github.com/mementum/backtrader/blob/master/backtrader/brokers/bbroker.py#L547-L572)
- [Official volume-filling documentation](https://www.backtrader.com/docu/filler/)
- [Wiley's legally public Paleologo chapter](https://catalogimages.wiley.com/images/db/pdf/9781394265459.excerpt.pdf)
- [Boyd et al., Multi-Period Trading via Convex Optimization](https://stanford.edu/~boyd/papers/cvx_portfolio.html)

Paleologo's public sample motivates attention to mechanisms and market
participants; Boyd et al. motivate explicit costs and constraints. Neither proves
engine parity or supplies this experiment. No full-book access or unverified
theorem constants are assumed. The GitHub master link is explanatory; actual
observations use the pinned installed version and recorded source hashes.
