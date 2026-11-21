# Backtrader sell, price basis, and expiry controls

Fixed, offline follow-on to the [admission capability screen](EXECUTION_CAPABILITY_PROFILE.md).
This is not a broker adapter, live or historical market data, parity evidence,
or a strategy performance calculation. The screen's classifications are unchanged.

## Reproduce

From the repository root in an isolated environment:

```sh
python -m pip install -r experiments/requirements-cost-probe.txt
python -m experiments.backtrader_sell_quote_probe
python -m pytest -q python/tests/test_backtrader_sell_quote_probe.py
```

The probe observes pinned Backtrader 1.9.78.123 with zero commissions, fixed
one-unit slippage applied at open, and the order-local FixedSize filler.
Cash starts at 100. On bar 0 it requests a two-share market buy. Only if
both shares have arrived by bar 1 does it submit a two-share sell; the
default sell limit is 9 and the validity timestamp is exactly bar 2's
timestamp. One control instead submits a market sell with the same timestamp;
another uses the same limit sell without a validity constraint. Bars use synthetic OHLC/volume. The independent valuation mark is
never a broker input. Fill bits are read once per order, distinct from the
final per-bar order state; no callback timestamp is inferred from fill bits.

| Zero-based bar | Open / high / low / close | Volume | External mark |
|---|---|---:|---:|
| 0 | 10 / 10 / 10 / 10 | 0 | 10 |
| 1 | 10 / 10 / 9 / 9 | 2 | 12 |
| 2 | 9 / 10 / 8 / 10 | 1 | 12 |
| 3 | 11 / 11 / 11 / 11 | 2 | 13 |

## What ran

| Control | Fill path | End state |
|---|---|---|
| Match and clamp (`slip_match=True`, `slip_out=False`) | +2 at 10 on bar 1; −1 at limit 9 on bar 2 | Sell remainder −1 expires at bar 3, inventory 1, cash 89. |
| Allow unclamped slippage (`slip_match=True`, `slip_out=True`) | +2 at 11 on bar 1 (above high 10); −1 at 8 on bar 2 (below limit 9) | Sell remainder −1 expires; inventory 1, cash 86. |
| No market match at slipped price (`slip_match=False`, `slip_out=False`) | No bar 1 market fill; +1 at 10 on bar 2; no exit submitted | Entry remains incomplete; inventory 1, cash 90. |
| Clamped market sell, same validity time as limit | +2 at 10 on bar 1; −1 at 8 on bar 2; −1 at 11 on bar 3 | Sell completes after stated validity; inventory 0, cash 99. |
| Clamped limit sell, no validity deadline | +2 at 10 on bar 1; −1 at 9 on bar 2; −1 at 10 on bar 3 | Same pre-deadline path as timed limit; sell completes, inventory 0, cash 99. |

Tests calculate the four primary paths' exact cash as initial cash minus signed fill
quantity times execution price, then independently calculate value at both
close and mark. E.g., on the allow-outside path, the first fill leaves cash
78 and inventory 2. Broker value is 96 at close 9; reference value is 102
at mark 12. Neither valuation creates a cash flow. This fixture uses small
exactly representable integers; passing does not qualify general monetary
precision. The source hashes identify installed files, not authenticated
evidence.

A synthetic quoted ask of 11 on bar 1 coincides with the unclamped
price, but that one match does not show quote support. The configuration
also violates the explicit sell-price lower bound in the second bar; the
low of 8 is inside that bar, so the problem there is the **order limit**,
not its OHLC range. The pinned favorable-open sell-limit branch calls
`_slip_down` with the limit as its bound; `slip_out=True` returns the
unclamped result. For expiry, pinned `Order.expire()` explicitly exempts
market orders and otherwise uses a strict greater-than time comparison.
This explains the observed order-type dependence. Generic statements about
Backtrader, other slippage/order configurations, or another version are not
established by these controls.

## Product decision and next qualification

Do not map portable bid/ask to fixed slippage, portfolio mark to broker
close, or an explicit deadline to Backtrader `valid` without order-type
qualification. In particular, do not propose the unclamped configuration
as a portable adapter merely because it matches one desired price.

1. An adapter proof should take bid, ask, and mark as independent as-of
   fields; include same OHLC with two distinct quote paths and the same
   execution path with two marks. Preserve order limit as a separate
   invariant and reject an out-of-bound fill rather than normalizing it.
2. Record separately: decision time, earliest eligible fill, inclusive
   deadline, last fill, terminal-event time/reason (if any), unexecuted
   remainder, and live inventory. Demand no fills after the deadline for
   both sides and each claimed order type. Add explicit liquidation with
   its own fill and capacity constraints; an expired sell is not a
   liquidation.
3. Revisit the portable screen only after a **single** configuration or
   explicitly scoped adapter passes both price and time checks together
   with existing cash, shared-liquidity and numeric controls. Do not pool
   distinct modes into one capability claim. Preserve trace receipt
   ingestion and real data/trading as separate work.

## Primary sources

- [Backtrader slippage configuration](https://www.backtrader.com/docu/slippage/slippage/)
  and [official broker code](https://github.com/mementum/backtrader/blob/master/backtrader/brokers/bbroker.py);
  the public branch is explanatory, while the tests exercise the pinned install.
- [Backtrader official order implementation](https://github.com/mementum/backtrader/blob/master/backtrader/order.py).
- [Publisher's public Paleologo chapter](https://catalogimages.wiley.com/images/db/pdf/9781394265459.excerpt.pdf)
  frames before/during/after-trade activities and motivates separating
  valuation, execution, and post-trade checks. No full-book access claimed.
- [Wang et al., Machine Learning Meets Markowitz](https://www.nber.org/papers/w34861)
  is a separate upstream portfolio-research lead. A synthetic follow-on could
  compare forecast-then-optimize with a decision-aware baseline under identical
  time splits, cost models and feasible sets, holding this execution contract
  fixed. No reported asset returns or empirical gains are reproduced here.
