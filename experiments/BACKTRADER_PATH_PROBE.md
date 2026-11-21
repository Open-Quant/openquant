# External broker-path probe (November 16, 2026)

This continues the November 15 execution-cost acceptance work, not a new strategy.
It uses the optional pinned `backtrader==1.9.78.123` in an isolated environment,
synthetic local CSV bars, and no accounts, external feeds or paid services.

## What changed

`python -m experiments.backtrader_path_probe` now executes a real Backtrader
broker/strategy path rather than calling its commission primitive alone. A custom
commission scheme charges a fixed dollar amount per nonzero fill plus a fraction
of executed notional; `FixedSize` constrains each fill by the current bar's volume.
The one buy order is submitted on the first bar and may fill only later. Quotes
are collapsed OHLC values, not a qualified bid/ask/mark adapter. The diagnostic
prints observations, not a pass receipt; use the tests for acceptance.

Independent rational tests cover 50 seeded five-bar paths with enough cash, plus
three hand-checkable cases and invalid-input rejection. Paid partial path:
100 cash -> 78.98 after buying two at 10 -> 53.956 after buying three at 8.
The fee is 1 per fill plus .001 of notional. An inserted zero-volume bar leaves
cash and inventory unchanged and produces no fee or fill.

## Useful negative result

With starting cash 25, the default submission cash check refuses the whole
five-share order. No partial fill happens, even though the first two-share fill
would cost 21.02 and be affordable. The broker's `check_submitted` pseudo-executes
the order before accepting it. This differs from the portable fixture contract's
per-fill admission. It is a configuration/semantic mismatch, not a newly found
Backtrader bug. Disabling this check is not automatically equivalent: later
insufficient-cash behavior and residual-order lifecycle would still need tests.

## Decision and next bounded experiment

Keep `external_engine_validated=false`. Do not shoehorn the fixture into the
engine's behavior or report agreement from terminal cash alone. Next adapter
qualification must explicitly map whole-order admission versus per-fill rejection,
record refusal time and residual order state, then add bid/ask/mark separation,
sells, deadline handling and terminal liquidation. A schema that says only
“cash cannot be negative” is insufficient to identify admission semantics.
Acceptance requires every supported complete path to match, unsupported cases to
block qualification, and wrong-unit/same-bar/ignored-volume controls to fail.
No optimization, live execution, or portfolio-return conclusion follows here.

## Sources and limits

- [Official broker documentation](https://www.backtrader.com/docu/broker/)
- [Official volume-filling documentation](https://www.backtrader.com/docu/filler/)
- [Official commission schemes](https://www.backtrader.com/docu/commission-schemes/commission-schemes/)
- [Wiley's public Paleologo sample](https://catalogimages.wiley.com/images/db/pdf/9781394265459.excerpt.pdf)

The public chapter provides context about trading mechanisms and liquidity; it
does not supply this experiment. No full-book access assumed. Findings come from
the pinned installed code and synthetic checks, not general empirical market data.
Binary64 external values use stated numerical tolerances against exact rational
references. No native OpenQuant package build, full suite or notebook rerun in
this session. No production modules, release workflow or docs-site files edited.
