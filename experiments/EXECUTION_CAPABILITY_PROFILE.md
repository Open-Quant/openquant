# Execution-capability preflight

Research continuation, November 18, 2026. Owner: OpenQuant maintainers.
Source baseline: draft PR [#290](https://github.com/Open-Quant/openquant/pull/290),
commit `02c77035741d421af6e33dd61e309e2be8e51c1c`.

## Product decision

Do not treat Backtrader as a drop-in verifier of the
[portable execution fixtures](EXECUTION_COST_CONFORMANCE.md). First compare
explicit execution conventions, then compare every path under a qualified
adapter. A configuration that cannot represent a requirement is blocked, not
silently approximated, skipped or "fixed" by changing the reference fixture.

This experiment implements the next step from the
[admission study](BACKTRADER_ADMISSION_PROBE.md): a versioned, evidence-bearing
preflight. It does not add an external adapter to the acceptance gate and does
not change any fixture, production API, native function, dependency or workflow.

## What runs

From the repository root with the optional pinned
`experiments/requirements-cost-probe.txt` installed:

```sh
python -m experiments.execution_capability_profile --checksubmit on
python -m experiments.execution_capability_profile --checksubmit off
python -m experiments.execution_capability_profile --requirements observed-subset
python -m pytest -q python/tests/test_execution_capability_profile.py
```

The first two commands emit JSON and exit **2**: the portable-v2 screen is blocked.
The third exits **0**, meaning only **eligible for a subsequent case comparison**.
It does not run that comparison and is not an acceptance result. Reports always
set `external_engine_validated=false` and `case_comparison_executed=false`;
there is deliberately no `passed` field. Redirect stdout to a new file if desired.
Normal shell redirection can overwrite files; the command does not manage storage.

`characterize(checksubmit=True|False)` reruns six fixed controls on Backtrader
1.9.78.123. Each profile covers only one configuration: one asset, integer buy
orders, collapsed OHLC, `FixedSize` per-order filling, fixed-plus-proportional
commission, no leverage/interest/slippage or cheat mode. It does not combine
observations from `checksubmit=True` and `False` into one supposed capability.

| Dimension | Profile evidence | Portable-v2 screen |
|---|---|---|
| Submission precheck | Enabled rejects an unaffordable whole order before execution; disabled emits execution refusal instead | Requires no submission precheck |
| Execution cash refusal | Residual 3 becomes terminal `Margin`, including with the precheck enabled when repeated fixed fees exhaust cash | Expects an overdraw error, not a terminal broker order |
| Feed exhaustion | Residual 3 stays active after the first partial fill | Not a requirement here: the portable output does not expose order liveness |
| Capacity | Two orders fill four shares in a bar with volume two | Requires shared volume, so blocked |
| Fixed commission | Five shares across two fills pay two fixed charges | Per-fill basis matches this dimension only |
| Arithmetic | Pinned engine uses binary64; the decimal boundary control is retained | Requires exact rational arithmetic, so blocked |
| Separate bid/ask/mark, sells, deadlines, terminal liquidation | Not exercised by this probe | Unqualified, not supported |

Enabled-precheck output has four mismatches (admission, refusal, volume,
arithmetic) and four unqualified dimensions. Disabling precheck removes only
the admission mismatch. It does not repair liquidity, fees, arithmetic or
terminal order semantics. `Margin` must not be normalized to retryable pending.

The six controls are: cash refusal, execution refusal after a partial fill,
active residual at feed end, a completed two-fill order, the shared-volume
counterexample, and the exact-decimal cash boundary. The last retains rejection at the exact-decimal boundary, not an epsilon-adjusted
acceptance condition. The profile asserts fixed outcomes for all six and refuses
unexpected behavior. This is
regression evidence within these controls, not a universal engine theorem.

## Data contract and limits

`openquant_execution_capabilities_v1` contains version/configuration, capability
values, all six raw traces, their SHA-256 digests, installed-source digests and
lifecycle summaries. `assess(profile, requirements)` reruns bookkeeping audits,
checks the fixed configurations and expectations, rejects mixed source digests,
rederives all metadata, and compares its canonical JSON digest. This catches
accidental metadata edits and integer-for-boolean substitution. Dictionary order
is irrelevant; numeric types and values are not silently coerced.

Requirements are a nonempty, explicit map of allowlisted dimensions and meanings.
Unknown keys/values and malformed reports raise `ValueError`. A requirement cannot
request `unknown` or `unqualified` to manufacture a match. Missing dimensions are
listed in `omitted_dimensions`; matching only the fee dimension is not full
contract compatibility. `PORTABLE_V2` is a manually reviewed requirements screen
for the bundled v2 fixture, **not its complete executable specification**. Its
values must be revisited if the fixture/adapter semantics change.
`assess_portable` and the default CLI bind this map to the exact reviewed fixture
SHA-256 (`2f34d8b…7287a440`). Any byte change refuses the named screen until an
explicit review updates the mapping and pin; no fixture content is rewritten.
The lower-level `assess` accepts an explicit caller-owned map and makes no
assertion that such a map came from the portable fixture.

Source hashes and self-consistency are not authentication. A malicious producer
could fabricate consistent traces, and untrusted oversized reports are not a
safe input boundary. Run the pinned observer locally on controlled synthetic
inputs. The inherited audit intentionally does not validate admission shadow
arithmetic. Its relative/absolute tolerance is not a cent-level precision rule.
The profile records the distinction rather than upgrading that audit to proof.

## Useful next slice

Before any real integration, select an explicit adapter contract rather than
translating away mismatches. The [sell, price basis, and expiry probe](BACKTRADER_SELL_QUOTE_PROBE.md)
is a follow-on diagnostic with a matched-time limit/market counterexample. It
does not change any capability classification. Implement true separate
bid/ask/mark controls and retain exact reference paths and refusal outcomes. Only then
consider a **separate versioned approximate-comparison policy** with explicit
money/quantity tolerances and magnitude bounds. Do not apply a global epsilon
or use the existing broad bookkeeping tolerance as a trading-money guarantee.
Shared capacity and terminal-liquidation conventions need independent controls
and adapter work, not a new boolean flag. Keep this preflight independent of
actual receipt-journal ingestion and live trading permissions.

## Sources and attribution

- [Official Backtrader broker configuration](https://www.backtrader.com/docu/broker/)
  documents submission checks and execution settings. Installed pinned source
  and local raw traces, not the latest web documentation alone, support the
  specific observations.
- [Paleologo's legally public Wiley chapter sample](https://catalogimages.wiley.com/images/db/pdf/9781394265459.excerpt.pdf)
  motivates examining market mechanisms before choosing quantitative tools.
  No full-book access, copied exercise or new book-derived result is claimed.
- [Boyd et al., Multi-Period Trading via Convex Optimization](https://stanford.edu/~boyd/papers/cvx_portfolio.html)
  separates costs and constraints in portfolio optimization. It is context for
  explicit model assumptions, not evidence of Backtrader parity or performance.

All observations are synthetic. No strategy returns, calibrated market model,
statistical discovery, speed improvement or real brokerage readiness is claimed.
