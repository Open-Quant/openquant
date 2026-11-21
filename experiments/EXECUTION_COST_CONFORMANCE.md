# Portable execution-cost fixtures

An offline contract and five original hand-calculated cases, November 14, 2026.
The original v1 diagnostic implements one adapter: the experimental OpenQuant fill
ledger. The [November 15 continuation](#november-15-continuation-an-acceptance-gate-not-a-consensus-shortcut)
adds a fail-closed v2 gate, a second local rational interpreter, and a separate
external commission-only probe. None establishes full external-engine equivalence
or investment performance.

## Why a separate fixture format

[Yin et al.'s 2026 preprint](https://arxiv.org/html/2603.20319v1) motivates
cross-implementation comparison with costs held explicit. We have not reproduced
its empirical study. These original fixtures are a smaller engineering artifact:
they make a few execution conventions portable and testable, without copying
the paper's data, code or a book's exercises.

The public [Paleologo chapter-one sample](https://catalogimages.wiley.com/images/db/pdf/9781394265459.excerpt.pdf)
provides a useful before/during/after-trade framing. Only that public sample was
read. Neither the fixture schema nor its arithmetic is attributed to the book.

## Contract

`fixtures/execution_cost_conformance_v1.json` contains all inputs and expected
paths. One synthetic instrument, one currency (USD), long-only without borrowing:

- Timestamps are integer seconds after the declared UTC origin. A decision must
  precede its fill strictly; no implicit same-close execution.
- Proportional fee rate is a **fraction of absolute executed notional**:
  10 basis points = `1/1000`, not `10`, `0.1` or `0.00001`.
- A fixed fee is USD **per nonzero fill**, not per order, day or rebalance.
- Buys execute at the declared ask, sells at the declared bid. Marked equity uses
  the independent mark. No inferred slippage adjustment is added.
- Capacity is shared gross share volume and participation is one in these cases.
- Terminal liquidation uses a separate bid/volume and pays its own fee. Residual
  inventory remains visible and completion is false if it cannot all be sold.
- Zero execution has zero fee. Mark-only completion means the declared mark-only
  policy completed; it does not mean the inventory was liquidated.
- Numeric strings represent exact rational values. Decimal strings are exact
  decimals. Compare exact amounts for these fixtures, not float display strings.

All five cases start with cash 100 and no shares. One order requests five shares.
Regular buckets fill two at 10 then three at 8 when volume is available. Marks
are 10 and 9. A declared terminal bucket sells at 9.

The paid entry calculation is independently hand-checkable:
`100 - (2*10 + 1 + 20/1000) = 78.98`;
`78.98 - (3*8 + 1 + 24/1000) = 53.956`.
Full terminal sale yields `53.956 + 45 - 1 - 45/1000 = 97.911`.
Selling only two shares yields cash `70.938`, inventory 3 and marked equity
`97.938`, not terminal cash 97.938.

## Run and compare

```sh
python -m experiments.execution_cost_conformance
python -m unittest discover -s python/tests -p test_execution_cost_conformance.py
```

The report records fixture SHA-256 and observed paths. The adapter checks every
cash/inventory/equity/commission row, every fill quantity/price/commission and
both completion flags, not just final wealth. The comparator normalizes rational
spellings, rejects different field sets and does not silently reinterpret units.
The adapter refuses a changed version-one contract instead of pretending it
supports margin, another currency or a same-bar convention.

Four negative controls deliberately use wrong conventions: divide the rate by
100, charge one fixed fee per order, omit terminal charges, and erase residual
inventory. They demonstrate fixture sensitivity. They are **not newly discovered
bugs in OpenQuant or any other engine**.

## Original v1 next step: adding an external-engine adapter

Keep the fixture JSON unchanged. An independent adapter should translate the
declared event/fee conventions into its engine's settings, emit the same fields
and name the exact engine version/configuration. If the engine cannot express
a convention, return **unsupported**, not pass, and do not alter a fixture to
match the engine. Report all differing rows. Demonstrate the zero-cost control
and paid cases; the former alone does not qualify the latter.

Before calling this a conformance suite, add sell-order timing, opposing orders,
multiple currencies or explicit exclusions, cash exhaustion and tie-ordering
cases. The existing ledger tests cover some of those internally; this portable
version-one bank does not. Do not infer complete engine equivalence, model
validity, investment profitability or broker readiness from five synthetic cases.

The CLI is a diagnostic JSON report, not a CI exit-status gate. Tests assert the
case and negative-control results; consumers must inspect those fields rather
than infer success from process exit alone.

## November 15 continuation: an acceptance gate, not a consensus shortcut

The original five-case v1 fixture and diagnostic command are unchanged. Version
two puts a complete single-order scenario in each case and adds six original
cases: paid sells, decision/fill equal-time exclusion, inclusive order deadline,
cash exhaustion, short-inventory rejection, and sale fees exceeding available cash
plus proceeds. A reject case is an expected refusal, not an accepted fill. The
ledger rejects overdrawn fills rather than silently clipping affordable size.

Run the new, stricter gate:

    python -m experiments.execution_cost_acceptance
    python -m unittest discover -s python/tests -p 'test_execution_cost*.py'

Exit zero means all eleven declared fixtures matched both implementations.
Nonzero means failed, invalid or unsupported. A missing result is not acceptance.
The source-bound JSON includes exact row mismatches, the fixture digest and hashes
of the four implementation sources. These hashes identify bytes; they do not
authenticate them or prove a correct economic model. The runner is not a resource
sandbox for arbitrarily large untrusted inputs.

The new execution_cost_reference.py is a separately written rational interpreter
with no imports from the OpenQuant fill ledger. It reads only inputs, not expected
outputs. Both interpreters must independently match the complete hand-calculated
expected paths. A test changes expected cash while both interpreters still agree
with each other: the gate fails. Exact rational spellings remain interchangeable,
but missing fields/rows, duplicate JSON keys, unknown conventions and malformed
completion flags cannot pass. Expected rejections accept only a known reason or
that reason followed by the ledger's explicit space-for-space detail delimiter.

This is **two local implementations, not two independent external engines**.
They still share a declared contract and may share mistaken assumptions.
Four hundred seeded synthetic scenarios compare acceptance/rejection categories
and complete rational paths. They are differential testing, not a probability
of correctness or a market-performance experiment.

The earlier "adding a second adapter" section describes external-engine
qualification still to do; the new local reference does not satisfy that step.

### Bounded external probe

The optional dependency file pins backtrader==1.9.78.123. It is not added to the
package's runtime dependencies. Use an isolated research environment:

    python -m pip install -r experiments/requirements-cost-probe.txt
    python -m experiments.backtrader_commission_probe

This calls only CommInfoBase.getcommission with synthetic quantities, prices and
rates. No feeds, accounts, broker execution or strategy backtest are started.
The observed version and installed commission-source hash are recorded. A missing
or different package version returns unavailable/unsupported with a nonzero exit.
The optional external package is not redistributed in the review archive.

All 36 explicit fraction-rate probes matched exact-rational references to the
stated float tolerance. All 18 nonzero paid cases differed when the same fraction
was sent to CommInfoBase without percabs=True. A separate COMM_FIXED probe
charged 5 for five units at parameter 1, rather than the fixture's 1 per fill.
The [official Backtrader documentation](https://www.backtrader.com/docu/commission-schemes/commission-schemes/)
distinguishes percentage conventions and the legacy CommissionInfo default;
therefore this is a reproduced **configuration hazard**, not a claim that all
Backtrader entry points have the same default or that a new library bug was found.
Per-fill fixed charges require a deliberately qualified custom scheme.

This probe is not an external adapter for the full fixture bank. Order timing,
cash, partial fills, zero-volume attempts, inventory and terminal liquidation
remain unqualified in that engine. Keep external_engine_validated=false.

### Product choice and next experiment

The current portfolio already includes point-in-time, leakage, covariance,
risk-budget and execution labs. Another strategy-performance leaderboard would
not resolve the open execution-convention risk. Prefer a small adapter seam
over a new optimizer or a silent change to package defaults:

1. Keep this experimental gate outside production. Review same-source native
   package regression results separately from these stdlib-only tests.
2. Add a Backtrader full-path adapter only after explicitly deciding its
   cash-rejection, partial-fill and per-fill-charge mapping. Leave unsupported
   cases visible; never change the fixtures to make the adapter pass.
3. Acceptance: supported paid/zero-cost paths and expected rejections match,
   deliberate wrong-unit and same-time-fill mutants fail, source/configuration
   identity is retained, and unsupported cases prevent full qualification.
4. Extend the versioned contract for multiple opposing orders and shared
   capacity only after stating tie priority. Neither the single-order reference
   nor commission probe establishes those properties.
5. Agreement is not an investment-performance claim. The next stage remains
   synthetic and offline, without broker credentials or market-data cost.

The accessible [Wiley sample of Paleologo](https://catalogimages.wiley.com/images/db/pdf/9781394265459.excerpt.pdf)
reinforces interpreting methods in their trading context. It does not supply these
fixtures or acceptance rules, and this work does not claim to have read the full
book. The [2026 implementation-risk preprint](https://arxiv.org/html/2603.20319v1)
motivates checking cost conventions; its broad empirical findings have not been
reproduced here. Our evidence is the local fixtures and narrow pinned primitive
probe, not the paper's reported backtest results.
