---
title: "etf_trick"
description: "Turn a rebalanced basket of futures, or a single rolled contract, into one continuous series whose changes are achievable PnL."
status: authored
last_authored: '2026-09-25'
audience:
  - quant-dev
  - platform-engineering
module: "etf_trick"
api_surface: "rust-only"
afml_chapter:
  - "2"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 2, §2.4.1 The ETF Trick; §2.4.3 Single Future Roll; Snippets 2.2 and 2.3."
rust_api:
  - "EtfTrick"
  - "EtfTrick::from_tables"
  - "EtfTrick::from_csv"
  - "EtfTrick::get_etf_series"
  - "get_futures_roll_series"
  - "FuturesRollRow"
  - "Table"
  - "EtfTrickError"
sidebar:
  badge: Module
---

A futures strategy rarely trades one price. It holds a basket whose weights change, in
contracts that expire and are replaced, sometimes quoted in another currency and paying or
costing carry along the way. Concatenating the raw prices manufactures a return at every
roll and every rebalance, and a model trained on that series learns the calendar.

AFML §2.4.1 replaces the basket with the value of one dollar invested in it — a synthetic
total-return ETF. The series has no gaps at rolls, reflects rebalancing at prices that were
actually available, and can be fed to bars, filters and labels like any single instrument.
§2.4.3 is the single-contract special case: remove the roll gap from one chain.

This module is Rust-only. There is no Python binding.

## The value series

For instruments $i$ with allocation $\omega_{i,t}$, open $o_{i,t}$, close $p_{i,t}$,
exchange rate $\varphi_{i,t}$ to the account currency and carry or dividend $d_{i,t}$, the
holdings and the value of the dollar $K_t$ are

$$
\begin{aligned}
h_{i,t} &= \frac{\omega_{i,t}\,K_t}{o_{i,t+1}\,\varphi_{i,t}\sum_j \lvert\omega_{j,t}\rvert}
  && \text{at a rebalance; otherwise } h_{i,t}=h_{i,t-1} \\[4pt]
\delta_{i,t} &= \begin{cases} p_{i,t}-o_{i,t} & \text{the bar after a rebalance} \\ p_{i,t}-p_{i,t-1} & \text{otherwise} \end{cases} \\[4pt]
K_t &= K_{t-1} + \sum_i h_{i,t-1}\,\varphi_{i,t}\,\bigl(\delta_{i,t} + d_{i,t}\bigr)
\end{aligned}
$$

Three things in those lines do the work. Dividing by $\sum_j\lvert\omega_j\rvert$ de-levers
the allocation, so weights of $(2,-1)$ and $(\tfrac23,-\tfrac13)$ give the same series.
Holdings are sized at the *next* open, $o_{i,t+1}$, because a rebalance decided on bar $t$
cannot trade before then. And the bar after a rebalance earns only open-to-close, since the
new position did not exist overnight.

Note the indices in the last line: bar $t$'s move is earned by $h_{i,t-1}$, which was set from
bar $t-1$'s allocation $\omega_{i,t-1}$, value $K_{t-1}$ and exchange rate, and bought at bar
$t$'s open $o_{i,t}$. Nothing from bar $t$ other than its open enters the size of the position
that earns bar $t$.

`EtfTrick` takes five tables with identical row index and columns — `open`, `close`, `alloc`,
`costs` and optionally `rates` — and `get_etf_series` returns `(index, K)` pairs starting
from $K=1$.

```rust
use openquant::etf_trick::{EtfTrick, Table};

fn table(columns: &[&str], rows: &[(&str, [f64; 2])]) -> Table {
    Table {
        index: rows.iter().map(|(day, _)| day.to_string()).collect(),
        columns: columns.iter().map(|c| c.to_string()).collect(),
        values: rows.iter().map(|(_, v)| v.to_vec()).collect(),
    }
}

// Two futures, six sessions. The basket is 50/50 until the fourth session, then 80/20.
let cols = ["CL", "NG"];
let days = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"];
let open = [[70.0, 2.50], [70.5, 2.52], [71.4, 2.49], [71.0, 2.55], [72.2, 2.60], [72.0, 2.58]];
let close = [[70.4, 2.51], [71.2, 2.50], [71.1, 2.54], [72.0, 2.61], [72.1, 2.57], [72.6, 2.59]];
let alloc = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.8, 0.2], [0.8, 0.2], [0.8, 0.2]];
let zeros = [[0.0, 0.0]; 6];

let rows = |v: &[[f64; 2]; 6]| days.iter().copied().zip(v.iter().copied()).collect::<Vec<_>>();
let etf = EtfTrick::from_tables(
    table(&cols, &rows(&open)),
    table(&cols, &rows(&close)),
    table(&cols, &rows(&alloc)),
    table(&cols, &rows(&zeros)), // carry / dividends, in price units; none here
    None,                        // no FX: every contract is quoted in the account currency
)?;
for (day, value) in etf.get_etf_series(100)? {
    println!("{day}  K = {value:.6}");
}
```

```text
2024-01-03  K = 1.000000
2024-01-04  K = 1.007939
2024-01-05  K = 1.028298
2024-01-08  K = 1.024786
```

The second value can be checked by hand. On 01-03 the dollar is split 50/50 and sized at the
next opens, so $h = (0.5/71.4,\; 0.5/2.49)$. The position is bought at the 01-04 opens, so
01-04 earns open-to-close: crude closed 0.30 below its open and gas 0.05 above:
$K = 1 + \tfrac{0.5}{71.4}(-0.30) + \tfrac{0.5}{2.49}(0.05) = 1.007939$. No rebalance on
01-04, so 01-05 earns close-to-close on the same holdings; the 80/20 allocation of 01-05 is
bought at the 01-08 opens and first earns on 01-08.

Six input rows give four values. The series starts on the second row with $K=1$, which
counts as a rebalance (the first row is not used), and the last row is dropped because
sizing a position there would need an open that has not happened. The row index matches
mlfinlab's output. The values do not: mlfinlab, and earlier versions of this crate, sized
the holdings that earn bar $t$ from bar $t$'s allocation and bar $t+1$'s open, one bar
later than §2.4.1.

## Rolling one contract

`get_futures_roll_series` computes the cumulative roll gap for a single chain. Each row says
which contract it quotes (`security`) and which contract is front on that date
(`current_security`); rows where the two differ are ignored, so you can pass the whole
quote table. At each change of front contract the gap is the new contract's open minus the
old one's previous close, and the gaps are accumulated.

```rust
use chrono::NaiveDate;
use openquant::etf_trick::{get_futures_roll_series, FuturesRollRow};

// CLG4 until the roll on 01-05, CLH4 after. The new contract opens 0.90 above the old
// one's last close, and that gap is not a return.
let d = |day| NaiveDate::from_ymd_opt(2024, 1, day).unwrap();
let chain: Vec<FuturesRollRow> = [(2, 70.0, 70.4, "CLG4"), (3, 70.5, 71.2, "CLG4"), (4, 71.4, 71.1, "CLG4"),
    (5, 72.0, 72.9, "CLH4"), (8, 73.1, 73.0, "CLH4")]
    .into_iter()
    .map(|(day, open, close, contract)| FuturesRollRow {
        date: d(day),
        open,
        close,
        security: contract.to_string(),
        current_security: contract.to_string(),
    })
    .collect();

let gaps = get_futures_roll_series(&chain, "absolute", true)?;
for (row, gap) in chain.iter().zip(&gaps) {
    println!("{}  {}  close {:.2}  gap {:+.2}  rolled {:.2}", row.date, row.security, row.close, gap, row.close - gap);
}
```

```text
2024-01-02  CLG4  close 70.40  gap -0.90  rolled 71.30
2024-01-03  CLG4  close 71.20  gap -0.90  rolled 72.10
2024-01-04  CLG4  close 71.10  gap -0.90  rolled 72.00
2024-01-05  CLH4  close 72.90  gap +0.00  rolled 72.90
2024-01-08  CLH4  close 73.00  gap +0.00  rolled 73.00
```

Subtract the gap series from the raw prices to get the rolled series. With
`roll_backward = true` the most recent prices are left untouched and history is shifted
(Snippet 2.2), which keeps the series aligned with the live market; with `false`, history is
fixed and recent prices move. `"absolute"` accumulates price differences; `"relative"`
accumulates ratios, returned as a factor to divide by.

An absolute backward roll can push old prices negative after years of contango. AFML's
remedy (Snippet 2.3) is to work with returns of the rolled series against the *raw* previous
price and compound them into a price-of-one-dollar series; that step is not implemented
here, and `"relative"` is the simpler way to stay positive.

Both examples on this page are the program in `crates/openquant/examples/docs_etf_trick.rs`;
the output shown is what it prints.

## What to watch for

- **`costs` is added, not subtracted.** The table holds $d_{i,t}$ — carry, dividends, coupons
  — in price units, with the sign of a credit. To charge a cost, pass it negative. The name
  comes from mlfinlab and is misleading.
- **Transaction costs are not modelled.** AFML §2.4.1 also tracks rebalancing cost
  $c_t=\sum_i (\lvert h_{i,t-1}\rvert p_{i,t} + \lvert h_{i,t}\rvert o_{i,t+1})\tau_i$ and
  bid-ask cost; neither is computed. $K_t$ is gross.
- **A rebalance is detected by comparing allocation rows for exact equality.** Weights that
  drift in the last decimal place from a floating-point calculation register as a rebalance
  on every bar, and every bar then earns only open-to-close. Round allocations before passing
  them in.
- **`from_csv` reads every file in full.** The `batch_size` argument of `get_etf_series`
  exists for mlfinlab compatibility: it must be at least 3 for the CSV source and does not
  bound memory.
- **The roll calendar is yours.** `current_security` decides when the roll happens. Verify it
  against the exchange's last-trade dates rather than inferring it from volume.

## Related modules

- [`data-structures`](/modules/data-structures/) — build bars on the value series.
- [`backtesting-engine`](/modules/backtesting-engine/) and
  [`backtest-statistics`](/modules/backtest-statistics/) — evaluate strategies on it.
