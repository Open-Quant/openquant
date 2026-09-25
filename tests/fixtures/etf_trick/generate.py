"""Reference series for the ETF trick and futures roll gaps, independent of this library.

    uv run --with pandas python tests/fixtures/etf_trick/generate.py

Writes reference.json next to this file. Imports neither openquant nor mlfinlab. Inputs are the
five BSD-3 CSVs in this directory (mlfinlab v0.8.0 test data; see tests/FIXTURES.md).

ETF trick, AFML section 2.4.1, implemented as written:

    K_t = K_{t-1} + sum_i h_{i,t-1} phi_{i,t} (delta_{i,t} + d_{i,t})
    h_{i,t} = omega_{i,t} K_t / (o_{i,t+1} phi_{i,t} sum_i |omega_{i,t}|)   if t in B, else h_{i,t-1}
    delta_{i,t} = p_{i,t} - o_{i,t} if t-1 in B, else p_{i,t} - p_{i,t-1}

with o = open_df, p = close_df, omega = alloc_df, d = costs_df, phi = rates_df (or 1 when no
rates are given), and B the bars whose allocation differs from the previous bar's (in this data
every bar). Two conventions follow the library rather than the book, because the book leaves
them open: the series starts at the second row with K = 1, and it stops one row before the end
(h at the last row would need the next open).

Futures roll gaps, AFML snippet 2.2 (rollGaps): at each roll date, gap = open of the new contract
minus the previous close; gaps are cumulated, and "roll backward" subtracts the last value so
the series ends at 0. The relative variant is the multiplicative analogue (open / previous close,
cumulative product, divided by the last value when rolling backward). The roll schedule is the
one crates/openquant/tests/futures_roll.rs builds from the spx column: contract 1 to 2017-03-20,
contract 2 to 2018-01-17, contract 3 after.
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent


def read(name):
    return pd.read_csv(HERE / name, index_col=0)


o, p, w, d = read("open_df.csv"), read("close_df.csv"), read("alloc_df.csv"), read("costs_df.csv")
rates = read("rates_df.csv")[o.columns]


def etf_trick(phi, start=1):
    rebalance = (w != w.shift()).any(axis=1)
    rebalance.iloc[0] = True
    k = {o.index[start]: 1.0}
    h = w.iloc[start] * 1.0 / (o.iloc[start + 1] * phi.iloc[start] * w.iloc[start].abs().sum())
    for t in range(start + 1, len(o) - 1):
        delta = (p.iloc[t] - o.iloc[t]) if rebalance.iloc[t - 1] else (p.iloc[t] - p.iloc[t - 1])
        k[o.index[t]] = k[o.index[t - 1]] + float((h * phi.iloc[t] * (delta + d.iloc[t])).sum())
        if rebalance.iloc[t]:
            h = w.iloc[t] * k[o.index[t]] / (o.iloc[t + 1] * phi.iloc[t] * w.iloc[t].abs().sum())
    return pd.Series(k)


with_rates = etf_trick(rates)
without_rates = etf_trick(rates * 0 + 1)

dates = pd.to_datetime(o.index)
contract = pd.Series(["futures_1"] * len(dates), index=o.index)
contract[dates > "2017-03-20"] = "futures_2"
contract[dates > "2018-01-17"] = "futures_3"
roll_dates = contract.drop_duplicates(keep="first").index
positions = [list(o.index).index(r) for r in roll_dates]
opens, closes = o["spx"], p["spx"]

abs_gaps = closes * 0.0
rel_gaps = closes * 0.0 + 1.0
for pos in positions[1:]:
    abs_gaps.iloc[pos] = opens.iloc[pos] - closes.iloc[pos - 1]
    rel_gaps.iloc[pos] = opens.iloc[pos] / closes.iloc[pos - 1]
abs_forward, rel_forward = abs_gaps.cumsum(), rel_gaps.cumprod()

out = {
    "source": "tests/fixtures/etf_trick/generate.py: AFML section 2.4.1 and snippet 2.2, pandas %s" % pd.__version__,
    "etf_trick": {
        "dates": list(with_rates.index),
        "with_rates": [float(x) for x in with_rates],
        "without_rates": [float(x) for x in without_rates],
    },
    "futures_roll": {
        "roll_dates": list(roll_dates),
        "absolute_forward": [float(x) for x in abs_forward],
        "absolute_backward": [float(x) for x in abs_forward - abs_forward.iloc[-1]],
        "relative_forward": [float(x) for x in rel_forward],
        "relative_backward": [float(x) for x in rel_forward / rel_forward.iloc[-1]],
    },
}
(HERE / "reference.json").write_text(json.dumps(out, indent=2) + "\n")
print("etf rows", len(with_rates), "K[20]", with_rates.iloc[20], without_rates.iloc[20])
print("roll dates", list(roll_dates), "abs last", abs_forward.iloc[-1], "rel last", rel_forward.iloc[-1])
