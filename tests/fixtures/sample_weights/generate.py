"""Reference for sample weights: the AFML snippets, run in pandas, independent of this library.

    uv run --with pandas python tests/fixtures/sample_weights/generate.py

Pipeline (the same one mlfinlab's test_sample_weights sets up):
  daily vol (3.1) -> CUSUM filter on log prices (2.4) -> 2-day vertical barrier (3.4)
  -> events with pt/sl = 4x target, min_ret 0.005, side = 1 (3.3 / 3.6)
  -> concurrency (4.1) -> weights by return attribution (4.10) -> time decay (4.11).

mlfinlab's own test compares its results with a tolerance of 1e5, so the numbers quoted
there (0.781807, 1.627944, 0.582191) were never checked by anything. This script is the check.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
close = pd.read_csv(HERE.parent / "filters" / "dollar_bar_sample.csv", index_col=0, parse_dates=[0])["close"]


def daily_vol(close, lookback=100):
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    return df0.ewm(span=lookback).std()


def cusum_filter(close, threshold):
    events, s_pos, s_neg = [], 0.0, 0.0
    diff = np.log(close).diff().dropna()
    for t, d in diff.items():
        s_pos, s_neg = max(0.0, s_pos + d), min(0.0, s_neg + d)
        if s_neg < -threshold:
            s_neg = 0.0
            events.append(t)
        elif s_pos > threshold:
            s_pos = 0.0
            events.append(t)
    return pd.DatetimeIndex(events)


def vertical_barrier(t_events, close, days):
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=days))
    t1 = t1[t1 < close.shape[0]]
    return pd.Series(close.index[t1], index=t_events[: t1.shape[0]])


def get_events(close, t_events, pt_sl, target, min_ret, vertical, side):
    target = target.reindex(t_events)
    target = target[target > min_ret]
    events = pd.concat({"t1": vertical, "trgt": target, "side": side.reindex(target.index)}, axis=1).dropna(subset=["trgt"])
    first = {}
    for loc, vert in events["t1"].fillna(close.index[-1]).items():
        path = (close[loc:vert] / close[loc] - 1) * events.at[loc, "side"]
        pt, sl = pt_sl[0] * events.at[loc, "trgt"], -pt_sl[1] * events.at[loc, "trgt"]
        touches = [vert if pd.notna(events.at[loc, "t1"]) else pd.NaT,
                   path[path > pt].index.min(), path[path < sl].index.min()]
        first[loc] = pd.Series(touches).dropna().min() if any(pd.notna(t) for t in touches) else pd.NaT
    events["t1"] = pd.Series(first)
    return events


def num_concurrent(close_index, t1):
    count = pd.Series(0, index=close_index[close_index.searchsorted(t1.index[0]): close_index.searchsorted(t1.max()) + 1])
    for t_in, t_out in t1.items():
        count.loc[t_in:t_out] += 1
    return count


def weights_by_return(t1, close):
    conc = num_concurrent(close.index, t1)
    ret = np.log(close).diff()
    w = pd.Series({t_in: (ret.loc[t_in:t_out] / conc.loc[t_in:t_out]).sum() for t_in, t_out in t1.items()}).abs()
    return w * w.shape[0] / w.sum()


def weights_by_time_decay(t1, close, decay):
    conc = num_concurrent(close.index, t1)
    uniq = pd.Series({t_in: (1.0 / conc.loc[t_in:t_out]).mean() for t_in, t_out in t1.items()})
    cum = uniq.sort_index().cumsum()
    slope = (1.0 - decay) / cum.iloc[-1] if decay >= 0 else 1.0 / ((decay + 1) * cum.iloc[-1])
    out = (1.0 - slope * cum.iloc[-1]) + slope * cum
    out[out < 0] = 0.0
    return out


vol = daily_vol(close)
t_events = cusum_filter(close, 0.02)
events = get_events(close, t_events, [4, 4], vol, 0.005, vertical_barrier(t_events, close, 2), pd.Series(1.0, index=close.index))
t1 = events["t1"].dropna()
fmt = "%Y-%m-%d %H:%M:%S.%f"
out = {
    "source": "AFML snippets 2.4, 3.1, 3.3, 3.4, 3.6, 4.1, 4.10, 4.11 in pandas %s" % pd.__version__,
    "n_cusum_events": int(len(t_events)),
    "events": [{"t0": t0.strftime(fmt), "t1": t.strftime(fmt), "trgt": float(events.at[t0, "trgt"])} for t0, t in t1.items()],
    "weights_by_return": [float(x) for x in weights_by_return(t1, close)],
    "time_decay": {str(d): [float(x) for x in weights_by_time_decay(t1, close, d)] for d in (0.5, 1.0, -0.5, 0.0, 1.5)},
}
(HERE / "reference.json").write_text(json.dumps(out, indent=2) + "\n")
print("cusum events:", out["n_cusum_events"], "| events with t1:", len(out["events"]))
print("weights_by_return:", [round(x, 6) for x in out["weights_by_return"]])
print("time_decay 0.5:   ", [round(x, 6) for x in out["time_decay"]["0.5"]])
