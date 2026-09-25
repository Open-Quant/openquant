"""Reference triple-barrier events and labels, independent of this library.

    uv run --with pandas python tests/fixtures/labeling/generate.py

Writes reference.json next to this file. Imports neither openquant nor mlfinlab.

Pipeline, AFML chapter 3 on tests/fixtures/filters/dollar_bar_sample.csv (the setup of
crates/openquant/tests/labeling.rs):
  daily volatility, span 100 (snippet 3.1) -> CUSUM filter on log prices, h = 0.02 (snippet 2.4)
  -> vertical barrier one day after each event (snippet 3.4) -> events with pt = sl = 1 x target,
  min_ret 0.005, no side (snippets 3.2, 3.3) -> labels (snippet 3.5: ret = p[t1] / p[t0] - 1,
  bin = sign(ret), times the side when a side is given; meta-labels are 1 if ret > 0 else 0).
The events are computed with and without the vertical barrier, and with side = 1 (meta-labelling).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
close = pd.read_csv(HERE.parent / "filters" / "dollar_bar_sample.csv", index_col=0, parse_dates=[0])["close"]


def daily_vol(close, lookback=100):  # snippet 3.1
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    return df0.ewm(span=lookback).std()


def cusum_filter(close, h):  # snippet 2.4 on log prices
    events, s_pos, s_neg = [], 0.0, 0.0
    diff = np.log(close).diff().dropna()
    for t, d in diff.items():
        s_pos, s_neg = max(0.0, s_pos + d), min(0.0, s_neg + d)
        if s_neg < -h:
            s_neg = 0.0
            events.append(t)
        elif s_pos > h:
            s_pos = 0.0
            events.append(t)
    return pd.DatetimeIndex(events)


def vertical_barrier(t_events, close, days):  # snippet 3.4
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=days))
    t1 = t1[t1 < close.shape[0]]
    return pd.Series(close.index[t1], index=t_events[: t1.shape[0]])


def get_events(close, t_events, pt_sl, target, min_ret, vertical, side):  # snippets 3.2, 3.3
    target = target.reindex(t_events)
    target = target[target > min_ret]
    if vertical is None:
        vertical = pd.Series(pd.NaT, index=t_events)
    side_ = pd.Series(1.0, index=target.index) if side is None else side.reindex(target.index)
    events = pd.concat({"t1": vertical, "trgt": target, "side": side_}, axis=1).dropna(subset=["trgt"])
    first = {}
    for loc, vert in events["t1"].fillna(close.index[-1]).items():
        path = (close[loc:vert] / close[loc] - 1) * events.at[loc, "side"]
        pt = pt_sl[0] * events.at[loc, "trgt"] if pt_sl[0] > 0 else np.inf
        sl = -pt_sl[1] * events.at[loc, "trgt"] if pt_sl[1] > 0 else -np.inf
        touches = [events.at[loc, "t1"], path[path > pt].index.min(), path[path < sl].index.min()]
        touches = [t for t in touches if pd.notna(t)]
        first[loc] = min(touches) if touches else pd.NaT
    events["t1"] = pd.Series(first)
    if side is None:
        events = events.drop(columns="side")
    return events


def get_bins(events, close):  # snippet 3.5
    ev = events.dropna(subset=["t1"])
    ret = close.loc[ev["t1"].values].values / close.loc[ev.index].values - 1
    if "side" in ev:
        ret = ret * ev["side"].values
        return ret, np.where(ret > 0, 1, 0)
    return ret, np.sign(ret).astype(int)


vol = daily_vol(close)
t_events = cusum_filter(close, 0.02)
vertical = vertical_barrier(t_events, close, 1)
fmt = "%Y-%m-%d %H:%M:%S.%f"


def dump(events):
    ret, bins = get_bins(events, close)
    return {
        "t0": [t.strftime(fmt) for t in events.index],
        "t1": [None if pd.isna(t) else t.strftime(fmt) for t in events["t1"]],
        "trgt": [float(x) for x in events["trgt"]],
        "ret": [float(x) for x in ret],
        "bin": [int(x) for x in bins],
    }


out = {
    "source": "tests/fixtures/labeling/generate.py: AFML snippets 2.4, 3.1-3.5, pandas %s" % pd.__version__,
    "cusum_events": [t.strftime(fmt) for t in t_events],
    "events": dump(get_events(close, t_events, [1, 1], vol, 0.005, vertical, None)),
    "events_no_vertical": dump(get_events(close, t_events, [1, 1], vol, 0.005, None, None)),
    "meta_events": dump(get_events(close, t_events, [1, 1], vol, 0.005, vertical, pd.Series(1.0, index=close.index))),
}
(HERE / "reference.json").write_text(json.dumps(out, indent=2) + "\n")
for key in ("events", "events_no_vertical", "meta_events"):
    print(key, len(out[key]["t0"]), "bins", out[key]["bin"], "trgt[0], trgt[-1]", out[key]["trgt"][0], out[key]["trgt"][-1])
