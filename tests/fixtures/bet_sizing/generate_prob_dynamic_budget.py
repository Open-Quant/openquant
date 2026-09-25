"""Reference bet sizes (probability, dynamic, budget), independent of this library.

    uv run --with pandas --with scipy python tests/fixtures/bet_sizing/generate_prob_dynamic_budget.py

Writes prob_dynamic_budget.json next to this file. Imports neither openquant nor mlfinlab.

Inputs are the small hand-written cases of crates/openquant/tests/bet_sizing.rs:
  five events starting daily from 2000-01-01, each ending 24, 36, 48, 60, 72 hours after its
  start; predicted probabilities 0.55, 0.7, 0.95, 0.65, 0.85; sides +1, -1, +1, -1, +1; two
  classes.

Methods (AFML chapter 10):
  prob_default  snippet 10.1: z = (p - 1/K) / sqrt(p (1 - p)), m = side * (2 Phi(z) - 1).
  prob_avg      snippet 10.2: at every event start and every end time, the mean of the signals
                of the bets active then (start <= t < end), 0 if none is active.
  prob_step     snippet 10.3: m rounded to the nearest multiple of 0.1, clipped to [-1, 1].
  dynamic       snippet 10.4: sigmoid sizing, w calibrated so that a divergence of 10 gives
                m = 0.95 (getW), target position int(m * maxPos) (getTPos), and the
                limit price as written in the snippet (limitPrice), in l_p.
                l_p_path is the limit price as openquant defines it wherever the snippet's
                loop does not describe the move (reducing, short targets, crossing zero):
                the mean of invPrice(f, w, k / maxPos) over the signed positions
                k = pos + sgn, pos + 2 sgn, ..., tPos. It equals l_p when 0 <= pos < tPos.
  budget        section 10.2: m_t = c_long,t / max_i c_long,i - c_short,t / max_i c_short,i,
                where c counts the bets active at t (start <= t < end), t running over event starts.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).parent

start = pd.date_range("2000-01-01", periods=5, freq="D")
t1 = pd.Series([s + pd.Timedelta(hours=int(24 * (0.5 * i + 1))) for i, s in enumerate(start)], index=start)
prob = pd.Series([0.55, 0.7, 0.95, 0.65, 0.85], index=start)
side = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0], index=start)


def get_signal(prob, num_classes, side):  # snippet 10.1
    z = (prob - 1.0 / num_classes) / (prob * (1.0 - prob)) ** 0.5
    return side * (2 * norm.cdf(z) - 1)


def avg_active_signals(signals, t1):  # snippet 10.2, single process
    t_pnts = sorted(set(t1.dropna().values) | set(signals.index.values))
    out = pd.Series(0.0, index=pd.DatetimeIndex(t_pnts))
    for loc in out.index:
        act = (signals.index.values <= loc) & ((loc < t1.values) | pd.isnull(t1.values))
        act = signals[act].index
        out[loc] = signals.loc[act].mean() if len(act) > 0 else 0.0
    return out


def discrete_signal(signal0, step_size):  # snippet 10.3
    signal1 = (signal0 / step_size).round() * step_size
    return signal1.clip(-1, 1)


def bet_size(w, x):  # snippet 10.4
    return x * (w + x ** 2) ** -0.5


def get_t_pos(w, f, m_p, max_pos):
    return int(bet_size(w, f - m_p) * max_pos)


def inv_price(f, w, m):
    return f - m * (w / (1 - m ** 2)) ** 0.5


def limit_price(t_pos, pos, f, w, max_pos):
    sgn = 1 if t_pos >= pos else -1
    l_p = 0.0
    for j in range(abs(pos + sgn), abs(t_pos + 1)):
        l_p += inv_price(f, w, j / float(max_pos))
    return l_p / (t_pos - pos)


def limit_price_path(t_pos, pos, f, w, max_pos):
    step = 1 if t_pos > pos else -1
    path = list(range(pos + step, t_pos + step, step))
    return sum(inv_price(f, w, k / float(max_pos)) for k in path) / len(path)


def get_w(x, m):
    return x ** 2 * (m ** -2 - 1)


signal0 = pd.Series(get_signal(prob, 2, side), index=start)
prob_default = list(signal0)
prob_avg = list(avg_active_signals(signal0, t1))
prob_step = list(discrete_signal(signal0, 0.1))

pos = [25, 35, 45, 40, 30]
max_pos = 55
m_p = [75.5, 76.9, 74.1, 67.75, 62.0]
f = [80.0, 75.0, 72.5, 65.0, 70.8]
w = get_w(10, 0.95)
dynamic = {"bet_size": [], "t_pos": [], "l_p": [], "l_p_path": []}
for p_, mp_, f_ in zip(pos, m_p, f):
    t_pos = get_t_pos(w, f_, mp_, max_pos)
    dynamic["bet_size"].append(bet_size(w, f_ - mp_))
    dynamic["t_pos"].append(t_pos)
    # The snippet's loop is empty when the target is on the other side of zero from the
    # current position, so its limit price is 0 there; recorded as written.
    dynamic["l_p"].append(limit_price(t_pos, p_, f_, w, max_pos) + 0.0)
    dynamic["l_p_path"].append(limit_price_path(t_pos, p_, f_, w, max_pos))


def concurrent(t1, side):
    long_, short_ = [], []
    for t in t1.index:
        active = (t1.index <= t) & (t1.values > t)
        long_.append(int(((side > 0) & active).sum()))
        short_.append(int(((side < 0) & active).sum()))
    return np.array(long_, float), np.array(short_, float)


c_long, c_short = concurrent(t1, side)
budget = c_long / c_long.max() - c_short / c_short.max()

out = {
    "source": "tests/fixtures/bet_sizing/generate_prob_dynamic_budget.py: AFML snippets 10.1-10.4 "
              "and section 10.2, pandas %s, scipy.stats.norm" % pd.__version__,
    "prob_default": [float(x) for x in prob_default],
    "prob_avg": [float(x) for x in prob_avg],
    "prob_step": [float(x) for x in prob_step],
    "dynamic": {"bet_size": [float(x) for x in dynamic["bet_size"]],
                "t_pos": dynamic["t_pos"],
                "l_p": [float(x) for x in dynamic["l_p"]],
                "l_p_path": [float(x) for x in dynamic["l_p_path"]]},
    "budget": {"bet_size": [float(x) for x in budget]},
}
(HERE / "prob_dynamic_budget.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
