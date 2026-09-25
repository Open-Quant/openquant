"""Reference for the "reserve" bet-sizing method on a seeded synthetic event set.

    uv run --with pandas --with scipy python tests/fixtures/bet_sizing/generate_reserve.py

Writes reserve_fixture.json next to this file. Imports neither openquant nor mlfinlab.

Method: AFML section 10.2 (strategy-independent bet sizing, "reserve" approach).
  1. For each event start t, c_t = (bets long and active at t) - (bets short and active at t),
     where a bet is active at t when start <= t < t1.
  2. Fit a mixture of two Gaussians to {c_t}. AFML uses EF3M (Lopez de Prado and Foreman,
     2014); any fit will do here, because the test takes the fitted parameters as given and
     checks what is done with them. This script uses maximum likelihood by EM with numpy,
     from fixed starting points.
  3. m_t = (F(c_t) - F(0)) / (1 - F(0)) if c_t >= 0, else (F(c_t) - F(0)) / F(0), where F is
     the fitted mixture CDF.

Synthetic data (numpy default_rng(138)): 500 events starting daily from 2000-01-01, each
lasting a uniform 2 to 20 days (whole microseconds), side +1 or -1 with equal probability.

fit = [mu_1, mu_2, sigma_1, sigma_2, p_1].
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).parent
rng = np.random.default_rng(138)
n = 500
start = pd.date_range("2000-01-01", periods=n, freq="D")
duration_us = np.round(rng.uniform(2, 20, n) * 86400e6).astype(np.int64)
t1 = start + pd.to_timedelta(duration_us, unit="us")
side = rng.choice([-1, 1], size=n)

active_long = np.zeros(n, dtype=int)
active_short = np.zeros(n, dtype=int)
for i, t in enumerate(start):
    active = (start <= t) & (t1 > t)
    active_long[i] = int((active & (side > 0)).sum())
    active_short[i] = int((active & (side < 0)).sum())
c_t = active_long - active_short


def em_two_normals(x, starts, tol=1e-12, max_iter=10_000):
    best = None
    for mu1, mu2 in starts:
        s1 = s2 = x.std()
        p1 = 0.5
        ll_prev = -np.inf
        for _ in range(max_iter):
            d1 = p1 * norm.pdf(x, mu1, s1)
            d2 = (1 - p1) * norm.pdf(x, mu2, s2)
            ll = np.log(d1 + d2).sum()
            g = d1 / (d1 + d2)
            n1, n2 = g.sum(), (1 - g).sum()
            p1 = n1 / len(x)
            mu1, mu2 = (g * x).sum() / n1, ((1 - g) * x).sum() / n2
            s1 = np.sqrt((g * (x - mu1) ** 2).sum() / n1)
            s2 = np.sqrt(((1 - g) * (x - mu2) ** 2).sum() / n2)
            if abs(ll - ll_prev) < tol:
                break
            ll_prev = ll
        if best is None or ll > best[0]:
            best = (ll, [mu1, mu2, s1, s2, p1])
    return best[1]


x = c_t.astype(float)
q = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])
fit = em_two_normals(x, [(q[0], q[4]), (q[1], q[3]), (q[0], q[2]), (q[2], q[4])])
mu1, mu2, s1, s2, p1 = fit


def cdf(v):
    return p1 * norm.cdf(v, mu1, s1) + (1 - p1) * norm.cdf(v, mu2, s2)


f0 = cdf(0.0)
bet_size = np.where(c_t >= 0, (cdf(x) - f0) / (1 - f0), (cdf(x) - f0) / f0)

fmt = "%Y-%m-%d %H:%M:%S.%f"
out = {
    "source": "tests/fixtures/bet_sizing/generate_reserve.py: AFML section 10.2, numpy %s, "
              "scipy.stats.norm; synthetic events from numpy default_rng(138)" % np.__version__,
    "fit": [float(v) for v in fit],
    "events_active": {
        "index": [t.strftime(fmt) for t in start],
        "t1": [t.strftime(fmt) for t in t1],
        "side": [int(v) for v in side],
        "active_long": [int(v) for v in active_long],
        "active_short": [int(v) for v in active_short],
        "c_t": [int(v) for v in c_t],
        "bet_size": [float(v) for v in bet_size],
    },
}
(HERE / "reserve_fixture.json").write_text(json.dumps(out, indent=2) + "\n")
print("fit", fit, "c_t range", c_t.min(), c_t.max(), "F(0)", f0)
