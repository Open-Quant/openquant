"""Reference values for the structural-break tests: AFML chapter 17, in numpy.

    uv run --with pandas python tests/fixtures/structural_breaks/generate.py

Independent of this library (imports neither openquant nor mlfinlab). Input is
the log of `close` in dollar_bar_sample.csv next to this file. Writes
reference.json next to this file.

* 17.3.1 Chow-type Dickey-Fuller: dy_t = delta * y_{t-1} * D_t + e_t, no
  intercept, DFC = delta_hat / se(delta_hat), one value per break bar in
  [min_length, T - min_length).
* 17.3.2 Chu-Stinchcombe-White CUSUM on levels:
  S_{n,t} = (y_t - y_n) / (sigma_t * sqrt(t - n)),
  sigma_t^2 = (t - 1)^-1 * sum_{i=2..t} (dy_i)^2 (1-based t),
  c_alpha[n, t] = sqrt(b_alpha + log(t - n)), b_alpha = 4.6. The reported
  statistic is max_n S_{n,t} (|y_t - y_n| for the two-sided test) and the
  critical value is the one at the arg-max n.
* 17.4.2 SADF, snippets 17.1-17.4 (get_bsadf, getYX, lagDF, getBetas):
  regress dy_t on [y_{t-1}, dy_{t-l} for each lag l, const, trend(, trend^2)],
  t-stat of the y_{t-1} coefficient with Var = e'e / (n - k) * (X'X)^-1, and
  SADF_t = sup over window starts of that t-stat.
* 17.4.3 sub/super-martingale tests: SM-Poly1 (y on 1, t, t^2), SM-Poly2
  (log y on 1, t, t^2), SM-Exp (log y on 1, t), SM-Power (log y on 1, log t);
  SMT_t = sup over window starts of |beta_hat| / se(beta_hat).

Conventions shared with the library (not specified by AFML): the regression
rows are bars max_lag+1 .. T-1 for every model (so all models give series of
equal length); the trend is the 0-based row position over the full sample; one
SADF value per row position pos >= min_length, taking the sup over windows of
at least min_length rows; "linear" with add_const=False keeps the trend and
drops only the constant; a lag list [1, 2, 5, 7] means dy lags 1, 2, 5, 7 in
addition to the lagged level; the SM models take y (or log y) of the series
passed in.

The SM-Power time index is t = row position + 1 (AFML's log t needs t >= 1).
Where the library departs from AFML the AFML value is the reference and the
library's convention is recomputed under "library_convention" (see "notes").
The SADF models match AFML since #166, so only the Chu-Stinchcombe-White
statistic is recomputed there. "sadf_prefix" also records every value on the
60-bar prefix, so each model is checked value by value.

Each regression is computed from a Givens-updated QR factor of [X | y] (one per
window start, grown one row at a time), so no normal equations are formed.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
LOG_P = np.log(pd.read_csv(HERE / "dollar_bar_sample.csv")["close"].to_numpy(dtype=float))


# --- 17.3.1 Chow-type DF -------------------------------------------------------------
def chow_type_stat(y, min_length):
    dy = np.diff(y)
    lag = y[:-1]
    out = []
    for index in range(min_length, len(y) - min_length):
        x = lag.copy()
        x[:index] = 0.0  # D_t = 0 before the break
        b = x @ dy / (x @ x)
        err = dy - b * x
        var = (err @ err) / (len(x) - 1) / (x @ x)
        out.append(b / np.sqrt(var))
    return np.array(out)


# --- 17.3.2 Chu-Stinchcombe-White ----------------------------------------------------
def chu_stinchcombe_white(y, test_type, afml=True):
    dy2 = np.diff(y) ** 2
    crit, stat = [], []
    for t in range(2, len(y)):  # 0-based t; AFML's 1-based t is t + 1
        ssq = dy2[:t].sum()  # the t differences up to bar t
        if afml:
            sigma = np.sqrt(ssq / t)  # (t_afml - 1)^-1 * sum, then sigma (not sigma^2)
        else:
            sigma = ssq / (t - 1)  # library: divisor t - 1 and variance in the denominator
        diff = y[t] - y[:t]
        if test_type == "two_sided":
            diff = np.abs(diff)
        dist = t - np.arange(t)
        s = diff / (sigma * np.sqrt(dist))
        j = int(np.argmax(s))  # first arg-max
        stat.append(s[j])
        crit.append(np.sqrt(4.6 + np.log(dist[j])))
    return np.array(crit), np.array(stat)


# --- 17.4.2 / 17.4.3 SADF -------------------------------------------------------------
def get_y_x(series, model, lags, add_const):
    """Snippet 17.2 getYX (with snippet 17.3 lagDF) plus the 17.4.3 SM designs.

    Returns y, X and the column of X whose t-stat is taken.
    """
    lags = list(range(1, lags + 1)) if isinstance(lags, int) else list(lags)
    dy = np.diff(series)
    rows = np.arange(max(lags) + 1, len(series))  # bar index of each regression row
    trend = np.arange(len(rows), dtype=float)
    if model in ("linear", "quadratic"):
        cols = [series[rows - 1]] + [dy[rows - 1 - l] for l in lags]
        if add_const:
            cols.append(np.ones(len(rows)))
        if model == "quadratic":
            cols += [trend, trend ** 2]  # AFML 'ctt'
        else:
            cols.append(trend)
        return dy[rows - 1], np.column_stack(cols)
    if model == "sm_poly_1":
        return series[rows], np.column_stack([trend ** 2, np.ones(len(rows)), trend])
    if model == "sm_poly_2":
        return np.log(series[rows]), np.column_stack([trend ** 2, np.ones(len(rows)), trend])
    if model == "sm_exp":
        return np.log(series[rows]), np.column_stack([trend, np.ones(len(rows))])
    if model == "sm_power":
        return np.log(series[rows]), np.column_stack([np.log(trend + 1.0), np.ones(len(rows))])
    raise ValueError(model)


def sadf(series, model, lags, add_const, min_length):
    """SADF_t for every row position t >= min_length (snippets 17.1 and 17.4).

    R[s] is the R factor of [X | y] over rows s..t for window start s; row t is
    folded into every R[s] with s <= t by Givens rotations. With k regressors,
    R[s][:k, :k] b = R[s][:k, k] gives beta and R[s][k, k]^2 is e'e.
    """
    y, x = get_y_x(series, model, lags, add_const)
    n, k = x.shape
    use_abs = model.startswith("sm_")
    a = np.column_stack([x, y])
    m = k + 1
    R = np.zeros((n, m, m))
    out = []
    for t in range(n):
        v = np.tile(a[t], (t + 1, 1))
        Rt = R[: t + 1]
        for j in range(m):
            p, q = Rt[:, j, j].copy(), v[:, j].copy()
            r = np.hypot(p, q)
            safe = r > 0
            c = np.where(safe, p / np.where(safe, r, 1.0), 1.0)
            s = np.where(safe, q / np.where(safe, r, 1.0), 0.0)
            top, bot = Rt[:, j, j:].copy(), v[:, j:].copy()
            Rt[:, j, j:] = c[:, None] * top + s[:, None] * bot
            v[:, j:] = -s[:, None] * top + c[:, None] * bot
        if t < min_length:
            continue
        starts = np.arange(t - min_length + 2)  # windows of >= min_length rows
        Rs = R[starts]
        rinv = np.linalg.inv(Rs[:, :k, :k])
        beta0 = np.einsum("ij,ij->i", rinv[:, 0, :], Rs[:, :k, k])
        sse = Rs[:, k, k] ** 2
        dof = (t - starts + 1) - k
        se0 = np.sqrt(sse / dof * np.einsum("ij,ij->i", rinv[:, 0, :], rinv[:, 0, :]))
        stat = beta0 / se0
        out.append(np.max(np.abs(stat) if use_abs else stat))
    return np.array(out)


def stats(v):
    return {"len": int(len(v)), "max": float(np.max(v)), "mean": float(np.mean(v))}


started = time.time()

chow = chow_type_stat(LOG_P, 10)
csw = {}
csw_lib = {}
for tt in ("one_sided", "two_sided"):
    crit, st = chu_stinchcombe_white(LOG_P, tt, afml=True)
    _, st_lib = chu_stinchcombe_white(LOG_P, tt, afml=False)
    csw[tt] = {"critical_value": {**stats(crit), "at_20": float(crit[20])},
               "stat": {**stats(st), "at_20": float(st[20])}}
    csw_lib[tt] = {"stat": {**stats(st_lib), "at_20": float(st_lib[20])}}

MIN_LENGTH, LAGS = 20, 5
CASES = {  # name -> (model, lags, add_const)
    "sm_power": ("sm_power", LAGS, True),
    "linear": ("linear", LAGS, True),
    "linear_no_const": ("linear", [1, 2, 5, 7], False),
    "quadratic": ("quadratic", LAGS, True),
    "sm_poly_1": ("sm_poly_1", LAGS, True),
    "sm_poly_2": ("sm_poly_2", LAGS, True),
    "sm_exp": ("sm_exp", LAGS, True),
}
PREFIX = 60

sadf_out, prefix_out = {}, {}
for name, (model, lags, add_const) in CASES.items():
    full = sadf(LOG_P, model, lags, add_const, MIN_LENGTH)
    pre = sadf(LOG_P[:PREFIX], model, lags, add_const, MIN_LENGTH)
    assert np.array_equal(pre, full[: len(pre)]), name
    sadf_out[name] = {**stats(full), "at_29": float(full[29])}
    prefix_out[name] = {"len": int(len(pre)), "at_29": float(pre[29]),
                        "values": [float(v) for v in pre]}

out = {
    "source": "AFML ch. 17 (17.3.1, 17.3.2, snippets 17.1-17.4, 17.4.3) in numpy %s on "
              "log(close) of tests/fixtures/structural_breaks/dollar_bar_sample.csv" % np.__version__,
    "n_bars": int(len(LOG_P)),
    "chow": {"min_length": 10, **stats(chow), "at_3": float(chow[3])},
    "chu_stinchcombe_white": csw,
    "sadf": {"min_length": MIN_LENGTH, "lags": LAGS, "lags_array": [1, 2, 5, 7], "models": sadf_out},
    "sadf_prefix": {"n_bars": PREFIX, "models": prefix_out},
    "library_convention": {
        "chu_stinchcombe_white": csw_lib,
    },
    "notes": {
        "chu_stinchcombe_white": "AFML divides y_t - y_n by sigma_t (a standard deviation) with "
            "sigma_t^2 = sum of the t-1 squared differences / (t-1). The library divides by the "
            "variance and uses t-2 as the divisor. Critical values do not depend on either.",
        "sadf": "Until #166 the library's 'quadratic' omitted the linear t of AFML's 'ctt', its "
            "sm_* models took the sup of the signed beta / se rather than |beta| / se, and "
            "sm_power used log(0) on the first row. It now matches the values here.",
    },
}
(HERE / "reference.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
print("took %.1f s" % (time.time() - started))
