"""Reference values for backtest statistics: AFML ch. 14, run in numpy/scipy/pandas.

    uv run --with pandas --with scipy python tests/fixtures/backtest_statistics/generate.py

Independent of this library (imports neither openquant nor mlfinlab). Writes
reference.json next to this file. Inputs are the ones the tests use.

Sources and formulas
--------------------
* Concentration of bets, AFML snippet 14.3 (`getHHI`): for returns r_1..r_n (n > 2),
  w_i = r_i / sum(r), HHI = sum(w_i^2), normalised HHI = (HHI - 1/n) / (1 - 1/n).
  `bets_concentration` is getHHI on all log returns of dollar_bar_sample.csv (the sum
  of the weights is 1 but they have mixed signs, so the value may exceed 1).
  `all_bets_concentration`: getHHI(ret[ret >= 0]), getHHI(ret[ret < 0]) and getHHI of the
  number of returns per calendar period, empty periods counted as 0.
  Library convention: the period is one calendar DAY. The book's snippet uses
  `TimeGrouper(freq='M')`; the sample covers January 2015 only, so a monthly grouping
  would give a single count and getHHI would return NaN, so the daily form is the
  only one with a value on this sample.
* Sharpe ratio: (mean(r) - rf) / std(r, ddof=1) * sqrt(periods per year), with rf a
  per-period rate subtracted from every return (library convention; the ddof=1 sample
  standard deviation is what pandas' `.std()` gives). Information ratio: the Sharpe
  ratio of r - benchmark with rf = 0. Returns are [0.03, 0.02, 0.01, -0.01, 0.02, 0.01,
  0.0, -0.01, 0.01], 12 periods a year, rf 0.005 and benchmark 0.006.
* Probabilistic Sharpe ratio, AFML section 14.7.1 / Bailey & Lopez de Prado (2012):
  PSR = Phi[(SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2)], with g3 the
  skewness and g4 the (non-excess) kurtosis.
* Deflated Sharpe ratio, AFML section 14.7.2 / Bailey & Lopez de Prado (2014): PSR with
  SR* = sqrt(V[SR_n]) ((1 - gamma) Phi^-1[1 - 1/N] + gamma Phi^-1[1 - 1/(N e)]),
  gamma the Euler-Mascheroni constant.
  - From trial Sharpe ratios [3.5, 1.01, 1.02]: N = 3 and V[SR_n] their variance.
    Library convention: the population variance (ddof=0, numpy's default `.std()`);
    `dsr_sample_variance` records the ddof=1 value for comparison.
  - From parameters [0.4, 100]: sqrt(V[SR_n]) = 0.4, N = 100 (estimates_param=True);
    `benchmark_out=True` returns SR* itself.
* Minimum track record length, Bailey & Lopez de Prado (2012), eq. for MinTRL:
  1 + (1 - g3 SR + (g4 - 1)/4 SR^2) (Phi^-1[1 - alpha] / (SR - SR*))^2.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm

HERE = Path(__file__).parent
EULER_GAMMA = float(np.euler_gamma)

bars = pd.read_csv(HERE.parent / "shared" / "dollar_bar_sample.csv", index_col=0, parse_dates=[0])
close = bars["close"]
log_ret = np.log(close).diff().dropna()


def get_hhi(bet_ret):
    # Snippet 14.3, verbatim apart from names.
    if bet_ret.shape[0] <= 2:
        return float("nan")
    wght = bet_ret / bet_ret.sum()
    hhi = (wght ** 2).sum()
    hhi = (hhi - bet_ret.shape[0] ** -1) / (1.0 - bet_ret.shape[0] ** -1)
    return float(hhi)


def sharpe(returns, periods, rf):
    r = pd.Series(returns)
    return float((r.mean() - rf) / r.std(ddof=1) * np.sqrt(periods))


def psr(sr, sr_star, t, skew, kurt):
    z = (sr - sr_star) * np.sqrt(t - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(norm.cdf(z))


def expected_max_sr(sd, n):
    return float(sd * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n)
                       + EULER_GAMMA * norm.ppf(1 - 1 / (n * np.e))))


def min_trl(sr, sr_star, skew, kurt, alpha):
    return float(1 + (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) * (norm.ppf(1 - alpha) / (sr - sr_star)) ** 2)


returns = [0.03, 0.02, 0.01, -0.01, 0.02, 0.01, 0.0, -0.01, 0.01]
trials = np.array([3.5, 1.01, 1.02])
sr_star_trials = expected_max_sr(trials.std(ddof=0), len(trials))
sr_star_param = expected_max_sr(0.4, 100)

out = {
    "source": "AFML ch. 14 (snippet 14.3, section 14.7) and Bailey & Lopez de Prado (2012, 2014) "
              "in numpy %s / scipy %s / pandas %s" % (np.__version__, scipy.__version__, pd.__version__),
    "bets_concentration": get_hhi(log_ret),
    "bets_concentration_negated": get_hhi(-log_ret),
    "all_bets_concentration": {
        "positive": get_hhi(log_ret[log_ret >= 0]),
        "negative": get_hhi(log_ret[log_ret < 0]),
        "time": get_hhi(log_ret.groupby(pd.Grouper(freq="D")).count()),
    },
    "sharpe_ratio": sharpe(returns, 12, 0.005),
    "information_ratio": sharpe([r - 0.006 for r in returns], 12, 0.0),
    "psr": psr(1.14, 1.0, 250, 0.0, 3.0),
    "dsr": psr(1.14, sr_star_trials, 250, 0.0, 3.0),
    "dsr_sample_variance": psr(1.14, expected_max_sr(trials.std(ddof=1), len(trials)), 250, 0.0, 3.0),
    "dsr_benchmark_from_params": sr_star_param,
    "dsr_from_params": psr(1.14, sr_star_param, 250, 0.0, 3.0),
    "min_trl": min_trl(1.14, 1.0, 0.0, 3.0, 0.05),
}
(HERE / "reference.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
