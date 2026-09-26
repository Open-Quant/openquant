"""Reference values for the bar-based microstructural features, from AFML ch. 19.

    uv run --with pandas --with scipy python tests/fixtures/microstructural_features/generate.py

Independent of this library. Reads tests/fixtures/shared/dollar_bar_sample.csv and writes reference.json
next to this file. Every feature uses a 20-bar window.

Sources (Lopez de Prado, "Advances in Financial Machine Learning", 2018):
  * Roll measure, section 19.3.1: 2 * sqrt(|cov(dp_t, dp_{t-1})|), with the
    covariance taken over a rolling window (sample covariance, ddof=1).
  * Roll impact: the Roll measure divided by the bar's dollar volume.
  * Corwin-Schultz spread, snippets 19.1 (beta, gamma, alpha, spread).
  * Becker-Parkinson volatility, snippet 19.2 (sigma from beta and gamma).
  * Kyle, Amihud and Hasbrouck lambdas, sections 19.4.1-19.4.3. The book states
    each as a regression; the bar-based versions here are the rolling mean of the
    per-bar ratio of the regression's left-hand side to its regressor:
        Kyle       dp_t / (b_t * V_t)
        Amihud     |log(p_t / p_{t-1})| / (p_t V_t)
        Hasbrouck  log(p_t / p_{t-1}) / (b_t * sqrt(p_t V_t))
    with b_t the sign of the price change, carried forward over unchanged bars.
  * Bulk volume classification, section 19.3.2: V_buy = V * Z(dp / sigma_dp),
    sigma_dp the rolling standard deviation (ddof=1) of the price changes.
  * VPIN, section 19.5.1: sum |V_sell - V_buy| / (n V), taken here as the rolling
    mean of |V_buy - V_sell| over n bars divided by the current bar's volume.

Columns used, as the tests pass them: close; high and low; cum_dollar as the
per-bar dollar volume (Roll impact, Amihud, Hasbrouck); cum_vol as the per-bar
volume (Kyle, BVC, VPIN).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm

HERE = Path(__file__).parent
WINDOW = 20
bars = pd.read_csv(HERE.parent / "shared" / "dollar_bar_sample.csv", index_col=0, parse_dates=[0])
close, high, low = bars["close"], bars["high"], bars["low"]
dollar_volume, volume = bars["cum_dollar"], bars["cum_vol"]


def roll_measure(close, window):
    dp = close.diff()
    return 2 * np.sqrt(abs(dp.rolling(window).cov(dp.shift(1))))


def get_beta(high, low, sl):
    # Snippet 19.1, with pandas' current rolling API.
    hl = np.log(high / low) ** 2
    beta = hl.rolling(2).sum()
    return beta.rolling(sl).mean()


def get_gamma(high, low):
    h2 = high.rolling(2).max()
    l2 = low.rolling(2).min()
    return np.log(h2 / l2) ** 2


def get_alpha(beta, gamma):
    den = 3 - 2 * 2**0.5
    alpha = (2**0.5 - 1) * (beta**0.5) / den
    alpha -= (gamma / den) ** 0.5
    alpha[alpha < 0] = 0
    return alpha


def corwin_schultz(high, low, sl):
    alpha = get_alpha(get_beta(high, low, sl), get_gamma(high, low))
    return 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))


def becker_parkinson(high, low, sl):
    # Snippet 19.2.
    beta, gamma = get_beta(high, low, sl), get_gamma(high, low)
    k2 = (8 / np.pi) ** 0.5
    den = 3 - 2 * 2**0.5
    sigma = (2**-0.5 - 1) * beta**0.5 / (k2 * den)
    sigma += (gamma / (k2**2 * den)) ** 0.5
    sigma[sigma < 0] = 0
    return sigma


def carried_sign(x):
    # Sign of x; a zero takes the previous bar's sign (tick rule).
    return np.sign(x).replace(0, np.nan).ffill().where(x.notna())


def kyle_lambda(close, volume, window):
    dp = close.diff()
    return (dp / (volume * carried_sign(dp))).rolling(window).mean()


def amihud_lambda(close, dollar_volume, window):
    r = np.log(close / close.shift(1)).abs()
    return (r / dollar_volume).rolling(window).mean()


def hasbrouck_lambda(close, dollar_volume, window):
    r = np.log(close / close.shift(1))
    return (r / (carried_sign(r) * np.sqrt(dollar_volume))).rolling(window).mean()


def bvc_buy_volume(close, volume, window):
    dp = close.diff()
    return volume * norm.cdf(dp / dp.rolling(window).std())


def vpin(volume, buy_volume, window):
    imbalance = (buy_volume - (volume - buy_volume)).abs()
    return imbalance.rolling(window).mean() / volume


def summary(series, position):
    finite = series[np.isfinite(series)]
    return {
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "position": position,
        "at_position": float(series.iloc[position]),
        "first_finite": int(np.flatnonzero(np.isfinite(series.to_numpy()))[0]),
    }


buy = bvc_buy_volume(close, volume, WINDOW)
out = {
    "source": "AFML ch. 19 in pandas %s / scipy %s on dollar_bar_sample.csv"
    % (pd.__version__, scipy.__version__),
    "window": WINDOW,
    "n_bars": int(close.shape[0]),
    "roll_measure": summary(roll_measure(close, WINDOW), 25),
    "roll_impact": summary(roll_measure(close, WINDOW) / dollar_volume, 25),
    "corwin_schultz": summary(corwin_schultz(high, low, WINDOW), 25),
    "becker_parkinson": summary(becker_parkinson(high, low, WINDOW), 25),
    "kyle_lambda": summary(kyle_lambda(close, volume, WINDOW), 25),
    "amihud_lambda": summary(amihud_lambda(close, dollar_volume, WINDOW), 25),
    "hasbrouck_lambda": summary(hasbrouck_lambda(close, dollar_volume, WINDOW), 25),
    "vpin_1": summary(vpin(volume, buy, 1), 25),
    "vpin_20": summary(vpin(volume, buy, 20), 45),
}
(HERE / "reference.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
