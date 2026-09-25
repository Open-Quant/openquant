"""Reference values for the range-based volatility estimators, run in pandas.

    uv run --with pandas python tests/fixtures/volatility/generate_range.py

Independent of this library (imports neither openquant nor mlfinlab). Writes
range_reference.json next to this file. Input: the OHLC columns of
tests/fixtures/backtest_statistics/dollar_bar_sample.csv, rolling window n = 20. Each
estimator is a rolling variance over the last n bars (NaN until n values exist), then
its square root; the tests compare the mean of the non-NaN values plus a few samples.

* Parkinson (1980), eq. for sigma^2: 1/(4 ln 2) * mean over the window of ln(H/L)^2.
* Garman & Klass (1980), the "practical" estimator sigma_5^2 without the opening jump:
  mean over the window of 0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2.
* Yang & Zhang (2000), eq. (9)-(10): V = V_O + k V_C + (1 - k) V_RS with
  k = 0.34 / (1.34 + (n + 1)/(n - 1)),
  o_i = ln(O_i / C_{i-1}) (overnight), c_i = ln(C_i / O_i) (open to close),
  V_O = 1/(n-1) sum (o_i - mean o)^2, V_C = 1/(n-1) sum (c_i - mean c)^2,
  V_RS = 1/n sum [ln(H/C) ln(H/O) + ln(L/C) ln(L/O)] (Rogers & Satchell 1991).
  Recorded as `yang_zhang_paper`.

  Until #165 the library departed from the paper in three ways, and this script also
  recorded that form (`yang_zhang`) and a zero-mean variant: its "close" term was
  ln(C_i / O_{i-1}) rather than ln(C_i / O_i); V_O and V_C were raw second moments
  sum(x^2)/(n-1) rather than demeaned variances; and V_RS was divided by n - 1. The
  library now implements the paper, and only the paper's form is recorded.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
WINDOW = 20
bars = pd.read_csv(HERE.parent / "backtest_statistics" / "dollar_bar_sample.csv", index_col=0, parse_dates=[0])
o, h, l, c = (bars[k] for k in ("open", "high", "low", "close"))
n = WINDOW
k = 0.34 / (1.34 + (n + 1) / (n - 1))

parkinson = np.sqrt((np.log(h / l) ** 2 / (4 * np.log(2))).rolling(n).mean())
garman_klass = np.sqrt((0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2).rolling(n).mean())

overnight = np.log(o / c.shift(1))
open_close = np.log(c / o)
rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)

yz_paper = np.sqrt(overnight.rolling(n).var(ddof=1) + k * open_close.rolling(n).var(ddof=1)
                   + (1 - k) * rs.rolling(n).mean())


def summary(series):
    picks = [n - 1, n, 100, len(series) // 2, len(series) - 1]
    return {
        "n_nan": int(series.isna().sum()),
        "mean_excluding_nan": float(series.dropna().mean()),
        "samples": [{"position": int(i), "value": None if pd.isna(series.iloc[i]) else float(series.iloc[i])}
                    for i in picks],
    }


out = {
    "source": "Parkinson (1980), Garman & Klass (1980), Yang & Zhang (2000) in pandas %s "
              "on tests/fixtures/backtest_statistics/dollar_bar_sample.csv" % pd.__version__,
    "window": WINDOW,
    "n_bars": int(len(bars)),
    "parkinson": summary(parkinson),
    "garman_klass": summary(garman_klass),
    "yang_zhang_paper": summary(yz_paper),
}
(HERE / "range_reference.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
